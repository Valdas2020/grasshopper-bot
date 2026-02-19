"""
GrassShopper — семейный список покупок, управляемый голосом.
Архитектура: aiogram 3.x + OpenAI Whisper + GPT-4o-mini + SQLite

Режимы запуска:
  - Production (Render): webhook через aiohttp, если задан RENDER_EXTERNAL_URL
  - Local dev:           long polling (python bot.py)
"""

import asyncio
import io
import json
import logging
import os
import sqlite3

from aiogram import Bot, Dispatcher, F, Router
from aiogram.filters import Command, StateFilter
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram.types import Message
from aiogram.webhook.aiohttp_server import SimpleRequestHandler, setup_application
from aiohttp import web
from dotenv import load_dotenv
from openai import AsyncOpenAI

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ── Config ─────────────────────────────────────────────────────────────────────

TELEGRAM_TOKEN: str = os.environ["TELEGRAM_TOKEN"]

# Whisper transcription — direct OpenAI API
OPENAI_API_KEY: str = os.environ["OPENAI_API_KEY"]

# GPT-4o-mini NLU — RouteLLM proxy (cheaper)
ROUTELLM_API_KEY: str = os.environ["ROUTELLM_API_KEY"]
ROUTELLM_BASE_URL: str = os.getenv("ROUTELLM_BASE_URL", "https://routellm.abacus.ai/v1/")
ROUTELLM_MODEL: str = os.getenv("ROUTELLM_MODEL", "gpt-4o-mini")

ALLOWED_USERS: list[int] = [
    int(uid.strip())
    for uid in os.environ["ALLOWED_USERS"].split(",")
    if uid.strip().isdigit()
]

# Render injects RENDER_EXTERNAL_URL automatically (e.g. https://grasshopper-bot.onrender.com)
RENDER_EXTERNAL_URL: str = os.getenv("RENDER_EXTERNAL_URL", "").rstrip("/")
PORT: int = int(os.getenv("PORT", "8080"))

# Webhook path includes the token so only Telegram can reach it
WEBHOOK_PATH = f"/webhook/{TELEGRAM_TOKEN}"

# SQLite lives on the persistent disk on Render (/data), locally in cwd
DB_PATH: str = os.getenv("DB_PATH", "shopping.db")

if not ALLOWED_USERS:
    raise ValueError("ALLOWED_USERS is not set or contains no valid IDs")

# ── Bot & clients ──────────────────────────────────────────────────────────────

bot = Bot(token=TELEGRAM_TOKEN)
dp = Dispatcher(storage=MemoryStorage())
router = Router()
dp.include_router(router)

# Whisper: direct OpenAI (audio transcriptions endpoint)
whisper_client = AsyncOpenAI(api_key=OPENAI_API_KEY)

# GPT-4o-mini NLU: RouteLLM proxy
routellm_client = AsyncOpenAI(api_key=ROUTELLM_API_KEY, base_url=ROUTELLM_BASE_URL)

# ── Database ───────────────────────────────────────────────────────────────────


def init_db() -> None:
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True) if os.path.dirname(DB_PATH) else None
    con = sqlite3.connect(DB_PATH)
    con.execute("""
        CREATE TABLE IF NOT EXISTS shopping_list (
            id       INTEGER PRIMARY KEY AUTOINCREMENT,
            item     TEXT    NOT NULL,
            amount   TEXT,
            location TEXT
        )
    """)
    con.execute("""
        CREATE TABLE IF NOT EXISTS glossary (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            spoken    TEXT    NOT NULL UNIQUE,
            canonical TEXT    NOT NULL
        )
    """)
    con.commit()
    con.close()


def db_get_list() -> list[tuple]:
    con = sqlite3.connect(DB_PATH)
    rows = con.execute(
        "SELECT item, amount, location FROM shopping_list ORDER BY id"
    ).fetchall()
    con.close()
    return rows


def db_add_item(item: str, amount: str | None, location: str | None) -> None:
    con = sqlite3.connect(DB_PATH)
    con.execute(
        "INSERT INTO shopping_list (item, amount, location) VALUES (?, ?, ?)",
        (item, amount, location),
    )
    con.commit()
    con.close()


def db_remove_item(item: str) -> int:
    """Fuzzy delete: matches if stored item CONTAINS the given substring."""
    con = sqlite3.connect(DB_PATH)
    cur = con.execute(
        "DELETE FROM shopping_list WHERE item LIKE ?",
        (f"%{item}%",),
    )
    deleted = cur.rowcount
    con.commit()
    con.close()
    return deleted


def db_get_glossary() -> list[tuple]:
    con = sqlite3.connect(DB_PATH)
    rows = con.execute(
        "SELECT spoken, canonical FROM glossary ORDER BY id"
    ).fetchall()
    con.close()
    return rows


def db_save_glossary_pair(spoken: str, canonical: str) -> None:
    con = sqlite3.connect(DB_PATH)
    con.execute(
        "INSERT OR REPLACE INTO glossary (spoken, canonical) VALUES (?, ?)",
        (spoken.lower().strip(), canonical.strip()),
    )
    con.commit()
    con.close()


# ── Glossary substitution ──────────────────────────────────────────────────────


def apply_glossary(text: str) -> str:
    """Replace all spoken variants with canonical store names."""
    pairs = db_get_glossary()
    result = text.lower()
    for spoken, canonical in pairs:
        result = result.replace(spoken.lower(), canonical)
    return result


# ── Formatting ─────────────────────────────────────────────────────────────────

HEADER = "🛒 *GrassShopper — список покупок*"


def format_shopping_list(rows: list[tuple]) -> str:
    if not rows:
        return f"{HEADER}\n\n_Список пуст_ ✨"
    lines = [f"{HEADER}\n"]
    for i, (item, amount, location) in enumerate(rows, 1):
        line = f"{i}. *{item}*"
        if amount:
            line += f" — {amount}"
        if location:
            line += f" ({location})"
        lines.append(line)
    return "\n".join(lines)


# ── OpenAI helpers ─────────────────────────────────────────────────────────────


async def transcribe_voice(file_bytes: bytes) -> str:
    """Transcribe OGG audio bytes using OpenAI Whisper (direct OpenAI API)."""
    buf = io.BytesIO(file_bytes)
    buf.name = "voice.ogg"
    response = await whisper_client.audio.transcriptions.create(
        model="whisper-1",
        file=buf,
        language="ru",
    )
    return response.text.strip()


NLU_SYSTEM_PROMPT = """\
Ты — помощник для управления семейным списком покупок.
Из сообщения пользователя извлеки намерение и верни ТОЛЬКО валидный JSON-объект с полями:
- "action": строго "add" или "remove"
- "item": название товара в именительном падеже, строчными буквами (обязательно)
- "amount": количество, например "2 литра", "1 кг", "пачка" (null если не указано)
- "location": магазин или место покупки (null если не указано)

Примеры:
Вход: "Нужно купить 2 литра молока в Ароме"
Выход: {"action":"add","item":"молоко","amount":"2 литра","location":"Арома"}

Вход: "Купил молоко, убери из списка"
Выход: {"action":"remove","item":"молоко","amount":null,"location":null}

Вход: "Добавь хлеб"
Выход: {"action":"add","item":"хлеб","amount":null,"location":null}
"""


async def parse_nlu(text: str) -> dict:
    """Extract shopping intent from text using GPT-4o-mini via RouteLLM."""
    response = await routellm_client.chat.completions.create(
        model=ROUTELLM_MODEL,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": NLU_SYSTEM_PROMPT},
            {"role": "user", "content": text},
        ],
        temperature=0,
        max_tokens=200,
    )
    return json.loads(response.choices[0].message.content)


# ── FSM States ─────────────────────────────────────────────────────────────────


class GlossaryStates(StatesGroup):
    waiting_canonical = State()
    waiting_spoken = State()


# ── Utilities ──────────────────────────────────────────────────────────────────


def is_allowed(user_id: int) -> bool:
    return user_id in ALLOWED_USERS


async def download_voice(voice) -> bytes:
    file = await bot.get_file(voice.file_id)
    buf = await bot.download_file(file.file_path)
    return buf.read()


async def broadcast_list() -> None:
    """Push the current shopping list to every allowed user."""
    rows = db_get_list()
    text = format_shopping_list(rows)
    for uid in ALLOWED_USERS:
        try:
            await bot.send_message(uid, text, parse_mode="Markdown")
        except Exception as exc:
            logger.warning("Failed to send list to %d: %s", uid, exc)


# ── Command Handlers ───────────────────────────────────────────────────────────


@router.message(Command("start"))
async def cmd_start(message: Message) -> None:
    uid = message.from_user.id
    status = "✅ У вас есть доступ." if is_allowed(uid) else "❌ У вас нет доступа к боту."
    await message.answer(
        f"👋 Добро пожаловать в *GrassShopper*!\n"
        f"Семейный список покупок, управляемый голосом.\n\n"
        f"🆔 Ваш Telegram ID: `{uid}`\n"
        f"{status}\n\n"
        f"*Команды:*\n"
        f"/list — показать список\n"
        f"/glossary — добавить произношение магазина\n"
        f"/list\\_glossary — показать глоссарий\n\n"
        f"Просто отправь 🎙 голосовое — и список обновится!",
        parse_mode="Markdown",
    )


@router.message(Command("list"))
async def cmd_list(message: Message) -> None:
    if not is_allowed(message.from_user.id):
        return
    await message.answer(format_shopping_list(db_get_list()), parse_mode="Markdown")


@router.message(Command("list_glossary"))
async def cmd_list_glossary(message: Message) -> None:
    if not is_allowed(message.from_user.id):
        return
    pairs = db_get_glossary()
    if not pairs:
        await message.answer("📖 Глоссарий пуст.\n\nДобавь магазины через /glossary")
        return
    lines = ["📖 *Глоссарий магазинов:*\n"]
    for spoken, canonical in pairs:
        lines.append(f"• _{spoken}_ → *{canonical}*")
    await message.answer("\n".join(lines), parse_mode="Markdown")


# ── /glossary FSM dialog ───────────────────────────────────────────────────────


@router.message(Command("glossary"))
async def cmd_glossary(message: Message, state: FSMContext) -> None:
    if not is_allowed(message.from_user.id):
        return
    await state.set_state(GlossaryStates.waiting_canonical)
    await message.answer(
        "📝 *Добавление магазина в глоссарий*\n\n"
        "Шаг 1 из 2\n"
        "Напиши *официальное название* магазина:\n"
        "_(например: HDL, Magnum, SPAR, Fix Price)_",
        parse_mode="Markdown",
    )


@router.message(GlossaryStates.waiting_canonical, F.text)
async def glossary_got_canonical(message: Message, state: FSMContext) -> None:
    canonical = message.text.strip()
    await state.update_data(canonical=canonical)
    await state.set_state(GlossaryStates.waiting_spoken)
    await message.answer(
        f"✅ Сохранено: *{canonical}*\n\n"
        f"Шаг 2 из 2\n"
        f"Теперь отправь 🎙 *голосовое сообщение* — "
        f"произнеси название так, как ты его обычно говоришь.",
        parse_mode="Markdown",
    )


@router.message(GlossaryStates.waiting_canonical)
async def glossary_canonical_wrong_type(message: Message) -> None:
    await message.answer(
        "Пожалуйста, напиши название магазина *текстом*.", parse_mode="Markdown"
    )


@router.message(GlossaryStates.waiting_spoken, F.voice)
async def glossary_got_spoken(message: Message, state: FSMContext) -> None:
    data = await state.get_data()
    canonical: str = data["canonical"]

    try:
        voice_bytes = await download_voice(message.voice)
        spoken_raw = await transcribe_voice(voice_bytes)
    except Exception as exc:
        logger.error("Glossary transcription error: %s", exc)
        await message.answer(
            "❌ Ошибка при распознавании. Попробуй ещё раз или начни заново: /glossary"
        )
        return

    spoken_clean = spoken_raw.lower().rstrip(". !,")
    db_save_glossary_pair(spoken_clean, canonical)
    await state.clear()

    await message.answer(
        f"✅ *Пара сохранена в глоссарий!*\n\n"
        f"🗣 Произношение: _{spoken_clean}_\n"
        f"🏪 Официальное: *{canonical}*\n\n"
        f"Теперь если Whisper распознает «{spoken_clean}», "
        f"бот заменит это на «{canonical}» перед анализом.",
        parse_mode="Markdown",
    )


@router.message(GlossaryStates.waiting_spoken)
async def glossary_spoken_wrong_type(message: Message) -> None:
    await message.answer(
        "Пожалуйста, отправь *голосовое сообщение* 🎙", parse_mode="Markdown"
    )


# ── Main voice handler ─────────────────────────────────────────────────────────
# Registered AFTER all FSM handlers → FSM states take priority.


@router.message(F.voice)
async def handle_voice(message: Message) -> None:
    if not is_allowed(message.from_user.id):
        await message.answer("❌ У вас нет доступа к этому боту.")
        return

    status_msg = await message.answer("🎙 Распознаю речь...")

    # Step 1: Transcribe
    try:
        voice_bytes = await download_voice(message.voice)
        transcription = await transcribe_voice(voice_bytes)
        logger.info("Transcription [uid=%d]: %s", message.from_user.id, transcription)
    except Exception as exc:
        logger.error("Voice download/transcription error: %s", exc)
        await status_msg.edit_text(
            "❌ Не удалось распознать голосовое сообщение. Попробуй ещё раз."
        )
        return

    # Step 2: Glossary substitution
    processed_text = apply_glossary(transcription)
    if processed_text != transcription:
        logger.info("After glossary substitution: %s", processed_text)

    # Step 3: NLU parsing
    await status_msg.edit_text("🧠 Анализирую команду...")
    try:
        intent = await parse_nlu(processed_text)
        logger.info("Intent: %s", intent)
    except Exception as exc:
        logger.error("NLU parsing error: %s", exc)
        await status_msg.edit_text(
            f"❌ Не удалось разобрать команду.\n\n_Распознано: {transcription}_",
            parse_mode="Markdown",
        )
        return

    action: str = intent.get("action", "")
    item: str = intent.get("item", "")
    amount: str | None = intent.get("amount") or None
    location: str | None = intent.get("location") or None

    if not item:
        await status_msg.edit_text(
            f"🤔 Не удалось определить товар.\n\n_Распознано: {transcription}_",
            parse_mode="Markdown",
        )
        return

    # Step 4: Apply DB action
    if action == "add":
        db_add_item(item, amount, location)
        confirmation = f"✅ Добавлено: *{item}*"
        if amount:
            confirmation += f" — {amount}"
        if location:
            confirmation += f" ({location})"

    elif action == "remove":
        deleted_count = db_remove_item(item)
        confirmation = (
            f"🗑 Удалено: *{item}*"
            if deleted_count
            else f"⚠️ Товар «{item}» не найден в списке."
        )
    else:
        await status_msg.edit_text(
            f"🤔 Не понял команду (action=«{action}»).\n\n_Распознано: {transcription}_",
            parse_mode="Markdown",
        )
        return

    await status_msg.delete()
    await message.answer(
        f"{confirmation}\n\n_Распознано: {transcription}_",
        parse_mode="Markdown",
    )

    # Step 5: Broadcast updated list to all users
    await broadcast_list()


# ── Text fallback (no active FSM state) ────────────────────────────────────────


@router.message(StateFilter(None))
async def handle_text_fallback(message: Message) -> None:
    if not is_allowed(message.from_user.id):
        return
    await message.answer(
        "🎙 Отправь голосовое сообщение для управления списком.\n\n"
        "*Примеры фраз:*\n"
        "• «Добавь 2 литра молока»\n"
        "• «Нужен хлеб в Магнуме»\n"
        "• «Купил яйца, убери из списка»",
        parse_mode="Markdown",
    )


# ── Webhook lifecycle ──────────────────────────────────────────────────────────


async def on_startup(bot: Bot) -> None:
    webhook_url = f"{RENDER_EXTERNAL_URL}{WEBHOOK_PATH}"
    await bot.set_webhook(webhook_url)
    logger.info("Webhook set: %s", webhook_url)


async def on_shutdown(bot: Bot) -> None:
    await bot.delete_webhook()
    logger.info("Webhook deleted")


# ── Entry point ────────────────────────────────────────────────────────────────


def main() -> None:
    init_db()

    if RENDER_EXTERNAL_URL:
        # ── Webhook mode (production on Render) ───────────────────────────────
        logger.info(
            "Starting in WEBHOOK mode. URL: %s%s", RENDER_EXTERNAL_URL, WEBHOOK_PATH
        )
        dp.startup.register(on_startup)
        dp.shutdown.register(on_shutdown)

        app = web.Application()

        # Health check for Render's uptime monitoring
        async def health(_: web.Request) -> web.Response:
            return web.Response(text="OK")

        app.router.add_get("/", health)

        SimpleRequestHandler(dispatcher=dp, bot=bot).register(app, path=WEBHOOK_PATH)
        setup_application(app, dp, bot=bot)

        web.run_app(app, host="0.0.0.0", port=PORT)

    else:
        # ── Polling mode (local development) ─────────────────────────────────
        logger.info("Starting in POLLING mode (local dev). Allowed users: %s", ALLOWED_USERS)
        asyncio.run(dp.start_polling(bot, skip_updates=True))


if __name__ == "__main__":
    main()
