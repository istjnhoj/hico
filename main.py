import os
import json
import asyncio
import logging
from collections import defaultdict, deque

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))

from google import genai
from google.genai import types
from telegram import Update
from telegram.constants import ChatAction
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters,
)

# ---------------- SETTINGS ----------------
MODEL = "gemini-3.8-flash"   # change this one line if Google renames the model
HISTORY_SIZE = 20            # recent messages Hico remembers per chat
RETRIES = 4                  # how many times to try when Gemini is busy
PERSONA_FILE = "persona.txt"
ALLOWED_FILE = "allowed.json"

DEFAULT_PERSONA = (
    "You are Hico, a friendly and funny friend in a Telegram chat. "
    "Reply in the same language the person writes in: English, Burmese, or Myanglish. "
    "Keep replies short and casual, like texting a friend. "
    "If someone sincerely asks, you can say you are a chat bot."
)

TELEGRAM_TOKEN = os.environ["TELEGRAM_TOKEN"]
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
OWNER_ID = int(os.environ.get("OWNER_ID", "0"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logging.getLogger("httpx").setLevel(logging.WARNING)  # keeps the bot token out of logs
log = logging.getLogger("hico")

client = genai.Client(api_key=GEMINI_API_KEY)
history = defaultdict(lambda: deque(maxlen=HISTORY_SIZE))  # chat_id -> (role, text)


# ---------------- PERSONA ----------------
def load_persona() -> str:
    try:
        with open(PERSONA_FILE, "r", encoding="utf-8") as f:
            text = f.read().strip()
            return text or DEFAULT_PERSONA
    except FileNotFoundError:
        return DEFAULT_PERSONA


def save_persona(text: str) -> None:
    with open(PERSONA_FILE, "w", encoding="utf-8") as f:
        f.write(text)


# ---------------- PRIVACY / ALLOW LIST ----------------
def load_allowed() -> set:
    try:
        with open(ALLOWED_FILE, "r", encoding="utf-8") as f:
            return set(json.load(f))
    except (FileNotFoundError, ValueError):
        return set()


def save_allowed(ids: set) -> None:
    with open(ALLOWED_FILE, "w", encoding="utf-8") as f:
        json.dump(sorted(ids), f)


def is_allowed(update: Update) -> bool:
    user = update.effective_user
    chat = update.effective_chat
    if user and user.id == OWNER_ID:
        return True
    allowed = load_allowed()
    return bool((user and user.id in allowed) or (chat and chat.id in allowed))


def is_owner(update: Update) -> bool:
    return bool(update.effective_user and update.effective_user.id == OWNER_ID)


async def deny(update: Update):
    # Stay silent in groups, be polite in private chats
    if update.effective_chat.type == "private":
        await update.effective_message.reply_text("Sorry, this bot is private.")


def parse_target(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """No argument = this chat. A number = that user or chat ID. Anything else = None."""
    if not context.args:
        return update.effective_chat.id
    arg = context.args[0]
    if arg.lstrip("-").isdigit():
        return int(arg)
    return None


async def allow_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_owner(update):
        return
    target = parse_target(update, context)
    if target is None:
        await update.message.reply_text("Usage: /allow  or  /allow <number id>")
        return
    ids = load_allowed()
    ids.add(target)
    save_allowed(ids)
    await update.message.reply_text(f"Allowed: {target}")


async def disallow_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_owner(update):
        return
    target = parse_target(update, context)
    if target is None:
        await update.message.reply_text("Usage: /disallow  or  /disallow <number id>")
        return
    ids = load_allowed()
    ids.discard(target)
    save_allowed(ids)
    await update.message.reply_text(f"Removed: {target}")


async def whoami(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        f"Your user ID: {update.effective_user.id}\n"
        f"This chat ID: {update.effective_chat.id}"
    )


# ---------------- COMMANDS ----------------
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_allowed(update):
        await deny(update)
        return
    await update.message.reply_text(
        "Hico here! Talk to me in English, Burmese, or Myanglish.\n"
        'In groups, just say "hico" or reply to me.'
    )


async def reset(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_allowed(update):
        await deny(update)
        return
    history[update.effective_chat.id].clear()
    await update.message.reply_text("Memory cleared.")


async def persona_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_owner(update):
        return
    new_text = " ".join(context.args).strip()
    if not new_text:
        await update.message.reply_text(
            "Usage: /persona <new personality>\n"
            "Use /persona reset to go back to the default."
        )
        return
    if new_text.lower() == "reset":
        save_persona(DEFAULT_PERSONA)
        await update.message.reply_text("Personality reset to default.")
        return
    save_persona(new_text)
    await update.message.reply_text("Personality updated.")


# ---------------- GEMINI (with quiet retry) ----------------
def is_busy_error(e: Exception) -> bool:
    s = str(e)
    return "503" in s or "UNAVAILABLE" in s or "429" in s or "RESOURCE_EXHAUSTED" in s


async def ask_gemini(contents):
    """Try up to RETRIES times when Gemini is busy. Returns (reply, error)."""
    last_error = None
    for attempt in range(RETRIES):
        try:
            response = await client.aio.models.generate_content(
                model=MODEL,
                contents=contents,
                config=types.GenerateContentConfig(system_instruction=load_persona()),
            )
            return (response.text or "").strip() or "...", None
        except Exception as e:
            last_error = e
            if is_busy_error(e) and attempt < RETRIES - 1:
                log.info("Gemini busy, retrying (%d/%d)", attempt + 1, RETRIES - 1)
                await asyncio.sleep(2 * (attempt + 1))  # waits 2s, 4s, 6s
                continue
            break
    return None, last_error


# ---------------- CHAT ----------------
async def on_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = update.effective_message
    if not msg or not msg.text:
        return

    if not is_allowed(update):
        await deny(update)
        return

    chat = update.effective_chat
    text = msg.text

    # In groups, only answer when called
    if chat.type != "private":
        bot_username = (context.bot.username or "").lower()
        replied_to_bot = (
            msg.reply_to_message
            and msg.reply_to_message.from_user
            and msg.reply_to_message.from_user.id == context.bot.id
        )
        called = (
            "hico" in text.lower()
            or "ဟီကို" in text
            or (bot_username and f"@{bot_username}" in text.lower())
            or replied_to_bot
        )
        if not called:
            return

    name = update.effective_user.first_name or "Friend"
    chat_hist = history[chat.id]
    chat_hist.append(("user", f"{name}: {text}"))

    contents = [
        types.Content(role=role, parts=[types.Part(text=t)])
        for role, t in chat_hist
    ]

    await context.bot.send_chat_action(chat_id=chat.id, action=ChatAction.TYPING)

    reply, error = await ask_gemini(contents)

    if reply is None:
        log.error("Gemini error: %s", type(error).__name__)
        chat_hist.pop()  # forget the message that failed
        if is_busy_error(error):
            await msg.reply_text("Gemini is busy right now. Try again in a minute.")
        else:
            await msg.reply_text(f"Error: {str(error)[:300]}")
        return

    chat_hist.append(("model", reply))

    # Telegram messages max out at 4096 characters
    for i in range(0, len(reply), 4000):
        await msg.reply_text(reply[i : i + 4000])


# ---------------- RUN ----------------
def main():
    if OWNER_ID == 0:
        log.warning("OWNER_ID is not set. Nobody can use owner commands.")
    app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("reset", reset))
    app.add_handler(CommandHandler("persona", persona_cmd))
    app.add_handler(CommandHandler("allow", allow_cmd))
    app.add_handler(CommandHandler("disallow", disallow_cmd))
    app.add_handler(CommandHandler("whoami", whoami))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, on_message))
    log.info("Hico is running with model %s", MODEL)
    app.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()
