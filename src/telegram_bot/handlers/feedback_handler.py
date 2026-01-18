# handlers/feedback_handler.py
import html

from telegram import Update
from telegram.constants import ParseMode
from telegram.ext import ContextTypes

from src.telegram_bot import constants
from src.telegram_bot.config_adapter import ADMIN_CHAT_ID
from src.telegram_bot.localization.manager import get_text, get_user_language
from src.utils.logger import get_logger

logger = get_logger(__name__)

async def feedback_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Обрабатывает команду /feedback и пересылает сообщение админу."""
    user = update.effective_user
    if not user or not update.message: return

    user_id = user.id
    lang_code = await get_user_language(user_id)
    args = context.args

    if not args:
        await update.message.reply_text(get_text(constants.MSG_FEEDBACK_PROMPT, lang_code))
        return

    feedback_text = " ".join(args)
    logger.info(f"Получен фидбек от user {user_id} ({user.username}): {feedback_text[:100]}...")

    if not ADMIN_CHAT_ID:
        logger.warning("ADMIN_CHAT_ID не установлен. Невозможно отправить фидбек.")
        await update.message.reply_text(get_text(constants.MSG_FEEDBACK_ERROR, lang_code))
        return

    # Формируем сообщение для админа
    admin_message = (
        f"✉️ Новый фидбек от пользователя:\n"
        f"ID: <code>{user_id}</code>\n"
        f"Username: @{html.escape(user.username or 'N/A')}\n"
        f"Имя: {html.escape(user.full_name)}\n"
        f"Язык: {lang_code}\n"
        f"--------------------\n"
        f"{html.escape(feedback_text)}"
    )

    try:
        # Отправляем сообщение админу
        await context.bot.send_message(
            chat_id=ADMIN_CHAT_ID,
            text=admin_message,
            parse_mode=ParseMode.HTML
        )
        # Отвечаем пользователю об успехе
        await update.message.reply_text(get_text(constants.MSG_FEEDBACK_SENT, lang_code))
        logger.info(f"Фидбек от user {user_id} успешно отправлен админу {ADMIN_CHAT_ID}.")

    except Exception as e:
        logger.error(f"Не удалось отправить фидбек админу {ADMIN_CHAT_ID}: {e}", exc_info=True)
        await update.message.reply_text(get_text(constants.MSG_FEEDBACK_ERROR, lang_code))
