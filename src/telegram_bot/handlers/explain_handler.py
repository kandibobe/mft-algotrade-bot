# handlers/explain_handler.py
import html

from telegram import Update
from telegram.constants import ParseMode
from telegram.ext import ContextTypes

from src.telegram_bot import constants
from src.telegram_bot.localization.manager import (  # Removed get_user_text as get_text is sufficient
    get_text,
    get_user_language,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


async def explain_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.effective_user.id
    lang_code = await get_user_language(user_id)
    args = context.args

    # Получаем словарь с терминами и их объяснениями для текущего языка пользователя
    # get_text вернет словарь, если значение для ключа KEY_EXPLAIN_TERMS_COLLECTION является словарем
    localized_terms_collection = get_text(constants.KEY_EXPLAIN_TERMS_COLLECTION, lang_code)

    # Проверка, что localized_terms_collection действительно словарь
    if not isinstance(localized_terms_collection, dict):
        logger.error(
            f"Explain terms for lang '{lang_code}' is not a dict. Received: {type(localized_terms_collection)}. Fallback to empty dict."
        )
        localized_terms_collection = {}
        # Можно отправить сообщение об ошибке пользователю или просто показать пустой список терминов
        # await update.message.reply_text(get_text(constants.MSG_ERROR_GENERAL, lang_code))
        # return

    available_terms_list = ", ".join(
        f"<code>{term}</code>" for term in localized_terms_collection.keys()
    )

    if not args:
        logger.debug(f"Команда /explain без аргументов user_id: {user_id}")
        text = (
            get_text(constants.MSG_EXPLAIN_PROMPT, lang_code)
            + f"\n\n<b>Доступные термины:</b>\n{available_terms_list}"
        )
        await update.message.reply_text(text, parse_mode=ParseMode.HTML)
        return

    term_query = args[0].upper()
    logger.info(f"Запрос объяснения для '{term_query}' от user_id: {user_id}")

    explanation = localized_terms_collection.get(term_query)

    if explanation:
        text = explanation
    else:
        text = get_text(
            constants.MSG_EXPLAIN_NOT_FOUND,
            lang_code,
            term=html.escape(term_query),
            available_terms=available_terms_list,
        )

    await update.message.reply_text(text, parse_mode=ParseMode.HTML)
