# handlers/news_handler.py
import asyncio
import html

from telegram import Update
from telegram.constants import ParseMode
from telegram.ext import ContextTypes

from src.telegram_bot import constants
from src.telegram_bot.config_adapter import NEWS_API_ORG_KEY, NEWS_API_PAGE_SIZE
from src.telegram_bot.localization.manager import get_text, get_user_language
from src.telegram_bot.services import data_fetcher
from src.utils.logger import get_logger

logger = get_logger(__name__)

# --- НАЧАЛО ИЗМЕНЕНИЙ: УДАЛЕНА ФУНКЦИЯ crypto_news_command ---

# --- КОНЕЦ ИЗМЕНЕНИЙ ---


# --- Обработчик Общих Новостей ---
async def run_sync_in_thread(func, *args, **kwargs):
    """Запускает синхронную функцию в отдельном потоке."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, lambda: func(*args, **kwargs))


async def news_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик команды /news."""
    effective_message = update.message or (
        update.callback_query.message if update.callback_query else None
    )
    if not effective_message:
        logger.error("news_command: Не найдено сообщение для ответа!")
        return

    user_id = update.effective_user.id
    lang_code = await get_user_language(user_id)
    query_list = context.args

    if not query_list:
        await effective_message.reply_text(
            get_text(constants.MSG_NEWS_PROMPT, lang_code), parse_mode=ParseMode.HTML
        )
        return

    query = " ".join(query_list).strip()
    if len(query) < 3:
        await effective_message.reply_text(get_text(constants.MSG_NEWS_QUERY_TOO_SHORT, lang_code))
        return

    logger.info(f"Запрос /news от user_id {user_id}, запрос: '{query}'")

    if not NEWS_API_ORG_KEY:
        await effective_message.reply_text(get_text(constants.MSG_NEWS_API_KEY_MISSING, lang_code))
        return

    loading_msg = await effective_message.reply_text(get_text(constants.MSG_LOADING, lang_code))

    result_tuple = await run_sync_in_thread(
        data_fetcher.fetch_general_news,
        query=query,
        language=lang_code if lang_code in ["ru", "en"] else "en",
        page_size=NEWS_API_PAGE_SIZE,
    )

    news_articles, status = (
        result_tuple
        if isinstance(result_tuple, tuple)
        else (None, data_fetcher.STATUS_UNKNOWN_ERROR)
    )

    try:
        await loading_msg.delete()
    except Exception:
        pass

    if status == data_fetcher.STATUS_OK and isinstance(news_articles, list) and news_articles:
        lines = [get_text(constants.MSG_NEWS_HEADER, lang_code, query=html.escape(query))]
        for article in news_articles:
            if isinstance(article, dict):
                lines.append(
                    get_text(
                        constants.MSG_NEWS_ITEM_WITH_SOURCE,
                        lang_code,
                        url=article.get("url", "#"),
                        title=html.escape(article.get("title", "N/A")),
                        source=html.escape(article.get("source", "N/A")),
                    )
                )
        lines.append("\n<i>Powered by NewsAPI.org</i>")
        reply_text = "\n".join(lines)
        await effective_message.reply_text(
            reply_text, parse_mode=ParseMode.HTML, disable_web_page_preview=True
        )
    elif status == data_fetcher.STATUS_NO_DATA or (
        status == data_fetcher.STATUS_OK and not news_articles
    ):
        await effective_message.reply_text(
            get_text(constants.MSG_NEWS_NO_RESULTS, lang_code, query=html.escape(query)),
            parse_mode=ParseMode.HTML,
        )
    elif status == data_fetcher.STATUS_INVALID_KEY:
        await effective_message.reply_text(get_text(constants.MSG_INVALID_API_KEY, lang_code))
    elif status == data_fetcher.STATUS_CONFIG_ERROR:
        await effective_message.reply_text(get_text(constants.MSG_NEWS_API_KEY_MISSING, lang_code))
    else:
        error_msg_key = (
            constants.MSG_RATE_LIMIT_ERROR
            if status == data_fetcher.STATUS_RATE_LIMIT
            else constants.MSG_TIMEOUT_ERROR
            if status == data_fetcher.STATUS_TIMEOUT
            else constants.MSG_ERROR_FETCH
        )
        await effective_message.reply_text(get_text(error_msg_key, lang_code))
