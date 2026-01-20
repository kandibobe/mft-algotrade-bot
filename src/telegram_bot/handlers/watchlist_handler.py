# handlers/watchlist_handler.py
import html

from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.constants import ParseMode
from telegram.ext import ContextTypes

from src.telegram_bot import constants

# ИСПРАВЛЕНИЕ: Импортируем alert_handler напрямую, но используем его осторожно
from src.telegram_bot.handlers import alert_handler
from src.telegram_bot.localization.manager import get_text, get_user_language
from src.telegram_bot.services import user_manager
from src.utils.logger import get_logger

logger = get_logger(__name__)


async def watchlist_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Показывает список наблюдения с кнопками удаления и добавления алерта."""
    user_id = update.effective_user.id
    lang_code = await get_user_language(user_id)
    logger.info(f"Запрос watchlist от user_id {user_id}")

    watchlist = user_manager.get_user_watchlist(user_id)
    limits = user_manager.get_user_limits(user_id)

    header_text = get_text(
        constants.TITLE_WATCHLIST, lang_code, count=len(watchlist), limit=limits["watchlist"]
    )

    if not watchlist:
        reply_text = header_text + "\n" + get_text(constants.MSG_WATCHLIST_EMPTY, lang_code)
        reply_markup = None
    else:
        reply_text = header_text + "\n"
        keyboard = []
        for item in watchlist:
            ticker = constants.REVERSE_ASSET_MAP.get(item["asset_id"], item["asset_id"])
            button_row = [
                InlineKeyboardButton(
                    f"🔔 {ticker}", callback_data=f"{constants.CB_ACTION_QUICK_ADD_ALERT}{ticker}"
                ),
                InlineKeyboardButton(
                    "❌", callback_data=f"{constants.CB_ACTION_DEL_WATCH}{ticker}"
                ),
            ]
            keyboard.append(button_row)
            reply_text += (
                get_text(
                    constants.MSG_WATCHLIST_ITEM,
                    lang_code,
                    asset_id=ticker,
                    asset_type=item["asset_type"],
                )
                + "\n"
            )
        reply_markup = InlineKeyboardMarkup(keyboard)

    effective_message = update.message or (
        update.callback_query.message if update.callback_query else None
    )
    if not effective_message:
        return

    if update.callback_query:
        try:
            await update.callback_query.edit_message_text(
                reply_text, reply_markup=reply_markup, parse_mode=ParseMode.HTML
            )
        except Exception as e:
            logger.error(f"Ошибка обновления сообщения watchlist: {e}")
    else:
        await effective_message.reply_text(
            reply_text, parse_mode=ParseMode.HTML, reply_markup=reply_markup
        )


async def delwatch_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обрабатывает нажатие кнопки удаления актива из списка наблюдения."""
    query = update.callback_query
    user_id = update.effective_user.id
    lang_code = await get_user_language(user_id)
    await query.answer()

    try:
        ticker_to_delete = query.data[len(constants.CB_ACTION_DEL_WATCH) :]
    except IndexError:
        logger.error(f"Не удалось извлечь тикер из callback_data: {query.data}")
        await query.edit_message_text(get_text(constants.MSG_ERROR_GENERAL, lang_code))
        return

    logger.info(f"Попытка удалить '{ticker_to_delete}' из watchlist user_id {user_id} через кнопку")
    result_code = user_manager.remove_asset_from_watchlist(user_id, ticker_to_delete)

    await watchlist_command(update, context)

    if result_code == user_manager.OPERATION_SUCCESS:
        msg_key = constants.MSG_DELWATCH_SUCCESS
    else:
        msg_key = constants.ERROR_DELWATCH_NOTFOUND

    await context.bot.send_message(
        chat_id=user_id, text=get_text(msg_key, lang_code, asset_id=ticker_to_delete)
    )


async def addwatch_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик команды /addwatch."""
    user_id = update.effective_user.id
    args = context.args
    lang_code = await get_user_language(user_id)

    if not args:
        await update.message.reply_text(get_text(constants.PROMPT_ADDWATCH, lang_code))
        return

    ticker = args[0].upper()
    logger.info(f"Попытка добавить '{ticker}' в watchlist user_id {user_id}")

    result_code = user_manager.add_asset_to_watchlist(user_id, ticker)

    reply_text = ""
    if result_code == user_manager.OPERATION_SUCCESS:
        reply_text = get_text(constants.MSG_ADDWATCH_SUCCESS, lang_code, asset_id=ticker)
    elif result_code == user_manager.OPERATION_FAILED_LIMIT:
        limits = user_manager.get_user_limits(user_id)
        premium_ad_text = get_text(constants.MSG_PREMIUM_AD_TEXT, lang_code, default="")
        reply_text = get_text(
            constants.ERROR_ADDWATCH_LIMIT,
            lang_code,
            limit=limits["watchlist"],
            premium_ad=premium_ad_text,
        )
    elif result_code == user_manager.OPERATION_FAILED_EXISTS:
        reply_text = get_text(constants.ERROR_ADDWATCH_EXISTS, lang_code, asset_id=ticker)
    elif result_code == user_manager.OPERATION_FAILED_INVALID:
        reply_text = get_text(
            constants.ERROR_ADDWATCH_INVALID, lang_code, ticker=html.escape(ticker)
        )
    else:
        reply_text = get_text(constants.MSG_ERROR_DB, lang_code)
        logger.error(f"Ошибка добавления '{ticker}' в watchlist user {user_id}, код: {result_code}")

    await update.message.reply_text(reply_text, parse_mode=ParseMode.HTML)


async def delwatch_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик команды /delwatch."""
    user_id = update.effective_user.id
    args = context.args
    lang_code = await get_user_language(user_id)

    if not args:
        watchlist = user_manager.get_user_watchlist(user_id)
        limits = user_manager.get_user_limits(user_id)
        header_text = get_text(
            constants.TITLE_WATCHLIST, lang_code, count=len(watchlist), limit=limits["watchlist"]
        )
        prompt_text = get_text(constants.PROMPT_DELWATCH, lang_code)

        reply_text = prompt_text
        if watchlist:
            items = [
                f"• <code>{constants.REVERSE_ASSET_MAP.get(item['asset_id'], item['asset_id'])}</code>"
                for item in watchlist
            ]
            reply_text += f"\n\n{header_text}\n" + "\n".join(items)
        else:
            reply_text += "\n" + get_text(constants.MSG_WATCHLIST_EMPTY, lang_code)

        await update.message.reply_text(reply_text, parse_mode=ParseMode.HTML)
        return

    ticker = args[0].upper()
    logger.info(f"Попытка удалить '{ticker}' из watchlist user_id {user_id} по команде")
    result_code = user_manager.remove_asset_from_watchlist(user_id, ticker)

    if result_code == user_manager.OPERATION_SUCCESS:
        message_key = constants.MSG_DELWATCH_SUCCESS
    elif result_code == user_manager.OPERATION_FAILED_NOT_FOUND:
        message_key = constants.ERROR_DELWATCH_NOTFOUND
    else:
        message_key = constants.MSG_ERROR_DB
        logger.error(
            f"Ошибка удаления '{ticker}' из watchlist user {user_id} по команде, код: {result_code}"
        )

    reply_text = get_text(message_key, lang_code, asset_id=ticker)
    await update.message.reply_text(reply_text, parse_mode=ParseMode.HTML)


async def quick_add_alert_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Обрабатывает нажатие кнопки '🔔' для быстрого добавления алерта.
    Перенаправляет на основной обработчик /addalert.
    """
    query = update.callback_query
    if not query:
        return

    await query.answer()

    try:
        ticker = query.data[len(constants.CB_ACTION_QUICK_ADD_ALERT) :]
    except IndexError:
        logger.error(f"Не удалось извлечь тикер из callback_data для быстрого алерта: {query.data}")
        return

    # Подготавливаем аргументы для хендлера addalert_start
    context.args = [ticker]

    try:
        # Удаляем сообщение со списком, чтобы не засорять чат
        await query.delete_message()
    except Exception as e:
        logger.warning(f"Не удалось удалить сообщение watchlist после нажатия '🔔': {e}")

    # Создаем фейковое сообщение, чтобы запустить ConversationHandler
    # Это необходимо, т.к. addalert_start ожидает update.message
    class MockMessage:
        def __init__(self, text, chat_id, bot):
            self.text = text
            self.chat_id = chat_id
            self._bot = bot

        async def reply_text(self, *args, **kwargs):
            return await self._bot.send_message(chat_id=self.chat_id, *args, **kwargs)

    fake_message = MockMessage(f"/addalert {ticker}", update.effective_chat.id, context.bot)
    fake_update = Update(update_id=update.update_id, message=fake_message)

    # Вызываем хендлер напрямую
    await alert_handler.addalert_start(fake_update, context)
