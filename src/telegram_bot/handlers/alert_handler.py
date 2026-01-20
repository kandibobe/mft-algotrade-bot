# handlers/alert_handler.py
import html
import re

# УБИРАЕМ ЦИКЛИЧЕСКИЕ ИМПОРТЫ
# from src.telegram_bot.handlers import report_handler, signal_handler, watchlist_handler, misc_handler
# from src.telegram_bot.handlers import news_handler, settings_handler, volatility_handler, explain_handler
# from src.telegram_bot.handlers import language_handler, feedback_handler
from datetime import datetime

from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.constants import ParseMode
from telegram.error import BadRequest
from telegram.ext import (
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    ConversationHandler,
    MessageHandler,
    filters,
)

from src.telegram_bot import constants
from src.telegram_bot.handlers import common as common_handlers
from src.telegram_bot.localization.manager import get_text, get_user_language
from src.telegram_bot.services import data_fetcher, user_manager
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Состояния
ASK_ASSET, CHOOSE_TYPE, ASK_PRICE_CONDITION, ASK_PRICE_VALUE, ASK_RSI_CONDITION = range(5)
EDIT_ALERT_CHOICE, EDIT_ALERT_CONDITION, EDIT_ALERT_VALUE, EDIT_ALERT_CONFIRM = range(5, 9)

# Константы CallbackData
CB_EA_START = constants.CB_ACTION_EDIT_ALERT_START
CB_EA_SET_COND_PREFIX = constants.CB_ACTION_EDIT_ALERT_SET_COND
CB_EA_SET_VAL = constants.CB_ACTION_EDIT_ALERT_SET_VAL
CB_EA_SAVE = constants.CB_ACTION_EDIT_ALERT_SAVE
CB_EA_CANCEL_SINGLE = constants.CB_ACTION_EDIT_ALERT_CANCEL_SINGLE


async def alerts_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    lang_code = await get_user_language(user_id)
    logger.info(f"Запрос списка алертов от user_id {user_id}")
    alerts = user_manager.get_user_price_alerts(user_id)
    limits = user_manager.get_user_limits(user_id)

    header_text = get_text(
        constants.TITLE_ALERTS, lang_code, count=len(alerts), limit=limits["alerts"]
    )
    keyboard_rows = []

    if not alerts:
        reply_text = header_text + "\n" + get_text(constants.MSG_ALERT_EMPTY, lang_code)
    else:
        reply_text = header_text + "\n"
        for alert in alerts:
            ticker = constants.REVERSE_ASSET_MAP.get(alert["asset_id"], alert["asset_id"])
            alert_type = alert.get("alert_type", constants.ALERT_TYPE_PRICE)
            created_at_str = "N/A"
            if isinstance(alert.get("created_at"), str):
                try:
                    created_at_str = datetime.strptime(
                        alert["created_at"], "%Y-%m-%d %H:%M:%S"
                    ).strftime("%d.%m.%y")
                except ValueError:
                    pass

            if alert_type == constants.ALERT_TYPE_RSI:
                alert_text_display = get_text(
                    constants.MSG_ALERT_ITEM_RSI,
                    lang_code,
                    alert_id=alert["id"],
                    asset_id=ticker,
                    condition=html.escape(alert["condition"]),
                    value=int(alert["target_value"]),
                    created_at=created_at_str,
                )
            else:
                alert_text_display = get_text(
                    constants.MSG_ALERT_ITEM_PRICE,
                    lang_code,
                    alert_id=alert["id"],
                    asset_id=ticker,
                    condition=html.escape(alert["condition"]),
                    value=alert["target_value"],
                    created_at=created_at_str,
                )

            row = [
                InlineKeyboardButton(
                    f"{get_text(constants.BTN_EDIT, lang_code)} (ID: {alert['id']})",
                    callback_data=f"{CB_EA_START}{alert['id']}",
                ),
                InlineKeyboardButton(
                    get_text(constants.BTN_DELETE, lang_code),
                    callback_data=f"{constants.CB_ACTION_DEL_ALERT}{alert['id']}",
                ),
            ]
            keyboard_rows.append(row)
            reply_text += f"• {alert_text_display}\n"
        reply_text += f"\n{get_text(constants.PROMPT_DELALERT, lang_code)}"

    reply_markup = InlineKeyboardMarkup(keyboard_rows) if keyboard_rows else None
    effective_message = update.message or (
        update.callback_query.message if update.callback_query else None
    )

    if effective_message:
        if update.callback_query:
            try:
                await update.callback_query.edit_message_text(
                    reply_text, parse_mode=ParseMode.HTML, reply_markup=reply_markup
                )
            except BadRequest as e:
                if "message is not modified" not in str(e).lower():
                    logger.warning(f"Не удалось отредактировать список алертов: {e}.")
                await update.callback_query.answer()
        else:
            await effective_message.reply_text(
                reply_text, parse_mode=ParseMode.HTML, reply_markup=reply_markup
            )


async def delalert_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    user_id = update.effective_user.id
    lang_code = await get_user_language(user_id)

    try:
        alert_id = int(query.data[len(constants.CB_ACTION_DEL_ALERT) :])
    except (IndexError, ValueError):
        logger.error(f"Не удалось извлечь ID алерта из callback_data: {query.data}")
        await query.answer(get_text(constants.MSG_ERROR_GENERAL, lang_code), show_alert=True)
        return

    alert_data = next(
        (a for a in user_manager.get_user_price_alerts(user_id) if a["id"] == alert_id), None
    )
    if not alert_data:
        await query.answer(
            get_text(constants.ERROR_DELALERT_NOTFOUND, lang_code, alert_id=alert_id),
            show_alert=True,
        )
        await alerts_command(update, context)
        return

    ticker = constants.REVERSE_ASSET_MAP.get(alert_data["asset_id"], alert_data["asset_id"])
    confirmation_text = get_text(
        constants.MSG_DELALERT_CONFIRM,
        lang_code,
        alert_id=alert_id,
        ticker=ticker,
        condition=html.escape(alert_data["condition"]),
        value=alert_data["target_value"],
    )
    keyboard = [
        [
            InlineKeyboardButton(
                get_text(constants.BTN_YES, lang_code),
                callback_data=f"{constants.CB_ACTION_DEL_ALERT_CONFIRMED}{alert_id}",
            ),
            InlineKeyboardButton(
                get_text(constants.BTN_NO, lang_code),
                callback_data=f"{constants.CB_ACTION_DEL_ALERT_CANCELLED}{alert_id}",
            ),
        ]
    ]
    await query.edit_message_text(
        confirmation_text, reply_markup=InlineKeyboardMarkup(keyboard), parse_mode=ParseMode.HTML
    )


async def delalert_confirmed_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    user_id = update.effective_user.id
    lang_code = await get_user_language(user_id)

    try:
        alert_id = int(query.data[len(constants.CB_ACTION_DEL_ALERT_CONFIRMED) :])
    except (IndexError, ValueError):
        logger.error(f"Не удалось извлечь ID алерта из callback_data подтверждения: {query.data}")
        await query.answer(get_text(constants.MSG_ERROR_GENERAL, lang_code), show_alert=True)
        return

    await query.answer()
    result_code = user_manager.delete_user_price_alert(user_id, alert_id)

    if result_code == user_manager.OPERATION_SUCCESS:
        await context.bot.send_message(
            user_id, get_text(constants.MSG_DELALERT_SUCCESS, lang_code, alert_id=alert_id)
        )
    else:
        await context.bot.send_message(
            user_id, get_text(constants.ERROR_DELALERT_NOTFOUND, lang_code, alert_id=alert_id)
        )

    await alerts_command(update, context)


async def delalert_cancelled_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.callback_query.answer("Удаление отменено.")
    await alerts_command(update, context)


async def delalert_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    lang_code = await get_user_language(user_id)
    if not context.args:
        await alerts_command(update, context)
        return
    try:
        alert_id = int(context.args[0])
    except ValueError:
        await update.message.reply_text(get_text(constants.ERROR_DELALERT_INVALID_ID, lang_code))
        return

    result_code = user_manager.delete_user_price_alert(user_id, alert_id)
    if result_code == user_manager.OPERATION_SUCCESS:
        await update.message.reply_text(
            get_text(constants.MSG_DELALERT_SUCCESS, lang_code, alert_id=alert_id)
        )
    else:
        await update.message.reply_text(
            get_text(constants.ERROR_DELALERT_NOTFOUND, lang_code, alert_id=alert_id)
        )
    await alerts_command(update, context)


# --- Add Alert Conversation ---


async def addalert_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    user_id = update.effective_user.id
    lang_code = await get_user_language(user_id)

    limits = user_manager.get_user_limits(user_id)
    if user_manager.get_price_alert_count(user_id) >= limits["alerts"]:
        await update.message.reply_text(
            get_text(
                constants.ERROR_ADDALERT_LIMIT,
                lang_code,
                limit=limits["alerts"],
                premium_ad=get_text(constants.MSG_PREMIUM_AD_TEXT, lang_code),
            )
        )
        return ConversationHandler.END

    if context.args and await _handle_quick_price_alert(update, context, " ".join(context.args)):
        return ConversationHandler.END

    await update.message.reply_text(get_text(constants.PROMPT_ADDALERT_ASSET, lang_code))
    return ASK_ASSET


async def _handle_quick_price_alert(
    update: Update, context: ContextTypes.DEFAULT_TYPE, text: str
) -> bool:
    user_id = update.effective_user.id
    lang_code = await get_user_language(user_id)

    match_abs = re.match(constants.ALERT_QUICK_ADD_REGEX, text, re.IGNORECASE)
    if match_abs:
        ticker, condition, value_str = match_abs.groups()
        ticker = ticker.upper()
        asset_info = constants.SUPPORTED_ASSETS.get(ticker)
        if not asset_info:
            await update.message.reply_text(
                get_text(
                    constants.ERROR_ADDALERT_INVALID_ASSET, lang_code, ticker=html.escape(ticker)
                )
            )
            return True
        try:
            target_value = float(value_str.replace(",", "."))
            if target_value <= 0:
                await update.message.reply_text(
                    get_text(constants.ERROR_VALUE_MUST_BE_POSITIVE, lang_code)
                )
                return True
        except ValueError:
            await update.message.reply_text(
                get_text(constants.ERROR_ADDALERT_INVALID_VALUE, lang_code)
            )
            return True

        asset_type, asset_id = asset_info
        result_code, new_alert_id = user_manager.add_user_alert(
            user_id, asset_type, asset_id, constants.ALERT_TYPE_PRICE, condition, target_value
        )

        if result_code == user_manager.OPERATION_SUCCESS:
            reply_text = get_text(
                constants.MSG_ADDALERT_QUICK_SUCCESS,
                lang_code,
                alert_id=new_alert_id,
                asset_id=ticker,
                condition=html.escape(condition),
                value=target_value,
            )
            await update.message.reply_text(reply_text, parse_mode=ParseMode.HTML)
        else:
            await update.message.reply_text(get_text(constants.ERROR_ADDALERT_GENERIC, lang_code))
        return True

    match_pct = re.match(constants.ALERT_QUICK_ADD_PERCENT_REGEX, text, re.IGNORECASE)
    if match_pct:
        ticker, sign, percent_str = match_pct.groups()
        ticker = ticker.upper()
        asset_info = constants.SUPPORTED_ASSETS.get(ticker)
        if not asset_info or asset_info[0] not in [constants.ASSET_CRYPTO, constants.ASSET_FOREX]:
            await update.message.reply_text(
                get_text(
                    constants.ERROR_ADDALERT_INVALID_ASSET, lang_code, ticker=html.escape(ticker)
                )
            )
            return True
        try:
            percent_change = float(percent_str.replace(",", "."))
            if percent_change <= 0:
                await update.message.reply_text(
                    get_text(constants.ERROR_ADDALERT_PERCENT_VALUE, lang_code)
                )
                return True
        except ValueError:
            await update.message.reply_text(
                get_text(constants.ERROR_ADDALERT_PERCENT_VALUE, lang_code)
            )
            return True

        session = context.bot_data.get("aiohttp_session")
        if not session or session.closed:
            await update.message.reply_text(get_text(constants.MSG_ERROR_GENERAL, lang_code))
            return True
        loading_msg = await update.message.reply_text(get_text(constants.MSG_LOADING, lang_code))

        asset_type, asset_id = asset_info
        price_data: tuple[float | None, float | None, str] | None = None
        if asset_type == constants.ASSET_CRYPTO:
            prices = await data_fetcher.fetch_current_crypto_data(session, [asset_id])
            price_data = prices.get(asset_id)
        elif asset_type == constants.ASSET_FOREX:
            rates = await data_fetcher.fetch_current_forex_rates(session, [asset_id])
            rate_tuple = rates.get(asset_id)
            if rate_tuple:
                price_data = (rate_tuple[0], None, rate_tuple[1])
        await loading_msg.delete()

        if not price_data or price_data[0] is None or price_data[2] != data_fetcher.STATUS_OK:
            await update.message.reply_text(
                get_text(constants.ERROR_ADDALERT_FETCH_PRICE, lang_code, ticker=ticker)
            )
            return True

        current_price = price_data[0]
        condition = ">" if sign == "+" else "<"
        target_price = current_price * (
            1 + (percent_change / 100) if sign == "+" else -(percent_change / 100)
        )
        result_code, new_alert_id = user_manager.add_user_alert(
            user_id, asset_type, asset_id, constants.ALERT_TYPE_PRICE, condition, target_price
        )

        if result_code == user_manager.OPERATION_SUCCESS:
            reply_text = get_text(
                constants.MSG_ADDALERT_PERCENT_SUCCESS,
                lang_code,
                alert_id=new_alert_id,
                asset_id=ticker,
                condition=html.escape(condition),
                target_price=target_price,
                sign=sign,
                percent=percent_change,
            )
            await update.message.reply_text(reply_text, parse_mode=ParseMode.HTML)
        else:
            await update.message.reply_text(get_text(constants.ERROR_ADDALERT_GENERIC, lang_code))
        return True
    return False


async def ask_asset_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    if not update.message or not update.message.text:
        return ASK_ASSET
    user_id = update.effective_user.id
    lang_code = await get_user_language(user_id)

    if await _handle_quick_price_alert(update, context, update.message.text):
        return ConversationHandler.END

    ticker = update.message.text.strip().upper()
    asset_info = constants.SUPPORTED_ASSETS.get(ticker)
    if not asset_info or asset_info[0] not in [constants.ASSET_CRYPTO, constants.ASSET_FOREX]:
        await update.message.reply_text(
            get_text(constants.ERROR_ADDALERT_INVALID_ASSET, lang_code, ticker=html.escape(ticker))
        )
        return ASK_ASSET

    context.user_data["alert_asset_type"], context.user_data["alert_asset_id"] = asset_info
    context.user_data["alert_ticker"] = ticker

    keyboard = [
        [
            InlineKeyboardButton(
                get_text(constants.BTN_ALERT_TYPE_PRICE, lang_code),
                callback_data=f"{constants.CB_ACTION_CHOOSE_ALERT_TYPE}{constants.ALERT_TYPE_PRICE}",
            )
        ]
    ]
    if asset_info[0] == constants.ASSET_CRYPTO:
        keyboard[0].append(
            InlineKeyboardButton(
                get_text(constants.BTN_ALERT_TYPE_RSI, lang_code),
                callback_data=f"{constants.CB_ACTION_CHOOSE_ALERT_TYPE}{constants.ALERT_TYPE_RSI}",
            )
        )

    await update.message.reply_text(
        get_text(constants.PROMPT_ADDALERT_TYPE, lang_code, ticker=html.escape(ticker)),
        reply_markup=InlineKeyboardMarkup(keyboard),
    )
    return CHOOSE_TYPE


async def received_alert_type(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    query = update.callback_query
    await query.answer()
    chosen_type = query.data[len(constants.CB_ACTION_CHOOSE_ALERT_TYPE) :]
    context.user_data["alert_type"] = chosen_type
    await query.edit_message_reply_markup(reply_markup=None)

    if chosen_type == constants.ALERT_TYPE_PRICE:
        return await ask_price_condition(update, context)
    if chosen_type == constants.ALERT_TYPE_RSI:
        return await ask_rsi_condition(update, context)
    return ConversationHandler.END


async def ask_price_condition(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    user_id = update.effective_user.id
    lang_code = await get_user_language(user_id)
    ticker = context.user_data.get("alert_ticker", "")
    asset_type = context.user_data.get("alert_asset_type")
    asset_id = context.user_data.get("alert_asset_id")

    current_price_text = ""
    session = context.bot_data.get("aiohttp_session")
    if session and not session.closed:
        price_data: tuple[float | None, float | None, str] | None = None
        try:
            if asset_type == constants.ASSET_CRYPTO:
                prices = await data_fetcher.fetch_current_crypto_data(session, [asset_id])
                price_data = prices.get(asset_id)
            elif asset_type == constants.ASSET_FOREX:
                rates = await data_fetcher.fetch_current_forex_rates(session, [asset_id])
                rate_tuple = rates.get(asset_id)
                if rate_tuple:
                    price_data = (rate_tuple[0], None, rate_tuple[1])

            if price_data and price_data[2] == data_fetcher.STATUS_OK and price_data[0] is not None:
                price_format = "{:.5f}" if asset_type == constants.ASSET_FOREX else "{:,.2f}"
                current_price_text = get_text(
                    constants.MSG_ADDALERT_CURRENT_PRICE,
                    lang_code,
                    price=price_format.format(price_data[0]),
                )
        except Exception as e:
            logger.warning(
                f"Ошибка получения текущей цены для '{ticker}' в /addalert: {e}", exc_info=True
            )

    base_prompt_text = get_text(constants.PROMPT_ADDALERT_CONDITION_BASE, lang_code)
    final_prompt_text = f"{base_prompt_text} {current_price_text}".strip()

    keyboard = [
        [
            InlineKeyboardButton(
                get_text(constants.BTN_CONDITION_GT, lang_code),
                callback_data=constants.CB_ALERT_COND_GT,
            ),
            InlineKeyboardButton(
                get_text(constants.BTN_CONDITION_LT, lang_code),
                callback_data=constants.CB_ALERT_COND_LT,
            ),
        ]
    ]
    await context.bot.send_message(
        chat_id=user_id, text=final_prompt_text, reply_markup=InlineKeyboardMarkup(keyboard)
    )
    return ASK_PRICE_CONDITION


async def received_price_condition_button(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> int:
    query = update.callback_query
    await query.answer()
    chosen_condition = ">" if query.data == constants.CB_ALERT_COND_GT else "<"
    context.user_data["alert_condition"] = chosen_condition
    ticker = context.user_data.get(
        "alert_ticker",
        get_text(constants.TEXT_ASSET_DEFAULT, await get_user_language(update.effective_user.id)),
    )
    await query.edit_message_reply_markup(reply_markup=None)
    prompt_text = get_text(
        constants.PROMPT_ADDALERT_VALUE,
        await get_user_language(update.effective_user.id),
        asset_id=html.escape(ticker),
        condition=html.escape(chosen_condition),
    )
    await context.bot.send_message(chat_id=update.effective_user.id, text=prompt_text)
    return ASK_PRICE_VALUE


async def add_price_alert_finish(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    user_id = update.effective_user.id
    lang_code = await get_user_language(user_id)
    try:
        target_value = float(update.message.text.strip().replace(",", "."))
        if target_value <= 0:
            await update.message.reply_text(
                get_text(constants.ERROR_VALUE_MUST_BE_POSITIVE, lang_code)
            )
            return ASK_PRICE_VALUE
    except ValueError:
        await update.message.reply_text(get_text(constants.ERROR_ADDALERT_INVALID_VALUE, lang_code))
        return ASK_PRICE_VALUE

    data = context.user_data
    result_code, new_alert_id = user_manager.add_user_alert(
        user_id,
        data["alert_asset_type"],
        data["alert_asset_id"],
        data["alert_type"],
        data["alert_condition"],
        target_value,
    )

    if result_code == user_manager.OPERATION_SUCCESS:
        reply_text = get_text(
            constants.MSG_ADDALERT_SUCCESS,
            lang_code,
            alert_id=new_alert_id,
            asset_id=html.escape(data["alert_ticker"]),
            condition=html.escape(data["alert_condition"]),
            value=target_value,
        )
        await update.message.reply_text(
            reply_text, reply_markup=common_handlers.get_main_keyboard(lang_code)
        )
    else:
        await update.message.reply_text(
            get_text(constants.ERROR_ADDALERT_GENERIC, lang_code),
            reply_markup=common_handlers.get_main_keyboard(lang_code),
        )

    context.user_data.clear()
    return ConversationHandler.END


async def ask_rsi_condition(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    user_id = update.effective_user.id
    lang_code = await get_user_language(user_id)
    keyboard = [
        [
            InlineKeyboardButton(
                get_text(constants.BTN_RSI_COND_GT70, lang_code),
                callback_data=f"{constants.CB_ACTION_SET_RSI_COND}gt70",
            ),
            InlineKeyboardButton(
                get_text(constants.BTN_RSI_COND_LT30, lang_code),
                callback_data=f"{constants.CB_ACTION_SET_RSI_COND}lt30",
            ),
        ]
    ]
    await context.bot.send_message(
        chat_id=user_id,
        text=get_text(constants.PROMPT_ADDALERT_RSI_CONDITION, lang_code),
        reply_markup=InlineKeyboardMarkup(keyboard),
    )
    return ASK_RSI_CONDITION


async def received_rsi_condition_button(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    query = update.callback_query
    user_id = update.effective_user.id
    lang_code = await get_user_language(user_id)
    await query.answer()

    chosen_rsi_cond = query.data[len(constants.CB_ACTION_SET_RSI_COND) :]
    condition, target_value = (">", 70) if chosen_rsi_cond == "gt70" else ("<", 30)

    await query.edit_message_reply_markup(reply_markup=None)
    data = context.user_data
    result_code, new_alert_id = user_manager.add_user_alert(
        user_id,
        data["alert_asset_type"],
        data["alert_asset_id"],
        data["alert_type"],
        condition,
        float(target_value),
    )

    if result_code == user_manager.OPERATION_SUCCESS:
        reply_text = get_text(
            constants.MSG_ADDALERT_SUCCESS_RSI,
            lang_code,
            alert_id=new_alert_id,
            asset_id=html.escape(data["alert_ticker"]),
            condition=html.escape(condition),
            value=target_value,
        )
        await context.bot.send_message(
            chat_id=user_id,
            text=reply_text,
            reply_markup=common_handlers.get_main_keyboard(lang_code),
        )
    else:
        await context.bot.send_message(
            chat_id=user_id,
            text=get_text(constants.ERROR_ADDALERT_GENERIC, lang_code),
            reply_markup=common_handlers.get_main_keyboard(lang_code),
        )

    context.user_data.clear()
    return ConversationHandler.END


async def cancel_addalert(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    user_id = update.effective_user.id
    lang_code = await get_user_language(user_id)
    logger.info(f"Отмена добавления алерта для user_id {user_id}")
    context.user_data.clear()
    await update.message.reply_text(
        get_text(constants.MSG_ADDALERT_CANCEL, lang_code),
        reply_markup=common_handlers.get_main_keyboard(lang_code),
    )
    return ConversationHandler.END


# --- Edit Alert Conversation ---
async def edit_alert_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    query = update.callback_query
    user_id = update.effective_user.id
    lang_code = await get_user_language(user_id)
    await query.answer()
    try:
        alert_id = int(query.data[len(CB_EA_START) :])
    except (IndexError, ValueError):
        await query.edit_message_text(get_text(constants.MSG_ERROR_GENERAL, lang_code))
        return ConversationHandler.END

    alert_data = next(
        (a for a in user_manager.get_user_price_alerts(user_id) if a["id"] == alert_id), None
    )
    if not alert_data:
        await query.edit_message_text(
            get_text(constants.ERROR_DELALERT_NOTFOUND, lang_code, alert_id=alert_id)
        )
        return ConversationHandler.END

    if alert_data.get("alert_type", constants.ALERT_TYPE_PRICE) != constants.ALERT_TYPE_PRICE:
        await context.bot.send_message(
            chat_id=user_id, text="Редактирование этого типа алертов пока не поддерживается."
        )
        return ConversationHandler.END

    context.user_data["edit_alert_id"] = alert_id
    context.user_data["edit_alert_data"] = alert_data
    ticker = constants.REVERSE_ASSET_MAP.get(alert_data["asset_id"], alert_data["asset_id"])
    text = get_text(
        constants.PROMPT_EDIT_ALERT_CHOICE,
        lang_code,
        ticker=ticker,
        condition=html.escape(alert_data["condition"]),
        value=alert_data["target_value"],
        alert_id=alert_id,
    )
    keyboard = [
        [
            InlineKeyboardButton(
                get_text(constants.BTN_EDIT_CONDITION, lang_code),
                callback_data=f"{CB_EA_SET_COND_PREFIX}{alert_id}",
            )
        ],
        [
            InlineKeyboardButton(
                get_text(constants.BTN_EDIT_VALUE, lang_code),
                callback_data=f"{CB_EA_SET_VAL}{alert_id}",
            )
        ],
        [
            InlineKeyboardButton(
                get_text(constants.BTN_CANCEL, lang_code),
                callback_data=f"{CB_EA_CANCEL_SINGLE}{alert_id}",
            )
        ],
    ]
    await query.edit_message_text(
        text, reply_markup=InlineKeyboardMarkup(keyboard), parse_mode=ParseMode.HTML
    )
    return EDIT_ALERT_CHOICE


async def edit_alert_ask_condition(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    query = update.callback_query
    user_id = update.effective_user.id
    lang_code = await get_user_language(user_id)
    await query.answer()
    alert_data = context.user_data["edit_alert_data"]
    ticker = constants.REVERSE_ASSET_MAP.get(alert_data["asset_id"], alert_data["asset_id"])
    text = get_text(
        constants.PROMPT_EDIT_ALERT_NEW_CONDITION,
        lang_code,
        ticker=ticker,
        current_condition=html.escape(alert_data["condition"]),
    )
    keyboard = [
        [
            InlineKeyboardButton(
                get_text(constants.BTN_CONDITION_GT, lang_code),
                callback_data=f"{CB_EA_SET_COND_PREFIX}{alert_data['id']}_>",
            ),
            InlineKeyboardButton(
                get_text(constants.BTN_CONDITION_LT, lang_code),
                callback_data=f"{CB_EA_SET_COND_PREFIX}{alert_data['id']}_<",
            ),
        ],
        [
            InlineKeyboardButton(
                get_text(constants.BTN_CANCEL, lang_code),
                callback_data=f"{CB_EA_CANCEL_SINGLE}{alert_data['id']}",
            )
        ],
    ]
    await query.edit_message_text(
        text, parse_mode=ParseMode.HTML, reply_markup=InlineKeyboardMarkup(keyboard)
    )
    return EDIT_ALERT_CONDITION


async def edit_alert_receive_condition(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    query = update.callback_query
    await query.answer()
    try:
        _, new_condition = query.data[len(CB_EA_SET_COND_PREFIX) :].split("_", 1)
    except (IndexError, ValueError):
        await query.edit_message_text(
            get_text(constants.MSG_ERROR_GENERAL, await get_user_language(update.effective_user.id))
        )
        return ConversationHandler.END

    context.user_data["new_alert_condition"] = new_condition
    await query.edit_message_reply_markup(reply_markup=None)
    return await edit_alert_confirm(update, context)


async def edit_alert_ask_value(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    query = update.callback_query
    user_id = update.effective_user.id
    lang_code = await get_user_language(user_id)
    await query.answer()
    alert_data = context.user_data["edit_alert_data"]
    ticker = constants.REVERSE_ASSET_MAP.get(alert_data["asset_id"], alert_data["asset_id"])
    text = get_text(
        constants.PROMPT_EDIT_ALERT_NEW_VALUE,
        lang_code,
        ticker=ticker,
        current_value=alert_data["target_value"],
    )
    await query.edit_message_reply_markup(reply_markup=None)
    await context.bot.send_message(chat_id=user_id, text=text, parse_mode=ParseMode.HTML)
    return EDIT_ALERT_VALUE


async def edit_alert_receive_value(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    lang_code = await get_user_language(update.effective_user.id)
    try:
        new_value = float(update.message.text.strip().replace(",", "."))
        if new_value <= 0:
            await update.message.reply_text(
                get_text(constants.ERROR_VALUE_MUST_BE_POSITIVE, lang_code)
            )
            return EDIT_ALERT_VALUE
    except ValueError:
        await update.message.reply_text(get_text(constants.ERROR_ADDALERT_INVALID_VALUE, lang_code))
        return EDIT_ALERT_VALUE
    context.user_data["new_alert_value"] = new_value
    return await edit_alert_confirm(update, context)


async def edit_alert_confirm(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    user_id = update.effective_user.id
    lang_code = await get_user_language(user_id)
    alert_id = context.user_data["edit_alert_id"]
    original_data = context.user_data["edit_alert_data"]
    new_condition = context.user_data.get("new_alert_condition", original_data["condition"])
    new_value = context.user_data.get("new_alert_value", original_data["target_value"])
    ticker = constants.REVERSE_ASSET_MAP.get(original_data["asset_id"], original_data["asset_id"])
    text = get_text(
        constants.PROMPT_EDIT_ALERT_CONFIRM,
        lang_code,
        ticker=ticker,
        new_condition=html.escape(new_condition),
        new_value=new_value,
        alert_id=alert_id,
    )
    keyboard = [
        [
            InlineKeyboardButton(
                get_text(constants.BTN_YES, lang_code), callback_data=f"{CB_EA_SAVE}{alert_id}"
            ),
            InlineKeyboardButton(
                get_text(constants.BTN_NO, lang_code),
                callback_data=f"{CB_EA_CANCEL_SINGLE}{alert_id}",
            ),
        ]
    ]
    if update.callback_query:
        await update.callback_query.edit_message_text(
            text, reply_markup=InlineKeyboardMarkup(keyboard), parse_mode=ParseMode.HTML
        )
    else:
        await update.message.reply_text(
            text, reply_markup=InlineKeyboardMarkup(keyboard), parse_mode=ParseMode.HTML
        )
    return EDIT_ALERT_CONFIRM


async def edit_alert_save(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    query = update.callback_query
    user_id = update.effective_user.id
    lang_code = await get_user_language(user_id)
    await query.answer()
    alert_id = context.user_data["edit_alert_id"]
    updates = {}
    if "new_alert_condition" in context.user_data:
        updates["condition"] = context.user_data["new_alert_condition"]
    if "new_alert_value" in context.user_data:
        updates["target_value"] = context.user_data["new_alert_value"]

    if not updates:
        final_text = get_text(constants.MSG_EDIT_ALERT_NO_CHANGES, lang_code)
    else:
        success = user_manager.update_user_price_alert_fields(user_id, alert_id, updates)
        final_text = (
            get_text(constants.MSG_EDIT_ALERT_SUCCESS, lang_code, alert_id=alert_id)
            if success
            else get_text(constants.MSG_ERROR_DB, lang_code)
        )

    await query.edit_message_text(final_text, parse_mode=ParseMode.HTML)
    context.user_data.clear()
    await alerts_command(update, context)
    return ConversationHandler.END


async def edit_alert_cancel(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    query = update.callback_query
    lang_code = await get_user_language(update.effective_user.id)
    if query:
        await query.answer()
    context.user_data.clear()
    cancel_message = get_text(constants.MSG_EDIT_ALERT_CANCELLED, lang_code)
    if query and query.message:
        await query.edit_message_text(f"<i>{cancel_message}</i>", parse_mode=ParseMode.HTML)
        await alerts_command(update, context)
    else:
        await update.message.reply_text(
            cancel_message, reply_markup=common_handlers.get_main_keyboard(lang_code)
        )
    return ConversationHandler.END


# --- Fallback Handlers ---
DIALOG_NAME_ALERT = "добавления/редактирования алерта"
common_fallbacks_for_alert_dialogs = [
    CommandHandler(
        c,
        lambda u, c: common_handlers.cancel_conversation_and_call_command(
            u, c, h, DIALOG_NAME_ALERT
        ),
    )
    for c, h_name in [
        (constants.CMD_START, "start"),
        (constants.CMD_REPORT, "report_command_handler"),
        (constants.CMD_SIGNAL, "signal_command_handler"),
        (constants.CMD_SETTINGS, "settings_command_handler"),
        (constants.CMD_HELP, "help_command"),
        (constants.CMD_WATCHLIST, "watchlist_command"),
        (constants.CMD_ALERTS, "alerts_command"),
    ]
    for h in [
        getattr(globals().get(h_name.split("_")[0] + "_handler", common_handlers), h_name, None)
    ]
    if h
]

add_alert_conv_handler = ConversationHandler(
    entry_points=[CommandHandler(constants.CMD_ADDALERT, addalert_start)],
    states={
        ASK_ASSET: [MessageHandler(filters.TEXT & ~filters.COMMAND, ask_asset_handler)],
        CHOOSE_TYPE: [
            CallbackQueryHandler(
                received_alert_type, pattern=f"^{constants.CB_ACTION_CHOOSE_ALERT_TYPE}"
            )
        ],
        ASK_PRICE_CONDITION: [
            CallbackQueryHandler(
                received_price_condition_button, pattern=f"^{constants.CB_ALERT_COND_GT}$"
            ),
            CallbackQueryHandler(
                received_price_condition_button, pattern=f"^{constants.CB_ALERT_COND_LT}$"
            ),
        ],
        ASK_PRICE_VALUE: [MessageHandler(filters.TEXT & ~filters.COMMAND, add_price_alert_finish)],
        ASK_RSI_CONDITION: [
            CallbackQueryHandler(
                received_rsi_condition_button, pattern=f"^{constants.CB_ACTION_SET_RSI_COND}"
            )
        ],
    },
    fallbacks=[
        *common_fallbacks_for_alert_dialogs,
        CommandHandler(constants.CMD_CANCEL, cancel_addalert),
    ],
    conversation_timeout=300.0,
    per_user=True,
    per_chat=True,
    per_message=False,
)

edit_alert_conv_handler = ConversationHandler(
    entry_points=[
        CallbackQueryHandler(
            edit_alert_start, pattern=f"^{CB_EA_START}{constants.ALERT_ID_REGEX_PART}$"
        )
    ],
    states={
        EDIT_ALERT_CHOICE: [
            CallbackQueryHandler(
                edit_alert_ask_condition,
                pattern=f"^{CB_EA_SET_COND_PREFIX}{constants.ALERT_ID_REGEX_PART}$",
            ),
            CallbackQueryHandler(
                edit_alert_ask_value, pattern=f"^{CB_EA_SET_VAL}{constants.ALERT_ID_REGEX_PART}$"
            ),
        ],
        EDIT_ALERT_CONDITION: [
            CallbackQueryHandler(
                edit_alert_receive_condition,
                pattern=f"^{CB_EA_SET_COND_PREFIX}{constants.ALERT_ID_REGEX_PART}_(>|<)$",
            )
        ],
        EDIT_ALERT_VALUE: [
            MessageHandler(filters.TEXT & ~filters.COMMAND, edit_alert_receive_value)
        ],
        EDIT_ALERT_CONFIRM: [
            CallbackQueryHandler(
                edit_alert_save, pattern=f"^{CB_EA_SAVE}{constants.ALERT_ID_REGEX_PART}$"
            ),
            CallbackQueryHandler(
                edit_alert_cancel, pattern=f"^{CB_EA_CANCEL_SINGLE}{constants.ALERT_ID_REGEX_PART}$"
            ),
        ],
    },
    fallbacks=[
        *common_fallbacks_for_alert_dialogs,
        CallbackQueryHandler(
            edit_alert_cancel, pattern=f"^{CB_EA_CANCEL_SINGLE}{constants.ALERT_ID_REGEX_PART}$"
        ),
        CommandHandler(constants.CMD_CANCEL, edit_alert_cancel),
    ],
    conversation_timeout=300.0,
    per_user=True,
    per_chat=True,
    per_message=False,
)

application_add_handler_alert_deletion_confirmation = [
    CallbackQueryHandler(
        delalert_confirmed_callback,
        pattern=f"^{constants.CB_ACTION_DEL_ALERT_CONFIRMED}{constants.ALERT_ID_REGEX_PART}$",
    ),
    CallbackQueryHandler(
        delalert_cancelled_callback,
        pattern=f"^{constants.CB_ACTION_DEL_ALERT_CANCELLED}{constants.ALERT_ID_REGEX_PART}$",
    ),
]
