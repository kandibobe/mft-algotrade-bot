# handlers/common.py
import html

from telegram import (
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    KeyboardButton,
    ReplyKeyboardMarkup,
    Update,
)
from telegram.constants import ParseMode
from telegram.ext import ContextTypes, ConversationHandler

from src.telegram_bot import constants
from src.telegram_bot.config_adapter import (  # ИМПОРТИРОВАНО ИЗ config.py
    DEFAULT_LANGUAGE,
    SUPPORTED_LANGUAGES,
)
from src.telegram_bot.localization.manager import (
    get_text,
    get_user_language,
    set_user_language_cache,
)
from src.telegram_bot.services import user_manager
from src.utils.logger import get_logger

logger = get_logger(__name__)


def get_main_keyboard(lang_code: str) -> ReplyKeyboardMarkup:
    """
    Создает ГЛАВНУЮ клавиатуру с кнопками, соответствующими текстовым командам.
    Это основной способ навигации для пользователя.
    """
    if lang_code not in SUPPORTED_LANGUAGES:
        lang_code = DEFAULT_LANGUAGE

    # Структурируем меню для лучшего восприятия
    layout = [
        [KeyboardButton(get_text("menu_btn_report", lang_code, default="📊 Мой Отчет"))],
        [
            KeyboardButton(get_text("menu_btn_signal", lang_code, default="📡 Сигнал")),
            KeyboardButton(get_text("menu_btn_ta", lang_code, default="📈 Тех. Анализ")),
        ],
        [
            KeyboardButton(get_text("menu_btn_watchlist", lang_code, default="⭐️ Списки")),
            KeyboardButton(get_text("menu_btn_market_data", lang_code, default="🌐 Рынок")),
        ],
        [
            KeyboardButton(get_text(constants.BTN_KEY_ANALYTICS_CHAT, lang_code)),  # НОВАЯ КНОПКА
            KeyboardButton(get_text("btn_settings", lang_code)),
            KeyboardButton(get_text("btn_help", lang_code)),
        ],
    ]
    return ReplyKeyboardMarkup(layout, resize_keyboard=True)


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    if not user:
        return

    user_id = user.id
    username = user.username or "N/A"
    first_name = html.escape(user.first_name or "Пользователь")

    settings = user_manager.get_settings(user_id)
    lang_code = settings.get("language_code", DEFAULT_LANGUAGE)

    set_user_language_cache(user_id, lang_code)
    logger.info(f"Команда /start от user_id: {user_id} (username: {username}, lang: {lang_code})")

    welcome_text = get_text(constants.MSG_WELCOME, lang_code, user_first_name=first_name)

    await update.message.reply_text(
        welcome_text, reply_markup=get_main_keyboard(lang_code), parse_mode=ParseMode.HTML
    )


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    lang_code = await get_user_language(user_id)
    button_text = get_text(constants.BTN_KEY_HELP, lang_code)

    log_source = (
        f"кнопка '{button_text}'"
        if update.message and update.message.text == button_text
        else "/help"
    )
    logger.info(f"{log_source} от user_id: {user_id}")

    help_text = get_text(constants.MSG_HELP, lang_code)

    await update.message.reply_text(
        help_text,
        parse_mode=ParseMode.HTML,
        reply_markup=get_main_keyboard(lang_code),
        disable_web_page_preview=True,
    )


async def premium_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    lang_code = await get_user_language(user_id)

    query = update.callback_query
    if query:
        await query.answer()
        logger.info(f"Запрос Premium Info (callback) от user_id: {user_id}")
    else:
        logger.info(f"Команда /premium от user_id: {user_id}")

    is_premium = user_manager.is_user_premium(user_id)
    limits = user_manager.get_user_limits(user_id)

    premium_status_key = (
        constants.MSG_PREMIUM_STATUS_ACTIVE if is_premium else constants.MSG_PREMIUM_STATUS_INACTIVE
    )
    premium_status_text = get_text(premium_status_key, lang_code)

    premium_info_text = get_text(
        constants.MSG_PREMIUM_INFO,
        lang_code,
        premium_status_text=premium_status_text,
        limit_watch=limits["watchlist"],
        limit_alerts=limits["alerts"],
    )

    effective_message = query.message if query else update.message
    if effective_message:
        edit_success = False
        if query:
            try:
                await query.edit_message_text(
                    premium_info_text, parse_mode=ParseMode.HTML, disable_web_page_preview=True
                )
                edit_success = True
            except Exception as e:
                logger.warning(f"Не удалось отредактировать сообщение Premium Info: {e}")

        if not edit_success:
            await effective_message.reply_text(
                premium_info_text, parse_mode=ParseMode.HTML, disable_web_page_preview=True
            )
    else:
        await context.bot.send_message(
            chat_id=user_id,
            text=premium_info_text,
            parse_mode=ParseMode.HTML,
            disable_web_page_preview=True,
        )


async def cancel_conversation_and_call_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    command_handler_func,
    dialog_name: str = "текущий диалог",
) -> int:
    """
    Cancels the current conversation and calls the specified command handler.
    To be used in ConversationHandler fallbacks.
    """
    user_id = update.effective_user.id
    lang_code = await get_user_language(user_id)

    logger.info(
        f"Диалог '{dialog_name}' прерван командой '{update.message.text}' от user {user_id}"
    )

    keys_to_clear = [
        k
        for k in context.user_data
        if k.startswith("alert_") or k.startswith("edit_alert_") or k.startswith("chat_")
    ]  # ОБНОВЛЕНО
    for key in keys_to_clear:
        try:
            del context.user_data[key]
        except KeyError:
            pass

    await update.message.reply_text(
        get_text(constants.MSG_DIALOG_CANCELLED_BY_COMMAND, lang_code),
        reply_markup=get_main_keyboard(lang_code),
    )

    await command_handler_func(update, context)
    return ConversationHandler.END


async def market_data_menu(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Отображает меню команд с рыночными данными."""
    user_id = update.effective_user.id
    lang_code = await get_user_language(user_id)

    text = f"<b>{get_text('menu_category_market_data', lang_code)}</b>"
    keyboard = [
        [
            InlineKeyboardButton(
                get_text("btn_fear_greed", lang_code), callback_data="command:/feargreed"
            ),
            InlineKeyboardButton(
                f"{get_text('menu_cmd_gas', lang_code)}", callback_data="command:/gas"
            ),
        ],
        [
            InlineKeyboardButton(
                get_text("btn_volatility", lang_code), callback_data="command:/volatility"
            ),
            InlineKeyboardButton(
                f"{get_text('menu_cmd_trending', lang_code)}", callback_data="command:/trending"
            ),
        ],
        [
            InlineKeyboardButton(
                f"{get_text('menu_cmd_funding', lang_code)}", callback_data="command:/funding"
            ),
            InlineKeyboardButton(
                f"{get_text('menu_cmd_tvl', lang_code)}", callback_data="command:/tvl"
            ),
        ],
        [
            InlineKeyboardButton(
                f"{get_text('menu_cmd_marketcap', lang_code)}", callback_data="command:/marketcap"
            ),
            InlineKeyboardButton(
                f"{get_text('menu_cmd_events', lang_code)}", callback_data="command:/events"
            ),
        ],
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    await update.message.reply_text(text, reply_markup=reply_markup, parse_mode=ParseMode.HTML)
