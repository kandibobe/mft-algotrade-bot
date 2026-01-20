# handlers/navigation.py
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.constants import ParseMode
from telegram.ext import ContextTypes

from src.telegram_bot import constants
from src.telegram_bot.localization.manager import get_text, get_user_language
from src.utils.logger import get_logger

logger = get_logger(__name__)

# --- Уровни меню ---
MENU_MAIN, MENU_ANALYTICS, MENU_MARKET_DATA, MENU_LISTS = range(4)

# --- Callback-данные для навигации ---
CB_NAV_MAIN = "nav_main"
CB_NAV_ANALYTICS = "nav_analytics"
CB_NAV_MARKET_DATA = "nav_market"
CB_NAV_LISTS = "nav_lists"


async def menu_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Отправляет главное меню."""
    await show_main_menu(update, context)


async def show_main_menu(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Отображает главный уровень меню."""
    query = update.callback_query
    user_id = update.effective_user.id
    lang_code = await get_user_language(user_id)

    if query:
        await query.answer()

    text = get_text(
        "menu_main_header",
        lang_code,
        default="<b>🤖 Главное меню</b>\n\nВыберите категорию, чтобы увидеть доступные команды.",
    )

    keyboard = [
        [
            InlineKeyboardButton(
                f"📈 {get_text('menu_category_analytics', lang_code, default='Аналитика')}",
                callback_data=CB_NAV_ANALYTICS,
            )
        ],
        [
            InlineKeyboardButton(
                f"📊 {get_text('menu_category_market_data', lang_code, default='Рыночные данные')}",
                callback_data=CB_NAV_MARKET_DATA,
            )
        ],
        [
            InlineKeyboardButton(
                f"⭐️ {get_text('menu_category_lists', lang_code, default='Управление списками')}",
                callback_data=CB_NAV_LISTS,
            )
        ],
        [
            InlineKeyboardButton(
                f"⚙️ {get_text('btn_settings', lang_code)}", callback_data=constants.CB_MAIN_SETTINGS
            )
        ],
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)

    effective_message = update.message or (query.message if query else None)
    if query:
        await query.edit_message_text(text, reply_markup=reply_markup, parse_mode=ParseMode.HTML)
    elif effective_message:
        await effective_message.reply_text(
            text, reply_markup=reply_markup, parse_mode=ParseMode.HTML
        )


async def show_analytics_menu(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Отображает меню аналитических команд."""
    query = update.callback_query
    user_id = update.effective_user.id
    lang_code = await get_user_language(user_id)
    await query.answer()

    text = f"<b>📈 {get_text('menu_category_analytics', lang_code, default='Аналитика')}</b>"
    keyboard = [
        [
            InlineKeyboardButton(
                get_text("btn_my_report", lang_code), callback_data="command:/report"
            )
        ],
        [InlineKeyboardButton(get_text("btn_signal", lang_code), callback_data="command:/signal")],
        [
            InlineKeyboardButton(
                f"{get_text('menu_cmd_ta', lang_code, default='Тех. анализ')} (/ta)",
                callback_data="command:/ta",
            )
        ],
        [
            InlineKeyboardButton(
                f"⬅️ {get_text('menu_btn_back', lang_code, default='Назад')}",
                callback_data=CB_NAV_MAIN,
            )
        ],
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    await query.edit_message_text(text, reply_markup=reply_markup, parse_mode=ParseMode.HTML)


async def show_market_data_menu(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Отображает меню команд с рыночными данными."""
    query = update.callback_query
    user_id = update.effective_user.id
    lang_code = await get_user_language(user_id)
    await query.answer()

    text = (
        f"<b>📊 {get_text('menu_category_market_data', lang_code, default='Рыночные данные')}</b>"
    )
    keyboard = [
        [
            InlineKeyboardButton(
                get_text("btn_fear_greed", lang_code), callback_data="command:/feargreed"
            ),
            InlineKeyboardButton(
                f"{get_text('menu_cmd_gas', lang_code, default='Газ')} (/gas)",
                callback_data="command:/gas",
            ),
        ],
        [
            InlineKeyboardButton(
                get_text("btn_volatility", lang_code), callback_data="command:/volatility"
            ),
            InlineKeyboardButton(
                f"{get_text('menu_cmd_trending', lang_code, default='Тренды')} (/trending)",
                callback_data="command:/trending",
            ),
        ],
        [
            InlineKeyboardButton(
                f"{get_text('menu_cmd_funding', lang_code, default='Фандинг')} (/funding)",
                callback_data="command:/funding",
            ),
            InlineKeyboardButton(
                f"{get_text('menu_cmd_tvl', lang_code, default='TVL')} (/tvl)",
                callback_data="command:/tvl",
            ),
        ],
        [
            InlineKeyboardButton(
                f"⬅️ {get_text('menu_btn_back', lang_code, default='Назад')}",
                callback_data=CB_NAV_MAIN,
            )
        ],
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    await query.edit_message_text(text, reply_markup=reply_markup, parse_mode=ParseMode.HTML)


async def show_lists_menu(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Отображает меню управления списками."""
    query = update.callback_query
    user_id = update.effective_user.id
    lang_code = await get_user_language(user_id)
    await query.answer()

    text = f"<b>⭐️ {get_text('menu_category_lists', lang_code, default='Управление списками')}</b>"
    keyboard = [
        [
            InlineKeyboardButton(
                get_text("btn_watchlist", lang_code), callback_data="command:/watchlist"
            )
        ],
        [InlineKeyboardButton(get_text("btn_alerts", lang_code), callback_data="command:/alerts")],
        [
            InlineKeyboardButton(
                f"⬅️ {get_text('menu_btn_back', lang_code, default='Назад')}",
                callback_data=CB_NAV_MAIN,
            )
        ],
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    await query.edit_message_text(text, reply_markup=reply_markup, parse_mode=ParseMode.HTML)


async def navigate_to_command_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обрабатывает нажатие на кнопку с командой, 'удаляя' меню и вызывая команду."""
    query = update.callback_query
    if not query or not query.data:
        return

    command_to_run = query.data.split(":", 1)[1]

    # Имитируем вызов команды от пользователя
    update.message = query.message  # "Пересаживаем" сообщение из query в update
    update.message.text = command_to_run

    # Очищаем аргументы, если они были от предыдущей команды
    context.args = []

    # Удаляем сообщение с меню, чтобы не засорять чат
    try:
        await query.delete_message()
    except Exception as e:
        logger.warning(f"Не удалось удалить сообщение с меню: {e}")

    # Находим и вызываем нужный обработчик команды
    # Это упрощенный диспетчер. В реальном приложении может потребоваться более сложная логика.
    # Мы будем полагаться на то, что Application сам найдет нужный CommandHandler.
    # Для этого мы должны "пропустить" обновление дальше.
    # Но так как мы уже в callback-обработчике, стандартный механизм не сработает.
    # Поэтому мы напрямую вызовем нужную функцию-обработчик.

    # Простой маппинг для примера.
    from src.telegram_bot.handlers import (
        alert_handler,
        misc_handler,
        report_handler,
        signal_handler,
        watchlist_handler,
    )

    command_map = {
        "/report": report_handler.report_command_handler,
        "/signal": signal_handler.signal_command_handler,
        "/ta": misc_handler.ta_command,
        "/feargreed": misc_handler.fear_greed_command,
        "/gas": misc_handler.gas_command,
        "/volatility": misc_handler.volatility_command_handler,
        "/trending": misc_handler.trending_command,
        "/funding": misc_handler.funding_command,
        "/tvl": misc_handler.tvl_command,
        "/watchlist": watchlist_handler.watchlist_command,
        "/alerts": alert_handler.alerts_command,
    }

    handler_func = command_map.get(command_to_run)
    if handler_func:
        logger.info(
            f"Навигация: вызов команды {command_to_run} для user {update.effective_user.id}"
        )
        await handler_func(update, context)
    else:
        logger.warning(f"Навигация: не найден обработчик для команды {command_to_run}")
