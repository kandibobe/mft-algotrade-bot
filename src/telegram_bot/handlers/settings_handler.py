# handlers/settings_handler.py

from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.constants import ParseMode
from telegram.ext import ContextTypes

from src.telegram_bot import constants
from src.telegram_bot.config_adapter import ANALYSIS_PERIODS, DEFAULT_ANALYSIS_PERIOD
from src.telegram_bot.handlers import common
from src.telegram_bot.localization.manager import get_text, get_user_language
from src.telegram_bot.services import user_manager
from src.utils.logger import get_logger

logger = get_logger(__name__)

async def settings_command_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id; logger.info(f"Запрос настроек от user_id {user_id}"); await show_settings_menu(update, context)

async def settings_main_menu_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
     query = update.callback_query; await query.answer(); await show_settings_menu(update, context)

async def show_settings_menu(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    if query: await query.answer()
    user_id = update.effective_user.id; lang_code = await get_user_language(user_id)
    settings = user_manager.get_settings(user_id); current_period = settings.get('analysis_period', DEFAULT_ANALYSIS_PERIOD); notifications_on = settings.get('notifications_enabled', True); is_premium = user_manager.is_user_premium(user_id)
    alerts_status_key = constants.MSG_ALERTS_ON if notifications_on else constants.MSG_ALERTS_OFF; alerts_status = get_text(alerts_status_key, lang_code); toggle_button_text_key = constants.MSG_ALERTS_OFF if notifications_on else constants.MSG_ALERTS_ON; toggle_button_text = get_text(toggle_button_text_key, lang_code); premium_status_text_key = constants.MSG_PREMIUM_STATUS_ACTIVE if is_premium else constants.MSG_PREMIUM_STATUS_INACTIVE; premium_status_text = get_text(premium_status_text_key, lang_code)
    menu_text = get_text(constants.MSG_SETTINGS_MENU, lang_code, lang=lang_code.upper(), period=current_period, alerts_status=alerts_status, premium_status=premium_status_text)
    keyboard = [ [InlineKeyboardButton(f"⏳ Период ({current_period} дн.)", callback_data=constants.CB_SETTINGS_PERIOD)], [InlineKeyboardButton(toggle_button_text, callback_data=constants.CB_SETTINGS_ALERTS_TOGGLE)], [InlineKeyboardButton("🌐 Язык / Language", callback_data=constants.CB_SETTINGS_LANG)], [InlineKeyboardButton("⭐️ Премиум Инфо", callback_data=constants.CB_ACTION_SHOW_PREMIUM)] ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    effective_message = update.message or (query.message if query else None)
    if not effective_message: logger.error(f"Не удалось найти сообщение для ответа/редактирования в show_settings_menu для user {user_id}"); return
    if query and query.message:
        try: await query.edit_message_text(menu_text, reply_markup=reply_markup, parse_mode=ParseMode.HTML)
        except Exception as e: logger.warning(f"Не удалось отредактировать меню настроек: {e}, отправка нового."); await effective_message.reply_text(menu_text, reply_markup=reply_markup, parse_mode=ParseMode.HTML)
    else: await effective_message.reply_text(menu_text, reply_markup=reply_markup, parse_mode=ParseMode.HTML)

async def show_period_selection(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query; await query.answer(); user_id = update.effective_user.id; lang_code = await get_user_language(user_id)
    period_buttons = [ InlineKeyboardButton(f"{p} дн.", callback_data=f"{constants.CB_ACTION_SET_PERIOD}{p}") for p in ANALYSIS_PERIODS ]
    back_button = InlineKeyboardButton("⬅️ Назад", callback_data=constants.CB_MAIN_SETTINGS); keyboard = [period_buttons, [back_button]]; reply_markup = InlineKeyboardMarkup(keyboard); prompt_text = get_text(constants.MSG_SELECT_PERIOD, lang_code)
    try: await query.edit_message_text(prompt_text, reply_markup=reply_markup)
    except Exception as e: logger.error(f"Ошибка редактирования сообщения для выбора периода: {e}")
    if query.message: await context.bot.send_message(chat_id=query.message.chat_id, text=prompt_text, reply_markup=reply_markup)

async def set_analysis_period(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query; user_id = update.effective_user.id; lang_code = await get_user_language(user_id)
    try:
        period_str = query.data.split('_')[-1]; period = int(period_str); logger.info(f"User {user_id} выбрал период анализа: {period}")
        if period in ANALYSIS_PERIODS:
            success = user_manager.update_analysis_period(user_id, period)
            if success: await query.answer(get_text(constants.MSG_PERIOD_SET, lang_code, period=period)); await show_settings_menu(update, context)
            else: await query.answer(get_text(constants.MSG_ERROR_DB, lang_code), show_alert=True); await show_settings_menu(update, context)
        else: logger.warning(f"Неверный период {period} от user {user_id} в callback: {query.data}"); await query.answer("Ошибка: неверный период.", show_alert=True); await show_settings_menu(update, context)
    except (IndexError, ValueError) as e: logger.error(f"Ошибка извлечения периода из callback '{query.data}': {e}"); await query.answer("Ошибка обработки.", show_alert=True); await show_settings_menu(update, context)
    except Exception as e: logger.error(f"Непредвиденная ошибка в set_analysis_period: {e}", exc_info=True); await query.answer(get_text(constants.MSG_ERROR_GENERAL, lang_code), show_alert=True); await show_settings_menu(update, context)

async def toggle_alerts_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query; user_id = update.effective_user.id; lang_code = await get_user_language(user_id); logger.info(f"User {user_id} нажал кнопку переключения уведомлений.")
    new_status = user_manager.toggle_notifications(user_id)
    if new_status is None: await query.answer(get_text(constants.MSG_ERROR_DB, lang_code), show_alert=True)
    else: await query.answer(get_text(constants.MSG_ALERT_TOGGLE_CONFIRM, lang_code))
    await show_settings_menu(update, context)

async def premium_info_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query; await query.answer(); await common.premium_command(update, context)
