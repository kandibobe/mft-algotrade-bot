# handlers/language_handler.py
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.constants import ParseMode
from telegram.ext import ContextTypes

from src.telegram_bot import constants
from src.telegram_bot.config_adapter import SUPPORTED_LANGUAGES
from src.telegram_bot.handlers.common import (
    get_main_keyboard,  # Импортируем для обновления клавиатуры
)
from src.telegram_bot.localization.manager import (
    get_text,
    get_user_language,
)
from src.telegram_bot.services import user_manager
from src.utils.logger import get_logger

logger = get_logger(__name__)


async def language_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Отображает кнопки для выбора языка."""
    user_id = update.effective_user.id
    lang_code = await get_user_language(user_id)
    logger.info(f"Запрос смены языка от user_id {user_id}")

    keyboard = []
    for lang in SUPPORTED_LANGUAGES:
        lang_name = "Русский" if lang == "ru" else "English"
        # Добавляем галочку к текущему языку
        prefix = "✅ " if lang == lang_code else ""
        keyboard.append(
            [
                InlineKeyboardButton(
                    f"{prefix}{lang_name}",
                    callback_data=f"{constants.CB_ACTION_SET_LANG}{lang}",  # Префикс + код языка
                )
            ]
        )

    reply_markup = InlineKeyboardMarkup(keyboard)
    prompt_text = get_text(constants.MSG_LANG_SELECT, lang_code)

    effective_message = update.message or (
        update.callback_query.message if update.callback_query else None
    )
    if not effective_message:
        logger.warning("Не удалось найти сообщение для ответа в language_command")
        return

    # Пытаемся редактировать, если это callback, иначе отправляем новое
    if update.callback_query:
        try:
            await update.callback_query.edit_message_text(prompt_text, reply_markup=reply_markup)
        except Exception as e:
            logger.warning(f"Не удалось отредактировать сообщение для выбора языка: {e}")
            # Если редактирование не удалось, отправим новое сообщение
            await context.bot.send_message(
                chat_id=user_id, text=prompt_text, reply_markup=reply_markup
            )
    else:
        await effective_message.reply_text(prompt_text, reply_markup=reply_markup)


async def set_language_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обрабатывает выбор языка и обновляет настройки."""
    query = update.callback_query
    if not query:
        return

    user_id = update.effective_user.id

    try:
        # <<< Начало try блока >>>
        # Извлекаем код языка из callback_data
        callback_prefix = constants.CB_ACTION_SET_LANG
        if not query.data or not query.data.startswith(callback_prefix):
            raise ValueError("Некорректный callback_data для языка")
        new_lang_code = query.data[len(callback_prefix) :]

        # <<< ИСПРАВЛЕНО: Проверка языка ПЕРЕМЕЩЕНА внутрь try >>>
        if new_lang_code not in SUPPORTED_LANGUAGES:
            raise ValueError(f"Неподдерживаемый код языка: {new_lang_code}")

        # <<< Конец try блока >>>
    except (IndexError, ValueError) as e:
        logger.error(f"Ошибка извлечения/валидации языка из callback_data '{query.data}': {e}")
        await query.answer("Ошибка!", show_alert=True)
        return

    logger.info(f"User {user_id} выбрал язык: {new_lang_code}")

    # Обновляем язык в БД и кеше
    success = user_manager.set_language(user_id, new_lang_code)

    if success:
        reply_text = get_text(constants.MSG_LANG_SET, new_lang_code)  # Сообщение на НОВОМ языке
        await query.answer(reply_text)  # Краткий ответ на callback

        # Редактируем исходное сообщение, добавляя напоминание о /start
        final_text = (
            reply_text + f"\n\n<i>{get_text(constants.MSG_RESTART_NEEDED, new_lang_code)}</i>"
        )
        try:
            await query.edit_message_text(final_text, parse_mode=ParseMode.HTML)
            # Отправляем новое сообщение с обновленной клавиатурой
            await context.bot.send_message(
                chat_id=user_id,
                text=f"✅ {reply_text}",  # Добавим галочку для ясности
                reply_markup=get_main_keyboard(new_lang_code),  # Обновляем ReplyKeyboard
            )
        except Exception as e:
            logger.warning(
                f"Не удалось отредактировать сообщение/отправить клавиатуру после смены языка: {e}"
            )
            # Если не удалось, все равно обновим клавиатуру новым сообщением
            try:
                await context.bot.send_message(
                    chat_id=user_id,
                    text=f"✅ {reply_text}",
                    reply_markup=get_main_keyboard(new_lang_code),
                )
            except Exception as send_err:
                logger.error(
                    f"Не удалось отправить сообщение с новой клавиатурой user {user_id}: {send_err}"
                )

    else:
        # Если ошибка БД при сохранении
        error_db_text = get_text(constants.MSG_ERROR_DB, new_lang_code)  # Сообщение на НОВОМ языке
        await query.answer(error_db_text, show_alert=True)
