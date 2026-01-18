# handlers/analytics_chat_handler.py
import asyncio
from datetime import datetime

from openai import OpenAI
from telegram import Update
from telegram.ext import CommandHandler, ContextTypes, ConversationHandler, MessageHandler, filters

from src.telegram_bot import constants
from src.telegram_bot.config_adapter import OPENAI_API_KEY
from src.telegram_bot.handlers import common as common_handlers
from src.telegram_bot.localization.manager import get_text, get_user_language
from src.utils.logger import get_logger

logger = get_logger(__name__)

# --- Инициализация клиента OpenAI ---
# Делаем это один раз при загрузке модуля
client = None
if OPENAI_API_KEY:
    try:
        client = OpenAI(api_key=OPENAI_API_KEY)
        logger.info("Клиент OpenAI успешно инициализирован.")
    except Exception as e:
        logger.error(f"Не удалось инициализировать клиент OpenAI: {e}")

# --- Вспомогательная функция для вызова LLM ---
async def get_llm_response(user_input: str, user_id: int) -> str:
    """Отправляет запрос к OpenAI и получает ответ для аналитического чата."""
    if not client:
        logger.error(f"Запрос от user {user_id} не выполнен: клиент OpenAI не инициализирован.")
        return get_text("error_general", "ru") # Возвращаем общую ошибку

    system_prompt = f"""
    Ты — "Crypto Analyst", экспертный ассистент в Telegram-боте.
    Твоя задача — помочь пользователю, отвечая на его вопросы о криптовалютах, финансах и экономике.
    Отвечай кратко, по делу и на том же языке, что и пользователь.
    Если ты не знаешь ответа, скажи "Я не уверен в этом, но могу поискать информацию.".
    Не придумывай данные, такие как цены или даты. Категорически запрещено давать прямые финансовые советы.
    Текущая дата: {datetime.now().strftime('%Y-%m-%d')}.
    """
    try:
        logger.info(f"Отправка запроса в OpenAI для user {user_id}: '{user_input[:50]}...'")

        # Запускаем синхронный вызов API в отдельном потоке, чтобы не блокировать бота
        response = await asyncio.to_thread(
            client.chat.completions.create,
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_input}
            ],
            temperature=0.3,
            max_tokens=300
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"Ошибка при вызове OpenAI API для user {user_id}: {e}", exc_info=True)
        return get_text("error_general", "ru")

# --- Обработчики состояний ---
async def analytics_chat_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Начинает диалог Аналитического Чата."""
    user_id = update.effective_user.id
    lang_code = await get_user_language(user_id)
    logger.info(f"Начало Аналитического Чата для user_id {user_id}")

    if not client:
        await update.message.reply_text("Функция AI-чата временно недоступна из-за ошибки конфигурации.")
        return ConversationHandler.END

    await update.message.reply_text(get_text(constants.MSG_ANALYTICS_CHAT_WELCOME, lang_code))
    return constants.ANALYTICS_CHAT_STATE_START

async def handle_chat_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Обрабатывает все текстовые сообщения внутри чата."""
    user_id = update.effective_user.id
    user_input = update.message.text.strip()

    loading_msg = await update.message.reply_text("🤖 AI-аналитик думает...")
    llm_answer = await get_llm_response(user_input, user_id)

    # Используем parse_mode=None, так как ответ от LLM может содержать Markdown-разметку
    await loading_msg.edit_text(llm_answer, parse_mode=None)

    # Остаемся в том же состоянии, чтобы продолжить диалог
    return constants.ANALYTICS_CHAT_STATE_START

async def cancel_analytics_chat(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Отменяет диалог Аналитического Чата."""
    user_id = update.effective_user.id
    lang_code = await get_user_language(user_id)
    logger.info(f"Отмена Аналитического Чата для user_id {user_id}")

    await update.message.reply_text(
        get_text(constants.MSG_ANALYTICS_CHAT_CANCELLED, lang_code),
        reply_markup=common_handlers.get_main_keyboard(lang_code)
    )
    return ConversationHandler.END

# --- Создание ConversationHandler ---
analytics_chat_conv_handler = ConversationHandler(
    entry_points=[CommandHandler(constants.CMD_ANALYTICS_CHAT, analytics_chat_start)],
    states={
        constants.ANALYTICS_CHAT_STATE_START: [
            MessageHandler(filters.TEXT & ~filters.COMMAND, handle_chat_message)
        ]
    },
    fallbacks=[
        CommandHandler(constants.CMD_CANCEL, cancel_analytics_chat),
        MessageHandler(filters.Regex(r"(?i)^(отмена|cancel)$"), cancel_analytics_chat)
    ],
    conversation_timeout=600.0, # Увеличим таймаут для чата
    per_message=False
)
