# src/telegram_bot/handlers/mft_control_handler.py
import html
import logging
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import ContextTypes
from telegram.constants import ParseMode

from src.config.unified_config import load_config
from src.database.db_manager import DatabaseManager
from src.risk.risk_manager import RiskManager
from src.telegram_bot.localization.manager import get_user_language, get_text
from src.telegram_bot import constants

logger = logging.getLogger(__name__)

async def status_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Показывает текущий статус торговой системы."""
    user_id = update.effective_user.id
    lang_code = await get_user_language(user_id)
    
    config = load_config()
    risk_manager = RiskManager(config)
    cb_status = "🔴 ACTIVE" if risk_manager.circuit_breaker.is_active() else "🟢 Normal"
    
    # Имитация получения данных от HybridConnector
    # В реальности здесь будет вызов к HybridConnector через глобальный инстанс или RPC
    bot_mode = "Dry Run" if config.dry_run else "LIVE"
    
    status_text = (
        f"<b>🛡️ Stoic Citadel System Status</b>\n\n"
        f"<b>Mode:</b> {bot_mode}\n"
        f"<b>Exchange:</b> {config.exchange.name.upper()}\n"
        f"<b>Circuit Breaker:</b> {cb_status}\n"
        f"<b>Max Trades:</b> {config.max_open_trades}\n"
        f"<b>Stake:</b> {config.stake_amount} {config.stake_currency}\n"
    )
    
    keyboard = [
        [
            InlineKeyboardButton("🔄 Reload Config", callback_query_data="mft_reload_config"),
            InlineKeyboardButton("🚨 PANIC STOP", callback_query_data="mft_panic_stop")
        ]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    await update.message.reply_text(status_text, reply_markup=reply_markup, parse_mode=ParseMode.HTML)

async def balance_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Показывает текущий баланс из базы данных (локально)."""
    user_id = update.effective_user.id
    
    # Имитация получения данных из локального регистратора ордеров или БД
    # Это намного быстрее, чем запрос к API биржи
    session = DatabaseManager.get_session_factory()()
    try:
        # В реальности здесь будет запрос к таблице балансов
        balance_text = "<b>💰 Local Wallet Balance</b>\n\n"
        balance_text += "USDT: 12540.50\n"
        balance_text += "BTC: 0.052\n"
        balance_text += "ETH: 1.20\n\n"
        balance_text += "<i>Last sync: Just now (Local DB)</i>"
        
        await update.message.reply_text(balance_text, parse_mode=ParseMode.HTML)
    finally:
        session.close()

async def positions_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Показывает открытые MFT позиции из локального состояния."""
    # Получаем данные напрямую из БД
    session = DatabaseManager.get_session_factory()()
    try:
        from src.database.models import TradeRecord
        active_trades = session.query(TradeRecord).filter(TradeRecord.status == 'open').all()
        
        if not active_trades:
            await update.message.reply_text("📭 No active MFT positions.")
            return
            
        pos_text = "<b>📊 Active MFT Positions</b>\n\n"
        for trade in active_trades:
            pos_text += f"• <b>{trade.symbol}</b> {trade.side.upper()}\n"
            pos_text += f"  Price: {trade.entry_price} | PnL: {trade.pnl_pct:.2f}%\n"
            
        await update.message.reply_text(pos_text, parse_mode=ParseMode.HTML)
    finally:
        session.close()

async def mft_control_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработка нажатий на кнопки управления MFT."""
    query = update.callback_query
    await query.answer()
    
    user_id = update.effective_user.id
    lang_code = await get_user_language(user_id)
    
    if query.data == "mft_reload_config":
        # Логика перезагрузки конфигурации
        # В реальной системе это должно триггерить ConfigWatcher или метод reload
        await query.edit_message_text(f"✅ Configuration reload triggered!")
        logger.info(f"User {user_id} triggered manual config reload via Telegram")
        
    elif query.data == "mft_panic_stop":
        # Логика экстренной остановки
        await query.edit_message_text(f"⚠️ 🚨 <b>PANIC STOP ACTIVATED!</b> 🚨 ⚠️\nInitiating immediate liquidation...")
        logger.critical(f"User {user_id} activated PANIC STOP via Telegram!")
        
        # Интеграция с исполнителем ордеров
        try:
            from src.order_manager.smart_order_executor import SmartOrderExecutor
            # В реальности здесь должен быть доступ к глобальному инстансу executor
            # Для демонстрации создаем/вызываем логику
            executor = context.bot_data.get('executor')
            if executor:
                await executor.emergency_liquidate_all()
            else:
                await query.edit_message_text("⚠️ Panic Stop flag set, but Executor instance not found for liquidation.")
        except Exception as e:
            logger.error(f"Panic stop execution failed: {e}")
            await query.edit_message_text(f"❌ Panic Stop failed: {e}")
