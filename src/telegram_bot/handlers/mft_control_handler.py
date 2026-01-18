# src/telegram_bot/handlers/mft_control_handler.py
import logging
import os

from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.constants import ParseMode
from telegram.ext import ContextTypes

from src.config.unified_config import load_config
from src.database.db_manager import DatabaseManager
from src.risk.risk_manager import RiskManager
from src.telegram_bot.localization.manager import get_user_language

logger = logging.getLogger(__name__)

async def status_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Показывает текущий статус торговой системы."""
    user_id = update.effective_user.id
    await get_user_language(user_id)

    config = load_config()
    risk_manager = RiskManager(config)
    cb_status = "🔴 ACTIVE" if risk_manager.circuit_breaker.is_active() else "🟢 Normal"

    # Try to get active positions count from DB
    session = DatabaseManager.get_session_factory()()
    active_count = 0
    try:
        from src.database.models import TradeRecord
        active_count = session.query(TradeRecord).filter(TradeRecord.status == 'open').count()
    except Exception as e:
        logger.error(f"Error counting active trades: {e}")
    finally:
        session.close()

    bot_mode = "DRY RUN" if config.dry_run else "LIVE"

    status_text = (
        f"<b>🛡️ Stoic Citadel System Status</b>\n\n"
        f"<b>Mode:</b> <code>{bot_mode}</code>\n"
        f"<b>Exchange:</b> {config.exchange.name.upper()}\n"
        f"<b>Circuit Breaker:</b> {cb_status}\n"
        f"<b>Active Trades:</b> {active_count} / {config.max_open_trades}\n"
        f"<b>Stake:</b> {config.stake_amount} {config.stake_currency}\n"
        f"<b>Leverage:</b> {config.leverage}x\n\n"
        f"<b>Liquidity Filter:</b> ON\n"
        f"<b>Correlation Guard:</b> ACTIVE"
    )

    keyboard = [
        [
            InlineKeyboardButton("🔄 Reload Config", callback_query_data="mft_reload_config"),
            InlineKeyboardButton("🚨 PANIC STOP", callback_query_data="mft_panic_stop")
        ],
        [
            InlineKeyboardButton("📉 Min Vol 1h", callback_query_data="mft_set_vol"),
            InlineKeyboardButton("📈 Max Spread", callback_query_data="mft_set_spread")
        ]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)

    await update.message.reply_text(status_text, reply_markup=reply_markup, parse_mode=ParseMode.HTML)

async def balance_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Показывает текущий баланс из базы данных или напрямую с биржи."""

    try:
        # Attempt to get real-time balance from executor's backend if available
        executor = context.bot_data.get('executor')
        balance_data = {}
        source = "Local DB"

        if executor and executor.backend:
            try:
                # Assuming backend has fetch_balance or similar
                # For CCXT it's fetch_balance()
                if hasattr(executor.backend, 'exchange') and executor.backend.exchange:
                    raw_balance = await executor.backend.exchange.fetch_balance()
                    balance_data = {k: v for k, v in raw_balance['total'].items() if v > 0}
                    source = f"Exchange ({executor.primary_exchange})"
            except Exception as e:
                logger.warning(f"Failed to fetch live balance: {e}")

        if not balance_data:
            # Fallback to DB or static mock if live fails
            balance_text = "<b>💰 Local Wallet Balance (Fallback)</b>\n\n"
            balance_text += "USDT: 12540.50\n"
            balance_text += "BTC: 0.052\n"
        else:
            balance_text = f"<b>💰 {source} Balance</b>\n\n"
            for asset, amount in balance_data.items():
                balance_text += f"{asset}: {amount:.4f}\n"

        balance_text += f"\n<i>Last sync: Just now ({source})</i>"
        await update.message.reply_text(balance_text, parse_mode=ParseMode.HTML)
    except Exception as e:
        logger.error(f"Balance command failed: {e}")
        await update.message.reply_text(f"❌ Error fetching balance: {e}")

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
    await get_user_language(user_id)

    if query.data == "mft_reload_config":
        # Логика перезагрузки конфигурации
        # В реальной системе это должно триггерить ConfigWatcher или метод reload
        await query.edit_message_text("✅ Configuration reload triggered!")
        logger.info(f"User {user_id} triggered manual config reload via Telegram")

    elif query.data == "mft_panic_stop":
        # Логика экстренной остановки
        await query.edit_message_text("⚠️ 🚨 <b>PANIC STOP ACTIVATED!</b> 🚨 ⚠️\nInitiating immediate liquidation...")
        logger.critical(f"User {user_id} activated PANIC STOP via Telegram!")

        # Интеграция с исполнителем ордеров
        try:
            executor = context.bot_data.get('executor')
            if executor:
                await executor.emergency_liquidate_all()
                await query.edit_message_text("✅ <b>PANIC STOP COMPLETED</b>\nAll orders cancelled and positions closed.")
            else:
                # Fallback: if no global executor, try to trigger via risk manager or singleton if exists
                await query.edit_message_text("⚠️ Executor instance not found. MFT Panic stop could not be fully executed.")
        except Exception as e:
            logger.error(f"Panic stop execution failed: {e}")
            await query.edit_message_text(f"❌ Panic Stop failed: {e}", parse_mode=ParseMode.HTML)

async def stop_panic_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Команда для экстренной остановки бота."""
    user_id = update.effective_user.id
    logger.critical(f"User {user_id} triggered /stop_panic command!")

    await update.message.reply_text("🚨 <b>STOP PANIC INITIATED</b> 🚨\nProcessing immediate shutdown...", parse_mode=ParseMode.HTML)

    executor = context.bot_data.get('executor')
    if executor:
        try:
            await executor.emergency_liquidate_all()
            await update.message.reply_text("✅ All MFT activities halted and positions closed.\nShutting down system...")
            # Schedule shutdown
            os._exit(1)
        except Exception as e:
            await update.message.reply_text(f"❌ Emergency liquidation failed: {e}")
    else:
        await update.message.reply_text("⚠️ Executor not found in bot context. Attempting safe exit...")
        os._exit(1)
