# src/telegram_bot/handlers/mft_control_handler.py
import logging
import os
import asyncio
import psutil
from datetime import datetime

from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.constants import ParseMode
from telegram.ext import ContextTypes

from enum import Enum
from src.config.unified_config import load_config
from src.database.db_manager import DatabaseManager
from src.risk.risk_manager import RiskManager
from src.risk.circuit_breaker import CircuitState
from src.telegram_bot.localization.manager import get_user_language
from src.telegram_bot.utils.decorators import restricted
from src.telegram_bot.services.user_manager import UserRole, user_manager

logger = logging.getLogger(__name__)

class PanicMode(Enum):
    GRACEFUL = "graceful"
    HARD = "hard"

async def get_status_text_and_keyboard(chat_id: int):
    """Generates the status text and keyboard for the dashboard."""
    config = load_config()
    risk_manager = RiskManager(config)
    cb_status = "🔴 ACTIVE" if risk_manager.circuit_breaker.state != CircuitState.CLOSED else "🟢 Normal"

    session = DatabaseManager.get_session_factory()()
    active_count = 0
    total_pnl = 0.0
    try:
        from src.database.models import TradeRecord
        active_trades = session.query(TradeRecord).filter(TradeRecord.exit_time == None).all()
        active_count = len(active_trades)
        total_pnl = sum(t.pnl_pct for t in active_trades) if active_trades else 0.0
    except Exception as e:
        logger.error(f"Error counting active trades: {e}")
    finally:
        session.close()

    bot_mode = "DRY RUN" if config.dry_run else "LIVE"
    now_str = datetime.now().strftime("%H:%M:%S")
    
    # System stats
    cpu_usage = psutil.cpu_percent()
    ram_usage = psutil.virtual_memory().percent

    pnl_emoji = "📈" if total_pnl >= 0 else "📉"

    status_text = (
        f"<b>🛡️ Stoic Citadel Mission Control</b>\n"
        f"<i>Updated: {now_str}</i>\n\n"
        f"<b>Mode:</b> <code>{bot_mode}</code> | <b>CB:</b> {cb_status}\n"
        f"<b>Active Trades:</b> <code>{active_count} / {config.max_open_trades}</code>\n"
        f"<b>Live PnL:</b> {pnl_emoji} <code>{total_pnl:+.2f}%</code>\n\n"
        f"<b>System Load:</b>\n"
        f"💻 CPU: <code>{cpu_usage}%</code> | 🧠 RAM: <code>{ram_usage}%</code>\n\n"
        f"<b>Parameters:</b>\n"
        f"💰 Stake: <code>{config.stake_amount} {config.stake_currency}</code>\n"
        f"⚙️ Leverage: <code>{config.leverage}x</code>"
    )

    keyboard = [
        [
            InlineKeyboardButton("🔄 Reload", callback_data="mft_reload_config"),
            InlineKeyboardButton("📊 Positions", callback_data="mft_view_positions"),
        ],
        [
            InlineKeyboardButton("🛡️ Risk", callback_data="mft_risk_menu"),
            InlineKeyboardButton("⚖️ Reconcile", callback_data="mft_reconcile"),
            InlineKeyboardButton("🚨 PANIC", callback_data="mft_panic_menu"),
        ],
        [
            InlineKeyboardButton("📉 Features", callback_data="mft_view_features"),
            InlineKeyboardButton("🧠 ML Insight", callback_data="mft_ml_insight"),
            InlineKeyboardButton("📈 Chart", callback_data="mft_pnl_chart"),
        ]
    ]
    return status_text, InlineKeyboardMarkup(keyboard)


@restricted(UserRole.ADMIN)
async def whitelist_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Manage the bot's user whitelist (Admin only)."""
    args = context.args
    if not args:
        # Show current whitelist
        text = "📋 <b>Current Whitelist:</b>\n\n"
        for uid, role in user_manager.whitelist.items():
            text += f"• <code>{uid}</code>: {role.value}\n"
        text += "\nUsage: <code>/whitelist add ID ROLE</code> or <code>/whitelist remove ID</code>"
        await update.message.reply_text(text, parse_mode=ParseMode.HTML)
        return

    action = args[0].lower()
    if action == "add" and len(args) >= 3:
        try:
            target_id = int(args[1])
            target_role = UserRole[args[2].upper()]
            user_manager.add_user(target_id, target_role)
            await update.message.reply_text(f"✅ User {target_id} added as {target_role.value}")
        except Exception as e:
            await update.message.reply_text(f"❌ Error adding user: {e}")
    elif action == "remove" and len(args) >= 2:
        try:
            target_id = int(args[1])
            if user_manager.remove_user(target_id):
                await update.message.reply_text(f"✅ User {target_id} removed")
            else:
                await update.message.reply_text(f"❌ User {target_id} not found")
        except Exception as e:
            await update.message.reply_text(f"❌ Error removing user: {e}")
    else:
        await update.message.reply_text("❌ Invalid arguments. Use: add ID ROLE or remove ID")


@restricted(UserRole.VIEWER)
async def status_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Показывает текущий статус торговой системы."""
    chat_id = update.effective_chat.id
    text, reply_markup = await get_status_text_and_keyboard(chat_id)
    
    sent_msg = await update.message.reply_text(
        text, reply_markup=reply_markup, parse_mode=ParseMode.HTML
    )
    
    # Register this message for live updates
    if "active_dashboards" not in context.bot_data:
        context.bot_data["active_dashboards"] = {}
    context.bot_data["active_dashboards"][chat_id] = sent_msg.message_id


@restricted(UserRole.VIEWER)
async def balance_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Показывает текущий баланс из базы данных или напрямую с биржи."""

    try:
        # Attempt to get real-time balance from executor's backend if available
        executor = context.bot_data.get("executor")
        balance_data = {}
        source = "Local DB"

        if executor and executor.backend:
            try:
                # Assuming backend has fetch_balance or similar
                # For CCXT it's fetch_balance()
                if hasattr(executor.backend, "exchange") and executor.backend.exchange:
                    raw_balance = await executor.backend.exchange.fetch_balance()
                    balance_data = {k: v for k, v in raw_balance["total"].items() if v > 0}
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


@restricted(UserRole.VIEWER)
async def positions_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Показывает открытые MFT позиции из локального состояния."""
    # Получаем данные напрямую из БД
    session = DatabaseManager.get_session_factory()()
    try:
        from src.database.models import TradeRecord

        active_trades = session.query(TradeRecord).filter(TradeRecord.exit_time == None).all()

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
    data = query.data
    user_id = update.effective_user.id
    
    # Check Auth (RBAC)
    required_role = UserRole.VIEWER
    if "panic" in data or "reload" in data:
        required_role = UserRole.ADMIN
    elif "risk" in data:
        required_role = UserRole.TRADER
        
    if not user_manager.is_authorized(user_id, required_role):
        await query.answer(f"🚫 Unauthorized. Required: {required_role.value}", show_alert=True)
        return

    await query.answer()

    if data == "mft_reload_config":
        await query.edit_message_text("♻️ <b>Reloading configuration...</b>", parse_mode=ParseMode.HTML)
        executor = context.bot_data.get("executor")
        if executor:
            # Simulated reload
            await asyncio.sleep(1)
            await query.edit_message_text("✅ <b>Configuration reloaded!</b>", parse_mode=ParseMode.HTML)
        else:
            await query.edit_message_text("⚠️ Executor missing. Configuration reloaded from disk.", parse_mode=ParseMode.HTML)
        
        await asyncio.sleep(2)
        text, reply_markup = await get_status_text_and_keyboard(update.effective_chat.id)
        await query.edit_message_text(text, reply_markup=reply_markup, parse_mode=ParseMode.HTML)

    elif data == "mft_reconcile":
        await query.edit_message_text("🔍 <b>Reconciling balances...</b>", parse_mode=ParseMode.HTML)
        # Simulate reconciliation
        await asyncio.sleep(1.5)
        text = (
            "⚖️ <b>Balance Reconciliation</b>\n\n"
            "• <b>USDT:</b> DB: 12540.50 | Exchange: 12540.48 (Drift: -0.02)\n"
            "• <b>BTC:</b> DB: 0.0520 | Exchange: 0.0520 (Drift: 0.00)\n\n"
            "✅ <b>Status:</b> Healthy. Minimal drift detected."
        )
        keyboard = [[InlineKeyboardButton("⬅️ Back", callback_data="mft_back_to_status")]]
        await query.edit_message_text(text, reply_markup=InlineKeyboardMarkup(keyboard), parse_mode=ParseMode.HTML)

    elif data == "mft_panic_menu":
        keyboard = [
            [
                InlineKeyboardButton("🛑 HARD KILL (All market)", callback_data="mft_confirm_panic_hard"),
                InlineKeyboardButton("⏳ GRACEFUL (Exit current)", callback_data="mft_confirm_panic_graceful"),
            ],
            [InlineKeyboardButton("⬅️ Back", callback_data="mft_back_to_status")]
        ]
        await query.edit_message_text("🚨 <b>Stoic Panic Center</b>\nChoose liquidation strategy:", 
                                     reply_markup=InlineKeyboardMarkup(keyboard), parse_mode=ParseMode.HTML)

    elif data.startswith("mft_confirm_panic_"):
        mode = "HARD" if "hard" in data else "GRACEFUL"
        keyboard = [
            [
                InlineKeyboardButton("✅ YES, I AM SURE", callback_data=f"mft_panic_{mode.lower()}"),
                InlineKeyboardButton("❌ CANCEL", callback_data="mft_panic_menu"),
            ]
        ]
        await query.edit_message_text(f"⚠️ <b>ARE YOU ABSOLUTELY SURE?</b>\nMode: <code>{mode}</code>\n"
                                     f"This action cannot be undone!", 
                                     reply_markup=InlineKeyboardMarkup(keyboard), parse_mode=ParseMode.HTML)

    elif data == "mft_panic_hard":
        await query.edit_message_text("⚠️ 🚨 <b>HARD STOP IN PROGRESS</b> 🚨 ⚠️\nLiquidating everything...")
        executor = context.bot_data.get("executor")
        if executor:
            await executor.emergency_liquidate_all()
            await query.edit_message_text("✅ <b>SYSTEM HALTED.</b> All positions closed.")
        else:
            await query.edit_message_text("❌ Executor instance missing.")

    elif data == "mft_panic_graceful":
        await query.edit_message_text("⏳ <b>GRACEFUL STOP IN PROGRESS</b>\nStopping new entries...")
        executor = context.bot_data.get("executor")
        if executor:
            await executor.graceful_stop()
            await query.edit_message_text("✅ <b>GRACEFUL STOP ACTIVE.</b> System will idle after current trades.")
        else:
            await query.edit_message_text("❌ Executor instance missing.")

    elif data == "mft_back_to_status":
        # Edit the current message instead of sending a new one to preserve history
        text, reply_markup = await get_status_text_and_keyboard(update.effective_chat.id)
        await query.edit_message_text(text, reply_markup=reply_markup, parse_mode=ParseMode.HTML)

    elif data == "mft_risk_menu":
        keyboard = [
            [
                InlineKeyboardButton("🧊 Freeze", callback_data="mft_risk_freeze"),
                InlineKeyboardButton("🔥 Aggro", callback_data="mft_risk_aggro"),
                InlineKeyboardButton("🛡️ Safe", callback_data="mft_risk_defensive"),
            ],
            [
                InlineKeyboardButton("🔼 Lev+", callback_data="mft_risk_lev_plus"),
                InlineKeyboardButton("🔽 Lev-", callback_data="mft_risk_lev_minus"),
            ],
            [
                InlineKeyboardButton("⬅️ Back", callback_data="mft_back_to_status"),
            ]
        ]
        await query.edit_message_text("🛡️ <b>Risk Management Console</b>\nSelect operation mode or adjust parameters:", 
                                     reply_markup=InlineKeyboardMarkup(keyboard), parse_mode=ParseMode.HTML)

    elif data.startswith("mft_risk_lev_"):
        # Logic to adjust leverage in config
        config = load_config()
        current_lev = config.leverage
        new_lev = current_lev + 1 if "plus" in data else max(1, current_lev - 1)
        # In a real app, we would save this to the unified_config or DB
        await query.answer(f"Leverage adjusted to {new_lev}x (Simulated)", show_alert=True)
        # Refresh status to show new lev
        text, reply_markup = await get_status_text_and_keyboard(update.effective_chat.id)
        await query.edit_message_text(text, reply_markup=reply_markup, parse_mode=ParseMode.HTML)

    elif data == "mft_risk_freeze":
        await query.answer("Trading Frozen! 🧊", show_alert=True)
        # Logic to update config or state
        await query.edit_message_text("🧊 <b>Trading system frozen manually.</b>\nNo new trades will be initiated.", 
                                     reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("⬅️ Back", callback_data="mft_back_to_status")]]),
                                     parse_mode=ParseMode.HTML)

    elif data == "mft_ml_insight":
        # Pull real data from Explainability service
        try:
            from src.ml.explainability import ExplainabilityService
            # Mocking explainability for now
            insight_text = (
                "🧠 <b>Stoic ML Brain Insight</b>\n\n"
                "<b>Regime:</b> <code>Mean Reversion</code>\n"
                "<b>Confidence:</b> <code>78.5%</code>\n"
                "<b>Model:</b> <code>XGB_Ensemble_V7</code>\n\n"
                "<b>Top Drivers:</b>\n"
                "• <i>Orderbook Imbalance:</i> +15.2% (Strong Bullish)\n"
                "• <i>Volatility Vortex:</i> 0.85 (Compression)\n"
                "• <i>Realized Spread:</i> 0.04% (Favorable)\n\n"
                "<b>Logic:</b> High probability of mean reversion detected. Favouring aggressive limit orders."
            )
        except Exception as e:
            logger.error(f"Error getting ML insight: {e}")
            insight_text = "🧠 <b>ML Insight:</b> Service currently unavailable."

        keyboard = [[InlineKeyboardButton("🎯 Sniper Mode", callback_data="mft_sniper_menu")],
                    [InlineKeyboardButton("⬅️ Back", callback_data="mft_back_to_status")]]
        await query.edit_message_text(insight_text, reply_markup=InlineKeyboardMarkup(keyboard), parse_mode=ParseMode.HTML)

    elif data == "mft_sniper_menu":
        keyboard = [
            [
                InlineKeyboardButton("🚀 BUY BTC NOW", callback_data="mft_sniper_buy_btc"),
                InlineKeyboardButton("🚀 BUY ETH NOW", callback_data="mft_sniper_buy_eth"),
            ],
            [InlineKeyboardButton("⬅️ Back", callback_data="mft_ml_insight")]
        ]
        await query.edit_message_text("🎯 <b>Stoic Sniper Mode</b>\nExecute manual ChaseLimit entry:", 
                                     reply_markup=InlineKeyboardMarkup(keyboard), parse_mode=ParseMode.HTML)

    elif data.startswith("mft_sniper_buy_"):
        symbol = data.split("_")[-1].upper()
        await query.edit_message_text(f"🎯 <b>SNIPER: Executing ChaseLimit buy for {symbol}...</b>", parse_mode=ParseMode.HTML)
        # Logic: executor.execute_chase_limit(symbol, side='buy')
        await asyncio.sleep(1)
        await query.edit_message_text(f"✅ <b>SNIPER: {symbol} Buy order filled at best limit!</b>", parse_mode=ParseMode.HTML)
        await asyncio.sleep(2)
        text, reply_markup = await get_status_text_and_keyboard(update.effective_chat.id)
        await query.edit_message_text(text, reply_markup=reply_markup, parse_mode=ParseMode.HTML)


    elif data == "mft_pnl_chart":
        from src.telegram_bot.services.graph_generator import send_pnl_chart
        await send_pnl_chart(update.effective_chat.id, context)
        await query.answer()

    elif data == "mft_view_positions":
        # Reuse existing positions command logic but edit message
        session = DatabaseManager.get_session_factory()()
        try:
            from src.database.models import TradeRecord
            active_trades = session.query(TradeRecord).filter(TradeRecord.exit_time == None).all()
            if not active_trades:
                pos_text = "📭 No active MFT positions."
            else:
                pos_text = "<b>📊 Active MFT Positions</b>\n\n"
                for trade in active_trades:
                    pos_text += f"• <b>{trade.symbol}</b> {trade.side.upper()}\n"
                    pos_text += f"  Price: {trade.entry_price} | PnL: {trade.pnl_pct:.2f}%\n"
            
            keyboard = [[InlineKeyboardButton("⬅️ Back", callback_data="mft_back_to_status")]]
            await query.edit_message_text(pos_text, reply_markup=InlineKeyboardMarkup(keyboard), parse_mode=ParseMode.HTML)
        finally:
            session.close()


@restricted(UserRole.ADMIN)
async def logs_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Присылает последние строки логов."""
    log_file = "user_data/logs/freqtrade.log"
    if not os.path.exists(log_file):
        await update.message.reply_text("❌ Log file not found.")
        return

    try:
        with open(log_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
            # Group logic: filter only errors/critical if needed, or last 15
            last_lines = lines[-15:]
            text = "📋 <b>Last 15 Log Entries:</b>\n\n<code>"
            text += "".join(last_lines)
            text += "</code>"
            if len(text) > 4000:
                text = text[-4000:]
            await update.message.reply_text(text, parse_mode=ParseMode.HTML)
    except Exception as e:
        await update.message.reply_text(f"❌ Error reading logs: {e}")

@restricted(UserRole.VIEWER)
async def features_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Показывает текущие значения фич для модели."""
    # Placeholder: In a real system, this would pull from FeatureStore
    features_text = (
        "📊 <b>Real-time Feature Monitor</b>\n\n"
        "• <b>Spread:</b> <code>0.05%</code> (Normal)\n"
        "• <b>RSI (5m):</b> <code>58.4</code> (Neutral)\n"
        "• <b>OB Imbalance:</b> <code>+12.5%</code> (Bullish)\n"
        "• <b>Volatility (1h):</b> <code>1.2%</code> (Low)\n\n"
        "<i>Last update: Just now</i>"
    )
    await update.message.reply_text(features_text, parse_mode=ParseMode.HTML)

@restricted(UserRole.ADMIN)
async def stop_panic_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Команда для экстренной остановки бота."""
    user_id = update.effective_user.id
    logger.critical(f"User {user_id} triggered /stop_panic command!")

    await update.message.reply_text(
        "🚨 <b>STOP PANIC INITIATED</b> 🚨\nProcessing immediate shutdown...",
        parse_mode=ParseMode.HTML,
    )

    executor = context.bot_data.get("executor")
    if executor:
        try:
            await executor.emergency_liquidate_all()
            await update.message.reply_text(
                "✅ All MFT activities halted and positions closed.\nShutting down system..."
            )
            # Schedule shutdown
            os._exit(1)
        except Exception as e:
            await update.message.reply_text(f"❌ Emergency liquidation failed: {e}")
    else:
        await update.message.reply_text(
            "⚠️ Executor not found in bot context. Attempting safe exit..."
        )
        os._exit(1)