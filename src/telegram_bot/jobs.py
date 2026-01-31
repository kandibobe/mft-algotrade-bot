from telegram.ext import ContextTypes
from src.telegram_bot.handlers.mft_control_handler import get_status_text_and_keyboard
from src.utils.logger import get_logger
from telegram.constants import ParseMode

logger = get_logger(__name__)

async def update_live_dashboard(context: ContextTypes.DEFAULT_TYPE):
    """Background job to update all active dashboards."""
    active_dashboards = context.bot_data.get("active_dashboards", {})
    if not active_dashboards:
        return

    for chat_id, message_id in list(active_dashboards.items()):
        try:
            text, reply_markup = await get_status_text_and_keyboard(chat_id)
            await context.bot.edit_message_text(
                chat_id=chat_id,
                message_id=message_id,
                text=text,
                reply_markup=reply_markup,
                parse_mode=ParseMode.HTML
            )
        except Exception as e:
            # If message is deleted or cannot be edited, remove from active
            logger.warning(f"Could not update dashboard for chat {chat_id}: {e}")
            active_dashboards.pop(chat_id, None)

# Placeholder for other jobs
async def check_price_alerts(context: ContextTypes.DEFAULT_TYPE):
    pass

async def fetch_volatility_job(context: ContextTypes.DEFAULT_TYPE):
    pass

async def fetch_shared_data_job(context: ContextTypes.DEFAULT_TYPE):
    pass

async def fetch_index_data_job(context: ContextTypes.DEFAULT_TYPE):
    pass

async def fetch_onchain_data_job(context: ContextTypes.DEFAULT_TYPE):
    pass

async def send_daily_digest(context: ContextTypes.DEFAULT_TYPE):
    pass

async def system_heartbeat(context: ContextTypes.DEFAULT_TYPE):
    """Send periodic system heartbeat to admin."""
    chat_id = context.job.chat_id if context.job.chat_id else context.bot_data.get("admin_chat_id")
    if not chat_id:
        # Try finding chat_id from env if not in job/data
        import os
        chat_id = os.getenv("TELEGRAM_CHAT_ID")
    
    if chat_id:
        try:
            await context.bot.send_message(
                chat_id=chat_id,
                text="=“ <b>System Heartbeat:</b> Stoic Citadel is online and healthy.",
                parse_mode=ParseMode.HTML,
                disable_notification=True
            )
        except Exception as e:
            logger.error(f"Failed to send heartbeat: {e}")
