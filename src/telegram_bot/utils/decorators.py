from functools import wraps
from telegram import Update
from telegram.ext import ContextTypes
from src.telegram_bot.services.user_manager import user_manager, UserRole

def restricted(required_role: UserRole = UserRole.VIEWER):
    """Decorator to restrict command access based on UserRole."""
    def decorator(func):
        @wraps(func)
        async def wrapped(update: Update, context: ContextTypes.DEFAULT_TYPE, *args, **kwargs):
            user_id = update.effective_user.id
            if not user_manager.is_authorized(user_id, required_role):
                await update.message.reply_text(
                    f"🚫 <b>Access Denied</b>\nRequired role: <code>{required_role.value}</code>",
                    parse_mode='HTML'
                )
                return
            return await func(update, context, *args, **kwargs)
        return wrapped
    return decorator