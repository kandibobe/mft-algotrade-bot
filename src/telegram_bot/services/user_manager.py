from enum import Enum
from typing import Any, Optional
from src.telegram_bot.config_adapter import ADMIN_CHAT_ID
from src.utils.logger import get_logger

logger = get_logger(__name__)

class UserRole(Enum):
    ADMIN = "ADMIN"
    TRADER = "TRADER"
    VIEWER = "VIEWER"
    UNAUTHORIZED = "UNAUTHORIZED"

class UserManager:
    """Manages user roles and access control for the Telegram Bot."""
    
    def __init__(self):
        # In a production system, this would be loaded from a database
        # For Stoic Citadel, we start with a hardcoded admin from config
        # and allow adding other users via commands.
        self.whitelist = {
            str(ADMIN_CHAT_ID): UserRole.ADMIN
        } if ADMIN_CHAT_ID else {}
        
    def get_user_role(self, user_id: int) -> UserRole:
        role = self.whitelist.get(str(user_id), UserRole.UNAUTHORIZED)
        return role

    def is_authorized(self, user_id: int, required_role: UserRole = UserRole.VIEWER) -> bool:
        user_role = self.get_user_role(user_id)
        
        if user_role == UserRole.ADMIN:
            return True
        
        if required_role == UserRole.TRADER:
            return user_role == UserRole.TRADER
            
        if required_role == UserRole.VIEWER:
            return user_role in [UserRole.TRADER, UserRole.VIEWER]
            
        return False

    def add_user(self, user_id: int, role: UserRole) -> bool:
        """Adds or updates a user in the whitelist."""
        self.whitelist[str(user_id)] = role
        logger.info(f"User {user_id} added/updated with role {role.value}")
        return True

    def remove_user(self, user_id: int) -> bool:
        """Removes a user from the whitelist."""
        if str(user_id) in self.whitelist:
            del self.whitelist[str(user_id)]
            logger.info(f"User {user_id} removed from whitelist")
            return True
        return False

# Global instance
user_manager = UserManager()