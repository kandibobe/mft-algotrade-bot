import asyncio
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from telegram import Update, User, Chat, Message
from telegram.ext import ContextTypes

# Mock external dependencies before importing handlers
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

class TestTelegramBotHandlers(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        # Mock Config
        self.patcher_config = patch('src.config.unified_config.load_config')
        self.mock_config = self.patcher_config.start()
        self.mock_config.return_value = MagicMock()
        self.mock_config.return_value.telegram.token = "123:abc"
        self.mock_config.return_value.telegram.chat_id = "12345"
        self.mock_config.return_value.telegram.enabled = True

        # Mock DB
        self.patcher_db = patch('src.database.db_manager.DatabaseManager.get_session_factory')
        self.mock_db = self.patcher_db.start()
        self.mock_session = MagicMock()
        self.mock_db.return_value = MagicMock(return_value=self.mock_session)

        # Import handlers now that mocks are set
        from src.telegram_bot.handlers import mft_control_handler
        self.mft_handler = mft_control_handler

    async def asyncTearDown(self):
        self.patcher_config.stop()
        self.patcher_db.stop()

    def create_mock_update(self, chat_id=12345, user_id=12345, text="/status"):
        update = MagicMock(spec=Update)
        update.effective_chat = MagicMock(spec=Chat)
        update.effective_chat.id = chat_id
        update.effective_user = MagicMock(spec=User)
        update.effective_user.id = user_id
        update.message = AsyncMock(spec=Message)
        update.message.text = text
        update.callback_query = AsyncMock()
        update.callback_query.data = ""
        update.callback_query.message = update.message
        return update

    async def test_status_command(self):
        update = self.create_mock_update()
        context = MagicMock(spec=ContextTypes.DEFAULT_TYPE)
        context.bot_data = {}
        
        await self.mft_handler.status_command(update, context)
        
        # Verify message was sent
        update.message.reply_text.assert_called()
        args, kwargs = update.message.reply_text.call_args
        self.assertIn("Stoic Citadel System Status", args[0])
        self.assertIn("active_dashboards", context.bot_data)

    async def test_balance_command(self):
        update = self.create_mock_update(text="/balance")
        context = MagicMock(spec=ContextTypes.DEFAULT_TYPE)
        context.bot_data = {}
        
        await self.mft_handler.balance_command(update, context)
        update.message.reply_text.assert_called()
        args, kwargs = update.message.reply_text.call_args
        self.assertIn("Balance", args[0])

    async def test_unauthorized_access(self):
        # Admin is 12345 (from mock config adapter which uses load_config)
        # We need to mock ADMIN_CHAT_ID in config_adapter specifically
        with patch('src.telegram_bot.config_adapter.ADMIN_CHAT_ID', "12345"):
            update = self.create_mock_update(user_id=99999) # Wrong user
            update.callback_query.data = "mft_reload_config"
            context = MagicMock(spec=ContextTypes.DEFAULT_TYPE)
            
            await self.mft_handler.mft_control_callback(update, context)
            
            update.callback_query.answer.assert_called_with("🚫 Unauthorized access", show_alert=True)

if __name__ == '__main__':
    unittest.main()