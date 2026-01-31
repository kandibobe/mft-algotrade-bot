from telegram.ext import Application
from src.utils.logger import get_logger
import asyncio

logger = get_logger(__name__)

class NotificationService:
    """Group and send notifications with anti-spam logic."""
    
    def __init__(self, application: Application):
        self.application = application
        self.message_queue = asyncio.Queue()
        self.group_delay = 1.0  # seconds to wait for grouping
        self._batch = []
        self._lock = asyncio.Lock()
        
        # Start background sender
        asyncio.create_task(self._sender_loop())

    async def notify(self, message: str, chat_id: int):
        """Public interface to send a notification."""
        await self.message_queue.put((message, chat_id))

    async def _sender_loop(self):
        while True:
            message, chat_id = await self.message_queue.get()
            self._batch.append(message)
            
            # Wait a bit for more messages to group
            await asyncio.sleep(self.group_delay)
            
            while not self.message_queue.empty():
                msg, _ = self.message_queue.get_nowait()
                self._batch.append(msg)
            
            if self._batch:
                await self._send_batch(chat_id)
                self._batch = []
            
            self.message_queue.task_done()

    async def _send_batch(self, chat_id: int):
        if not self._batch:
            return
            
        if len(self._batch) == 1:
            final_text = self._batch[0]
        else:
            final_text = f"📦 <b>Stoic Group Notification ({len(self._batch)} events)</b>\n\n"
            final_text += "\n".join([f"• {m}" for m in self._batch])
            
        try:
            await self.application.bot.send_message(
                chat_id=chat_id,
                text=final_text,
                parse_mode='HTML'
            )
        except Exception as e:
            logger.error(f"Failed to send grouped notification: {e}")