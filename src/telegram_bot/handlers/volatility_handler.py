# handlers/volatility_handler.py
import html

from telegram import Update
from telegram.constants import ParseMode
from telegram.ext import ContextTypes

from src.telegram_bot import constants
from src.telegram_bot.handlers.misc_handler import _get_cached_or_fetch
from src.telegram_bot.localization.manager import get_text, get_user_language
from src.telegram_bot.services import data_fetcher
from src.utils.logger import get_logger

logger = get_logger(__name__)

async def volatility_command_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обрабатывает команду /volatility и кнопку 'Волатильность'."""
    user_id = update.effective_user.id
    lang_code = await get_user_language(user_id)
    logger.info(f"Запрос /volatility от user_id: {user_id}")

    effective_message = update.message or (update.callback_query.message if update.callback_query else None)
    if not effective_message:
        logger.error("volatility_command_handler: Не найдено сообщение для ответа!")
        try: await context.bot.send_message(user_id, get_text(constants.MSG_ERROR_GENERAL, lang_code))
        except Exception: pass
        return

    # <<< ИЗМЕНЕНИЕ: Убрана распаковка кортежа >>>
    volatility_data = await _get_cached_or_fetch(
        "volatility",
        data_fetcher.fetch_top_crypto_volatility,
        context,
        update,
        lang_code
    )

    if isinstance(volatility_data, dict):
        try:
            gainers = volatility_data.get("gainers", [])
            losers = volatility_data.get("losers", [])

            lines = [get_text(constants.MSG_VOLATILITY_HEADER, lang_code)]

            if gainers:
                for i, coin in enumerate(gainers, 1):
                    symbol = html.escape(coin.get('symbol', '?'))
                    change_val = coin.get('change')
                    price_val = coin.get('price')
                    change_str = f"+{change_val:.2f}%" if isinstance(change_val, (float, int)) and change_val > 0 else f"{change_val:.2f}%" if isinstance(change_val, (float, int)) else "N/A"
                    price_str = f"${price_val:,.2f}" if isinstance(price_val, (float, int)) else "N/A"
                    lines.append(get_text(constants.MSG_VOLATILITY_GAINER, lang_code, rank=i, symbol=symbol, change=change_str, price=price_str))

            if losers:
                lines.append(get_text(constants.MSG_VOLATILITY_SEPARATOR, lang_code))
                for i, coin in enumerate(losers, 1):
                    symbol = html.escape(coin.get('symbol', '?'))
                    change_val = coin.get('change')
                    price_val = coin.get('price')
                    change_str = f"{change_val:.2f}%" if isinstance(change_val, (float, int)) else "N/A"
                    price_str = f"${price_val:,.2f}" if isinstance(price_val, (float, int)) else "N/A"
                    lines.append(get_text(constants.MSG_VOLATILITY_LOSER, lang_code, rank=i, symbol=symbol, change=change_str, price=price_str))

            if not gainers and not losers:
                 lines.append(f"<i>({get_text(constants.MSG_NO_DATA_AVAILABLE, lang_code)})</i>")

            reply_text = "\n".join(lines)
            await effective_message.reply_text(reply_text, parse_mode=ParseMode.HTML, disable_web_page_preview=True)

        except Exception as e:
            logger.error(f"Ошибка форматирования/отправки Volatility: {e}", exc_info=True)
            try: await effective_message.reply_text(get_text(constants.MSG_ERROR_GENERAL, lang_code))
            except Exception: pass
