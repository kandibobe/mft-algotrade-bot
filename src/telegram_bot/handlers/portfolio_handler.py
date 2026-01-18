# handlers/portfolio_handler.py
import asyncio
import html

from telegram import Update
from telegram.constants import ParseMode
from telegram.ext import ContextTypes

from src.telegram_bot import constants
from src.telegram_bot.localization.manager import get_text, get_user_language
from src.telegram_bot.services import data_fetcher, user_manager
from src.utils.logger import get_logger

logger = get_logger(__name__)

async def portfolio_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Отображает портфель пользователя с расчетом P/L."""
    user_id = update.effective_user.id
    lang_code = await get_user_language(user_id)
    logger.info(f"Запрос /portfolio от user_id {user_id}")

    effective_message = update.message or (update.callback_query.message if update.callback_query else None)
    if not effective_message: return

    loading_msg = await effective_message.reply_text(get_text(constants.MSG_LOADING, lang_code))

    portfolio = user_manager.get_user_portfolio(user_id)

    if not portfolio:
        await loading_msg.edit_text(get_text(constants.MSG_PORTFOLIO_EMPTY, lang_code))
        return

    session = context.bot_data.get('aiohttp_session')
    if not session or session.closed:
        await loading_msg.edit_text(get_text(constants.MSG_ERROR_GENERAL, lang_code))
        return

    # Сбор ID активов для запроса цен
    crypto_ids_to_fetch = [item['asset_id'] for item in portfolio if item['asset_type'] == constants.ASSET_CRYPTO]
    forex_pairs_to_fetch = [item['asset_id'] for item in portfolio if item['asset_type'] == constants.ASSET_FOREX]

    # Асинхронный сбор текущих цен
    current_prices = {}
    tasks = {}
    if crypto_ids_to_fetch:
        tasks['crypto'] = data_fetcher.fetch_current_crypto_data(session, crypto_ids_to_fetch)
    if forex_pairs_to_fetch:
        tasks['forex'] = data_fetcher.fetch_current_forex_rates(session, forex_pairs_to_fetch)

    results = await asyncio.gather(*tasks.values(), return_exceptions=True)
    results_map = dict(zip(tasks.keys(), results, strict=False))

    # Обработка результатов
    if 'crypto' in results_map and not isinstance(results_map['crypto'], Exception):
        for asset_id, (price, _, status) in results_map['crypto'].items():
            if status == data_fetcher.STATUS_OK:
                current_prices[asset_id] = price
    if 'forex' in results_map and not isinstance(results_map['forex'], Exception):
        for asset_id, (price, status) in results_map['forex'].items():
            if status == data_fetcher.STATUS_OK:
                current_prices[asset_id] = price

    # Расчет и форматирование
    total_portfolio_value = 0
    total_pl = 0
    portfolio_lines = []

    for item in portfolio:
        asset_id = item['asset_id']
        quantity = item['quantity']
        avg_buy_price = item['avg_buy_price']
        ticker = constants.REVERSE_ASSET_MAP.get(asset_id, asset_id)

        current_price = current_prices.get(asset_id)
        if current_price is None:
            line = get_text(constants.MSG_PORTFOLIO_ITEM_NO_PRICE, lang_code, ticker=ticker, quantity=f"{quantity:,.6f}".rstrip('0').rstrip('.'))
            portfolio_lines.append(line)
            continue

        current_value = quantity * current_price
        total_cost = quantity * avg_buy_price
        pl_value = current_value - total_cost
        pl_percent = (pl_value / total_cost * 100) if total_cost > 0 else 0

        total_portfolio_value += current_value
        total_pl += pl_value

        pl_sign = "+" if pl_value >= 0 else ""

        line = get_text(
            constants.MSG_PORTFOLIO_ITEM, lang_code,
            ticker=html.escape(ticker),
            quantity=f"{quantity:,.6f}".rstrip('0').rstrip('.'),
            current_value=f"{current_value:,.2f}",
            pl_sign=pl_sign,
            pl_value=f"{pl_value:,.2f}",
            pl_percent=f"{pl_percent:,.2f}"
        )
        portfolio_lines.append(line)

    header = get_text(constants.MSG_PORTFOLIO_HEADER, lang_code)
    total_line = get_text(constants.MSG_PORTFOLIO_TOTAL, lang_code,
                          total_value=f"{total_portfolio_value:,.2f}",
                          pl_sign="+" if total_pl >= 0 else "",
                          total_pl=f"{total_pl:,.2f}")

    final_text = f"{header}\n\n" + "\n".join(portfolio_lines) + f"\n\n<b>{total_line}</b>"

    await loading_msg.edit_text(final_text, parse_mode=ParseMode.HTML)


async def p_add_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Добавляет актив в портфель."""
    user_id = update.effective_user.id
    lang_code = await get_user_language(user_id)
    args = context.args

    if len(args) != 3:
        await update.message.reply_text(get_text(constants.MSG_P_ADD_PROMPT, lang_code))
        return

    ticker, quantity_str, price_str = args
    try:
        quantity = float(quantity_str.replace(',', '.'))
        price = float(price_str.replace(',', '.'))
        if quantity <= 0 or price < 0:
            raise ValueError("Quantity must be positive, price must be non-negative.")
    except ValueError:
        await update.message.reply_text(get_text(constants.MSG_P_ADD_FAIL_INVALID_FORMAT, lang_code))
        return

    result_code = user_manager.add_asset_to_portfolio(user_id, ticker, quantity, price)

    if result_code == user_manager.OPERATION_SUCCESS:
        reply_text = get_text(constants.MSG_P_ADD_SUCCESS, lang_code, quantity=quantity, ticker=ticker.upper())
        await update.message.reply_text(reply_text)
        await portfolio_command(update, context) # Показать обновленный портфель
    elif result_code == user_manager.OPERATION_FAILED_INVALID:
        reply_text = get_text(constants.MSG_ADDWATCH_FAIL_INVALID, lang_code, ticker=html.escape(ticker))
        await update.message.reply_text(reply_text)
    else:
        reply_text = get_text(constants.MSG_ERROR_DB, lang_code)
        await update.message.reply_text(reply_text)

async def p_del_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Удаляет актив из портфеля."""
    user_id = update.effective_user.id
    lang_code = await get_user_language(user_id)
    args = context.args

    if len(args) != 1:
        await update.message.reply_text(get_text(constants.MSG_P_DEL_PROMPT, lang_code))
        return

    ticker = args[0]
    result_code = user_manager.remove_asset_from_portfolio(user_id, ticker)

    if result_code == user_manager.OPERATION_SUCCESS:
        reply_text = get_text(constants.MSG_P_DEL_SUCCESS, lang_code, ticker=ticker.upper())
        await update.message.reply_text(reply_text)
        await portfolio_command(update, context) # Показать обновленный портфель
    elif result_code == user_manager.OPERATION_FAILED_NOT_FOUND:
        reply_text = get_text(constants.MSG_P_DEL_FAIL_NOTFOUND, lang_code, ticker=ticker.upper())
        await update.message.reply_text(reply_text)
    else:
        reply_text = get_text(constants.MSG_ERROR_DB, lang_code)
        await update.message.reply_text(reply_text)
