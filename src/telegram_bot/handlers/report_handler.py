# handlers/report_handler.py
import asyncio
import html
from datetime import datetime, timedelta, timezone
from typing import Any

import aiohttp
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.constants import ParseMode
from telegram.error import BadRequest
from telegram.ext import ContextTypes

from src.telegram_bot import constants
from src.telegram_bot.config_adapter import DEFAULT_ANALYSIS_PERIOD
from src.telegram_bot.localization.manager import get_text, get_user_language
from src.telegram_bot.services import analysis, data_fetcher, graph_generator, user_manager
from src.utils.logger import get_logger

logger = get_logger(__name__)

# --- Вспомогательные функции форматирования ---

def _format_status_for_user(status: str, lang_code: str) -> str:
    if status == data_fetcher.STATUS_OK:
        return ""
    status_key_part = status.replace('✅', '').replace('ℹ️', '').replace('❌', '').strip().lower().replace(' ', '_')
    status_localization_key = f"api_status_{status_key_part}"
    localized_status = get_text(status_localization_key, lang_code, default=None)
    return f"({localized_status})" if localized_status else f"({status.replace('✅', '').replace('ℹ️', '').replace('❌', '').strip()})"

def _format_macro_value(key: str, value: float | None) -> str:
    if value is None: return "N/A"
    try:
        if key == "M2": return f"${value / 1000:.2f} трлн"
        elif key in ["CPI", "DXY"]: return f"{value:.2f}"
        elif key in ["FFR", "UNRATE"]: return f"{value:.2f}%"
        else: return f"{value:.2f}"
    except (TypeError, ValueError):
        logger.warning(f"Не удалось отформатировать макрозначение: key={key}, value={value}")
        return str(value)

def _get_macro_explanation(key: str, lang_code: str) -> str:
    explain_collection_key = f"macro_explain_{key.lower()}"
    explanation = get_text(explain_collection_key, lang_code, default="")
    return f"<i>({explanation})</i>" if explanation else ""

def _format_asset_line(ticker: str, price_data: tuple[float | None, float | None, str], lang_code: str, is_watchlist: bool = False, asset_type_override: str | None = None, ta_results: dict[str, Any] | None = None) -> str:
    price, _, status = price_data
    asset_type = asset_type_override or (constants.SUPPORTED_ASSETS.get(ticker, (None,))[0])

    price_str = get_text(constants.MSG_ERROR_FETCH_SHORT, lang_code)
    status_str = _format_status_for_user(status, lang_code)

    if status == data_fetcher.STATUS_OK and price is not None:
        price_format = "{:.5f}" if asset_type == constants.ASSET_FOREX else "${:,.2f}"
        try:
            price_str = price_format.format(price)
        except (ValueError, TypeError):
            price_str = get_text(constants.MSG_ERROR_DATA_FORMAT_DETAIL, lang_code, source=ticker)
            status_str = _format_status_for_user(data_fetcher.STATUS_FORMAT_ERROR, lang_code)
    elif status == data_fetcher.STATUS_NO_DATA:
        price_str = get_text(constants.MSG_NO_DATA_SHORT, lang_code)

    prefix = "⭐️" if is_watchlist else "•"
    base_line = f"{prefix} {ticker}: <code>{html.escape(price_str)}</code> {html.escape(status_str)}".strip()

    if asset_type == constants.ASSET_CRYPTO and ta_results:
        ta_lines = [get_text(constants.MSG_REPORT_TA_HEADER, lang_code)]
        if 'rsi' in ta_results:
            ta_lines.append(get_text(constants.MSG_REPORT_TA_RSI, lang_code, rsi_value=ta_results['rsi']))
        if ta_results.get('position_key'):
            position_text = get_text(ta_results['position_key'], lang_code, **ta_results.get('position_args', {}))
            ta_lines.append(get_text(constants.MSG_REPORT_TA_SMA_POSITION, lang_code, position_text=position_text))
        base_line += "\n" + "\n".join(ta_lines)

    return base_line

# --- Функции для формирования разделов отчета ---

def _get_main_dashboard_keyboard(lang_code: str) -> InlineKeyboardMarkup:
    keyboard = [
        [
            InlineKeyboardButton(get_text(constants.MSG_REPORT_BTN_CRYPTO, lang_code), callback_data=constants.CB_REPORT_NAV_CRYPTO),
            InlineKeyboardButton(get_text(constants.MSG_REPORT_BTN_MACRO, lang_code), callback_data=constants.CB_REPORT_NAV_MACRO)
        ],
        [
            InlineKeyboardButton(get_text(constants.MSG_REPORT_BTN_ONCHAIN, lang_code), callback_data=constants.CB_REPORT_NAV_ONCHAIN),
            InlineKeyboardButton(get_text(constants.MSG_REPORT_BTN_GRAPHS, lang_code), callback_data=constants.CB_REPORT_NAV_GRAPHS)
        ]
    ]
    return InlineKeyboardMarkup(keyboard)

def _get_back_to_dashboard_keyboard(lang_code: str) -> InlineKeyboardMarkup:
    keyboard = [[InlineKeyboardButton(f"⬅️ {get_text(constants.MSG_REPORT_BTN_BACK, lang_code)}", callback_data=constants.CB_REPORT_NAV_MAIN)]]
    return InlineKeyboardMarkup(keyboard)

async def _display_main_dashboard(update: Update, context: ContextTypes.DEFAULT_TYPE, message_to_edit=None):
    """Отображает главный дашборд отчета."""
    user_id = update.effective_user.id
    lang_code = await get_user_language(user_id)
    report_data = context.user_data.get('report_data', {})

    period_days = report_data.get('period_days', DEFAULT_ANALYSIS_PERIOD)
    update_time_str = report_data.get('update_time', datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M"))

    fng_data = report_data.get('fng_data', {})
    fng_line = ""
    if fng_data and fng_data.get('status') == data_fetcher.STATUS_OK:
        fng_line = get_text(constants.MSG_SIGNAL_FNG_INFO, lang_code, value=fng_data['value'], classification=html.escape(fng_data['classification']))

    header = get_text(constants.MSG_REPORT_DASHBOARD_HEADER, lang_code, period=period_days, update_time=update_time_str)
    text = f"{header}\n\n{fng_line}"

    reply_markup = _get_main_dashboard_keyboard(lang_code)

    if message_to_edit:
        await message_to_edit.edit_text(text, reply_markup=reply_markup, parse_mode=ParseMode.HTML)
    elif update.callback_query:
        await update.callback_query.edit_message_text(text, reply_markup=reply_markup, parse_mode=ParseMode.HTML)
    elif update.message:
        await update.message.reply_text(text, reply_markup=reply_markup, parse_mode=ParseMode.HTML)

# --- Основные обработчики ---

async def report_command_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Начинает создание отчета, собирает все данные и показывает главный дашборд."""
    user_id = update.effective_user.id
    lang_code = await get_user_language(user_id)
    logger.info(f"Запрос 'Мой отчет' от user_id: {user_id}")

    effective_message = update.message or (update.callback_query.message if update.callback_query else None)
    if not effective_message: return

    loading_msg = await effective_message.reply_text(get_text(constants.MSG_LOADING, lang_code))

    session: aiohttp.ClientSession | None = context.bot_data.get('aiohttp_session')
    if not session or session.closed:
        await loading_msg.edit_text(get_text(constants.MSG_ERROR_GENERAL, lang_code))
        return

    # --- Сбор данных ---
    user_settings = user_manager.get_settings(user_id)
    period_days = user_settings.get('analysis_period', DEFAULT_ANALYSIS_PERIOD)

    watchlist = user_manager.get_user_watchlist(user_id)
    crypto_in_watchlist_ids = {item['asset_id'] for item in watchlist if item['asset_type'] == constants.ASSET_CRYPTO}
    forex_in_watchlist_pairs = {item['asset_id'] for item in watchlist if item['asset_type'] == constants.ASSET_FOREX}

    crypto_to_fetch_ids = crypto_in_watchlist_ids.union({constants.CG_BTC, constants.CG_ETH})
    forex_to_fetch_pairs = forex_in_watchlist_pairs.union(constants.FOREX_PAIRS[:2])

    tasks = {}
    # Цены
    if crypto_to_fetch_ids: tasks["crypto_prices"] = data_fetcher.fetch_current_crypto_data(session, list(crypto_to_fetch_ids))
    if forex_to_fetch_pairs: tasks["forex_rates"] = data_fetcher.fetch_current_forex_rates(session, list(forex_to_fetch_pairs))
    tasks["index_prices"] = data_fetcher.fetch_index_prices(session, list(constants.MARKET_INDEX_SYMBOLS.keys()))
    # История для ТА
    for asset_id in crypto_to_fetch_ids: tasks[f"hist_{asset_id}"] = data_fetcher.fetch_historical_crypto_data(session, asset_id, 205)
    # Макро
    fred_start_dt = (datetime.now() - timedelta(days=max(period_days * 2, 100))).strftime('%Y-%m-%d')
    for key, series_id in constants.SUPPORTED_MACRO_FOR_REPORT.items(): tasks[f"macro_{key}"] = data_fetcher.fetch_fred_series(session, series_id, fred_start_dt)
    # On-Chain
    tasks["onchain_netflow"] = data_fetcher.fetch_glassnode_metric(session, constants.GLASSNODE_BTC_NET_TRANSFER_EXCHANGES, 'BTC', '24h')
    # F&G
    tasks["fng_data"] = data_fetcher.fetch_fear_greed_index(session)

    logger.debug(f"Запуск {len(tasks)} задач для сбора данных отчета...")
    all_task_results = await asyncio.gather(*tasks.values(), return_exceptions=True)
    data_map = dict(zip(tasks.keys(), all_task_results, strict=False))

    # Сохраняем все собранные данные в user_data для последующего использования
    context.user_data['report_data'] = {
        'data_map': data_map,
        'period_days': period_days,
        'update_time': datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M"),
        'watchlist_crypto_ids': crypto_in_watchlist_ids,
        'watchlist_forex_pairs': forex_in_watchlist_pairs,
        'all_crypto_ids': crypto_to_fetch_ids
    }

    await _display_main_dashboard(update, context, message_to_edit=loading_msg)

async def report_navigation_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обрабатывает навигацию по разделам отчета."""
    query = update.callback_query
    await query.answer()

    user_id = update.effective_user.id
    lang_code = await get_user_language(user_id)

    report_data = context.user_data.get('report_data')
    if not report_data:
        await query.edit_message_text(get_text(constants.MSG_REPORT_DATA_EXPIRED, lang_code), reply_markup=None)
        return

    action = query.data
    data_map = report_data['data_map']
    period_days = report_data['period_days']
    update_time = report_data['update_time']

    header = get_text(constants.MSG_REPORT_DASHBOARD_HEADER, lang_code, period=period_days, update_time=update_time)
    lines = [header]
    reply_markup = _get_back_to_dashboard_keyboard(lang_code)

    if action == constants.CB_REPORT_NAV_MAIN:
        await _display_main_dashboard(update, context, message_to_edit=query.message)
        return

    elif action == constants.CB_REPORT_NAV_CRYPTO:
        lines.append(f"\n<b>{get_text(constants.MSG_CRYPTO_HEADER, lang_code)}</b>")
        crypto_prices = data_map.get("crypto_prices", {})
        if isinstance(crypto_prices, Exception):
            lines.append(f"  <i>{get_text(constants.MSG_ERROR_FETCH, lang_code)} (Crypto)</i>")
        else:
            displayed_tickers = set()
            # Сначала watchlist
            for asset_id in report_data['watchlist_crypto_ids']:
                ticker = constants.REVERSE_ASSET_MAP.get(asset_id, asset_id)
                hist_data, _ = data_map.get(f"hist_{asset_id}", (None, None))
                ta_results = analysis.calculate_technical_indicators(hist_data) if hist_data else None
                lines.append(_format_asset_line(ticker, crypto_prices.get(asset_id, (None, None, "N/A")), lang_code, is_watchlist=True, ta_results=ta_results))
                displayed_tickers.add(ticker)
            # Потом базовые, если их нет в watchlist
            for ticker, asset_id in [("BTC", constants.CG_BTC), ("ETH", constants.CG_ETH)]:
                if ticker not in displayed_tickers:
                    hist_data, _ = data_map.get(f"hist_{asset_id}", (None, None))
                    ta_results = analysis.calculate_technical_indicators(hist_data) if hist_data else None
                    lines.append(_format_asset_line(ticker, crypto_prices.get(asset_id, (None, None, "N/A")), lang_code, ta_results=ta_results))

    elif action == constants.CB_REPORT_NAV_MACRO:
        lines.append(f"\n<b>{get_text(constants.MSG_MACRO_HEADER, lang_code)}</b>")
        for key, _ in constants.SUPPORTED_MACRO_FOR_REPORT.items():
            macro_data, status = data_map.get(f"macro_{key}", (None, data_fetcher.STATUS_UNKNOWN_ERROR))
            value_str, date_str = get_text(constants.MSG_ERROR_FETCH_SHORT, lang_code), ""
            if status == data_fetcher.STATUS_OK and macro_data:
                last_date, last_value = macro_data[-1]
                value_str = _format_macro_value(key, last_value)
                date_str = f" ({last_date.strftime('%Y-%m-%d')})"
            elif status == data_fetcher.STATUS_NO_DATA:
                value_str = get_text(constants.MSG_NO_DATA_SHORT, lang_code)
            lines.append(f" • {key}{date_str}: <code>{html.escape(value_str)}</code> {_format_status_for_user(status, lang_code)} {_get_macro_explanation(key, lang_code)}".strip())

        lines.append(f"\n<b>{get_text(constants.MSG_INDEX_HEADER, lang_code)}</b>")
        index_prices = data_map.get("index_prices", {})
        for ticker, name in constants.MARKET_INDEX_SYMBOLS.items():
            price, status = index_prices.get(ticker, (None, "N/A"))
            price_str = f"{price:,.2f}" if price is not None else get_text(constants.MSG_ERROR_FETCH_SHORT, lang_code)
            lines.append(f" • {ticker} ({name}): <code>{html.escape(price_str)}</code> {_format_status_for_user(status, lang_code)}".strip())

    elif action == constants.CB_REPORT_NAV_ONCHAIN:
        lines.append(f"\n<b>{get_text(constants.MSG_REPORT_ONCHAIN_HEADER, lang_code)}</b>")
        netflow_data, status = data_map.get("onchain_netflow", (None, data_fetcher.STATUS_UNKNOWN_ERROR))
        if status == data_fetcher.STATUS_OK and netflow_data:
            last_point = netflow_data[-1]
            netflow_value = last_point.get('v', 0)
            netflow_btc_str = f"{netflow_value:,.2f} BTC"
            desc_key = constants.MSG_REPORT_ONCHAIN_OUTFLOW if netflow_value < 0 else constants.MSG_REPORT_ONCHAIN_INFLOW
            lines.append(get_text(constants.MSG_REPORT_ONCHAIN_NETFLOW, lang_code, value=netflow_btc_str, description=get_text(desc_key, lang_code)))
        else:
            lines.append(f"  <i>{get_text(constants.MSG_ERROR_FETCH, lang_code)} (Netflow)</i>")

    elif action == constants.CB_REPORT_NAV_GRAPHS:
        lines.append(f"\n<b>{get_text(constants.MSG_REPORT_BTN_GRAPHS, lang_code)}</b>")
        lines.append(get_text(constants.MSG_REPORT_GRAPH_PROMPT, lang_code))

        graph_buttons = []
        for asset_id in report_data['all_crypto_ids']:
            ticker = constants.REVERSE_ASSET_MAP.get(asset_id, asset_id)
            graph_buttons.append(InlineKeyboardButton(f"📊 {ticker}", callback_data=f"{constants.CB_ACTION_REPORT_GRAPH}{ticker}"))

        keyboard = [graph_buttons[i:i + 2] for i in range(0, len(graph_buttons), 2)]
        keyboard.append([InlineKeyboardButton(f"⬅️ {get_text(constants.MSG_REPORT_BTN_BACK, lang_code)}", callback_data=constants.CB_REPORT_NAV_MAIN)])
        reply_markup = InlineKeyboardMarkup(keyboard)

    final_text = "\n".join(lines)
    try:
        await query.edit_message_text(final_text, reply_markup=reply_markup, parse_mode=ParseMode.HTML)
    except BadRequest as e:
        if "message is not modified" not in str(e).lower():
            logger.error(f"Ошибка редактирования сообщения отчета: {e}")

async def report_graph_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Отправляет график для выбранного актива из отчета."""
    query = update.callback_query
    await query.answer()

    user_id = update.effective_user.id
    lang_code = await get_user_language(user_id)

    report_data = context.user_data.get('report_data')
    if not report_data:
        await context.bot.send_message(user_id, get_text(constants.MSG_REPORT_DATA_EXPIRED, lang_code))
        return

    try:
        ticker = query.data[len(constants.CB_ACTION_REPORT_GRAPH):]
        asset_id = constants.SUPPORTED_ASSETS[ticker][1]
        period_days = report_data['period_days']

        loading_msg = await context.bot.send_message(user_id, get_text(constants.MSG_GRAPH_LOADING, lang_code, symbol=ticker))

        hist_data, status = report_data['data_map'].get(f"hist_{asset_id}", (None, data_fetcher.STATUS_UNKNOWN_ERROR))

        if status == data_fetcher.STATUS_OK and hist_data:
            hist_for_graph = hist_data[-(period_days + 5):] # Берем с запасом
            graph_bytes = graph_generator.generate_candlestick_graph(hist_for_graph, ticker, period_days)
            if graph_bytes:
                await context.bot.send_photo(user_id, photo=graph_bytes)
            else:
                await context.bot.send_message(user_id, get_text(constants.MSG_GRAPH_ERROR_GENERAL, lang_code, symbol=ticker))
        else:
            await context.bot.send_message(user_id, get_text(constants.MSG_TA_DATA_ERROR, lang_code, ticker=ticker))

        await loading_msg.delete()

    except (KeyError, IndexError) as e:
        logger.error(f"Ошибка в report_graph_callback: не найдены данные для {query.data}. {e}")
        await context.bot.send_message(user_id, get_text(constants.MSG_ERROR_GENERAL, lang_code))
    except Exception as e:
        logger.error(f"Непредвиденная ошибка в report_graph_callback: {e}", exc_info=True)
