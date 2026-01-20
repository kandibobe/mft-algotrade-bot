# handlers/misc_handler.py
import asyncio
import html
import time
from collections.abc import Callable, Coroutine
from datetime import datetime
from typing import Any

import aiohttp
from telegram import Update
from telegram.constants import ParseMode
from telegram.ext import ContextTypes

from src.telegram_bot import constants
from src.telegram_bot.config_adapter import (
    API_COOLDOWN,
    SHARED_DATA_FETCH_INTERVAL,
)
from src.telegram_bot.jobs import shared_data_cache
from src.telegram_bot.localization.manager import get_text, get_user_language
from src.telegram_bot.services import analysis, data_fetcher
from src.utils.logger import get_logger

logger = get_logger(__name__)

# <<< ИЗМЕНЕНИЕ: Определения типов перенесены НАВЕРХ, до их использования >>>
BaseFetchFunctionType = Callable[
    [aiohttp.ClientSession], Coroutine[Any, Any, tuple[Any | None, str]]
]
ArgFetchFunctionType = Callable[..., Coroutine[Any, Any, tuple[Any | None, str]]]


async def _get_cached_or_fetch(
    cache_key_base: str,
    fetch_function: BaseFetchFunctionType | ArgFetchFunctionType,
    context: ContextTypes.DEFAULT_TYPE,
    update: Update,
    lang_code: str,
    fetch_args: list[Any] | None = None,
    force_fetch: bool = False,
    max_cache_age: int | None = None,
) -> Any | None:
    global shared_data_cache
    cache_key = cache_key_base
    if fetch_args:
        args_str = "_".join(map(str, fetch_args))
        cache_key = f"{cache_key_base}_args_{args_str}"

    shared_data_cache.setdefault(cache_key_base, {"data": None, "last_fetch": 0, "args": None})
    cache_entry = shared_data_cache.setdefault(
        cache_key, {"data": None, "last_fetch": 0, "args": fetch_args if fetch_args else None}
    )

    cached_data = cache_entry.get("data")

    effective_cache_age = (
        max_cache_age if max_cache_age is not None else SHARED_DATA_FETCH_INTERVAL * 1.5
    )
    cache_age = int(time.time()) - cache_entry.get("last_fetch", 0)

    should_fetch_due_to_args_mismatch = fetch_args and (cache_entry.get("args") != fetch_args)

    if (
        not force_fetch
        and not should_fetch_due_to_args_mismatch
        and cached_data
        and cache_age < effective_cache_age
    ):
        logger.debug(
            f"Использование кешированных данных для '{cache_key}' (возраст: {cache_age} сек)."
        )
        return cached_data

    effective_message = update.message or (
        update.callback_query.message if update.callback_query else None
    )
    if not effective_message:
        logger.error(
            f"Не найден effective_message для ответа в _get_cached_or_fetch (cache_key: {cache_key})"
        )
        return cached_data

    loading_text_key = (
        constants.MSG_DATA_STALE_FETCHING
        if cached_data and not force_fetch and not should_fetch_due_to_args_mismatch
        else constants.MSG_LOADING
    )

    loading_msg_id_key = (
        f"loading_msg_id_{effective_message.chat_id}_{effective_message.message_id}"
    )
    loading_msg = None
    if context.chat_data.get(loading_msg_id_key):
        logger.debug(f"Сообщение 'Загрузка...' уже существует для {loading_msg_id_key}")
    else:
        try:
            loading_msg = await effective_message.reply_text(get_text(loading_text_key, lang_code))
            context.chat_data[loading_msg_id_key] = loading_msg.message_id
        except Exception as e:
            logger.warning(f"Не удалось отправить сообщение 'Загрузка...' для {cache_key}: {e}")

    session: aiohttp.ClientSession | None = context.bot_data.get("aiohttp_session")
    if not session or session.closed:
        logger.error(f"{cache_key_base.capitalize()}: Сессия aiohttp не найдена или закрыта!")
        if loading_msg:
            await loading_msg.edit_text(get_text(constants.MSG_ERROR_GENERAL, lang_code))
        elif effective_message:
            await effective_message.reply_text(get_text(constants.MSG_ERROR_GENERAL, lang_code))
        if loading_msg_id_key in context.chat_data:
            del context.chat_data[loading_msg_id_key]
        return cached_data

    fresh_data: Any | None = None
    status: str = data_fetcher.STATUS_UNKNOWN_ERROR
    error_msg_key: str = constants.MSG_ERROR_FETCH

    try:
        if fetch_args:
            fresh_data, status = await fetch_function(session, *fetch_args)
        else:
            fresh_data, status = await fetch_function(session)

        if status == data_fetcher.STATUS_OK:
            if fresh_data is not None:
                cache_entry["data"] = fresh_data
                cache_entry["last_fetch"] = int(time.time())
                cache_entry["args"] = fetch_args
                logger.debug(f"Кеш для '{cache_key}' (args: {fetch_args}) обновлен.")

                if loading_msg:
                    try:
                        await loading_msg.delete()
                    except Exception as del_err:
                        logger.debug(f"Не удалось удалить сообщение загрузки: {del_err}")
                if loading_msg_id_key in context.chat_data:
                    del context.chat_data[loading_msg_id_key]
                return fresh_data
            else:
                logger.warning(f"Статус OK, но данные для {cache_key} отсутствуют.")
                status = data_fetcher.STATUS_NO_DATA
                error_msg_key = constants.MSG_NO_DATA_AVAILABLE

        elif status == data_fetcher.STATUS_CONFIG_ERROR:
            if cache_key_base == "gas":
                error_msg_key = constants.MSG_GAS_API_KEY_MISSING
            elif cache_key_base == "cryptonews":
                error_msg_key = constants.MSG_CRYPTO_NEWS_API_KEY_MISSING
            elif cache_key_base == "news":
                error_msg_key = constants.MSG_NEWS_API_KEY_MISSING
            else:
                error_msg_key = constants.MSG_CONFIG_ERROR
        elif status == data_fetcher.STATUS_INVALID_KEY:
            error_msg_key = constants.MSG_INVALID_API_KEY
        elif status == data_fetcher.STATUS_RATE_LIMIT:
            error_msg_key = constants.MSG_RATE_LIMIT_ERROR
        elif status == data_fetcher.STATUS_TIMEOUT:
            error_msg_key = constants.MSG_TIMEOUT_ERROR
        elif status == data_fetcher.STATUS_NO_DATA:
            error_msg_key = constants.MSG_NO_DATA_AVAILABLE
        elif status == data_fetcher.STATUS_NOT_FOUND:
            if cache_key_base == "cryptonews":
                error_msg_key = constants.MSG_CRYPTO_NEWS_NO_NEWS
            elif cache_key_base == "tvl":
                error_msg_key = constants.MSG_TVL_NOT_FOUND
            else:
                error_msg_key = constants.MSG_ERROR_FETCH

    except Exception as e:
        logger.error(
            f"Непредвиденное исключение при обновлении данных для '{cache_key}': {e}", exc_info=True
        )
        error_msg_key = constants.MSG_ERROR_GENERAL
        status = data_fetcher.STATUS_UNKNOWN_ERROR

    logger.warning(
        f"Не удалось обновить данные для '{cache_key}'. Статус: {status}. Ключ сообщения: {error_msg_key}"
    )

    error_params = {}
    if cache_key_base == "cryptonews" and error_msg_key == constants.MSG_CRYPTO_NEWS_NO_NEWS:
        error_params["filter_name"] = constants.CRYPTO_PANIC_FILTER
    if cache_key_base == "tvl" and error_msg_key == constants.MSG_TVL_NOT_FOUND and fetch_args:
        error_params["protocol_name"] = html.escape(fetch_args[0])

    final_error_text = get_text(error_msg_key, lang_code, **error_params)

    if loading_msg:
        try:
            await loading_msg.edit_text(final_error_text)
        except Exception as edit_err:
            logger.error(
                f"Не удалось отредактировать сообщение об ошибке для {cache_key}: {edit_err}"
            )
    elif effective_message:
        try:
            await effective_message.reply_text(final_error_text)
        except Exception as reply_err:
            logger.error(
                f"Не удалось отправить сообщение об ошибке (reply) для {cache_key}: {reply_err}"
            )

    if loading_msg_id_key in context.chat_data:
        del context.chat_data[loading_msg_id_key]

    is_critical_error = status in [
        data_fetcher.STATUS_CONFIG_ERROR,
        data_fetcher.STATUS_INVALID_KEY,
    ]
    if cached_data and not is_critical_error and not should_fetch_due_to_args_mismatch:
        logger.debug(
            f"Возвращаем старые данные для '{cache_key}' из-за ошибки получения свежих ({status})."
        )
        return cached_data
    else:
        logger.debug(
            f"Возвращаем None для '{cache_key}' (ошибка: {status} или нет старых данных / новые аргументы)."
        )
        return None


async def fear_greed_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    lang_code = await get_user_language(user_id)
    logger.info(f"Запрос /feargreed от user_id: {user_id}")
    fng_data = await _get_cached_or_fetch(
        "fng", data_fetcher.fetch_fear_greed_index, context, update, lang_code
    )
    effective_message = update.message or (
        update.callback_query.message if update.callback_query else None
    )
    if not effective_message:
        return
    if isinstance(fng_data, dict):
        try:
            value = fng_data.get("value", "N/A")
            classification = fng_data.get("classification", "N/A")
            reply_text = (
                get_text(constants.MSG_FNG_HEADER, lang_code)
                + "\n"
                + get_text(
                    constants.MSG_FNG_DATA,
                    lang_code,
                    value=value,
                    classification=html.escape(classification),
                )
            )
            await effective_message.reply_text(reply_text, parse_mode=ParseMode.HTML)
        except Exception as e:
            logger.error(f"Ошибка форматирования/отправки F&G: {e}", exc_info=True)
            if not isinstance(e, (aiohttp.ClientError, asyncio.TimeoutError)):
                try:
                    await effective_message.reply_text(
                        get_text(constants.MSG_ERROR_GENERAL, lang_code)
                    )
                except Exception:
                    pass


async def gas_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    lang_code = await get_user_language(user_id)
    logger.info(f"Запрос /gas от user_id: {user_id}")
    gas_data = await _get_cached_or_fetch(
        "gas", data_fetcher.fetch_eth_gas_price, context, update, lang_code
    )
    effective_message = update.message or (
        update.callback_query.message if update.callback_query else None
    )
    if not effective_message:
        return
    if isinstance(gas_data, dict):
        try:
            safe = gas_data.get("safe", "N/A")
            propose = gas_data.get("propose", "N/A")
            fast = gas_data.get("fast", "N/A")
            base = gas_data.get("base_fee")
            base_fee_str = (
                f"{base:.2f}"
                if isinstance(base, float)
                else str(base)
                if base is not None
                else "N/A"
            )
            reply_text = (
                get_text(constants.MSG_GAS_HEADER, lang_code)
                + "\n"
                + get_text(
                    constants.MSG_GAS_DATA,
                    lang_code,
                    safe=safe,
                    propose=propose,
                    fast=fast,
                    base_fee=base_fee_str,
                )
            )
            await effective_message.reply_text(reply_text, parse_mode=ParseMode.HTML)
        except Exception as e:
            logger.error(f"Ошибка форматирования/отправки Gas: {e}", exc_info=True)
            if not isinstance(e, (aiohttp.ClientError, asyncio.TimeoutError)):
                try:
                    await effective_message.reply_text(
                        get_text(constants.MSG_ERROR_GENERAL, lang_code)
                    )
                except Exception:
                    pass


async def trending_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    lang_code = await get_user_language(user_id)
    logger.info(f"Запрос /trending от user_id: {user_id}")
    trending_data = await _get_cached_or_fetch(
        "trending", data_fetcher.fetch_coingecko_trending, context, update, lang_code
    )
    effective_message = update.message or (
        update.callback_query.message if update.callback_query else None
    )
    if not effective_message:
        return
    if isinstance(trending_data, list):
        try:
            lines = [get_text(constants.MSG_TRENDING_HEADER, lang_code)]
            if not trending_data:
                lines.append(f"<i>({get_text(constants.MSG_NO_DATA_AVAILABLE, lang_code)})</i>")
            else:
                for i, coin in enumerate(trending_data[:7], 1):
                    if not isinstance(coin, dict):
                        continue
                    rank_val = coin.get("market_cap_rank")
                    rank_str = f"(#{rank_val})" if rank_val else ""
                    name = coin.get("name", "?")
                    symbol = coin.get("symbol", "?").upper()
                    lines.append(
                        get_text(
                            constants.MSG_TRENDING_ITEM,
                            lang_code,
                            rank=i,
                            name=html.escape(name),
                            symbol=html.escape(symbol),
                            market_cap_rank=rank_str,
                        )
                    )
            reply_text = "\n".join(lines)
            await effective_message.reply_text(
                reply_text, parse_mode=ParseMode.HTML, disable_web_page_preview=True
            )
        except Exception as e:
            logger.error(f"Ошибка форматирования/отправки Trending: {e}", exc_info=True)
            if not isinstance(e, (aiohttp.ClientError, asyncio.TimeoutError)):
                try:
                    await effective_message.reply_text(
                        get_text(constants.MSG_ERROR_GENERAL, lang_code)
                    )
                except Exception:
                    pass


async def marketcap_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    lang_code = await get_user_language(user_id)
    args = context.args
    default_n = 10
    max_n = 50
    n_to_fetch = default_n
    if args:
        try:
            n_to_fetch = int(args[0])
            if not (1 <= n_to_fetch <= max_n):
                await update.message.reply_text(
                    get_text(
                        constants.MSG_MARKETCAP_INVALID_N,
                        lang_code,
                        default=f"⚠️ Please enter a number between 1 and {max_n}.",
                    )
                )
                return
        except ValueError:
            await update.message.reply_text(
                get_text(
                    constants.MSG_MARKETCAP_INVALID_N,
                    lang_code,
                    default=f"⚠️ Invalid number. Please enter a number between 1 and {max_n}.",
                )
            )
            return
    logger.info(f"Запрос /marketcap (N={n_to_fetch}) от user_id: {user_id}")

    async def _fetch_marketcap_data_adapter(
        session: aiohttp.ClientSession, limit: int
    ) -> tuple[list[dict[str, Any]] | None, str]:
        url = "https://api.coingecko.com/api/v3/coins/markets"
        params = {
            "vs_currency": "usd",
            "order": "market_cap_desc",
            "per_page": limit,
            "page": 1,
            "sparkline": "false",
            "price_change_percentage": "24h",
        }
        source_name = f"CoinGecko-MarketCapTop{limit}"
        data, status = await data_fetcher._fetch_with_retry(
            session, url, params, source_name=source_name, base_wait=API_COOLDOWN * 1.2
        )
        if status != data_fetcher.STATUS_OK:
            return None, status
        if isinstance(data, list):
            formatted_coins = []
            for coin_data in data:
                if isinstance(coin_data, dict):
                    try:
                        formatted_coins.append(
                            {
                                "name": coin_data.get("name"),
                                "symbol": coin_data.get("symbol", "").upper(),
                                "current_price": coin_data.get("current_price"),
                                "price_change_percentage_24h": coin_data.get(
                                    "price_change_percentage_24h"
                                ),
                                "market_cap": coin_data.get("market_cap"),
                                "market_cap_rank": coin_data.get("market_cap_rank"),
                            }
                        )
                    except Exception as e:
                        logger.warning(f"Error formatting marketcap coin data: {e} - {coin_data}")
            return formatted_coins, data_fetcher.STATUS_OK
        else:
            return None, data_fetcher.STATUS_FORMAT_ERROR

    marketcap_data = await _get_cached_or_fetch(
        "marketcap",
        _fetch_marketcap_data_adapter,
        context,
        update,
        lang_code,
        fetch_args=[n_to_fetch],
        force_fetch=False,
    )
    effective_message = update.message
    if not effective_message:
        return
    if isinstance(marketcap_data, list):
        try:
            lines = [get_text(constants.MSG_MARKETCAP_HEADER, lang_code, limit=n_to_fetch)]
            if not marketcap_data:
                lines.append(f"<i>({get_text(constants.MSG_NO_DATA_AVAILABLE, lang_code)})</i>")
            else:
                for i, coin in enumerate(marketcap_data, 1):
                    if not isinstance(coin, dict):
                        continue
                    name = coin.get("name", "N/A")
                    symbol = coin.get("symbol", "N/A")
                    price = coin.get("current_price")
                    change_24h = coin.get("price_change_percentage_24h")
                    market_cap = coin.get("market_cap")
                    rank = coin.get("market_cap_rank", i)
                    price_str = f"${price:,.2f}" if isinstance(price, (int, float)) else "N/A"
                    change_str = (
                        f"{change_24h:+.2f}%" if isinstance(change_24h, (int, float)) else "N/A"
                    )
                    market_cap_str = (
                        f"${market_cap:,.0f}" if isinstance(market_cap, (int, float)) else "N/A"
                    )
                    lines.append(
                        get_text(
                            constants.MSG_MARKETCAP_ITEM,
                            lang_code,
                            rank=rank,
                            name=html.escape(name),
                            symbol=html.escape(symbol),
                            price=price_str,
                            change_24h=change_str,
                            market_cap=market_cap_str,
                        )
                    )
            reply_text = "\n".join(lines)
            await effective_message.reply_text(
                reply_text, parse_mode=ParseMode.HTML, disable_web_page_preview=True
            )
        except Exception as e:
            logger.error(f"Ошибка форматирования/отправки Marketcap: {e}", exc_info=True)
            if not isinstance(e, (aiohttp.ClientError, asyncio.TimeoutError)):
                try:
                    await effective_message.reply_text(
                        get_text(constants.MSG_ERROR_GENERAL, lang_code)
                    )
                except Exception:
                    pass


async def events_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    lang_code = await get_user_language(user_id)
    args = context.args
    period = "week"
    period_key_for_header = constants.MSG_EVENTS_PERIOD_THIS_WEEK
    if args:
        arg_lower = args[0].lower()
        if arg_lower in [
            "today",
            get_text(constants.MSG_TODAY, lang_code, default="today").lower(),
        ]:
            period = "today"
            period_key_for_header = constants.MSG_EVENTS_PERIOD_TODAY
        elif arg_lower in [
            "tomorrow",
            get_text(constants.MSG_TOMORROW, lang_code, default="tomorrow").lower(),
        ]:
            period = "tomorrow"
            period_key_for_header = constants.MSG_EVENTS_PERIOD_TOMORROW
    logger.info(f"Запрос /events от user_id: {user_id}, период: {period}")
    session: aiohttp.ClientSession | None = context.bot_data.get("aiohttp_session")
    if not session or session.closed:
        await update.message.reply_text(get_text(constants.MSG_ERROR_GENERAL, lang_code))
        return
    loading_msg = await update.message.reply_text(get_text(constants.MSG_LOADING, lang_code))
    events, status = await data_fetcher.fetch_economic_events(session, period=period)
    await loading_msg.delete()
    if status == data_fetcher.STATUS_OK and isinstance(events, list):
        period_str = get_text(period_key_for_header, lang_code)
        header = get_text(constants.MSG_EVENTS_HEADER, lang_code, period=period_str)
        lines = [header]
        for event in events:
            try:
                event_time_utc = datetime.fromisoformat(event["time"])
                time_str = event_time_utc.strftime("%H:%M")
                flag = get_text(
                    f"events_country_flag_{event['country'].lower()}",
                    lang_code,
                    default=event["country"],
                )
                impact_icon = get_text(
                    f"events_impact_icon_{event['impact'].lower()}", lang_code, default=""
                )
                event_line = get_text(
                    constants.MSG_EVENTS_ITEM_FORMAT,
                    lang_code,
                    time=time_str,
                    flag=flag,
                    impact=impact_icon,
                    title=html.escape(event["title"]),
                )
                lines.append(event_line)
            except (KeyError, TypeError) as e:
                logger.warning(f"Ошибка форматирования события: {e}. Данные: {event}")
                continue
        reply_text = "\n".join(lines)
        await update.message.reply_text(reply_text, parse_mode=ParseMode.HTML)
    elif status == data_fetcher.STATUS_NO_DATA:
        period_str = get_text(period_key_for_header, lang_code)
        no_events_text = get_text(constants.MSG_EVENTS_NO_EVENTS, lang_code, period=period_str)
        await update.message.reply_text(no_events_text, parse_mode=ParseMode.HTML)
    else:
        error_text = get_text(constants.MSG_ERROR_FETCH, lang_code)
        await update.message.reply_text(error_text, parse_mode=ParseMode.HTML)


async def ta_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handles the /ta command for detailed technical analysis."""
    user_id = update.effective_user.id
    lang_code = await get_user_language(user_id)
    args = context.args

    if not args:
        await update.message.reply_text(get_text(constants.MSG_TA_PROMPT, lang_code))
        return

    ticker = args[0].upper()
    logger.info(f"Запрос /ta для '{ticker}' от user_id: {user_id}")

    asset_info = constants.SUPPORTED_ASSETS.get(ticker)
    if not asset_info or asset_info[0] != constants.ASSET_CRYPTO:
        await update.message.reply_text(
            get_text(constants.MSG_TA_INVALID_ASSET, lang_code, ticker=html.escape(ticker))
        )
        return

    asset_id = asset_info[1]
    session: aiohttp.ClientSession | None = context.bot_data.get("aiohttp_session")
    if not session or session.closed:
        await update.message.reply_text(get_text(constants.MSG_ERROR_GENERAL, lang_code))
        return

    loading_msg = await update.message.reply_text(get_text(constants.MSG_LOADING, lang_code))

    # Запрашиваем достаточно данных для индикаторов (250 дней для надежности)
    hist_data, status = await data_fetcher.fetch_historical_crypto_data(
        session, asset_id, days_to_fetch=250
    )

    if status != data_fetcher.STATUS_OK:
        await loading_msg.edit_text(get_text(constants.MSG_TA_DATA_ERROR, lang_code, ticker=ticker))
        return

    ta_results = analysis.calculate_detailed_technical_indicators(hist_data)

    await loading_msg.delete()

    if not ta_results:
        await update.message.reply_text(
            get_text(constants.MSG_TA_DATA_ERROR, lang_code, ticker=ticker)
        )
        return

    # Формируем красивый ответ
    lines = [get_text(constants.MSG_TA_HEADER, lang_code, ticker=ticker)]

    # RSI
    rsi_val = ta_results["rsi"]
    if rsi_val > 70:
        rsi_desc_key = constants.MSG_TA_RSI_DESC_OVERBOUGHT
    elif rsi_val < 30:
        rsi_desc_key = constants.MSG_TA_RSI_DESC_OVERSOLD
    else:
        rsi_desc_key = constants.MSG_TA_RSI_DESC_NEUTRAL
    rsi_desc = get_text(rsi_desc_key, lang_code)
    lines.append(get_text(constants.MSG_TA_RSI_LINE, lang_code, rsi=rsi_val, description=rsi_desc))

    # MACD
    if ta_results["macd_line"] > ta_results["macd_signal_line"]:
        macd_desc_key = constants.MSG_TA_MACD_DESC_BULLISH
    else:
        macd_desc_key = constants.MSG_TA_MACD_DESC_BEARISH
    macd_desc = get_text(macd_desc_key, lang_code)
    lines.append(get_text(constants.MSG_TA_MACD_LINE, lang_code, description=macd_desc))

    # Bollinger Bands
    price = ta_results["last_price"]
    if price > ta_results["bb_upper"]:
        bb_desc_key = constants.MSG_TA_BBANDS_DESC_ABOVE
    elif price < ta_results["bb_lower"]:
        bb_desc_key = constants.MSG_TA_BBANDS_DESC_BELOW
    else:
        bb_desc_key = constants.MSG_TA_BBANDS_DESC_INSIDE
    bb_desc = get_text(bb_desc_key, lang_code)
    lines.append(get_text(constants.MSG_TA_BBANDS_LINE, lang_code, description=bb_desc))

    reply_text = "\n".join(lines)
    await update.message.reply_text(reply_text, parse_mode=ParseMode.HTML)


async def funding_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handles the /funding command to show top funding rates."""
    user_id = update.effective_user.id
    lang_code = await get_user_language(user_id)
    logger.info(f"Запрос /funding от user_id: {user_id}")

    TRUSTED_EXCHANGES = [
        "Binance",
        "Bybit",
        "OKX",
        "dYdX",
        "Bitget",
        "Kraken",
        "KuCoin",
        "Gate.io",
        "MEXC",
        "BingX",
    ]

    funding_data = await _get_cached_or_fetch(
        "funding_rates", data_fetcher.fetch_funding_rates, context, update, lang_code
    )

    effective_message = update.message or (
        update.callback_query.message if update.callback_query else None
    )
    if not effective_message:
        return

    if isinstance(funding_data, list):
        try:
            filtered_rates = [
                item for item in funding_data if item.get("market") in TRUSTED_EXCHANGES
            ]

            if not filtered_rates:
                logger.warning("После фильтрации по биржам не осталось данных о фандинге.")
                await effective_message.reply_text(
                    get_text(constants.MSG_NO_DATA_AVAILABLE, lang_code)
                )
                return

            sorted_rates = sorted(filtered_rates, key=lambda x: x["rate"], reverse=True)

            top_10_high = sorted_rates[:10]
            top_10_low = sorted_rates[-10:]
            top_10_low.reverse()

            lines = [get_text(constants.MSG_FUNDING_HEADER, lang_code)]

            lines.append(get_text(constants.MSG_FUNDING_HIGH_RATES_HEADER, lang_code))
            for i, item in enumerate(top_10_high, 1):
                lines.append(
                    get_text(
                        constants.MSG_FUNDING_ITEM,
                        lang_code,
                        rank=i,
                        symbol=item["symbol"],
                        market=item["market"],
                        rate=item["rate"],
                    )
                )

            lines.append(get_text(constants.MSG_FUNDING_LOW_RATES_HEADER, lang_code))
            for i, item in enumerate(top_10_low, 1):
                lines.append(
                    get_text(
                        constants.MSG_FUNDING_ITEM,
                        lang_code,
                        rank=i,
                        symbol=item["symbol"],
                        market=item["market"],
                        rate=item["rate"],
                    )
                )

            reply_text = "\n".join(lines)
            await effective_message.reply_text(
                reply_text, parse_mode=ParseMode.HTML, disable_web_page_preview=True
            )

        except Exception as e:
            logger.error(f"Ошибка форматирования/отправки Funding Rates: {e}", exc_info=True)
            await effective_message.reply_text(get_text(constants.MSG_ERROR_GENERAL, lang_code))


async def tvl_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handles the /tvl command to show TVL for a DeFi protocol."""
    user_id = update.effective_user.id
    lang_code = await get_user_language(user_id)
    args = context.args

    if not args:
        await update.message.reply_text(get_text(constants.MSG_TVL_PROMPT, lang_code))
        return

    protocol_name = " ".join(args)
    logger.info(f"Запрос /tvl для '{protocol_name}' от user_id: {user_id}")

    tvl_value = await _get_cached_or_fetch(
        "tvl", data_fetcher.fetch_tvl, context, update, lang_code, fetch_args=[protocol_name]
    )

    effective_message = update.message or (
        update.callback_query.message if update.callback_query else None
    )
    if not effective_message:
        return

    if tvl_value is not None:
        if tvl_value > 1_000_000_000:
            tvl_str = f"{tvl_value / 1_000_000_000:.2f}B"
        elif tvl_value > 1_000_000:
            tvl_str = f"{tvl_value / 1_000_000:.2f}M"
        elif tvl_value > 1_000:
            tvl_str = f"{tvl_value / 1_000:.2f}K"
        else:
            tvl_str = f"{tvl_value:,.2f}"

        reply_text = get_text(
            constants.MSG_TVL_HEADER,
            lang_code,
            protocol_name=html.escape(protocol_name.capitalize()),
            tvl_value=tvl_str,
        )
        await effective_message.reply_text(reply_text, parse_mode=ParseMode.HTML)


async def ta_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    lang_code = await get_user_language(user_id)
    args = context.args

    if not args:
        await update.message.reply_text(get_text(constants.MSG_TA_PROMPT, lang_code))
        return

    ticker = args[0].upper()
    logger.info(f"Запрос /ta для '{ticker}' от user_id: {user_id}")

    asset_info = constants.SUPPORTED_ASSETS.get(ticker)
    if not asset_info or asset_info[0] != constants.ASSET_CRYPTO:
        await update.message.reply_text(
            get_text(constants.MSG_TA_INVALID_ASSET, lang_code, ticker=html.escape(ticker))
        )
        return

    asset_id = asset_info[1]

    # --- НАЧАЛО ИСПРАВЛЕНИЯ: Добавляем кеширование ---
    # Используем _get_cached_or_fetch для исторических данных с долгим временем жизни кеша (1 час)
    hist_data = await _get_cached_or_fetch(
        cache_key_base=f"hist_data_{asset_id}",
        fetch_function=data_fetcher.fetch_historical_crypto_data,
        context=context,
        update=update,
        lang_code=lang_code,
        fetch_args=[asset_id, 250],  # Запрашиваем 250 дней
        max_cache_age=3600,  # Кешируем на 1 час
    )
    # --- КОНЕЦ ИСПРАВЛЕНИЯ ---

    if not hist_data:
        # _get_cached_or_fetch уже отправит сообщение об ошибке, если данные не получены
        return

    ta_results = analysis.calculate_detailed_technical_indicators(hist_data)

    if not ta_results:
        await update.message.reply_text(
            get_text(constants.MSG_TA_DATA_ERROR, lang_code, ticker=ticker)
        )
        return

    # Формируем красивый ответ
    lines = [get_text(constants.MSG_TA_HEADER, lang_code, ticker=ticker)]

    # RSI
    rsi_val = ta_results["rsi"]
    if rsi_val > 70:
        rsi_desc_key = constants.MSG_TA_RSI_DESC_OVERBOUGHT
    elif rsi_val < 30:
        rsi_desc_key = constants.MSG_TA_RSI_DESC_OVERSOLD
    else:
        rsi_desc_key = constants.MSG_TA_RSI_DESC_NEUTRAL
    rsi_desc = get_text(rsi_desc_key, lang_code)
    lines.append(
        get_text(constants.MSG_TA_RSI_LINE, lang_code, rsi=f"{rsi_val:.2f}", description=rsi_desc)
    )

    # MACD
    if ta_results["macd_line"] > ta_results["macd_signal_line"]:
        macd_desc_key = constants.MSG_TA_MACD_DESC_BULLISH
    else:
        macd_desc_key = constants.MSG_TA_MACD_DESC_BEARISH
    macd_desc = get_text(macd_desc_key, lang_code)
    lines.append(get_text(constants.MSG_TA_MACD_LINE, lang_code, description=macd_desc))

    # Bollinger Bands
    price = ta_results["last_price"]
    if price > ta_results["bb_upper"]:
        bb_desc_key = constants.MSG_TA_BBANDS_DESC_ABOVE
    elif price < ta_results["bb_lower"]:
        bb_desc_key = constants.MSG_TA_BBANDS_DESC_BELOW
    else:
        bb_desc_key = constants.MSG_TA_BBANDS_DESC_INSIDE
    bb_desc = get_text(bb_desc_key, lang_code)
    lines.append(get_text(constants.MSG_TA_BBANDS_LINE, lang_code, description=bb_desc))

    reply_text = "\n".join(lines)
    await update.message.reply_text(reply_text, parse_mode=ParseMode.HTML)
