# jobs.py
import asyncio
import html
import time
from datetime import datetime, timezone
from typing import Any

from telegram.constants import ParseMode
from telegram.error import Forbidden
from telegram.ext import Application, ContextTypes

from src.telegram_bot import constants
from src.telegram_bot.config_adapter import (
    ALERT_COOLDOWN_SECONDS,
    API_COOLDOWN,
    CRYPTO_PANIC_FILTER,
    NEWS_API_PAGE_SIZE,
)
from src.telegram_bot.localization.manager import get_text, get_user_language
from src.telegram_bot.services import data_fetcher, user_manager
from src.utils.logger import get_logger

logger = get_logger(__name__)

# --- Кэш для данных, используемых несколькими пользователями ---
shared_data_cache: dict[str, dict[str, Any]] = {
    "fng": {"data": None, "last_fetch": 0},
    "gas": {"data": None, "last_fetch": 0},
    "trending": {"data": None, "last_fetch": 0},
    "indices": {"data": {}, "last_fetch": 0},
    "crypto_news": {"data": None, "last_fetch": 0},
    "volatility": {"data": None, "last_fetch": 0},
    "btc_dominance": {"data": None, "last_fetch": 0},
    "marketcap": {"data": None, "last_fetch": 0, "args": None},
    "onchain_netflow": {"data": None, "last_fetch": 0},
}

async def fetch_shared_data_job(context: ContextTypes.DEFAULT_TYPE):
    """Периодически обновляет общие данные: F&G, Gas, Тренды, Новости, Доминация."""
    global shared_data_cache
    application: Application = context.application
    session = application.bot_data.get('aiohttp_session')
    if not session or session.closed:
        logger.error("Нет активной сессии aiohttp для задачи fetch_shared_data_job.")
        return

    now_ts = int(time.time())
    logger.info("Запуск задачи обновления общих данных (F&G, Gas, Trending, News, Global Market)...")

    # F&G
    try:
        fng_data, fng_status = await data_fetcher.fetch_fear_greed_index(session)
        if fng_status == data_fetcher.STATUS_OK and isinstance(fng_data, dict):
            shared_data_cache["fng"]["data"] = fng_data
            shared_data_cache["fng"]["last_fetch"] = now_ts
            logger.info(f"Кеш F&G Index обновлен: {fng_data.get('value', 'N/A')}")
        else:
            logger.warning(f"Не удалось обновить F&G Index (статус: {fng_status}). Кеш не изменен.")
    except Exception as e:
        logger.error(f"Исключение при фоновом запросе F&G: {e}", exc_info=True)
    await asyncio.sleep(API_COOLDOWN)

    # Gas
    try:
        gas_data, gas_status = await data_fetcher.fetch_eth_gas_price(session)
        if gas_status == data_fetcher.STATUS_OK and isinstance(gas_data, dict):
            shared_data_cache["gas"]["data"] = gas_data
            shared_data_cache["gas"]["last_fetch"] = now_ts
            logger.info("Кеш ETH Gas обновлен.")
        elif gas_status == data_fetcher.STATUS_CONFIG_ERROR and shared_data_cache["gas"]["last_fetch"] == 0:
            logger.warning("Ключ Etherscan API не установлен, кеш Gas не будет обновляться.")
            shared_data_cache["gas"]["data"] = None
        else:
            logger.warning(f"Не удалось обновить ETH Gas (статус: {gas_status}). Кеш не изменен.")
    except Exception as e:
        logger.error(f"Исключение при фоновом запросе Gas: {e}", exc_info=True)
    await asyncio.sleep(API_COOLDOWN)

    # Trending
    try:
        trending_data, trending_status = await data_fetcher.fetch_coingecko_trending(session)
        if trending_status == data_fetcher.STATUS_OK and isinstance(trending_data, list):
            shared_data_cache["trending"]["data"] = trending_data
            shared_data_cache["trending"]["last_fetch"] = now_ts
            logger.info(f"Кеш Trending Coins обновлен ({len(trending_data)} монет).")
        else:
            logger.warning(f"Не удалось обновить Trending Coins (статус: {trending_status}). Кеш не изменен.")
    except Exception as e:
        logger.error(f"Исключение при фоновом запросе Trending: {e}", exc_info=True)
    await asyncio.sleep(API_COOLDOWN)

    # Crypto News
    try:
        news_data, news_status = await data_fetcher.fetch_crypto_news(session, news_filter=CRYPTO_PANIC_FILTER, limit=NEWS_API_PAGE_SIZE)
        if news_status == data_fetcher.STATUS_OK and isinstance(news_data, list):
            shared_data_cache["crypto_news"]["data"] = news_data
            shared_data_cache["crypto_news"]["last_fetch"] = now_ts
            logger.info(f"Кеш Crypto News обновлен ({len(news_data)} новостей).")
        elif news_status == data_fetcher.STATUS_CONFIG_ERROR and shared_data_cache["crypto_news"]["last_fetch"] == 0:
             logger.warning("Ключ CryptoPanic API не установлен, кеш крипто-новостей не будет обновляться.")
             shared_data_cache["crypto_news"]["data"] = None
        else:
            logger.warning(f"Не удалось обновить Crypto News (статус: {news_status}). Кеш не изменен.")
    except Exception as e:
        logger.error(f"Исключение при фоновом запросе Crypto News: {e}", exc_info=True)
    await asyncio.sleep(API_COOLDOWN)

    # Global Market Data (BTC Dominance)
    try:
        global_market_data, global_market_status = await data_fetcher.fetch_current_global_market_data(session)
        if global_market_status == data_fetcher.STATUS_OK and isinstance(global_market_data, dict):
            shared_data_cache["btc_dominance"]["data"] = global_market_data
            shared_data_cache["btc_dominance"]["last_fetch"] = now_ts
            logger.info(f"Кеш Global Market Data (BTC.D) обновлен: {global_market_data.get('btc_dominance', 'N/A')}%")
        else:
            logger.warning(f"Не удалось обновить Global Market Data (статус: {global_market_status}). Кеш не изменен.")
    except Exception as e:
        logger.error(f"Исключение при фоновом запросе Global Market Data: {e}", exc_info=True)

    logger.info("Задача обновления общих данных завершена.")


async def fetch_index_data_job(context: ContextTypes.DEFAULT_TYPE):
    """Периодически обновляет цены на рыночные индексы."""
    global shared_data_cache
    application: Application = context.application
    session = application.bot_data.get('aiohttp_session')
    if not session or session.closed:
        logger.error("Нет активной сессии aiohttp для задачи fetch_index_data_job.")
        return

    symbols_to_fetch = list(constants.MARKET_INDEX_SYMBOLS.keys())
    if not symbols_to_fetch: return

    logger.info(f"Запуск задачи обновления кеша индексов ({', '.join(symbols_to_fetch)})...")
    index_data_results = await data_fetcher.fetch_index_prices(session, symbols_to_fetch)

    current_time = int(time.time())
    updated_count = 0
    indices_cache_data = shared_data_cache.setdefault("indices", {}).setdefault("data", {})

    for symbol, (price, status) in index_data_results.items():
        if status == data_fetcher.STATUS_OK and price is not None:
            indices_cache_data[symbol] = {"price": price, "status": status, "updated_at": current_time}
            updated_count += 1
        else:
            indices_cache_data[symbol] = {"price": None, "status": status, "updated_at": current_time}
            logger.warning(f"Не удалось обновить цену для индекса {symbol} (статус: {status}).")

    if updated_count > 0:
        shared_data_cache["indices"]["last_fetch"] = current_time
    logger.info(f"Кеш индексов обновлен ({updated_count}/{len(symbols_to_fetch)} успешно).")


async def fetch_onchain_data_job(context: ContextTypes.DEFAULT_TYPE):
    """Периодически запрашивает и кэширует ключевые on-chain метрики."""
    global shared_data_cache
    application: Application = context.application
    session = application.bot_data.get('aiohttp_session')
    if not session or session.closed:
        logger.error("Нет активной сессии aiohttp для задачи fetch_onchain_data_job.")
        return

    logger.info("Запуск задачи обновления On-Chain данных...")

    try:
        netflow_data, status = await data_fetcher.fetch_glassnode_metric(
            session, constants.GLASSNODE_BTC_NET_TRANSFER_EXCHANGES, 'BTC', '24h'
        )
        if status == data_fetcher.STATUS_OK and isinstance(netflow_data, list) and netflow_data:
            shared_data_cache["onchain_netflow"]["data"] = netflow_data
            shared_data_cache["onchain_netflow"]["last_fetch"] = int(time.time())
            logger.info(f"Кеш On-Chain (Netflow) обновлен. Последняя точка: {netflow_data[-1]}")
        elif status == data_fetcher.STATUS_CONFIG_ERROR and shared_data_cache["onchain_netflow"]["last_fetch"] == 0:
             logger.warning("Ключ Glassnode API не установлен, кеш on-chain не будет обновляться.")
             shared_data_cache["onchain_netflow"]["data"] = None
        else:
            logger.warning(f"Не удалось обновить On-Chain Netflow (статус: {status}). Кеш не изменен.")
    except Exception as e:
        logger.error(f"Исключение при фоновом запросе On-Chain Netflow: {e}", exc_info=True)

    logger.info("Задача обновления On-Chain данных завершена.")


async def check_price_alerts(context: ContextTypes.DEFAULT_TYPE):
    """Проверяет все активные алерты пользователей и отправляет уведомления."""
    application: Application = context.application
    session = application.bot_data.get('aiohttp_session')
    if not session or session.closed:
        logger.error("Нет активной сессии aiohttp для задачи check_price_alerts.")
        return

    logger.info("Запуск задачи проверки алертов...")
    all_alerts = user_manager.get_all_price_alerts()
    if not all_alerts:
        logger.debug("Активных алертов нет.")
        return

    assets_to_check: dict[str, set] = {constants.ASSET_CRYPTO: set(), constants.ASSET_FOREX: set()}
    for alert in all_alerts:
        if alert.get('asset_type') in assets_to_check and alert.get('asset_id'):
             assets_to_check[alert['asset_type']].add(alert['asset_id'])

    current_prices = {}
    # Запрос цен для всех необходимых активов
    try:
        if assets_to_check[constants.ASSET_CRYPTO]:
            crypto_results = await data_fetcher.fetch_current_crypto_data(session, list(assets_to_check[constants.ASSET_CRYPTO]))
            for cid, (price, _, status) in crypto_results.items():
                if status == data_fetcher.STATUS_OK: current_prices[(constants.ASSET_CRYPTO, cid)] = price
        if assets_to_check[constants.ASSET_FOREX]:
            forex_results = await data_fetcher.fetch_current_forex_rates(session, list(assets_to_check[constants.ASSET_FOREX]))
            for pair, (rate, status) in forex_results.items():
                if status == data_fetcher.STATUS_OK: current_prices[(constants.ASSET_FOREX, pair)] = rate
    except Exception as e:
         logger.error(f"Критическая ошибка при запросе цен для алертов: {e}", exc_info=True)
         return

    current_ts = int(time.time())
    user_settings_cache = {uid: user_manager.get_settings(uid) for uid in {a['user_id'] for a in all_alerts}}

    for alert in all_alerts:
        user_id = alert['user_id']
        asset_key = (alert['asset_type'], alert['asset_id'])
        current_price = current_prices.get(asset_key)
        if current_price is None: continue

        if not user_settings_cache.get(user_id, {}).get('notifications_enabled', True):
            continue

        alert_id, condition, target_value = alert['id'], alert['condition'], alert['target_value']
        last_triggered = alert.get('last_triggered_at', 0)

        triggered = False
        try:
            if condition == '>' and current_price > target_value: triggered = True
            elif condition == '<' and current_price < target_value: triggered = True
        except TypeError:
            continue

        if triggered and (current_ts - last_triggered > ALERT_COOLDOWN_SECONDS):
            ticker = constants.REVERSE_ASSET_MAP.get(alert['asset_id'], alert['asset_id'])
            lang_code = await get_user_language(user_id)
            price_format = "{:.5f}" if alert['asset_type'] == constants.ASSET_FOREX else "{:,.4f}"

            message_text = get_text(
                constants.MSG_PRICE_ALERT_TRIGGERED, lang_code,
                asset_id=ticker, condition=html.escape(condition),
                target_value=price_format.format(target_value),
                current_value=price_format.format(current_price)
            )
            try:
                await application.bot.send_message(user_id, message_text, parse_mode=ParseMode.HTML)
                user_manager.update_alert_trigger(alert_id, current_ts)
                logger.info(f"Алерт ID {alert_id} отправлен user {user_id}.")
                await asyncio.sleep(0.1)
            except Forbidden:
                logger.warning(f"Не удалось отправить алерт user {user_id} (ID {alert_id}): Бот заблокирован.")
            except Exception as e:
                logger.error(f"Ошибка отправки алерта user {user_id}, alert_id {alert_id}: {e}", exc_info=True)

    logger.info("Проверка алертов завершена.")


async def send_daily_digest(context: ContextTypes.DEFAULT_TYPE):
    """Формирует и отправляет утренний дайджест всем подписчикам."""
    application: Application = context.application
    session = application.bot_data.get('aiohttp_session')
    if not session or session.closed:
        logger.error("Нет активной сессии aiohttp для задачи send_daily_digest.")
        return

    logger.info("Запуск задачи формирования утреннего дайджеста...")
    subscribed_users = user_manager.get_subscribed_user_ids()
    if not subscribed_users:
        logger.info("Нет подписчиков для дайджеста.")
        return

    # Сбор данных для дайджеста
    fng_data = shared_data_cache["fng"].get("data")
    top_crypto_news_list = shared_data_cache["crypto_news"].get("data")
    volatility_data = shared_data_cache["volatility"].get("data")
    btc_dominance_info = shared_data_cache.get("btc_dominance", {}).get("data")

    # Сбор данных по всем watchlist'ам пользователей
    all_watchlists = {uid: user_manager.get_user_watchlist(uid) for uid in subscribed_users}
    unique_crypto_ids = {item['asset_id'] for wl in all_watchlists.values() for item in wl if item['asset_type'] == constants.ASSET_CRYPTO}

    watchlist_prices = {}
    if unique_crypto_ids:
        crypto_results = await data_fetcher.fetch_current_crypto_data(session, list(unique_crypto_ids), include_change=True)
        watchlist_prices.update({(constants.ASSET_CRYPTO, cid): data for cid, data in crypto_results.items()})

    logger.info(f"Формирование и отправка дайджеста {len(subscribed_users)} пользователям...")
    for user_id in subscribed_users:
        try:
            lang_code = await get_user_language(user_id)
            digest_lines = [get_text("TITLE_DIGEST", lang_code, date=datetime.now(timezone.utc).strftime('%Y-%m-%d'))]

            if top_crypto_news_list:
                news_line = get_text("MSG_CRYPTO_NEWS_ITEM", lang_code, url=top_crypto_news_list[0].get('url','#'), title=html.escape(top_crypto_news_list[0].get('title', '')), domain=html.escape(top_crypto_news_list[0].get('domain','')))
                digest_lines.append(get_text("HEADER_DIGEST_NEWS", lang_code, news_line=news_line))

            if volatility_data:
                vol_parts = []
                for g in volatility_data.get("gainers", []): vol_parts.append(f"  📈 {g['symbol']}: <b>{g['change']:.2f}%</b>")
                for l in volatility_data.get("losers", []): vol_parts.append(f"  📉 {l['symbol']}: <b>{l['change']:.2f}%</b>")
                if vol_parts: digest_lines.append(get_text("HEADER_DIGEST_VOLATILITY", lang_code, volatility="\n".join(vol_parts)))

            if fng_data: digest_lines.append(f'\n{get_text("MSG_DIGEST_FNG_LINE", lang_code, value=fng_data["value"], classification=html.escape(fng_data["classification"]))}')
            if btc_dominance_info: digest_lines.append(get_text("MSG_DIGEST_DOMINANCE_LINE", lang_code, value=btc_dominance_info.get('btc_dominance', 0)))

            digest_lines.append(get_text("HEADER_DIGEST_WATCHLIST", lang_code))
            user_watchlist = all_watchlists.get(user_id, [])
            if user_watchlist:
                for item in user_watchlist:
                    price_data = watchlist_prices.get((item['asset_type'], item['asset_id']))
                    ticker = constants.REVERSE_ASSET_MAP.get(item['asset_id'], item['asset_id'])
                    if price_data and price_data[2] == data_fetcher.STATUS_OK:
                        price, change = price_data[0], price_data[1]
                        change_str = f" ({change:+.2f}%)" if change is not None else ""
                        digest_lines.append(get_text("MSG_DIGEST_WATCHLIST_ITEM", lang_code, symbol=ticker, price=f"${price:,.2f}", change=change_str))
            else:
                digest_lines.append(get_text("MSG_DIGEST_NO_WATCHLIST", lang_code))

            digest_lines.append("\n" + get_text("MSG_DIGEST_FOOTER", lang_code))
            await application.bot.send_message(user_id, "\n".join(digest_lines), parse_mode=ParseMode.HTML)
            await asyncio.sleep(0.1)
        except Exception as e:
            logger.error(f"Ошибка отправки дайджеста user {user_id}: {e}")

    logger.info("Отправка утреннего дайджеста завершена.")


async def fetch_volatility_job(context: ContextTypes.DEFAULT_TYPE):
    """Периодически обновляет данные по волатильности."""
    global shared_data_cache
    application: Application = context.application
    session = application.bot_data.get('aiohttp_session')
    if not session or session.closed:
        logger.error("Нет активной сессии aiohttp для задачи fetch_volatility_job.")
        return

    logger.info("Запуск задачи обновления кеша волатильности...")
    data, status = await data_fetcher.fetch_top_crypto_volatility(session, limit=3)

    if status == data_fetcher.STATUS_OK and isinstance(data, dict):
        shared_data_cache["volatility"]["data"] = data
        shared_data_cache["volatility"]["last_fetch"] = int(time.time())
        logger.info("Кеш волатильности обновлен.")
    else:
        logger.error(f"Не удалось обновить кеш волатильности (статус: {status}). Кеш не изменен.")
