# services/data_fetcher.py
import asyncio
from datetime import date, datetime, timedelta, timezone
from typing import Any, Union

import aiohttp
from newsapi import NewsApiClient

from src.telegram_bot import constants
from src.telegram_bot.config_adapter import (
    ALPHA_VANTAGE_API_KEY,
    API_COOLDOWN,
    CRYPTO_PANIC_API_KEY,
    DEFAULT_REQUEST_TIMEOUT,
    ETHERSCAN_API_KEY,
    FRED_API_KEY,
    GLASSNODE_API_KEY,
    NEWS_API_ORG_KEY,
    PROXY_URL,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)
DateType = Union[datetime, date]

STATUS_OK = "✅ OK"
STATUS_NO_DATA = "ℹ️ No Data"
STATUS_API_ERROR = "❌ API Error"
STATUS_TIMEOUT = "❌ Timeout"
STATUS_NETWORK_ERROR = "❌ Network Error"
STATUS_RATE_LIMIT = "❌ Rate Limit"
STATUS_NOT_FOUND = "❌ Not Found"
STATUS_FORBIDDEN = "❌ Forbidden"
STATUS_INVALID_KEY = "❌ Invalid Key"
STATUS_BAD_REQUEST = "❌ Bad Request"
STATUS_CONFIG_ERROR = "❌ Config Error"
STATUS_FORMAT_ERROR = "❌ Format Error"
STATUS_UNKNOWN_ERROR = "❌ Unknown Error"

newsapi_client: NewsApiClient | None = None
if NEWS_API_ORG_KEY:
    try:
        newsapi_client = NewsApiClient(api_key=NEWS_API_ORG_KEY)
        logger.info("NewsApiClient (NewsAPI.org) инициализирован.")
    except Exception as e:
        logger.error(f"Ошибка инициализации NewsApiClient: {e}")

async def _fetch_with_retry(
    session: aiohttp.ClientSession,
    url: str,
    params: dict | None = None,
    headers: dict | None = None,
    retries: int = 2,
    base_wait: float = API_COOLDOWN,
    source_name: str = "API"
) -> tuple[Any, str]:
    last_exception: Exception | None = None
    final_status: str = STATUS_UNKNOWN_ERROR

    safe_params = params.copy() if params else {}
    for key_to_mask in ["api_key", "apikey", "auth_token", "api-key"]:
        if key_to_mask in safe_params:
            safe_params[key_to_mask] = "***MASKED***"
        if headers and key_to_mask in headers:
            pass

    for attempt in range(retries + 1):
        current_base_wait = base_wait
        if "alphavantage" in source_name.lower():
            current_base_wait = max(base_wait, 12.5)

        wait_time = current_base_wait * (2 ** attempt)

        try:
            logger.debug(f"[{source_name}] Попытка {attempt+1}/{retries+1}: GET {url} (params: {safe_params})")
            async with session.get(url, params=params, headers=headers, timeout=DEFAULT_REQUEST_TIMEOUT, proxy=PROXY_URL) as response:
                status_code = response.status
                log_msg = f"[{source_name}] Статус {status_code} для {url} (попытка {attempt+1})"
                logger.debug(log_msg)
                error_body_text_preview = ""

                if status_code == 200:
                    content_type = response.headers.get('Content-Type', '')
                    if 'application/json' in content_type:
                        try:
                            json_data = await response.json()
                            if "alphavantage" in source_name.lower() and isinstance(json_data, dict):
                                if "Error Message" in json_data:
                                    logger.warning(f"[{source_name}] Alpha Vantage вернул ошибку в JSON (200 OK): {json_data['Error Message']}")
                                    final_status = STATUS_API_ERROR
                                    return None, final_status
                                if "Note" in json_data and "API call frequency" in json_data["Note"]:
                                    logger.warning(f"[{source_name}] Alpha Vantage вернул Rate Limit Note в JSON (200 OK): {json_data['Note']}")
                                    final_status = STATUS_RATE_LIMIT
                                    return None, final_status

                            logger.debug(f"[{source_name}] Успешно получен JSON с {url}")
                            return json_data, STATUS_OK
                        except aiohttp.ContentTypeError as json_err:
                            logger.error(f"[{source_name}] Ошибка декодирования JSON (ContentTypeError) с {url}. Content-Type: {content_type}. Ошибка: {json_err}")
                            final_status = STATUS_FORMAT_ERROR
                            return None, final_status
                        except Exception as e:
                             logger.error(f"[{source_name}] Неожиданная ошибка при чтении JSON с {url}: {e}", exc_info=True)
                             final_status = STATUS_FORMAT_ERROR
                             return None, final_status
                    else:
                        logger.warning(f"[{source_name}] Получен не-JSON ответ (200 OK) с {url}. Content-Type: {content_type}.")
                        final_status = STATUS_FORMAT_ERROR
                        try: text_response = await response.text(); logger.debug(f"[{source_name}] Текст ответа: {text_response[:200]}")
                        except Exception: pass
                        return None, final_status

                try: error_body_text_preview = (await response.text())[:250]
                except Exception: error_body_text_preview = "Не удалось прочитать тело ответа."

                if status_code == 429:
                    final_status = STATUS_RATE_LIMIT
                    retry_after_header = response.headers.get("Retry-After")
                    if retry_after_header:
                        try: wait_time = max(wait_time, float(retry_after_header))
                        except ValueError: logger.warning(f"[{source_name}] Не удалось распарсить Retry-After: {retry_after_header}")
                    wait_time = min(wait_time, 60.0)
                    logger.warning(f"[{source_name}] API Rate Limit (429) для {url}. Попытка {attempt+1}. Ожидание {wait_time:.1f} сек... Ответ: {error_body_text_preview}")
                    last_exception = aiohttp.ClientResponseError(response.request_info, response.history, status=429, message=f"Rate limit exceeded. Body: {error_body_text_preview}")

                elif status_code == 404:
                    final_status = STATUS_NOT_FOUND
                    logger.warning(f"[{source_name}] Ресурс не найден (404) по адресу {url}. Ответ: {error_body_text_preview}")
                    last_exception = aiohttp.ClientResponseError(response.request_info, response.history, status=404, message=f"Not Found. Body: {error_body_text_preview}")
                    return None, final_status

                elif status_code in [401, 403]:
                    final_status = STATUS_FORBIDDEN
                    if "invalid api key" in error_body_text_preview.lower() or \
                       "access restricted" in error_body_text_preview.lower() or \
                       "authentication failed" in error_body_text_preview.lower() or \
                       "invalid token" in error_body_text_preview.lower() or \
                       "api key" in error_body_text_preview.lower():
                         final_status = STATUS_INVALID_KEY
                         logger.error(f"[{source_name}] Ошибка доступа ({status_code}): Неверный или ограниченный API ключ для {url}. Ответ: {error_body_text_preview}")
                    else:
                         logger.error(f"[{source_name}] Ошибка доступа ({status_code}) для {url}. Ответ: {error_body_text_preview}")
                    last_exception = aiohttp.ClientResponseError(response.request_info, response.history, status=status_code, message=f"Forbidden/Unauthorized. Body: {error_body_text_preview}")
                    return None, final_status

                elif status_code == 400:
                    final_status = STATUS_BAD_REQUEST
                    logger.error(f"[{source_name}] Неверный запрос (400) к {url}. Параметры: {safe_params}. Ответ API: {error_body_text_preview}")
                    last_exception = aiohttp.ClientResponseError(response.request_info, response.history, status=400, message=f"Bad Request. Body: {error_body_text_preview}")
                    return None, final_status

                elif status_code >= 500:
                    final_status = STATUS_API_ERROR
                    logger.warning(f"[{source_name}] Ошибка сервера ({status_code}) при запросе к {url}. Попытка {attempt+1}. Ответ: {error_body_text_preview}")
                    last_exception = aiohttp.ClientResponseError(response.request_info, response.history, status=status_code, message=f"Server Error. Body: {error_body_text_preview}")

                else:
                     final_status = STATUS_UNKNOWN_ERROR
                     logger.warning(f"[{source_name}] Неожиданный HTTP статус {status_code} от {url}. Попытка {attempt+1}. Ответ: {error_body_text_preview}")
                     last_exception = aiohttp.ClientResponseError(response.request_info, response.history, status=status_code, message=f"Unexpected HTTP status. Body: {error_body_text_preview}")
                     return None, final_status

                if attempt < retries:
                    logger.info(f"[{source_name}] Ожидание {wait_time:.1f} сек перед ретраем...")
                    await asyncio.sleep(wait_time)
                    continue
                else:
                    logger.error(f"[{source_name}] Закончились ретраи ({retries}) для {url} после ошибки {status_code}.")
                    return None, final_status

        except (aiohttp.ClientConnectionError, aiohttp.ServerDisconnectedError, aiohttp.ClientPayloadError) as net_err:
            logger.warning(f"[{source_name}] Ошибка сети/соединения на попытке {attempt+1} для {url}: {net_err.__class__.__name__}")
            final_status = STATUS_NETWORK_ERROR
            last_exception = net_err
            if attempt == retries: return None, final_status
            await asyncio.sleep(wait_time)

        except asyncio.TimeoutError as timeout_err:
            logger.warning(f"[{source_name}] Таймаут на попытке {attempt+1} для {url}")
            final_status = STATUS_TIMEOUT
            last_exception = timeout_err
            if attempt == retries: return None, final_status
            await asyncio.sleep(wait_time)

        except Exception as e:
            logger.error(f"[{source_name}] Непредвиденное исключение при запросе к {url} на попытке {attempt+1}: {e.__class__.__name__} - {e}", exc_info=True)
            final_status = STATUS_UNKNOWN_ERROR
            last_exception = e
            return None, final_status

    logger.error(f"[{source_name}] Запрос для {url} завершился без явного успеха или ошибки после всех ретраев. Последний статус: {final_status}. Последнее исключение: {last_exception}")
    return None, final_status


async def fetch_fred_series(
    session: aiohttp.ClientSession,
    series_id: str,
    start_date: str | None = None,
    end_date: str | None = None
) -> tuple[list[tuple[date, float]] | None, str]:
    if not FRED_API_KEY:
        logger.warning(f"[FRED] Запрос ({series_id}) не выполнен: ключ FRED_API_KEY отсутствует.")
        return None, STATUS_CONFIG_ERROR

    base_url = "https://api.stlouisfed.org/fred/series/observations"
    params = {
        "series_id": series_id,
        "api_key": FRED_API_KEY,
        "file_type": "json",
        "sort_order": "asc",
        "limit": 100000
    }
    if start_date: params["observation_start"] = start_date
    if end_date: params["observation_end"] = end_date
    logger.debug(f"[FRED] Запрос: {series_id} с {start_date if start_date else 'начала'}")

    data, status = await _fetch_with_retry(session, base_url, params, source_name=f"FRED-{series_id}")

    if status != STATUS_OK:
        return None, status

    try:
        if not isinstance(data, dict): raise TypeError("Ответ FRED не является словарем")
        observations = data.get("observations")
        if not isinstance(observations, list): raise TypeError("Поле 'observations' в ответе FRED не является списком")

        result: list[tuple[date, float]] = []
        for item in observations:
            value_str = item.get('value')
            date_str = item.get('date')
            if value_str != '.' and date_str:
                try:
                    date_obj = datetime.strptime(date_str, '%Y-%m-%d').date()
                    value_fl = float(value_str)
                    result.append((date_obj, value_fl))
                except (ValueError, TypeError, KeyError) as parse_err:
                    logger.warning(f"[FRED] Ошибка парсинга записи ({series_id}): {item}. Ошибка: {parse_err}")
                    continue

        if not result:
            logger.warning(f"[FRED] ({series_id}): Получен пустой список наблюдений после фильтрации.")
            return None, STATUS_NO_DATA

        logger.debug(f"[FRED] ({series_id}): успешно получено {len(result)} точек.")
        return result, STATUS_OK
    except Exception as e:
        logger.error(f"[FRED] Критическая ошибка обработки данных ({series_id}): {e}", exc_info=True)
        return None, STATUS_FORMAT_ERROR


async def fetch_current_crypto_data(
    session: aiohttp.ClientSession,
    crypto_ids: list[str],
    include_change: bool = False
) -> dict[str, tuple[float | None, float | None, str]]:
    results: dict[str, tuple[float | None, float | None, str]] = dict.fromkeys(crypto_ids, (None, None, STATUS_UNKNOWN_ERROR))
    if not crypto_ids: return results

    ids_param = ",".join(crypto_ids)
    vs_currency = "usd"
    endpoint_name = "coins/markets" if include_change else "simple/price"
    url_base = "https://api.coingecko.com/api/v3/"
    url = f"{url_base}{endpoint_name}"
    source_name = f"CoinGecko-{endpoint_name.split('/')[0]}"

    if include_change:
        params = {
            "vs_currency": vs_currency, "ids": ids_param, "order": "market_cap_desc",
            "per_page": min(len(crypto_ids) + 5, 250), "page": 1, "sparkline": "false",
            "price_change_percentage": "24h"
        }
        data, status = await _fetch_with_retry(session, url, params, source_name=source_name)

        if status != STATUS_OK:
            return dict.fromkeys(crypto_ids, (None, None, status))

        if isinstance(data, list):
            data_dict = {item['id']: item for item in data if isinstance(item, dict) and 'id' in item}
            for crypto_id in crypto_ids:
                coin_data = data_dict.get(crypto_id)
                if coin_data:
                    try:
                        price = float(coin_data['current_price'])
                        change_raw = coin_data.get('price_change_percentage_24h')
                        change = float(change_raw) if change_raw is not None else None
                        results[crypto_id] = (price, change, STATUS_OK)
                    except (ValueError, TypeError, KeyError) as parse_err:
                        logger.warning(f"[{source_name}] Ошибка парсинга данных для {crypto_id}: {parse_err}. Данные: {coin_data}")
                        results[crypto_id] = (None, None, STATUS_FORMAT_ERROR)
                else:
                    logger.warning(f"[{source_name}] Данные для {crypto_id} не найдены в ответе.")
                    results[crypto_id] = (None, None, STATUS_NOT_FOUND)
        else:
            logger.error(f"[{source_name}] Неожиданный формат ответа: {type(data)}. Ожидался list.")
            return dict.fromkeys(crypto_ids, (None, None, STATUS_FORMAT_ERROR))

    else:
        params = {"ids": ids_param, "vs_currencies": vs_currency}
        data, status = await _fetch_with_retry(session, url, params, source_name=source_name)

        if status != STATUS_OK:
            return dict.fromkeys(crypto_ids, (None, None, status))

        if isinstance(data, dict):
            for crypto_id in crypto_ids:
                if crypto_id in data and isinstance(data[crypto_id], dict) and vs_currency in data[crypto_id]:
                    try:
                        price = float(data[crypto_id][vs_currency])
                        results[crypto_id] = (price, None, STATUS_OK)
                    except (ValueError, TypeError) as parse_err:
                         logger.warning(f"[{source_name}] Ошибка парсинга цены для {crypto_id}: {parse_err}. Данные: {data[crypto_id]}")
                         results[crypto_id] = (None, None, STATUS_FORMAT_ERROR)
                else:
                    logger.warning(f"[{source_name}] Цена для {crypto_id} не найдена в ответе.")
                    results[crypto_id] = (None, None, STATUS_NOT_FOUND)
        else:
             logger.error(f"[{source_name}] Неожиданный формат ответа: {type(data)}. Ожидался dict.")
             return dict.fromkeys(crypto_ids, (None, None, STATUS_FORMAT_ERROR))
    return results


async def fetch_historical_crypto_data(
    session: aiohttp.ClientSession,
    crypto_id: str,
    days_to_fetch: int,
    vs_currency: str = "usd",
    data_type: str = "prices"
) -> tuple[list[tuple[date, float]] | None, str]:
    api_request_days = days_to_fetch + 5
    url = f"https://api.coingecko.com/api/v3/coins/{crypto_id}/market_chart"
    params = {"vs_currency": vs_currency, "days": str(api_request_days), "interval": "daily"}
    source_name = f"CoinGecko-Hist-{crypto_id}-{data_type}"
    logger.debug(f"[{source_name}] Запрос: {api_request_days} дней.")

    data, status = await _fetch_with_retry(session, url, params, source_name=source_name)

    if status != STATUS_OK:
        return None, status

    try:
        if not isinstance(data, dict): raise TypeError("Ответ не является словарем")

        raw_data_list = data.get(data_type)
        if not isinstance(raw_data_list, list): raise TypeError(f"Поле '{data_type}' не является списком")
        if not raw_data_list: raise ValueError(f"Список '{data_type}' пуст")

        daily_values: dict[date, float] = {}
        for item in raw_data_list:
             if not isinstance(item, list) or len(item) != 2:
                 logger.warning(f"[{source_name}] Некорректный формат элемента в '{data_type}': {item}")
                 continue
             try:
                 timestamp_ms, value_val = item
                 dt_object_utc = datetime.fromtimestamp(timestamp_ms / 1000, tz=timezone.utc)
                 day_date = dt_object_utc.date()
                 daily_values[day_date] = float(value_val)
             except (ValueError, TypeError, IndexError, OverflowError) as proc_err:
                 logger.warning(f"[{source_name}] Ошибка обработки точки данных: {item}, {proc_err}")
                 continue

        if not daily_values:
             logger.warning(f"[{source_name}] Не осталось валидных данных после обработки.")
             return None, STATUS_NO_DATA

        historical_data = sorted(daily_values.items())
        today_utc = datetime.now(timezone.utc).date()

        if historical_data and historical_data[-1][0] >= today_utc:
            logger.debug(f"[{source_name}] Удаление потенциально неполных данных за сегодня ({historical_data[-1][0]}).")
            historical_data.pop()

        if len(historical_data) > days_to_fetch:
            historical_data = historical_data[-days_to_fetch:]

        if not historical_data:
            logger.warning(f"[{source_name}] Не осталось данных за {days_to_fetch} дн. после финальной обрезки.")
            return None, STATUS_NO_DATA

        logger.debug(f"[{source_name}] Обработано {len(historical_data)} точек за период <= {days_to_fetch} дн.")
        return historical_data, STATUS_OK

    except Exception as e:
        logger.error(f"[{source_name}] Критическая ошибка обработки исторических данных ({data_type}): {e}", exc_info=True)
        return None, STATUS_FORMAT_ERROR


async def fetch_top_crypto_volatility(
    session: aiohttp.ClientSession,
    limit: int = 5,
    vs_currency: str = "usd"
) -> tuple[dict[str, list[dict[str, Any]]] | None, str]:
    url = "https://api.coingecko.com/api/v3/coins/markets"
    params = {
        "vs_currency": vs_currency, "order": "market_cap_desc", "per_page": 100,
        "page": 1, "sparkline": "false", "price_change_percentage": "24h"
    }
    source_name = "CoinGecko-Volatility"
    logger.debug(f"[{source_name}] Запрос для волатильности...")

    markets_data, status = await _fetch_with_retry(session, url, params, source_name=source_name)

    if status != STATUS_OK:
         return None, status

    try:
        if not isinstance(markets_data, list):
            logger.error(f"[{source_name}] Ответ не список: {type(markets_data)}")
            return None, STATUS_FORMAT_ERROR

        valid_coins = [
            c for c in markets_data
            if isinstance(c, dict) and isinstance(c.get('price_change_percentage_24h'), (int, float))
               and c.get('symbol') and c.get('current_price') is not None
        ]

        if not valid_coins:
            logger.warning(f"[{source_name}] Не найдено монет с данными об изменении цены.")
            return None, STATUS_NO_DATA

        gainers = sorted(valid_coins, key=lambda x: x['price_change_percentage_24h'], reverse=True)
        losers = sorted(valid_coins, key=lambda x: x['price_change_percentage_24h'])

        def format_coin_data(coin_dict: dict) -> dict[str, Any] | None:
            try:
                return {
                    "symbol": str(coin_dict["symbol"]).upper(),
                    "change": float(coin_dict["price_change_percentage_24h"]),
                    "price": float(coin_dict["current_price"])
                }
            except (ValueError, TypeError, KeyError) as format_err:
                 logger.warning(f"[{source_name}] Ошибка форматирования данных монеты: {coin_dict}. Ошибка: {format_err}")
                 return None

        top_results = {
            "gainers": [formatted for c in gainers[:limit] if (formatted := format_coin_data(c)) is not None],
            "losers": [formatted for c in losers[:limit] if (formatted := format_coin_data(c)) is not None]
        }

        logger.debug(f"[{source_name}] Данные волатильности получены: {len(top_results['gainers'])} G, {len(top_results['losers'])} L")
        return top_results, STATUS_OK

    except Exception as e:
        logger.error(f"[{source_name}] Критическая ошибка при обработке волатильности: {e}", exc_info=True)
        return None, STATUS_FORMAT_ERROR

async def fetch_current_forex_rates(
    session: aiohttp.ClientSession,
    pairs: list[str] = constants.FOREX_PAIRS
) -> dict[str, tuple[float | None, str]]:
    results: dict[str, tuple[float | None, str]] = dict.fromkeys(pairs, (None, STATUS_UNKNOWN_ERROR))
    if not pairs: return results
    successful_pairs = set()
    source_name_av = "AlphaVantage-Forex"
    source_name_er = "ExchangeRate-Forex"

    if ALPHA_VANTAGE_API_KEY:
        logger.debug(f"[{source_name_av}] Попытка получить курсы для: {', '.join(pairs)}")
        av_tasks = {}
        for i, pair in enumerate(pairs):
            if len(pair) == 6:
                from_currency, to_currency = pair[:3].upper(), pair[3:].upper()
                url = "https://www.alphavantage.co/query"
                params = {"function": "CURRENCY_EXCHANGE_RATE", "from_currency": from_currency, "to_currency": to_currency, "apikey": ALPHA_VANTAGE_API_KEY}
                if i > 0: await asyncio.sleep(1)
                av_tasks[pair] = _fetch_with_retry(session, url, params, retries=1, source_name=f"{source_name_av}-{pair}")
            else:
                logger.warning(f"[{source_name_av}] Некорректный формат пары: {pair}")
                results[pair] = (None, STATUS_BAD_REQUEST)

        if av_tasks:
            av_results_list = await asyncio.gather(*av_tasks.values(), return_exceptions=True)
            av_data_map = dict(zip(av_tasks.keys(), av_results_list, strict=False))
            for pair, result_or_exc in av_data_map.items():
                if isinstance(result_or_exc, Exception):
                    logger.error(f"[{source_name_av}] Исключение при запросе {pair}: {result_or_exc}")
                    results[pair] = (None, STATUS_UNKNOWN_ERROR)
                    continue

                data, status = result_or_exc
                if status == STATUS_OK and isinstance(data, dict) and "Realtime Currency Exchange Rate" in data:
                    try:
                        rate_data = data["Realtime Currency Exchange Rate"]
                        rate_str = rate_data.get("5. Exchange Rate")
                        if rate_str is None: raise KeyError("Отсутствует '5. Exchange Rate'")
                        results[pair] = (round(float(rate_str), 5), STATUS_OK)
                        successful_pairs.add(pair)
                    except (ValueError, TypeError, KeyError) as e:
                        results[pair] = (None, STATUS_FORMAT_ERROR)
                        logger.warning(f"[{source_name_av}] Ошибка парсинга {pair}: {e}. Ответ: {data}")
                elif status == STATUS_RATE_LIMIT:
                    logger.warning(f"[{source_name_av}] Достигнут Rate Limit для {pair}.")
                    results[pair] = (None, status)
                else:
                    results[pair] = (None, status)

            if len(successful_pairs) == len(pairs):
                return results
            logger.warning(f"[{source_name_av}] Не все курсы получены ({len(successful_pairs)}/{len(pairs)}). Переход к ExchangeRate-API для оставшихся.")

    pairs_for_fallback = [p for p in pairs if p not in successful_pairs]
    if not pairs_for_fallback:
        return results

    logger.debug(f"[{source_name_er}] Попытка получить курсы для: {', '.join(pairs_for_fallback)} через ExchangeRate-API")
    url_er_public = "https://api.exchangerate-api.com/v4/latest/USD"

    data_er, status_er = await _fetch_with_retry(session, url_er_public, source_name=source_name_er)

    if status_er != STATUS_OK:
        for pair in pairs_for_fallback:
            if results[pair][1] != STATUS_UNKNOWN_ERROR and results[pair][1] != STATUS_RATE_LIMIT :
                 pass
            else:
                 results[pair] = (None, status_er)
        return results

    if isinstance(data_er, dict) and isinstance(data_er.get("rates"), dict):
        rates_er = data_er["rates"]
        base_currency_er = data_er.get("base", "USD")
        for pair in pairs_for_fallback:
            try:
                if len(pair) == 6:
                    from_curr, to_curr = pair[:3].upper(), pair[3:].upper()

                    if from_curr == base_currency_er:
                        if to_curr in rates_er:
                            results[pair] = (round(float(rates_er[to_curr]), 5), STATUS_OK)
                        else: results[pair] = (None, STATUS_NOT_FOUND)
                    elif to_curr == base_currency_er:
                        if from_curr in rates_er and float(rates_er[from_curr]) != 0:
                            results[pair] = (round(1 / float(rates_er[from_curr]), 5), STATUS_OK)
                        elif from_curr in rates_er :
                            results[pair] = (None, STATUS_FORMAT_ERROR)
                        else: results[pair] = (None, STATUS_NOT_FOUND)
                    else:
                        rate_from_base = rates_er.get(from_curr)
                        rate_to_base = rates_er.get(to_curr)
                        if rate_from_base is not None and rate_to_base is not None and float(rate_from_base) != 0:
                            results[pair] = (round(float(rate_to_base) / float(rate_from_base), 5), STATUS_OK)
                        elif rate_from_base is not None and float(rate_from_base) == 0:
                            results[pair] = (None, STATUS_FORMAT_ERROR)
                        else: results[pair] = (None, STATUS_NOT_FOUND)
                else:
                    results[pair] = (None, STATUS_BAD_REQUEST)
            except (ValueError, TypeError, ZeroDivisionError) as e:
                results[pair] = (None, STATUS_FORMAT_ERROR)
                logger.warning(f"[{source_name_er}] Ошибка расчета {pair}: {e}")
    else:
        for pair in pairs_for_fallback:
            if results[pair][1] != STATUS_UNKNOWN_ERROR and results[pair][1] != STATUS_RATE_LIMIT:
                pass
            else:
                results[pair] = (None, STATUS_FORMAT_ERROR)
    return results

async def fetch_fear_greed_index(session: aiohttp.ClientSession) -> tuple[dict[str, Any] | None, str]:
    url = "https://api.alternative.me/fng/?limit=1&format=json"
    source_name = "AlternativeMe-FNG"
    data, status = await _fetch_with_retry(session, url, source_name=source_name)
    if status != STATUS_OK: return None, status
    try:
        if not isinstance(data, dict) or data.get('name') != "Fear and Greed Index": raise ValueError("Неверный формат или имя ответа")
        fng_data_list = data.get('data', [])
        if not fng_data_list or not isinstance(fng_data_list, list): raise ValueError("Отсутствуют данные F&G в ответе")
        fng_data = fng_data_list[0]
        result = {'value': int(fng_data['value']), 'classification': str(fng_data['value_classification']), 'timestamp': int(fng_data['timestamp'])}
        logger.debug(f"[{source_name}] F&G получен: {result}")
        return result, STATUS_OK
    except (ValueError, KeyError, TypeError, IndexError) as e:
        logger.error(f"[{source_name}] Ошибка обработки данных F&G: {e}. Ответ: {data}", exc_info=True)
        return None, STATUS_FORMAT_ERROR

async def fetch_eth_gas_price(session: aiohttp.ClientSession) -> tuple[dict[str, int | float] | None, str]:
    if not ETHERSCAN_API_KEY: return None, STATUS_CONFIG_ERROR
    url = "https://api.etherscan.io/api"
    params = {"module": "gastracker", "action": "gasoracle", "apikey": ETHERSCAN_API_KEY}
    source_name = "Etherscan-Gas"
    data, status = await _fetch_with_retry(session, url, params, retries=1, base_wait=1.5, source_name=source_name)

    if status != STATUS_OK:
         if isinstance(data, dict) and data.get("message") == "NOTOK" and "invalid api key" in str(data.get('result','')).lower():
             logger.error(f"[{source_name}] Etherscan API Key невалиден. Ответ: {data.get('result','')}")
             return None, STATUS_INVALID_KEY
         if status == STATUS_FORBIDDEN:
             return None, STATUS_INVALID_KEY
         return None, status
    try:
        if not isinstance(data, dict) or data.get("status") != "1" or not isinstance(data.get("result"), dict):
            logger.error(f"[{source_name}] Неверный статус или формат ответа Etherscan: {data.get('message', '')}. Result: {data.get('result')}")
            if isinstance(data.get("result"), str) and "invalid api key" in data.get("result", "").lower():
                 return None, STATUS_INVALID_KEY
            return None, STATUS_FORMAT_ERROR

        gas_data = data["result"]
        result = {
            'safe': int(float(gas_data['SafeGasPrice'])), 'propose': int(float(gas_data['ProposeGasPrice'])),
            'fast': int(float(gas_data['FastGasPrice'])), 'base_fee': round(float(gas_data['suggestBaseFee']), 2)
        }
        logger.debug(f"[{source_name}] ETH Gas получен: {result}")
        return result, STATUS_OK
    except (ValueError, KeyError, TypeError) as e:
        logger.error(f"[{source_name}] Ошибка обработки данных ETH Gas: {e}. Ответ: {data}", exc_info=True)
        return None, STATUS_FORMAT_ERROR

async def fetch_coingecko_trending(session: aiohttp.ClientSession) -> tuple[list[dict[str, Any]] | None, str]:
    url = "https://api.coingecko.com/api/v3/search/trending"
    source_name = "CoinGecko-Trending"
    data, status = await _fetch_with_retry(session, url, source_name=source_name)
    if status != STATUS_OK: return None, status
    try:
        if not isinstance(data, dict) or not isinstance(data.get('coins'), list): raise ValueError("Неверный формат ответа")
        trending_coins = []
        for item_wrapper in data['coins']:
            coin_info = item_wrapper.get('item', {})
            if coin_info.get('id') and coin_info.get('name') and coin_info.get('symbol'):
                 trending_coins.append({'id': coin_info['id'], 'name': coin_info['name'], 'symbol': coin_info['symbol'].upper(),
                                        'thumb': coin_info.get('thumb'), 'market_cap_rank': coin_info.get('market_cap_rank')})
        if not trending_coins and data['coins']: return None, STATUS_FORMAT_ERROR
        if not trending_coins: return None, STATUS_NO_DATA
        logger.debug(f"[{source_name}] Получено {len(trending_coins)} трендовых монет.")
        return trending_coins, STATUS_OK
    except (ValueError, KeyError, TypeError, IndexError) as e:
        logger.error(f"[{source_name}] Критическая ошибка обработки Trending: {e}. Ответ: {data}", exc_info=True)
        return None, STATUS_FORMAT_ERROR

async def fetch_crypto_news(
    session: aiohttp.ClientSession,
    news_filter: str = "important",
    limit: int = 5
) -> tuple[list[dict[str, Any]] | None, str]:
    if not CRYPTO_PANIC_API_KEY:
        return None, STATUS_CONFIG_ERROR

    url = "https://cryptopanic.com/api/v1/posts/"
    params = {"auth_token": CRYPTO_PANIC_API_KEY, "public": "true", "filter": news_filter}
    source_name = "CryptoPanic-News"

    data, status = await _fetch_with_retry(session, url, params, source_name=source_name)

    if status != STATUS_OK:
        if status == STATUS_INVALID_KEY:
             logger.error(f"[{source_name}] CryptoPanic API ключ невалиден или ограничен.")
        elif status == STATUS_NOT_FOUND:
             logger.warning(f"[{source_name}] CryptoPanic API вернул 404 (Not Found) для фильтра '{news_filter}'.")
        return None, status

    try:
        if not isinstance(data, dict) or not isinstance(data.get('results'), list):
            logger.error(f"[{source_name}] Неверный формат ответа от CryptoPanic: 'results' не список или ответ не dict. Ответ: {str(data)[:200]}")
            return None, STATUS_FORMAT_ERROR

        news_items = []
        for item in data['results'][:limit]:
            # --- НАЧАЛО ИСПРАВЛЕНИЯ ---
            # Делаем проверку более гибкой. Нам нужны хотя бы заголовок и ссылка.
            if isinstance(item, dict) and item.get('title') and item.get('url'):
                news_items.append({
                    'title': item['title'],
                    'url': item['url'],
                    'domain': item.get('domain', 'N/A'), # .get() безопасен, вернет 'N/A' если ключа нет
                    'published_at': item.get('published_at'),
                    'source_title': item.get('source', {}).get('title') # Безопасное получение вложенного ключа
                })
            else:
                # Логируем, только если это действительно неожиданно (например, нет заголовка)
                if isinstance(item, dict):
                    logger.warning(f"[{source_name}] Пропуск элемента новости из-за отсутствия 'title' или 'url': {item.get('id', 'No ID')}")
                else:
                    logger.warning(f"[{source_name}] Пропуск невалидного элемента новости (не dict): {str(item)[:100]}")
            # --- КОНЕЦ ИСПРАВЛЕНИЯ ---

        if not news_items and data['results']:
             logger.warning(f"[{source_name}] Не удалось отформатировать ни одной новости из полученных {len(data['results'])}.")
             # Возвращаем пустой список, но со статусом OK, чтобы не показывать ошибку пользователю
             return [], STATUS_OK
        if not news_items:
             logger.info(f"[{source_name}] Новости не найдены для фильтра '{news_filter}'.")
             return None, STATUS_NO_DATA

        logger.debug(f"[{source_name}] Получено {len(news_items)} новостей.")
        return news_items, STATUS_OK

    except (ValueError, KeyError, TypeError, IndexError) as e:
        logger.error(f"[{source_name}] Критическая ошибка обработки новостей CryptoPanic: {e}. Ответ: {str(data)[:200]}", exc_info=True)
        return None, STATUS_FORMAT_ERROR


def fetch_general_news(query: str, language: str = 'en', page_size: int = 5, sort_by: str = 'relevancy') -> tuple[list[dict[str, Any]] | None, str]:
    if not newsapi_client: return None, STATUS_CONFIG_ERROR
    if not query: return None, STATUS_BAD_REQUEST
    source_name = "NewsAPI.org"
    logger.debug(f"[{source_name}] Запрос: query='{query}', lang='{language}', size={page_size}")
    try:
        all_articles = newsapi_client.get_everything(q=query, language=language, sort_by=sort_by, page_size=page_size)

        if all_articles.get('status') == 'ok':
            articles_list = all_articles.get('articles', [])
            if not articles_list: return None, STATUS_NO_DATA

            formatted_articles = []
            for article in articles_list:
                if article.get('title') and article.get('url'):
                    formatted_articles.append({
                        'title': article['title'],
                        'url': article['url'],
                        'source': article.get('source', {}).get('name'),
                        'published_at': article.get('publishedAt'),
                        'description': article.get('description')
                    })
            if not formatted_articles and articles_list: return None, STATUS_FORMAT_ERROR
            if not formatted_articles: return None, STATUS_NO_DATA

            logger.debug(f"[{source_name}] Получено {len(formatted_articles)} статей.")
            return formatted_articles, STATUS_OK
        else:
            err_code = all_articles.get('code', 'unknown_error_code')
            err_msg = all_articles.get('message', 'N/A')
            logger.error(f"[{source_name}] Ошибка от NewsAPI: {err_msg} (code: {err_code})")
            if err_code in ['apiKeyInvalid', 'apiKeyMissing', 'apiKeyDisabled']: return None, STATUS_INVALID_KEY
            if err_code == 'rateLimited': return None, STATUS_RATE_LIMIT
            return None, STATUS_API_ERROR
    except Exception as e:
        logger.error(f"[{source_name}] Исключение при запросе к NewsAPI: {e}", exc_info=True)
        if "timed out" in str(e).lower(): return None, STATUS_TIMEOUT
        return None, STATUS_UNKNOWN_ERROR

async def fetch_index_prices(session: aiohttp.ClientSession, symbols: list[str]) -> dict[str, tuple[float | None, str]]:
    results: dict[str, tuple[float | None, str]] = dict.fromkeys(symbols, (None, STATUS_UNKNOWN_ERROR))
    if not symbols: return results
    if not ALPHA_VANTAGE_API_KEY:
        logger.warning("[AlphaVantage-Index] Запрос не выполнен: ключ ALPHA_VANTAGE_API_KEY отсутствует.")
        return dict.fromkeys(symbols, (None, STATUS_CONFIG_ERROR))

    source_name = "AlphaVantage-Index"
    logger.debug(f"[{source_name}] Запрос цен для: {', '.join(symbols)}")

    for i, symbol in enumerate(symbols):
        if i > 0:
            await asyncio.sleep(API_COOLDOWN * 0.5)

        params = {"function": "GLOBAL_QUOTE", "symbol": symbol.upper(), "apikey": ALPHA_VANTAGE_API_KEY}
        data, status = await _fetch_with_retry(session, "https://www.alphavantage.co/query", params, retries=1, source_name=f"{source_name}-{symbol}")

        if status == STATUS_OK and isinstance(data, dict) and "Global Quote" in data:
            try:
                quote_data = data["Global Quote"]
                if not quote_data:
                     logger.warning(f"[{source_name}] Пустой 'Global Quote' для {symbol}. Вероятно, тикер не найден.")
                     results[symbol] = (None, STATUS_NOT_FOUND)
                     continue

                price_str = quote_data.get("05. price")
                if price_str is None:
                    logger.warning(f"[{source_name}] '05. price' отсутствует для {symbol}. Ответ: {data}")
                    results[symbol] = (None, STATUS_NO_DATA)
                    continue
                results[symbol] = (float(price_str), STATUS_OK)
            except (ValueError, TypeError, KeyError) as e:
                results[symbol] = (None, STATUS_FORMAT_ERROR)
                logger.warning(f"[{source_name}] Ошибка парсинга {symbol}: {e}. Ответ: {data}")
        elif status == STATUS_RATE_LIMIT or (isinstance(data, dict) and data.get("Note") and "API call frequency" in data["Note"]):
            logger.warning(f"[{source_name}] Достигнут Rate Limit для {symbol} (или всех последующих).")
            results[symbol] = (None, STATUS_RATE_LIMIT)
            for remaining_symbol in symbols[i+1:]:
                results[remaining_symbol] = (None, STATUS_RATE_LIMIT)
            break
        else:
            if isinstance(data, dict) and "Error Message" in data:
                 logger.warning(f"[{source_name}] Ошибка от AlphaVantage для {symbol}: {data['Error Message']}")
                 results[symbol] = (None, STATUS_API_ERROR)
            else:
                 results[symbol] = (None, status)
    return results


async def fetch_current_global_market_data(session: aiohttp.ClientSession) -> tuple[dict[str, Any] | None, str]:
    url = "https://api.coingecko.com/api/v3/global"
    source_name = "CoinGecko-Global"
    data, status = await _fetch_with_retry(session, url, source_name=source_name)

    if status != STATUS_OK:
        return None, status

    try:
        if not isinstance(data, dict) or "data" not in data or not isinstance(data["data"], dict):
            raise ValueError("Invalid format for global market data")

        global_data = data["data"]
        market_cap_percentage = global_data.get("market_cap_percentage", {})
        total_market_cap = global_data.get("total_market_cap", {})
        total_volume = global_data.get("total_volume", {})

        result = {
            "btc_dominance": market_cap_percentage.get("btc"),
            "eth_dominance": market_cap_percentage.get("eth"),
            "total_market_cap_usd": total_market_cap.get("usd"),
            "total_volume_usd": total_volume.get("usd"),
            "active_cryptocurrencies": global_data.get("active_cryptocurrencies"),
            "updated_at": global_data.get("updated_at")
        }

        if result.get("btc_dominance") is None:
            logger.warning(f"[{source_name}] BTC dominance not found in response: {global_data}")
            pass

        logger.debug(f"[{source_name}] Global market data fetched: BTC.D {result.get('btc_dominance')}%")
        return result, STATUS_OK

    except (ValueError, KeyError, TypeError) as e:
        logger.error(f"[{source_name}] Error processing global market data: {e}. Response: {data}", exc_info=True)
        return None, STATUS_FORMAT_ERROR

async def fetch_historical_btc_dominance_data(
    session: aiohttp.ClientSession,
    days_to_fetch: int,
    vs_currency: str = "usd"
) -> tuple[list[tuple[date, float]] | None, str]:
    logger.warning("[Derived-BTCDominanceHist] Historical BTC Dominance data fetching is currently disabled due to API limitations for free tiers.")
    return None, STATUS_NO_DATA

async def fetch_funding_rates(
    session: aiohttp.ClientSession
) -> tuple[list[dict[str, Any]] | None, str]:
    """Fetches all derivative tickers to find funding rates for perpetuals."""
    url = "https://api.coingecko.com/api/v3/derivatives"
    source_name = "CoinGecko-FundingRates"

    data, status = await _fetch_with_retry(session, url, source_name=source_name)

    if status != STATUS_OK:
        return None, status

    try:
        if not isinstance(data, list):
            raise TypeError("Derivatives response is not a list")

        funding_rates = []
        for item in data:
            if (isinstance(item, dict) and
                item.get('contract_type') == 'perpetual' and
                isinstance(item.get('funding_rate'), (int, float))):

                funding_rates.append({
                    'market': item.get('market', 'N/A'),
                    'symbol': item.get('symbol', 'N/A'),
                    'rate': float(item['funding_rate']) * 100
                })

        if not funding_rates:
            logger.warning(f"[{source_name}] No perpetual contracts with funding rates found in the response.")
            return None, STATUS_NO_DATA

        logger.debug(f"[{source_name}] Successfully processed {len(funding_rates)} perpetual contracts.")
        return funding_rates, STATUS_OK

    except (TypeError, KeyError, ValueError) as e:
        logger.error(f"[{source_name}] Critical error processing funding rates data: {e}", exc_info=True)
        return None, STATUS_FORMAT_ERROR

async def fetch_tvl(
    session: aiohttp.ClientSession,
    protocol_name: str
) -> tuple[float | None, str]:
    """Fetches the Total Value Locked (TVL) for a given DeFi protocol."""
    protocol_slug = protocol_name.lower().replace(" ", "-")
    url = f"https://api.llama.fi/tvl/{protocol_slug}"
    source_name = f"DeFiLlama-TVL-{protocol_slug}"

    try:
        logger.debug(f"[{source_name}] Попытка: GET {url}")
        async with session.get(url, timeout=DEFAULT_REQUEST_TIMEOUT) as response:
            status_code = response.status
            if status_code == 200:
                tvl_value_str = await response.text()
                try:
                    tvl_value = float(tvl_value_str)
                    logger.debug(f"[{source_name}] TVL получен: {tvl_value}")
                    return tvl_value, STATUS_OK
                except (ValueError, TypeError):
                    logger.error(f"[{source_name}] Не удалось преобразовать ответ в число: {tvl_value_str[:100]}")
                    return None, STATUS_FORMAT_ERROR
            elif status_code == 404:
                logger.warning(f"[{source_name}] Протокол не найден (404) по адресу {url}.")
                return None, STATUS_NOT_FOUND
            else:
                error_text = await response.text()
                logger.error(f"[{source_name}] Ошибка API ({status_code}): {error_text[:200]}")
                return None, STATUS_API_ERROR
    except asyncio.TimeoutError:
        logger.warning(f"[{source_name}] Таймаут при запросе к {url}")
        return None, STATUS_TIMEOUT
    except aiohttp.ClientError as e:
        logger.error(f"[{source_name}] Ошибка сети при запросе к {url}: {e}", exc_info=True)
        return None, STATUS_NETWORK_ERROR
    except Exception as e:
        logger.error(f"[{source_name}] Непредвиденное исключение при запросе TVL: {e}", exc_info=True)
        return None, STATUS_UNKNOWN_ERROR


async def fetch_economic_events(
    session: aiohttp.ClientSession,
    period: str = "week"
) -> tuple[list[dict[str, Any]] | None, str]:
    """
    Симулятор запроса экономических событий. Возвращает предопределенный список.
    """
    source_name = "Events-Simulator"
    logger.debug(f"[{source_name}] Запрос событий для периода: {period}")

    await asyncio.sleep(0.2)

    now = datetime.now(timezone.utc)
    today = now.date()

    all_simulated_events = [
        {'time': now.replace(hour=12, minute=30, second=0, microsecond=0), 'country': 'US', 'title': 'Initial Jobless Claims', 'impact': 'high', 'forecast': '215K', 'previous': '212K'},
        {'time': now.replace(hour=14, minute=0, second=0, microsecond=0), 'country': 'US', 'title': 'Existing Home Sales', 'impact': 'low', 'forecast': '4.20M', 'previous': '4.19M'},
        {'time': (now + timedelta(days=1)).replace(hour=9, minute=0, second=0, microsecond=0), 'country': 'EU', 'title': 'German Flash PMI', 'impact': 'medium', 'forecast': '46.5', 'previous': '45.7'},
        {'time': (now + timedelta(days=1)).replace(hour=13, minute=45, second=0, microsecond=0), 'country': 'US', 'title': 'Flash Services PMI', 'impact': 'high', 'forecast': '51.2', 'previous': '51.3'},
        {'time': (now + timedelta(days=3)).replace(hour=8, minute=30, second=0, microsecond=0), 'country': 'EU', 'title': 'ECB President Lagarde Speaks', 'impact': 'high', 'forecast': None, 'previous': None},
        {'time': (now + timedelta(days=4)).replace(hour=12, minute=30, second=0, microsecond=0), 'country': 'US', 'title': 'Core PCE Price Index m/m', 'impact': 'high', 'forecast': '0.3%', 'previous': '0.3%'},
        {'time': (now + timedelta(days=4)).replace(hour=14, minute=0, second=0, microsecond=0), 'country': 'US', 'title': 'Revised UoM Consumer Sentiment', 'impact': 'low', 'forecast': '67.7', 'previous': '67.4'},
        {'time': (now + timedelta(days=8)).replace(hour=14, minute=0, second=0, microsecond=0), 'country': 'US', 'title': 'ISM Manufacturing PMI', 'impact': 'high', 'forecast': '50.5', 'previous': '50.3'},
    ]

    filtered_events = []
    if period == "today":
        filtered_events = [e for e in all_simulated_events if e['time'].date() == today]
    elif period == "tomorrow":
        tomorrow = today + timedelta(days=1)
        filtered_events = [e for e in all_simulated_events if e['time'].date() == tomorrow]
    elif period == "week":
        end_of_week = today + timedelta(days=6 - today.weekday() + 1)
        filtered_events = [e for e in all_simulated_events if today <= e['time'].date() < end_of_week]

    result_list = []
    for event in sorted(filtered_events, key=lambda x: x['time']):
        event_copy = event.copy()
        event_copy['time'] = event['time'].isoformat()
        result_list.append(event_copy)

    if not result_list:
        logger.info(f"[{source_name}] Не найдено событий для периода '{period}'.")
        return None, STATUS_NO_DATA

    logger.debug(f"[{source_name}] Возвращено {len(result_list)} событий для периода '{period}'.")
    return result_list, STATUS_OK

async def fetch_glassnode_metric(
    session: aiohttp.ClientSession,
    endpoint: str,
    asset: str = 'BTC',
    resolution: str = '24h',
    since: int | None = None
) -> tuple[list[dict[str, Any]] | None, str]:
    """
    Fetches a specific metric from the Glassnode API v1.
    """
    if not GLASSNODE_API_KEY:
        logger.warning("[Glassnode] Запрос не выполнен: ключ GLASSNODE_API_KEY отсутствует.")
        return None, STATUS_CONFIG_ERROR

    base_url = "https://api.glassnode.com/v1/metrics/"
    url = f"{base_url}{endpoint}"
    source_name = f"Glassnode-{endpoint.split('/')[-1]}"

    params = {
        'a': asset,
        'i': resolution,
        's': since if since else int((datetime.now() - timedelta(days=2)).timestamp())
    }
    headers = {'X-Api-Key': GLASSNODE_API_KEY}

    data, status = await _fetch_with_retry(session, url, params=params, headers=headers, source_name=source_name)

    if status != STATUS_OK:
        return None, status

    try:
        if not isinstance(data, list):
            raise TypeError("Ответ Glassnode не является списком")

        formatted_data = [{'t': item['t'], 'v': item['v']} for item in data if 't' in item and 'v' in item]

        if not formatted_data:
            logger.warning(f"[{source_name}] Получен пустой список данных после форматирования.")
            return None, STATUS_NO_DATA

        logger.debug(f"[{source_name}] Успешно получено {len(formatted_data)} точек данных.")
        return formatted_data, STATUS_OK

    except (TypeError, KeyError) as e:
        logger.error(f"[{source_name}] Критическая ошибка обработки данных Glassnode: {e}", exc_info=True)
        return None, STATUS_FORMAT_ERROR
