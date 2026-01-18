from typing import Any

from src.telegram_bot import constants
from src.telegram_bot.config_adapter import (
    ANALYSIS_PERIODS,
    DEFAULT_LANGUAGE,
    PRICE_ALERT_LIMIT,
    PRICE_ALERT_LIMIT_PREMIUM,
    SUPPORTED_LANGUAGES,
    WATCHLIST_LIMIT,
    WATCHLIST_LIMIT_PREMIUM,
)
from src.telegram_bot.database import db_manager
from src.telegram_bot.localization.manager import set_user_language_cache
from src.utils.logger import get_logger

logger = get_logger(__name__)

OPERATION_SUCCESS = "success"
OPERATION_FAILED_LIMIT = "limit_reached"
OPERATION_FAILED_EXISTS = "already_exists"
OPERATION_FAILED_NOT_FOUND = "not_found"
OPERATION_FAILED_INVALID = "invalid_input"
OPERATION_FAILED_DB_ERROR = "db_error"

def get_settings(user_id: int) -> dict[str, Any]:
    return db_manager.get_user_settings(user_id)

def is_user_premium(user_id: int) -> bool:
    settings = get_settings(user_id)
    return settings.get(constants.DB_FIELD_IS_PREMIUM, False)

def update_analysis_period(user_id: int, period: int) -> bool:
    if period not in ANALYSIS_PERIODS:
         logger.warning(f"Попытка установить недопустимый период {period} для user {user_id}")
         return False
    return db_manager.update_user_settings(user_id, {'analysis_period': period})

def toggle_notifications(user_id: int) -> bool | None:
    settings = get_settings(user_id)
    current_status = settings.get('notifications_enabled', True)
    new_status = not current_status
    success = db_manager.update_user_settings(user_id, {'notifications_enabled': new_status})
    if success:
        logger.info(f"Статус уведомлений для user {user_id} изменен на {new_status}.")
        return new_status
    else:
        logger.error(f"Не удалось изменить статус уведомлений для user {user_id}.")
        return None

def set_language(user_id: int, lang_code: str) -> bool:
    if lang_code not in SUPPORTED_LANGUAGES:
        logger.warning(f"Неподдерживаемый язык '{lang_code}' для user {user_id}. Используется '{DEFAULT_LANGUAGE}'.")
        lang_code = DEFAULT_LANGUAGE
    success = db_manager.update_user_settings(user_id, {'language_code': lang_code})
    if success:
        try:
            set_user_language_cache(user_id, lang_code)
            logger.info(f"Язык для user {user_id} установлен на '{lang_code}' (БД и кеш).")
        except Exception as cache_err:
            logger.warning(f"Не удалось обновить кеш языка для user {user_id} после смены на '{lang_code}': {cache_err}")
    else:
         logger.error(f"Не удалось сохранить язык '{lang_code}' в БД для user {user_id}.")
    return success

def get_user_limits(user_id: int) -> dict[str, int]:
    premium = is_user_premium(user_id)
    limits = {
        'watchlist': WATCHLIST_LIMIT_PREMIUM if premium else WATCHLIST_LIMIT,
        'alerts': PRICE_ALERT_LIMIT_PREMIUM if premium else PRICE_ALERT_LIMIT
    }
    logger.debug(f"Лимиты для user {user_id} (premium={premium}): {limits}")
    return limits

def add_asset_to_watchlist(user_id: int, ticker: str) -> str:
    ticker_upper = ticker.upper()
    limits = get_user_limits(user_id)
    current_count = db_manager.get_watchlist_count(user_id)
    if current_count >= limits['watchlist']:
        logger.warning(f"User {user_id} достиг лимита watchlist ({current_count}/{limits['watchlist']}).")
        return OPERATION_FAILED_LIMIT
    asset_info = constants.SUPPORTED_ASSETS.get(ticker_upper)
    if not asset_info:
        logger.warning(f"Попытка добавить невалидный тикер '{ticker_upper}' в watchlist user {user_id}.")
        return OPERATION_FAILED_INVALID
    asset_type, asset_id = asset_info
    if asset_type not in [constants.ASSET_CRYPTO, constants.ASSET_FOREX]:
        logger.warning(f"Попытка добавить неподдерживаемый тип актива '{asset_type}' ({ticker_upper}) в watchlist user {user_id}.")
        return OPERATION_FAILED_INVALID
    added = db_manager.add_to_watchlist(user_id, asset_type, asset_id)
    if added:
        return OPERATION_SUCCESS
    else:
        return OPERATION_FAILED_EXISTS

def remove_asset_from_watchlist(user_id: int, ticker_or_id: str) -> str:
    ticker_upper = ticker_or_id.upper()
    asset_id_to_delete: str | None = None
    asset_info = constants.SUPPORTED_ASSETS.get(ticker_upper)
    if asset_info:
        asset_id_to_delete = asset_info[1]
    else:
        if ticker_or_id in constants.REVERSE_ASSET_MAP:
             asset_id_to_delete = ticker_or_id
        else:
             found_by_direct_id_check = False
             for _, (_, potential_id) in constants.SUPPORTED_ASSETS.items():
                 if ticker_or_id == potential_id:
                     asset_id_to_delete = ticker_or_id
                     found_by_direct_id_check = True
                     break
             if not found_by_direct_id_check:
                logger.warning(f"Попытка удалить неподдерживаемый тикер/ID '{ticker_or_id}' из watchlist user {user_id}.")
                return OPERATION_FAILED_NOT_FOUND

    if not asset_id_to_delete:
        logger.error(f"Не удалось определить asset_id для удаления '{ticker_or_id}' user {user_id}.")
        return OPERATION_FAILED_INVALID

    deleted = db_manager.remove_from_watchlist(user_id, asset_id_to_delete)
    return OPERATION_SUCCESS if deleted else OPERATION_FAILED_NOT_FOUND

def get_user_watchlist(user_id: int) -> list[dict[str, str]]:
    return db_manager.get_watchlist(user_id)

def get_watchlist_count(user_id: int) -> int:
    return db_manager.get_watchlist_count(user_id)

def add_user_alert(
    user_id: int,
    asset_type: str,
    asset_id: str,
    alert_type: str,
    condition: str,
    target_value: float
) -> tuple[str, int | None]:
    limits = get_user_limits(user_id)
    current_count = get_price_alert_count(user_id)
    if current_count >= limits['alerts']:
        logger.warning(f"User {user_id} достиг лимита алертов ({current_count}/{limits['alerts']}).")
        return OPERATION_FAILED_LIMIT, None

    if alert_type == constants.ALERT_TYPE_PRICE:
        if condition not in ['>', '<'] or not isinstance(target_value, (int, float)) or target_value <= 0:
            return OPERATION_FAILED_INVALID, None
    elif alert_type == constants.ALERT_TYPE_RSI:
        if condition not in ['>', '<'] or target_value not in [30, 70] or asset_type != constants.ASSET_CRYPTO:
            return OPERATION_FAILED_INVALID, None
    else:
        return OPERATION_FAILED_INVALID, None

    alert_id = db_manager.add_alert(user_id, asset_type, asset_id, alert_type, condition, target_value)
    return (OPERATION_SUCCESS, alert_id) if alert_id is not None else (OPERATION_FAILED_DB_ERROR, None)

def get_user_price_alerts(user_id: int) -> list[dict[str, Any]]:
    return db_manager.get_price_alerts(user_id)

def get_all_price_alerts() -> list[dict[str, Any]]:
    return db_manager.get_price_alerts()

def delete_user_price_alert(user_id: int, alert_id: int) -> str:
    if not isinstance(alert_id, int):
        return OPERATION_FAILED_INVALID
    return OPERATION_SUCCESS if db_manager.delete_price_alert(user_id, alert_id) else OPERATION_FAILED_NOT_FOUND

def update_alert_trigger(alert_id: int, timestamp: int) -> bool:
    return db_manager.update_alert_trigger_time(alert_id, timestamp)

def get_subscribed_user_ids() -> list[int]:
    return db_manager.get_subscribed_users()

def get_price_alert_count(user_id: int) -> int:
    return len(db_manager.get_price_alerts(user_id))

def update_user_price_alert_fields(user_id: int, alert_id: int, updates: dict[str, Any]) -> bool:
    if not isinstance(alert_id, int) or alert_id <= 0 or not updates:
        return False

    allowed_updates = {
        k: v for k, v in updates.items()
        if (k == 'condition' and v in ['>', '<']) or
           (k == 'target_value' and isinstance(v, (int, float)) and v > 0)
    }
    return db_manager.update_price_alert_fields(user_id, alert_id, allowed_updates) if allowed_updates else False

# --- Portfolio Management ---

def get_user_portfolio(user_id: int) -> list[dict[str, Any]]:
    return db_manager.get_portfolio(user_id)

def add_asset_to_portfolio(user_id: int, ticker: str, quantity: float, price: float) -> str:
    ticker_upper = ticker.upper()
    asset_info = constants.SUPPORTED_ASSETS.get(ticker_upper)
    if not asset_info:
        return OPERATION_FAILED_INVALID

    asset_type, asset_id = asset_info
    if db_manager.add_to_portfolio(user_id, asset_type, asset_id, quantity, price):
        return OPERATION_SUCCESS
    else:
        return OPERATION_FAILED_DB_ERROR

def remove_asset_from_portfolio(user_id: int, ticker: str) -> str:
    ticker_upper = ticker.upper()
    asset_info = constants.SUPPORTED_ASSETS.get(ticker_upper)
    if not asset_info:
        return OPERATION_FAILED_NOT_FOUND

    asset_id = asset_info[1]
    if db_manager.remove_from_portfolio(user_id, asset_id):
        return OPERATION_SUCCESS
    else:
        return OPERATION_FAILED_NOT_FOUND
