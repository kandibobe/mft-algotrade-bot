# database/db_manager.py
import sqlite3
from typing import Any

from src.telegram_bot import constants
from src.telegram_bot.config_adapter import (
    DATABASE_URL,
    DEFAULT_ANALYSIS_PERIOD,
    DEFAULT_LANGUAGE,
    SUPPORTED_LANGUAGES,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


def _get_connection() -> sqlite3.Connection:
    db_path = DATABASE_URL.replace("sqlite:///", "")
    try:
        conn = sqlite3.connect(db_path)
        return conn
    except sqlite3.Error as e:
        logger.critical(f"Не удалось подключиться к базе данных '{db_path}': {e}")
        raise


def initialize_db():
    conn = _get_connection()
    cursor = conn.cursor()
    logger.info("Инициализация/проверка структуры базы данных...")
    try:
        # Table for user settings
        cursor.execute(f"""
            CREATE TABLE IF NOT EXISTS {constants.DB_TABLE_USER_SETTINGS} (
                user_id INTEGER PRIMARY KEY,
                language_code TEXT DEFAULT '{DEFAULT_LANGUAGE}',
                analysis_period INTEGER DEFAULT {DEFAULT_ANALYSIS_PERIOD},
                notifications_enabled BOOLEAN DEFAULT TRUE,
                {constants.DB_FIELD_IS_PREMIUM} BOOLEAN DEFAULT FALSE
            )
        """)
        # Table for watchlist
        cursor.execute(f"""
            CREATE TABLE IF NOT EXISTS {constants.DB_TABLE_WATCHLIST} (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                asset_type TEXT NOT NULL,
                asset_id TEXT NOT NULL,
                added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES {constants.DB_TABLE_USER_SETTINGS}(user_id) ON DELETE CASCADE,
                UNIQUE(user_id, asset_id)
            )
        """)
        # Table for price alerts
        cursor.execute(f"""
            CREATE TABLE IF NOT EXISTS {constants.DB_TABLE_PRICE_ALERTS} (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                asset_type TEXT NOT NULL,
                asset_id TEXT NOT NULL,
                alert_type TEXT NOT NULL DEFAULT '{constants.ALERT_TYPE_PRICE}',
                condition TEXT NOT NULL CHECK(condition IN ('>', '<')),
                target_value REAL NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_triggered_at INTEGER DEFAULT 0,
                FOREIGN KEY (user_id) REFERENCES {constants.DB_TABLE_USER_SETTINGS}(user_id) ON DELETE CASCADE
            )
        """)
        # Table for portfolio
        cursor.execute(f"""
            CREATE TABLE IF NOT EXISTS {constants.DB_TABLE_USER_PORTFOLIO} (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                asset_type TEXT NOT NULL,
                asset_id TEXT NOT NULL,
                quantity REAL NOT NULL,
                avg_buy_price REAL NOT NULL,
                FOREIGN KEY (user_id) REFERENCES {constants.DB_TABLE_USER_SETTINGS}(user_id) ON DELETE CASCADE,
                UNIQUE(user_id, asset_id)
            )
        """)

        # Indexes
        cursor.execute(
            f"CREATE INDEX IF NOT EXISTS idx_watchlist_user_id ON {constants.DB_TABLE_WATCHLIST}(user_id)"
        )
        cursor.execute(
            f"CREATE INDEX IF NOT EXISTS idx_alerts_user_id ON {constants.DB_TABLE_PRICE_ALERTS}(user_id)"
        )
        cursor.execute(
            f"CREATE INDEX IF NOT EXISTS idx_portfolio_user_id ON {constants.DB_TABLE_USER_PORTFOLIO}(user_id)"
        )

        conn.commit()
        logger.info("Структура базы данных инициализирована/проверена.")
    except sqlite3.Error as e:
        logger.error(f"Ошибка при инициализации таблиц БД: {e}", exc_info=True)
        conn.rollback()
    finally:
        conn.close()


def _dict_factory(cursor, row):
    fields = [column[0] for column in cursor.description]
    return dict(zip(fields, row, strict=False))


def get_user_settings(user_id: int) -> dict[str, Any]:
    conn = _get_connection()
    conn.row_factory = _dict_factory
    cursor = conn.cursor()
    settings: dict[str, Any] | None = None
    try:
        cursor.execute(
            f"SELECT * FROM {constants.DB_TABLE_USER_SETTINGS} WHERE user_id = ?", (user_id,)
        )
        settings = cursor.fetchone()
        if settings is None:
            logger.info(f"Пользователь {user_id} не найден, создание настроек по умолчанию.")
            default_settings = {
                "user_id": user_id,
                "language_code": DEFAULT_LANGUAGE,
                "analysis_period": DEFAULT_ANALYSIS_PERIOD,
                "notifications_enabled": True,
                constants.DB_FIELD_IS_PREMIUM: False,
            }
            columns = ", ".join(default_settings.keys())
            placeholders = ", ".join(["?"] * len(default_settings))
            values = list(default_settings.values())
            cursor.execute(
                f"INSERT INTO {constants.DB_TABLE_USER_SETTINGS} ({columns}) VALUES ({placeholders})",
                values,
            )
            conn.commit()
            settings = default_settings
            logger.info(f"Настройки по умолчанию для пользователя {user_id} созданы.")
        else:
            if constants.DB_FIELD_IS_PREMIUM not in settings:
                logger.warning(
                    f"Отсутствует поле '{constants.DB_FIELD_IS_PREMIUM}' для user {user_id}. Добавление со значением False."
                )
                update_user_settings(user_id, {constants.DB_FIELD_IS_PREMIUM: False})
                settings[constants.DB_FIELD_IS_PREMIUM] = False
            if settings.get("language_code") not in SUPPORTED_LANGUAGES:
                logger.warning(
                    f"Некорректный язык '{settings.get('language_code')}' для user {user_id}. Установка '{DEFAULT_LANGUAGE}'."
                )
                update_user_settings(user_id, {"language_code": DEFAULT_LANGUAGE})
                settings["language_code"] = DEFAULT_LANGUAGE
    except sqlite3.Error as e:
        logger.error(f"Ошибка получения настроек для user {user_id}: {e}", exc_info=True)
        settings = {
            "user_id": user_id,
            "language_code": DEFAULT_LANGUAGE,
            "analysis_period": DEFAULT_ANALYSIS_PERIOD,
            "notifications_enabled": True,
            constants.DB_FIELD_IS_PREMIUM: False,
        }
    finally:
        conn.close()
    return (
        settings
        if settings is not None
        else {
            "user_id": user_id,
            "language_code": DEFAULT_LANGUAGE,
            "analysis_period": DEFAULT_ANALYSIS_PERIOD,
            "notifications_enabled": True,
            constants.DB_FIELD_IS_PREMIUM: False,
        }
    )


def update_user_settings(user_id: int, updates: dict[str, Any]) -> bool:
    if not updates:
        return False
    conn = _get_connection()
    cursor = conn.cursor()
    set_clause = ", ".join([f"{key} = ?" for key in updates.keys()])
    values = list(updates.values())
    values.append(user_id)
    success = False
    try:
        cursor.execute(
            f"UPDATE {constants.DB_TABLE_USER_SETTINGS} SET {set_clause} WHERE user_id = ?",
            tuple(values),
        )
        conn.commit()
        success = cursor.rowcount > 0
        if not success:
            logger.warning(
                f"Обновление настроек не затронуло строк для user {user_id}. (Настройки: {updates})"
            )
        else:
            logger.debug(f"Настройки для user {user_id} обновлены: {updates}.")
    except sqlite3.Error as e:
        logger.error(f"Ошибка обновления настроек для user {user_id}: {e}", exc_info=True)
        conn.rollback()
        success = False
    finally:
        conn.close()
    return success


def get_watchlist_count(user_id: int) -> int:
    conn = _get_connection()
    cursor = conn.cursor()
    count = 0
    try:
        cursor.execute(
            f"SELECT COUNT(*) FROM {constants.DB_TABLE_WATCHLIST} WHERE user_id = ?", (user_id,)
        )
        result = cursor.fetchone()
        if result:
            count = result[0]
    except sqlite3.Error as e:
        logger.error(
            f"Ошибка получения количества watchlist для user {user_id}: {e}", exc_info=True
        )
    finally:
        conn.close()
    return count


def add_to_watchlist(user_id: int, asset_type: str, asset_id: str) -> bool:
    conn = _get_connection()
    cursor = conn.cursor()
    success = False
    try:
        cursor.execute(
            f"INSERT INTO {constants.DB_TABLE_WATCHLIST} (user_id, asset_type, asset_id) VALUES (?, ?, ?)",
            (user_id, asset_type, asset_id),
        )
        conn.commit()
        logger.info(f"Актив '{asset_id}' ({asset_type}) добавлен в watchlist для user {user_id}.")
        success = True
    except sqlite3.IntegrityError:
        logger.warning(f"Актив '{asset_id}' уже существует в watchlist для user {user_id}.")
        success = False
    except sqlite3.Error as e:
        logger.error(
            f"Ошибка добавления в watchlist user {user_id}, актив {asset_id}: {e}", exc_info=True
        )
        conn.rollback()
        success = False
    finally:
        conn.close()
    return success


def remove_from_watchlist(user_id: int, asset_id: str) -> bool:
    conn = _get_connection()
    cursor = conn.cursor()
    success = False
    try:
        cursor.execute(
            f"DELETE FROM {constants.DB_TABLE_WATCHLIST} WHERE user_id = ? AND asset_id = ?",
            (user_id, asset_id),
        )
        conn.commit()
        success = cursor.rowcount > 0
        if success:
            logger.info(f"Актив '{asset_id}' удален из watchlist для user {user_id}.")
        else:
            logger.warning(
                f"Актив '{asset_id}' не найден в watchlist для user {user_id} при попытке удаления."
            )
    except sqlite3.Error as e:
        logger.error(
            f"Ошибка удаления из watchlist user {user_id}, актив {asset_id}: {e}", exc_info=True
        )
        conn.rollback()
        success = False
    finally:
        conn.close()
    return success


def get_watchlist(user_id: int) -> list[dict[str, str]]:
    conn = _get_connection()
    conn.row_factory = _dict_factory
    cursor = conn.cursor()
    watchlist = []
    try:
        cursor.execute(
            f"SELECT asset_type, asset_id FROM {constants.DB_TABLE_WATCHLIST} WHERE user_id = ? ORDER BY added_at ASC",
            (user_id,),
        )
        watchlist = cursor.fetchall()
    except sqlite3.Error as e:
        logger.error(f"Ошибка получения watchlist для user {user_id}: {e}", exc_info=True)
    finally:
        conn.close()
    return watchlist if watchlist is not None else []


def add_alert(
    user_id: int,
    asset_type: str,
    asset_id: str,
    alert_type: str,
    condition: str,
    target_value: float,
) -> int | None:
    conn = _get_connection()
    cursor = conn.cursor()
    alert_id: int | None = None
    try:
        cursor.execute(
            f"""
            INSERT INTO {constants.DB_TABLE_PRICE_ALERTS}
            (user_id, asset_type, asset_id, alert_type, condition, target_value)
            VALUES (?, ?, ?, ?, ?, ?)
        """,
            (user_id, asset_type, asset_id, alert_type, condition, target_value),
        )
        conn.commit()
        alert_id = cursor.lastrowid
        if alert_id:
            logger.info(
                f"Алерт (тип: {alert_type}) ID {alert_id} создан для user {user_id} ({asset_id} {condition} {target_value})."
            )
        else:
            logger.error(
                f"Не удалось получить lastrowid после добавления алерта для user {user_id}."
            )
    except sqlite3.Error as e:
        logger.error(f"Ошибка добавления алерта user {user_id}: {e}", exc_info=True)
        conn.rollback()
    finally:
        conn.close()
    return alert_id


def get_price_alerts(user_id: int | None = None) -> list[dict[str, Any]]:
    conn = _get_connection()
    conn.row_factory = _dict_factory
    cursor = conn.cursor()
    alerts = []
    try:
        if user_id:
            cursor.execute(
                f"SELECT * FROM {constants.DB_TABLE_PRICE_ALERTS} WHERE user_id = ? ORDER BY created_at ASC",
                (user_id,),
            )
        else:
            cursor.execute(f"SELECT * FROM {constants.DB_TABLE_PRICE_ALERTS}")
        alerts = cursor.fetchall()
    except sqlite3.Error as e:
        logger.error(
            f"Ошибка получения ценовых алертов (user: {user_id if user_id else 'all'}): {e}",
            exc_info=True,
        )
    finally:
        conn.close()
    return alerts if alerts is not None else []


def delete_price_alert(user_id: int, alert_id: int) -> bool:
    conn = _get_connection()
    cursor = conn.cursor()
    success = False
    try:
        cursor.execute(
            f"DELETE FROM {constants.DB_TABLE_PRICE_ALERTS} WHERE id = ? AND user_id = ?",
            (alert_id, user_id),
        )
        conn.commit()
        success = cursor.rowcount > 0
        if success:
            logger.info(f"Ценовой алерт ID {alert_id} удален для user {user_id}.")
        else:
            logger.warning(
                f"Ценовой алерт ID {alert_id} не найден или не принадлежит user {user_id} при попытке удаления."
            )
    except sqlite3.Error as e:
        logger.error(
            f"Ошибка удаления ценового алерта ID {alert_id} для user {user_id}: {e}", exc_info=True
        )
        conn.rollback()
        success = False
    finally:
        conn.close()
    return success


def update_alert_trigger_time(alert_id: int, timestamp: int) -> bool:
    conn = _get_connection()
    cursor = conn.cursor()
    success = False
    try:
        cursor.execute(
            f"UPDATE {constants.DB_TABLE_PRICE_ALERTS} SET last_triggered_at = ? WHERE id = ?",
            (timestamp, alert_id),
        )
        conn.commit()
        success = cursor.rowcount > 0
        if not success:
            logger.warning(
                f"Не удалось обновить время срабатывания для алерта ID {alert_id} (возможно, удален?)."
            )
    except sqlite3.Error as e:
        logger.error(
            f"Ошибка обновления времени срабатывания для алерта ID {alert_id}: {e}", exc_info=True
        )
        conn.rollback()
        success = False
    finally:
        conn.close()
    return success


def get_subscribed_users() -> list[int]:
    conn = _get_connection()
    cursor = conn.cursor()
    user_ids = []
    try:
        cursor.execute(
            f"SELECT user_id FROM {constants.DB_TABLE_USER_SETTINGS} WHERE notifications_enabled = TRUE"
        )
        user_ids = [row[0] for row in cursor.fetchall()]
    except sqlite3.Error as e:
        logger.error(f"Ошибка получения списка подписанных пользователей: {e}", exc_info=True)
    finally:
        conn.close()
    return user_ids


def update_price_alert_fields(user_id: int, alert_id: int, updates: dict[str, Any]) -> bool:
    if not updates:
        logger.warning(f"Нет данных для обновления алерта ID {alert_id} для user {user_id}")
        return False

    conn = _get_connection()
    cursor = conn.cursor()

    set_clause_parts = []
    values = []
    for key, value in updates.items():
        if key in ["condition", "target_value"]:
            set_clause_parts.append(f"{key} = ?")
            values.append(value)
        else:
            logger.warning(f"Попытка обновить неразрешенное поле '{key}' для алерта ID {alert_id}")

    if not set_clause_parts:
        logger.warning(
            f"Нет разрешенных полей для обновления в алерте ID {alert_id}. Updates: {updates}"
        )
        conn.close()
        return False

    set_clause = ", ".join(set_clause_parts)
    values.append(alert_id)
    values.append(user_id)

    success = False
    try:
        cursor.execute(
            f"UPDATE {constants.DB_TABLE_PRICE_ALERTS} SET {set_clause} WHERE id = ? AND user_id = ?",
            tuple(values),
        )
        conn.commit()
        success = cursor.rowcount > 0
        if success:
            logger.info(f"Алерт ID {alert_id} для user {user_id} обновлен: {updates}.")
        else:
            logger.warning(
                f"Обновление алерта ID {alert_id} не затронуло строк для user {user_id}. (Updates: {updates})"
            )
    except sqlite3.Error as e:
        logger.error(
            f"Ошибка обновления алерта ID {alert_id} для user {user_id}: {e}", exc_info=True
        )
        conn.rollback()
        success = False
    finally:
        conn.close()
    return success


# --- Portfolio Functions ---


def get_portfolio(user_id: int) -> list[dict[str, Any]]:
    conn = _get_connection()
    conn.row_factory = _dict_factory
    cursor = conn.cursor()
    portfolio = []
    try:
        cursor.execute(
            f"SELECT * FROM {constants.DB_TABLE_USER_PORTFOLIO} WHERE user_id = ?", (user_id,)
        )
        portfolio = cursor.fetchall()
    except sqlite3.Error as e:
        logger.error(f"Ошибка получения портфеля для user {user_id}: {e}", exc_info=True)
    finally:
        conn.close()
    return portfolio if portfolio is not None else []


def add_to_portfolio(
    user_id: int, asset_type: str, asset_id: str, quantity: float, price: float
) -> bool:
    conn = _get_connection()
    cursor = conn.cursor()
    try:
        # Using INSERT OR REPLACE to handle both new and existing assets
        cursor.execute(
            f"""
            INSERT INTO {constants.DB_TABLE_USER_PORTFOLIO} (user_id, asset_type, asset_id, quantity, avg_buy_price)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(user_id, asset_id) DO UPDATE SET
                quantity = excluded.quantity,
                avg_buy_price = excluded.avg_buy_price;
        """,
            (user_id, asset_type, asset_id, quantity, price),
        )
        conn.commit()
        logger.info(f"Актив '{asset_id}' добавлен/обновлен в портфеле user {user_id}.")
        return True
    except sqlite3.Error as e:
        logger.error(
            f"Ошибка добавления/обновления в портфель user {user_id}, актив {asset_id}: {e}",
            exc_info=True,
        )
        conn.rollback()
        return False
    finally:
        conn.close()


def remove_from_portfolio(user_id: int, asset_id: str) -> bool:
    conn = _get_connection()
    cursor = conn.cursor()
    try:
        cursor.execute(
            f"DELETE FROM {constants.DB_TABLE_USER_PORTFOLIO} WHERE user_id = ? AND asset_id = ?",
            (user_id, asset_id),
        )
        conn.commit()
        if cursor.rowcount > 0:
            logger.info(f"Актив '{asset_id}' удален из портфеля user {user_id}.")
            return True
        else:
            logger.warning(
                f"Актив '{asset_id}' не найден в портфеле user {user_id} при попытке удаления."
            )
            return False
    except sqlite3.Error as e:
        logger.error(
            f"Ошибка удаления из портфеля user {user_id}, актив {asset_id}: {e}", exc_info=True
        )
        conn.rollback()
        return False
    finally:
        conn.close()
