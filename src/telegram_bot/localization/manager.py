# localization/manager.py
from src.telegram_bot.config_adapter import DEFAULT_LANGUAGE, SUPPORTED_LANGUAGES
from src.telegram_bot.database import db_manager
from src.utils.logger import get_logger

from .strings import STRINGS

logger = get_logger(__name__)

user_lang_cache = {}


async def get_user_language(user_id: int) -> str:
    if user_id in user_lang_cache:
        return user_lang_cache[user_id]

    settings = db_manager.get_user_settings(user_id)
    lang = settings.get("language_code", DEFAULT_LANGUAGE)

    if lang not in SUPPORTED_LANGUAGES:
        lang = DEFAULT_LANGUAGE

    user_lang_cache[user_id] = lang
    return lang


def set_user_language_cache(user_id: int, lang_code: str):
    if lang_code in SUPPORTED_LANGUAGES:
        user_lang_cache[user_id] = lang_code
    else:
        user_lang_cache[user_id] = DEFAULT_LANGUAGE
    logger.debug(f"Обновлен кеш языка для user {user_id}: {user_lang_cache[user_id]}")


def get_text(
    key: str, lang_code: str = DEFAULT_LANGUAGE, default: str | None = None, **kwargs
) -> str:
    if lang_code not in SUPPORTED_LANGUAGES:
        lang_code = DEFAULT_LANGUAGE

    lang_strings = STRINGS.get(lang_code, STRINGS.get(DEFAULT_LANGUAGE, {}))
    base_string = lang_strings.get(key)

    if base_string is None and lang_code != DEFAULT_LANGUAGE:
        lang_strings = STRINGS.get(DEFAULT_LANGUAGE, {})
        base_string = lang_strings.get(key)

    if base_string is None:
        logger.warning(
            f"Ключ локализации '{key}' не найден ни для '{lang_code}', ни для '{DEFAULT_LANGUAGE}'."
        )
        final_string = default if default is not None else f"[{key}]"
    else:
        final_string = base_string

    try:
        # **ИСПРАВЛЕНИЕ**: Убрана ошибочная логика, которая приводила к падению.
        # Ответственность за добавление 'premium_ad' теперь лежит на хендлерах.
        if isinstance(final_string, dict):
            return final_string

        return final_string.format(**kwargs)
    except KeyError as e:
        logger.error(
            f"Ключ локализации '{key}' (язык: {lang_code}) ожидает аргумент форматирования: {e}. Строка: '{final_string}' Передано: {kwargs}"
        )
        return f"[{key}: Отсутствует Аргумент '{e}']"
    except Exception as e:
        logger.error(
            f"Ошибка форматирования строки локализации '{key}' (язык: {lang_code}): {e}. Строка: '{final_string}' Передано: {kwargs}"
        )
        return f"[{key}: Ошибка Формата]"


async def get_user_text(user_id: int, key: str, default: str | None = None, **kwargs) -> str:
    lang_code = await get_user_language(user_id)
    return get_text(key, lang_code, default=default, **kwargs)
