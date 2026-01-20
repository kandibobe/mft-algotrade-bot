# utils/logger.py
import logging
import os  # Добавлен для получения уровня лога из окружения
import sys

# Определяем уровень логирования из переменной окружения или по умолчанию INFO
# Это позволяет легко менять уровень детализации логов без изменения кода
log_level_name = os.getenv("LOG_LEVEL", "INFO").upper()
log_level = getattr(logging, log_level_name, logging.INFO)


def get_logger(name: str | None = None) -> logging.Logger:
    """
    Настраивает и возвращает логгер с заданным именем или именем вызывающего модуля.
    Логгер выводит сообщения в stdout с заданным форматом и уровнем.
    Предотвращает дублирование обработчиков.
    """
    logger_name = name or __name__  # Используем имя модуля, если имя не передано
    logger = logging.getLogger(logger_name)

    # Предотвращаем дублирование обработчиков, если логгер уже настроен
    if logger.hasHandlers():
        # Убедимся, что уровень логгера соответствует глобальной настройке
        if logger.level != log_level:
            logger.setLevel(log_level)
        return logger

    # Устанавливаем уровень логирования
    logger.setLevel(log_level)

    # Создаем обработчик для вывода в консоль (stdout)
    stream_handler = logging.StreamHandler(sys.stdout)

    # Создаем форматтер
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - [%(funcName)s:%(lineno)d] - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",  # Формат даты и времени
    )

    # Присваиваем форматтер обработчику
    stream_handler.setFormatter(formatter)

    # Добавляем обработчик к логгеру
    logger.addHandler(stream_handler)

    # Устанавливаем уровень для обработчика (обычно совпадает с логгером)
    stream_handler.setLevel(log_level)

    # Отключаем передачу сообщений родительским логгерам (если они есть)
    # logger.propagate = False # Обычно не требуется для корневых логгеров или при простой настройке

    return logger


# Пример использования в других модулях:
# from src.utils.logger import get_logger
# logger = get_logger(__name__) # Использовать __name__ для получения имени текущего модуля
# logger.debug("Это сообщение для отладки")
# logger.info("Это информационное сообщение")
# logger.warning("Это предупреждение")
# logger.error("Это ошибка")
# logger.critical("Это критическая ошибка")
