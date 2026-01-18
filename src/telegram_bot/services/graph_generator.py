# services/graph_generator.py
import io
from datetime import date, datetime
from typing import Union

import matplotlib
import mplfinance as mpf
import pandas as pd

from src.utils.logger import get_logger

matplotlib.use('Agg') # Используем бэкенд Agg для работы без GUI
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import linregress

logger = get_logger(__name__)
DateType = Union[datetime, date]

MIN_POINTS_FOR_GRAPH_RENDER = 3

def generate_candlestick_graph(
    historical_data: list[tuple[DateType, float]],
    asset_symbol: str,
    period_days: int,
) -> bytes | None:
    if not historical_data or len(historical_data) < MIN_POINTS_FOR_GRAPH_RENDER:
        logger.warning(f"Недостаточно данных для генерации графика {asset_symbol} ({len(historical_data)} < {MIN_POINTS_FOR_GRAPH_RENDER}).")
        return None

    logger.debug(f"Генерация свечного графика для {asset_symbol} ({period_days} дн., {len(historical_data)} точек)")
    fig = None

    try:
        df = pd.DataFrame(historical_data, columns=['Date', 'Close'])
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.set_index('Date')

        df['Open'] = df['Close'].shift(1).fillna(df['Close'])
        df['High'] = df[['Open', 'Close']].max(axis=1).fillna(df['Close'])
        df['Low'] = df[['Open', 'Close']].min(axis=1).fillna(df['Close'])
        df = df[['Open', 'High', 'Low', 'Close']]

        if df.isnull().values.any():
            logger.warning(f"DataFrame для графика {asset_symbol} содержит NaN значения после обработки. Попытка удалить строки с NaN.")
            df.dropna(inplace=True)
            if len(df) < MIN_POINTS_FOR_GRAPH_RENDER:
                logger.warning(f"После удаления NaN осталось недостаточно данных для графика {asset_symbol} ({len(df)} < {MIN_POINTS_FOR_GRAPH_RENDER}).")
                return None

        if df.empty:
            logger.warning(f"DataFrame для графика {asset_symbol} пуст после обработки.")
            return None

        style = 'yahoo'
        buf = io.BytesIO()

        fig, _axes = mpf.plot(
            df, type='candle', title=f"\n{asset_symbol} Price ({period_days}d)",
            ylabel='Price (USD)', figratio=(12, 6), figscale=1.1, style=style,
            volume=False, datetime_format='%y-%m-%d', tight_layout=True, returnfig=True
        )
        fig.savefig(buf, format='png', bbox_inches='tight', dpi=100)
        buf.seek(0)
        logger.debug(f"Свечной график '{asset_symbol}' успешно сгенерирован.")
        return buf.getvalue()
    except Exception as e:
        logger.error(f"Ошибка генерации свечного графика '{asset_symbol}': {e}", exc_info=True)
        return None
    finally:
        if fig:
            try: plt.close(fig); logger.debug(f"Фигура графика '{asset_symbol}' закрыта.")
            except Exception as close_err: logger.error(f"Ошибка при закрытии фигуры графика '{asset_symbol}': {close_err}")

def generate_trend_graph(
    data: list[tuple[DateType, float] | float],
    title: str,
    y_label: str
) -> bytes | None:
    """Генерирует линейный график с линией тренда."""
    if not data or len(data) < 2:
        logger.warning(f"Недостаточно данных для генерации графика тренда для '{title}' ({len(data)} < 2).")
        return None

    logger.debug(f"Генерация графика тренда для '{title}'")
    fig = None

    try:
        if isinstance(data[0], tuple):
            # Данные с датами
            dates = [item[0] for item in data]
            values = [item[1] for item in data]
            df = pd.DataFrame({'Date': pd.to_datetime(dates), 'Value': values}).set_index('Date')
            x_axis = df.index
        else:
            # Просто список значений
            values = data
            x_axis = range(len(values))

        # Расчет линии тренда
        x_numeric = np.arange(len(values))
        y_numeric = np.array(values, dtype=float)

        # Игнорируем NaN для расчета тренда
        mask = ~np.isnan(y_numeric)
        if np.sum(mask) < 2:
            logger.warning(f"Недостаточно валидных (не NaN) точек для тренда в '{title}'")
            return None

        slope, intercept, _, _, _ = linregress(x_numeric[mask], y_numeric[mask])
        trend_line = slope * x_numeric + intercept

        # Построение графика
        plt.style.use('seaborn-v0_8-darkgrid')
        fig, ax = plt.subplots(figsize=(10, 5), dpi=100)

        ax.plot(x_axis, values, label=y_label, color='royalblue', linewidth=2)
        ax.plot(x_axis, trend_line, label='Тренд', color='tomato', linestyle='--', linewidth=2)

        ax.set_title(title, fontsize=16)
        ax.set_ylabel(y_label, fontsize=12)
        ax.legend()

        if isinstance(x_axis, pd.DatetimeIndex):
            plt.gcf().autofmt_xdate() # Автоформатирование дат

        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)

        logger.debug(f"График тренда '{title}' успешно сгенерирован.")
        return buf.getvalue()
    except Exception as e:
        logger.error(f"Ошибка генерации графика тренда '{title}': {e}", exc_info=True)
        return None
    finally:
        if fig:
            try: plt.close(fig); logger.debug(f"Фигура графика '{title}' закрыта.")
            except Exception as close_err: logger.error(f"Ошибка при закрытии фигуры графика '{title}': {close_err}")
