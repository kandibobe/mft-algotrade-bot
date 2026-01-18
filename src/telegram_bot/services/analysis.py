# services/analysis.py
from datetime import date, datetime
from typing import Any, Union

import numpy as np
import pandas as pd
from scipy.stats import linregress, pearsonr

from src.telegram_bot import constants
from src.telegram_bot.localization.manager import get_text
from src.utils.logger import get_logger

logger = get_logger(__name__)
DateType = Union[datetime, date]

def calculate_correlation(data1: list[float], data2: list[float]) -> float | None:
    if not data1 or not data2 or len(data1) != len(data2) or len(data1) < 3:
        return None
    try:
        arr1, arr2 = np.array(data1, dtype=float), np.array(data2, dtype=float)
        if np.all(arr1 == arr1[0]) or np.all(arr2 == arr2[0]): return 0.0
        corr, _ = pearsonr(arr1, arr2)
        return round(corr, 3) if np.isfinite(corr) else None
    except Exception as e:
        logger.error(f"Ошибка вычисления корреляции: {e}", exc_info=True)
        return None

def calculate_trend(data: list[float]) -> float:
    if len(data) < 2: return 0.0
    try:
        x, y = np.arange(len(data)), np.array(data, dtype=float)
        if np.all(y == y[0]): return 0.0
        slope, _, r_value, _, _ = linregress(x, y)
        logger.debug(f"Расчет тренда: slope={slope:.4f}, R^2={r_value**2:.3f} для {len(data)} точек")
        return slope if np.isfinite(slope) else 0.0
    except Exception as e:
        logger.error(f"Ошибка расчета тренда: {e}", exc_info=True)
        return 0.0

def align_data_by_date(
    data1: list[tuple[DateType, float]],
    data2: list[tuple[DateType, float]]
) -> tuple[list[float], list[float], list[date]]:
    if not data1 or not data2: return [], [], []
    try:
        dict1 = { (item[0].date() if isinstance(item[0], datetime) else item[0]): item[1]
                  for item in data1 if isinstance(item, tuple) and len(item) == 2 and isinstance(item[0], (datetime, date)) }
        dict2 = { (item[0].date() if isinstance(item[0], datetime) else item[0]): item[1]
                  for item in data2 if isinstance(item, tuple) and len(item) == 2 and isinstance(item[0], (datetime, date)) }

        common_dates_set = set(dict1.keys()) & set(dict2.keys())
        if not common_dates_set: return [], [], []

        common_dates = sorted(common_dates_set)
        return [dict1[d] for d in common_dates], [dict2[d] for d in common_dates], common_dates
    except Exception as e:
        logger.error(f"Ошибка при выравнивании данных по дате: {e}", exc_info=True)
        return [], [], []

def get_factor_arrow_score(score_value: int) -> str:
    if score_value > 0: return f"🟩 (+{score_value} BUY)"
    if score_value < 0: return f"🟥 ({score_value} SELL)"
    return "🟨 (0)"

def calculate_technical_indicators(
    historical_prices: list[tuple[DateType, float]],
    rsi_period: int = 14,
    sma_short: int = 50,
    sma_long: int = 200
) -> dict[str, Any] | None:
    """
    Рассчитывает RSI, SMA и положение цены для отчета /report.
    """
    min_data_points = sma_long + 5
    if not historical_prices or len(historical_prices) < min_data_points:
        logger.debug(f"Недостаточно данных для ТА: получено {len(historical_prices)}, требуется {min_data_points}")
        return None

    try:
        df = pd.DataFrame(historical_prices, columns=['Date', 'Close'])
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.set_index('Date')

        df.ta.rsi(length=rsi_period, append=True)
        df.ta.sma(length=sma_short, append=True)
        df.ta.sma(length=sma_long, append=True)

        latest_data = df.iloc[-1]
        last_price = latest_data['Close']
        rsi_value = latest_data[f'RSI_{rsi_period}']
        sma_short_value = latest_data[f'SMA_{sma_short}']
        sma_long_value = latest_data[f'SMA_{sma_long}']

        if pd.isna(rsi_value) or pd.isna(sma_short_value) or pd.isna(sma_long_value):
            logger.warning("Один из индикаторов ТА для отчета рассчитан как NaN.")
            return None

        position_key = None
        position_args = {}
        if last_price > sma_short_value and last_price > sma_long_value:
            position_key = constants.MSG_REPORT_TA_PRICE_ABOVE_SMA
            position_args = {'sma_period': sma_short}
        elif last_price < sma_short_value and last_price < sma_long_value:
            position_key = constants.MSG_REPORT_TA_PRICE_BELOW_SMA
            position_args = {'sma_period': sma_long}
        else:
            position_key = constants.MSG_REPORT_TA_PRICE_BETWEEN_SMAS
            position_args = {}

        return {
            "rsi": rsi_value,
            "position_key": position_key,
            "position_args": position_args
        }

    except Exception as e:
        logger.error(f"Ошибка при расчете ТА для отчета: {e}", exc_info=True)
        return None

def calculate_detailed_technical_indicators(
    historical_prices: list[tuple[DateType, float]]
) -> dict[str, Any] | None:
    """
    Рассчитывает RSI, MACD, Bollinger Bands для команды /ta.
    """
    min_data_points = 40
    if not historical_prices or len(historical_prices) < min_data_points:
        logger.debug(f"Недостаточно данных для детального ТА: получено {len(historical_prices)}, требуется {min_data_points}")
        return None

    try:
        df = pd.DataFrame(historical_prices, columns=['Date', 'Close'])
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.set_index('Date')

        df.ta.rsi(append=True)
        df.ta.macd(append=True)
        df.ta.bbands(append=True)

        latest = df.iloc[-1]

        results = {
            'rsi': latest.get('RSI_14'),
            'macd_line': latest.get('MACD_12_26_9'),
            'macd_signal_line': latest.get('MACDs_12_26_9'),
            'bb_lower': latest.get('BBL_20_2.0'),
            'bb_upper': latest.get('BBU_20_2.0'),
            'last_price': latest.get('Close')
        }

        if any(pd.isna(v) for v in results.values()):
            logger.warning(f"Один из детальных индикаторов ТА рассчитан как NaN. Результаты: {results}")
            return None

        return results
    except Exception as e:
        logger.error(f"Ошибка при расчете детальных технических индикаторов: {e}", exc_info=True)
        return None


def generate_trading_signal(
    btc_trend_val: float,
    cpi_data_full: list[tuple[date, float]] | None,
    dxy_trend_val: float,
    correlation_btc_dxy_val: float | None,
    fng_value: int | None,
    btc_dominance_trend_val: float | None = None
    ) -> dict[str, Any]:

    buy_score = 0
    sell_score = 0
    details = {}
    trend_threshold_abs = 0.001

    # 1. Тренд BTC
    btc_factor_score = 0
    if btc_trend_val > trend_threshold_abs:
        btc_factor_score = 1
        details[constants.SIGNAL_FACTOR_BTC_TREND] = {'reason_key': constants.SIGNAL_DETAIL_BTC_UP}
    elif btc_trend_val < -trend_threshold_abs:
        btc_factor_score = -1
        details[constants.SIGNAL_FACTOR_BTC_TREND] = {'reason_key': constants.SIGNAL_DETAIL_BTC_DOWN}
    else:
        details[constants.SIGNAL_FACTOR_BTC_TREND] = {'reason_key': constants.SIGNAL_DETAIL_BTC_FLAT}
    details[constants.SIGNAL_FACTOR_BTC_TREND]['score_val'] = btc_factor_score
    if btc_factor_score > 0: buy_score += btc_factor_score
    else: sell_score += abs(btc_factor_score)
    details[constants.SIGNAL_FACTOR_BTC_TREND]['numeric_value'] = btc_trend_val


    # 2. Тренд CPI (улучшенный анализ)
    cpi_factor_score = 0
    cpi_numeric_value = None
    if cpi_data_full and len(cpi_data_full) >= 2:
        cpi_values = [item[1] for item in cpi_data_full]
        cpi_trend_val = calculate_trend(cpi_values)
        cpi_numeric_value = cpi_trend_val

        last_cpi, prev_cpi = cpi_values[-1], cpi_values[-2]
        cpi_change_direction_key = None

        current_cpi_trend_key = constants.SIGNAL_DETAIL_CPI_FLAT
        if cpi_trend_val < -trend_threshold_abs:
            cpi_factor_score = 1
            current_cpi_trend_key = constants.SIGNAL_DETAIL_CPI_DOWN
        elif cpi_trend_val > trend_threshold_abs:
            cpi_factor_score = -1
            current_cpi_trend_key = constants.SIGNAL_DETAIL_CPI_UP

        if current_cpi_trend_key == constants.SIGNAL_DETAIL_CPI_DOWN:
            if last_cpi < prev_cpi:
                cpi_change_direction_key = constants.SIGNAL_DETAIL_CPI_DECELERATES
        elif current_cpi_trend_key == constants.SIGNAL_DETAIL_CPI_UP:
            if last_cpi > prev_cpi:
                cpi_change_direction_key = constants.SIGNAL_DETAIL_CPI_ACCELERATES

        details[constants.SIGNAL_FACTOR_CPI_TREND] = {'reason_key': current_cpi_trend_key}
        if cpi_change_direction_key:
            details[constants.SIGNAL_FACTOR_CPI_TREND]['sub_reason_key'] = cpi_change_direction_key

    elif cpi_data_full and len(cpi_data_full) == 1:
        details[constants.SIGNAL_FACTOR_CPI_TREND] = {'reason_key': constants.SIGNAL_DETAIL_CPI_ONE_POINT}
        cpi_numeric_value = cpi_data_full[0][1]
    else:
        details[constants.SIGNAL_FACTOR_CPI_TREND] = {'reason_key': constants.SIGNAL_DETAIL_CPI_NA}

    details[constants.SIGNAL_FACTOR_CPI_TREND]['score_val'] = cpi_factor_score
    if cpi_numeric_value is not None: details[constants.SIGNAL_FACTOR_CPI_TREND]['numeric_value'] = cpi_numeric_value
    if cpi_factor_score > 0: buy_score += cpi_factor_score
    else: sell_score += abs(cpi_factor_score)


    # 3. Тренд DXY
    dxy_factor_score = 0
    if dxy_trend_val < -trend_threshold_abs:
        dxy_factor_score = 1; details[constants.SIGNAL_FACTOR_DXY_TREND] = {'reason_key': constants.SIGNAL_DETAIL_DXY_DOWN}
    elif dxy_trend_val > trend_threshold_abs:
        dxy_factor_score = -1; details[constants.SIGNAL_FACTOR_DXY_TREND] = {'reason_key': constants.SIGNAL_DETAIL_DXY_UP}
    else:
        details[constants.SIGNAL_FACTOR_DXY_TREND] = {'reason_key': constants.SIGNAL_DETAIL_DXY_FLAT}
    details[constants.SIGNAL_FACTOR_DXY_TREND]['score_val'] = dxy_factor_score
    if dxy_factor_score > 0: buy_score += dxy_factor_score
    else: sell_score += abs(dxy_factor_score)
    details[constants.SIGNAL_FACTOR_DXY_TREND]['numeric_value'] = dxy_trend_val


    # 4. Корреляция BTC/DXY
    corr_factor_score = 0
    corr_reason_key = constants.SIGNAL_DETAIL_CORR_NONE
    if correlation_btc_dxy_val is not None:
        if correlation_btc_dxy_val < -0.5:
            if dxy_trend_val < -trend_threshold_abs:
                corr_factor_score = 1; corr_reason_key = constants.SIGNAL_DETAIL_CORR_NEG_DXY_DOWN
            elif dxy_trend_val > trend_threshold_abs:
                corr_factor_score = -1; corr_reason_key = constants.SIGNAL_DETAIL_CORR_NEG_DXY_UP
    details[constants.SIGNAL_FACTOR_CORR] = {'reason_key': corr_reason_key, 'score_val': corr_factor_score, 'numeric_value': correlation_btc_dxy_val}
    if corr_factor_score > 0: buy_score += corr_factor_score
    else: sell_score += abs(corr_factor_score)

    # 5. F&G Индекс
    fng_factor_score = 0
    fng_reason_key = constants.SIGNAL_DETAIL_FNG_NA
    if fng_value is not None:
        if fng_value <= 25: fng_factor_score = 1; fng_reason_key = constants.SIGNAL_DETAIL_FNG_EXTREME_FEAR
        elif fng_value < 46: fng_reason_key = constants.SIGNAL_DETAIL_FNG_FEAR
        elif fng_value <= 54: fng_reason_key = constants.SIGNAL_DETAIL_FNG_NEUTRAL
        elif fng_value < 75: fng_reason_key = constants.SIGNAL_DETAIL_FNG_GREED
        else: fng_factor_score = -1; fng_reason_key = constants.SIGNAL_DETAIL_FNG_EXTREME_GREED
    details[constants.SIGNAL_FACTOR_FNG] = {'reason_key': fng_reason_key, 'score_val': fng_factor_score, 'numeric_value': fng_value}
    if fng_factor_score > 0: buy_score += fng_factor_score
    else: sell_score += abs(fng_factor_score)

    # 6. BTC Dominance
    btc_dom_factor_score = 0
    btc_dom_reason_key = constants.SIGNAL_DETAIL_BTC_DOMINANCE_NA
    dominance_trend_threshold = trend_threshold_abs / 5
    if btc_dominance_trend_val is not None:
        if btc_dominance_trend_val > dominance_trend_threshold:
            btc_dom_reason_key = constants.SIGNAL_DETAIL_BTC_DOMINANCE_RISING
        elif btc_dominance_trend_val < -dominance_trend_threshold:
            btc_dom_reason_key = constants.SIGNAL_DETAIL_BTC_DOMINANCE_FALLING
        else:
            pass

    details[constants.SIGNAL_FACTOR_BTC_DOMINANCE] = {
        'reason_key': btc_dom_reason_key,
        'score_val': btc_dom_factor_score,
        'numeric_value': btc_dominance_trend_val
    }

    strong_signal_threshold = 2
    if buy_score == 0 and sell_score == 0:
        signal_key = constants.SIGNAL_HOLD
    elif buy_score >= sell_score + strong_signal_threshold: signal_key = constants.SIGNAL_STRONG_BUY
    elif sell_score >= buy_score + strong_signal_threshold: signal_key = constants.SIGNAL_STRONG_SELL
    elif buy_score > sell_score: signal_key = constants.SIGNAL_WEAK_BUY
    elif sell_score > buy_score: signal_key = constants.SIGNAL_WEAK_SELL
    else: signal_key = constants.SIGNAL_HOLD

    final_signal_text_for_log = get_text(signal_key, 'ru')
    logger.info(f"Сгенерирован сигнал: {final_signal_text_for_log} (BUY: {buy_score}, SELL: {sell_score})")
    logger.debug(f"Детали сигнала: {details}")
    return {"signal_key": signal_key, "buy_score": buy_score, "sell_score": sell_score, "details": details}
