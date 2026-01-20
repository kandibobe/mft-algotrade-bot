# handlers/signal_handler.py
import asyncio
import html
from datetime import datetime, timedelta

import aiohttp
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.constants import ParseMode
from telegram.ext import ContextTypes

from src.telegram_bot import constants
from src.telegram_bot.config_adapter import DEFAULT_ANALYSIS_PERIOD, SIGNAL_DATA_CACHE_AGE
from src.telegram_bot.handlers.misc_handler import _get_cached_or_fetch
from src.telegram_bot.localization.manager import get_text, get_user_language
from src.telegram_bot.services import analysis, data_fetcher, graph_generator, user_manager
from src.utils.logger import get_logger

logger = get_logger(__name__)


async def signal_command_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    lang_code = await get_user_language(user_id)
    logger.info(f"Запрос сигнала от user_id: {user_id}")

    effective_message = update.message or (
        update.callback_query.message if update.callback_query else None
    )
    if not effective_message:
        return

    loading_msg = await effective_message.reply_text(get_text(constants.MSG_LOADING, lang_code))

    session: aiohttp.ClientSession | None = context.bot_data.get("aiohttp_session")
    if not session or session.closed:
        await loading_msg.edit_text(get_text(constants.MSG_ERROR_GENERAL, lang_code))
        return

    user_settings = user_manager.get_settings(user_id)
    analysis_period = user_settings.get("analysis_period", DEFAULT_ANALYSIS_PERIOD)

    # Запросы данных
    btc_hist_data = await _get_cached_or_fetch(
        "signal_btc_hist",
        data_fetcher.fetch_historical_crypto_data,
        context,
        update,
        lang_code,
        fetch_args=[constants.CG_BTC, analysis_period + 10],
        max_cache_age=SIGNAL_DATA_CACHE_AGE,
    )
    dxy_hist_data = await _get_cached_or_fetch(
        "signal_dxy_hist",
        data_fetcher.fetch_fred_series,
        context,
        update,
        lang_code,
        fetch_args=[
            constants.FRED_DXY,
            (datetime.now() - timedelta(days=analysis_period + 30)).strftime("%Y-%m-%d"),
        ],
        max_cache_age=SIGNAL_DATA_CACHE_AGE,
    )
    cpi_hist_data_full = await _get_cached_or_fetch(
        "signal_cpi_hist",
        data_fetcher.fetch_fred_series,
        context,
        update,
        lang_code,
        fetch_args=[
            constants.FRED_CPI,
            (datetime.now() - timedelta(days=max(analysis_period * 2, 180))).strftime("%Y-%m-%d"),
        ],
        max_cache_age=SIGNAL_DATA_CACHE_AGE,
    )
    fng_data = await _get_cached_or_fetch(
        "fng", data_fetcher.fetch_fear_greed_index, context, update, lang_code
    )

    await loading_msg.delete()

    try:
        if not isinstance(btc_hist_data, list) or not isinstance(dxy_hist_data, list):
            raise ValueError(
                get_text(constants.MSG_ERROR_DATA_FORMAT_DETAIL, lang_code, source="BTC/DXY")
            )

        btc_val_aligned, dxy_val_aligned, common_dates = analysis.align_data_by_date(
            btc_hist_data, dxy_hist_data
        )
        if not btc_val_aligned:
            raise ValueError(get_text(constants.MSG_SIGNAL_ERROR_NO_COMMON_DATES, lang_code))

        final_analysis_days = len(btc_val_aligned)
        if final_analysis_days < constants.MIN_POINTS_FOR_SIGNAL:
            raise ValueError(
                get_text(
                    constants.MSG_SIGNAL_ERROR_POINTS,
                    lang_code,
                    actual_points=final_analysis_days,
                    min_points=constants.MIN_POINTS_FOR_SIGNAL,
                    period=analysis_period,
                )
            )

        cpi_for_analysis = None
        if isinstance(cpi_hist_data_full, list) and len(cpi_hist_data_full) >= 1:
            points_to_take = (
                min(max(3, analysis_period // 30), len(cpi_hist_data_full))
                if len(cpi_hist_data_full) >= 2
                else 1
            )
            cpi_for_analysis = cpi_hist_data_full[-points_to_take:]

        # Сохраняем данные для генерации графиков по запросу
        context.user_data["signal_graph_data"] = {
            "btc": list(zip(common_dates, btc_val_aligned, strict=False)),
            "dxy": list(zip(common_dates, dxy_val_aligned, strict=False)),
            "cpi": cpi_for_analysis,
        }

        signal_result = analysis.generate_trading_signal(
            btc_trend_val=analysis.calculate_trend(btc_val_aligned),
            dxy_trend_val=analysis.calculate_trend(dxy_val_aligned),
            cpi_data_full=cpi_for_analysis,
            correlation_btc_dxy_val=analysis.calculate_correlation(
                btc_val_aligned, dxy_val_aligned
            ),
            fng_value=fng_data.get("value") if isinstance(fng_data, dict) else None,
        )

        parts = [get_text(constants.MSG_SIGNAL_HEADER, lang_code, period=final_analysis_days)]
        signal_desc = get_text(signal_result["signal_key"], lang_code)
        parts.append(
            f"{get_text(constants.SIGNAL_FINAL_SIGNAL, lang_code)}: <b>{signal_desc}</b> (Покупка: {signal_result['buy_score']} | Продажа: {signal_result['sell_score']})\n"
        )

        parts.append(f"\n<b>{get_text(constants.SIGNAL_DETAILS_HEADER, lang_code)}:</b>")

        keyboard = []
        for key, data in signal_result["details"].items():
            name = get_text(key, lang_code, default=key)
            reason = get_text(data["reason_key"], lang_code, default=data["reason_key"])
            score_viz = analysis.get_factor_arrow_score(data.get("score_val", 0))

            button_text = f"{name}: {reason} → {score_viz}"

            # Добавляем кнопку только если для фактора можно построить график
            if key in [
                constants.SIGNAL_FACTOR_BTC_TREND,
                constants.SIGNAL_FACTOR_DXY_TREND,
                constants.SIGNAL_FACTOR_CPI_TREND,
            ]:
                keyboard.append(
                    [
                        InlineKeyboardButton(
                            button_text, callback_data=f"{constants.CB_PREFIX_SIGNAL_DETAIL}{key}"
                        )
                    ]
                )
            else:
                parts.append(f" • {button_text}")

        signal_text = "\n".join(parts)
        await effective_message.reply_text(
            signal_text, parse_mode=ParseMode.HTML, reply_markup=InlineKeyboardMarkup(keyboard)
        )

    except ValueError as ve:
        await effective_message.reply_text(f"⚠️ {html.escape(str(ve))}", parse_mode=ParseMode.HTML)
    except Exception as e:
        logger.error(f"Непредвиденная ошибка генерации сигнала: {e}", exc_info=True)
        await effective_message.reply_text(
            get_text(constants.MSG_ERROR_ANALYSIS, lang_code), parse_mode=ParseMode.HTML
        )


async def signal_detail_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обрабатывает нажатие на кнопку фактора сигнала и отправляет график."""
    query = update.callback_query
    await query.answer()

    user_id = update.effective_user.id
    lang_code = await get_user_language(user_id)

    graph_data = context.user_data.get("signal_graph_data")
    if not graph_data:
        await query.message.reply_text(
            get_text(
                constants.MSG_REPORT_DATA_EXPIRED,
                lang_code,
                default="Данные для графика устарели. Пожалуйста, запросите новый /signal.",
            )
        )
        return

    factor_key = query.data[len(constants.CB_PREFIX_SIGNAL_DETAIL) :]

    data_to_plot = None
    title = ""
    y_label = ""

    if factor_key == constants.SIGNAL_FACTOR_BTC_TREND:
        data_to_plot = graph_data.get("btc")
        title = get_text("graph_title_btc_trend", lang_code, default="Тренд цены BTC")
        y_label = "Цена, USD"
    elif factor_key == constants.SIGNAL_FACTOR_DXY_TREND:
        data_to_plot = graph_data.get("dxy")
        title = get_text("graph_title_dxy_trend", lang_code, default="Тренд индекса доллара (DXY)")
        y_label = "Значение индекса"
    elif factor_key == constants.SIGNAL_FACTOR_CPI_TREND:
        data_to_plot = graph_data.get("cpi")
        title = get_text("graph_title_cpi_trend", lang_code, default="Тренд инфляции (CPI, США)")
        y_label = "Значение индекса"

    if not data_to_plot:
        await query.message.reply_text(get_text(constants.MSG_ERROR_GENERAL, lang_code))
        return

    loading_msg = await query.message.reply_text(
        get_text(constants.MSG_GRAPH_LOADING, lang_code, symbol=title)
    )

    graph_bytes = await asyncio.to_thread(
        graph_generator.generate_trend_graph, data_to_plot, title, y_label
    )

    await loading_msg.delete()

    if graph_bytes:
        await query.message.reply_photo(photo=graph_bytes)
    else:
        await query.message.reply_text(
            get_text(constants.MSG_GRAPH_ERROR_GENERAL, lang_code, symbol=title)
        )
