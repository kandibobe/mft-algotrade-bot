# src/telegram_bot/runner.py
from datetime import time, timezone

import aiohttp
from telegram import Update
from telegram.constants import ParseMode
from telegram.ext import (
    Application,
    ApplicationBuilder,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    Defaults,
    JobQueue,
    MessageHandler,
    filters,
)

from src.telegram_bot import constants, jobs

# Updated imports for project structure
from src.telegram_bot.config_adapter import (
    DIGEST_SEND_HOUR_UTC,
    INDEX_FETCH_INTERVAL,
    ONCHAIN_FETCH_INTERVAL,
    PRICE_ALERT_CHECK_INTERVAL,
    SHARED_DATA_FETCH_INTERVAL,
    SUPPORTED_LANGUAGES,
    TELEGRAM_BOT_TOKEN,
    VOLATILITY_FETCH_INTERVAL,
)
from src.telegram_bot.database import db_manager
from src.telegram_bot.handlers import (
    alert_handler,
    analytics_chat_handler,
    mft_control_handler,
    report_handler,
    settings_handler,
    signal_handler,
    volatility_handler,
    watchlist_handler,
)
from src.telegram_bot.services.notification_service import NotificationService
from src.telegram_bot.handlers import common as common_main_handlers
from src.telegram_bot.localization.manager import get_text
from src.utils.logger import log as logger

BOT_VERSION = "1.2.0-integrated"


async def post_init(application: Application):
    logger.info(f"Post-initialization of Bot v{BOT_VERSION}...")
    logger.info("Initializing database...")
    try:
        db_manager.initialize_db()
        logger.info("Database initialized successfully.")
    except Exception as e:
        logger.critical(f"CRITICAL DB INIT ERROR: {e}", exc_info=True)

    logger.info("Creating shared aiohttp session...")
    if (
        "aiohttp_session" not in application.bot_data
        or application.bot_data["aiohttp_session"].closed
    ):
        application.bot_data["aiohttp_session"] = aiohttp.ClientSession()

    jq: JobQueue | None = application.job_queue
    if jq:
        logger.info("Scheduling background jobs...")
        # Remove old jobs if any
        for job in jq.jobs():
            job.schedule_removal()

        # Add jobs
        jq.run_repeating(
            jobs.check_price_alerts,
            interval=PRICE_ALERT_CHECK_INTERVAL,
            first=10,
            name="PriceAlertCheck",
        )
        jq.run_repeating(
            jobs.fetch_volatility_job,
            interval=VOLATILITY_FETCH_INTERVAL,
            first=40,
            name="VolatilityFetch",
        )
        jq.run_repeating(
            jobs.fetch_shared_data_job,
            interval=SHARED_DATA_FETCH_INTERVAL,
            first=5,
            name="SharedDataFetch",
        )
        jq.run_repeating(
            jobs.fetch_index_data_job, interval=INDEX_FETCH_INTERVAL, first=15, name="IndexFetch"
        )
        jq.run_repeating(
            jobs.fetch_onchain_data_job,
            interval=ONCHAIN_FETCH_INTERVAL,
            first=60,
            name="OnChainDataFetch",
        )
        
        # System Heartbeat (every hour)
        jq.run_repeating(
            jobs.system_heartbeat,
            interval=3600,
            first=60,
            name="SystemHeartbeat"
        )

        try:
            digest_hour = int(DIGEST_SEND_HOUR_UTC)
            if 0 <= digest_hour <= 23:
                digest_time_utc = time(hour=digest_hour, minute=0, second=0, tzinfo=timezone.utc)
                jq.run_daily(jobs.send_daily_digest, time=digest_time_utc, name="DailyDigest")
        except (ValueError, TypeError):
            logger.error(f"Invalid DIGEST_SEND_HOUR_UTC: {DIGEST_SEND_HOUR_UTC}")

        logger.info("All jobs scheduled.")
    else:
        logger.warning("JobQueue not available!")


async def post_shutdown(application: Application):
    logger.info("Bot shutdown initiated...")
    session: aiohttp.ClientSession | None = application.bot_data.get("aiohttp_session")
    if session and not session.closed:
        await session.close()
    logger.info("Bot shutdown complete.")


async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE):
    logger.error("Exception while handling an update:", exc_info=context.error)


def get_button_texts(button_key: str) -> list[str]:
    texts = set()
    for lang in SUPPORTED_LANGUAGES:
        try:
            val = get_text(button_key, lang, default=None)
            if val:
                texts.add(val)
        except Exception:
            pass
    return list(texts)


def create_bot_application() -> Application:
    """Factory to create the bot application."""
    if not TELEGRAM_BOT_TOKEN:
        logger.warning("TELEGRAM_BOT_TOKEN not set! Bot will not start.")
        return None

    defaults = Defaults(parse_mode=ParseMode.HTML)
    application = (
        ApplicationBuilder()
        .token(TELEGRAM_BOT_TOKEN)
        .defaults(defaults)
        .post_init(post_init)
        .post_shutdown(post_shutdown)
        .connect_timeout(30)
        .read_timeout(30)
        .get_updates_read_timeout(60)
        .build()
    )

    # Handlers
    application.add_handler(alert_handler.add_alert_conv_handler)
    application.add_handler(alert_handler.edit_alert_conv_handler)
    application.add_handler(analytics_chat_handler.analytics_chat_conv_handler)

    # Initialize Notification Service
    notification_service = NotificationService(application)
    application.bot_data["notification_service"] = notification_service

    # Add Job to update live dashboard
    jq = application.job_queue
    if jq:
        jq.run_repeating(jobs.update_live_dashboard, interval=30, first=10, name="LiveDashboard")

    # Commands
    application.add_handler(CommandHandler(constants.CMD_START, common_main_handlers.start))
    application.add_handler(CommandHandler(constants.CMD_HELP, common_main_handlers.help_command))
    application.add_handler(
        CommandHandler(constants.CMD_REPORT, report_handler.report_command_handler)
    )
    application.add_handler(
        CommandHandler(constants.CMD_SIGNAL, signal_handler.signal_command_handler)
    )
    application.add_handler(
        CommandHandler(constants.CMD_VOLATILITY, volatility_handler.volatility_command_handler)
    )
    application.add_handler(
        CommandHandler(constants.CMD_WATCHLIST, watchlist_handler.watchlist_command)
    )
    application.add_handler(CommandHandler(constants.CMD_ALERTS, alert_handler.alerts_command))
    application.add_handler(
        CommandHandler(constants.CMD_SETTINGS, settings_handler.settings_command_handler)
    )

    # MFT Control Commands
    application.add_handler(CommandHandler("status", mft_control_handler.status_command))
    application.add_handler(CommandHandler("balance", mft_control_handler.balance_command))
    application.add_handler(CommandHandler("positions", mft_control_handler.positions_command))
    application.add_handler(CommandHandler("stop_panic", mft_control_handler.stop_panic_command))
    application.add_handler(CommandHandler("logs", mft_control_handler.logs_command))
    application.add_handler(CommandHandler("features", mft_control_handler.features_command))
    application.add_handler(CommandHandler("whitelist", mft_control_handler.whitelist_command))

    # Message Handlers for Menus
    application.add_handler(
        MessageHandler(
            filters.Text(get_button_texts("menu_btn_report")), report_handler.report_command_handler
        )
    )
    application.add_handler(
        MessageHandler(
            filters.Text(get_button_texts("menu_btn_watchlist")),
            watchlist_handler.watchlist_command,
        )
    )

    # Callbacks
    application.add_handler(
        CallbackQueryHandler(
            settings_handler.settings_main_menu_callback, pattern=f"^{constants.CB_MAIN_SETTINGS}$"
        )
    )
    application.add_handler(
        CallbackQueryHandler(
            alert_handler.delalert_callback,
            pattern=f"^{constants.CB_ACTION_DEL_ALERT}{constants.ALERT_ID_REGEX_PART}$",
        )
    )
    application.add_handler(
        CallbackQueryHandler(mft_control_handler.mft_control_callback, pattern="^mft_")
    )
    for h in alert_handler.application_add_handler_alert_deletion_confirmation:
        application.add_handler(h)

    application.add_error_handler(error_handler)
    return application


def run_bot():
    """Entry point for standalone execution."""
    app = create_bot_application()
    if app:
        logger.info("Starting Telegram Bot Polling...")
        app.run_polling(allowed_updates=Update.ALL_TYPES, drop_pending_updates=True)

if __name__ == "__main__":
    run_bot()