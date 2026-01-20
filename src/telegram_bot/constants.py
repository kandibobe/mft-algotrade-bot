# constants.py
"""
Центральный файл для всех констант проекта.
Содержит команды, ключи для локализации, callback-данные,
идентификаторы API и другие статические значения.
"""

# =============================================================================
# --- Команды Бота ---
# =============================================================================
CMD_START = "start"
CMD_HELP = "help"
CMD_MENU = "menu"
CMD_SETTINGS = "settings"
CMD_CANCEL = "cancel"
CMD_CANCEL_EDIT = "cancel_edit"

# Аналитика
CMD_REPORT = "report"
CMD_SIGNAL = "signal"
CMD_TA = "ta"
CMD_ANALYTICS_CHAT = "chat"

# Рыночные данные
CMD_FUNDING = "funding"
CMD_TVL = "tvl"
CMD_FEAR_GREED = "feargreed"
CMD_GAS = "gas"
CMD_TRENDING = "trending"
CMD_VOLATILITY = "volatility"
CMD_MARKETCAP = "marketcap"
CMD_EVENTS = "events"

# Новости
CMD_CRYPTO_NEWS = "cryptonews"
CMD_NEWS = "news"

# Управление списками и портфелем
CMD_WATCHLIST = "watchlist"
CMD_ADDWATCH = "addwatch"
CMD_DELWATCH = "delwatch"
CMD_ALERTS = "alerts"
CMD_ADDALERT = "addalert"
CMD_DELALERT = "delalert"
CMD_PORTFOLIO = "portfolio"
CMD_PORTFOLIO_ADD = "p_add"
CMD_PORTFOLIO_DEL = "p_del"


# Прочее
CMD_LANGUAGE = "language"
CMD_EXPLAIN = "explain"
CMD_FEEDBACK = "feedback"
CMD_PREMIUM = "premium"


# =============================================================================
# --- Ключи для ReplyKeyboardMarkup (Главное меню) ---
# =============================================================================
BTN_KEY_MY_REPORT = "btn_my_report"
BTN_KEY_SIGNAL = "btn_signal"
BTN_KEY_VOLATILITY = "btn_volatility"
BTN_KEY_WATCHLIST = "btn_watchlist"
BTN_KEY_ALERTS = "btn_alerts"
BTN_KEY_SETTINGS = "btn_settings"
BTN_KEY_HELP = "btn_help"
BTN_KEY_FEAR_GREED = "btn_fear_greed"
BTN_KEY_CRYPTO_NEWS = "btn_crypto_news"
BTN_KEY_ANALYTICS_CHAT = "btn_analytics_chat"
BTN_KEY_PORTFOLIO = "btn_portfolio"


# =============================================================================
# --- CallbackData: Префиксы (строительные блоки) ---
# =============================================================================
CB_PREFIX_SETTINGS = "set_"
CB_PREFIX_LANG = "lang_"
CB_PREFIX_ALERT_ADD = "aa_"
CB_PREFIX_ALERT_DEL = "ad_"
CB_PREFIX_ALERT_EDIT = "ae_"
CB_PREFIX_WATCH_DEL = "wd_"
CB_PREFIX_REPORT_GRAPH = "rg_"
CB_PREFIX_QUICK_ALERT = "qa_"
CB_PREFIX_REPORT_NAV = "r_nav_"
CB_PREFIX_SIGNAL_DETAIL = "sig_det_"


# =============================================================================
# --- CallbackData: Полные строки (конкретные действия) ---
# =============================================================================
# Настройки
CB_MAIN_SETTINGS = f"{CB_PREFIX_SETTINGS}main"
CB_SETTINGS_PERIOD = f"{CB_PREFIX_SETTINGS}period"
CB_SETTINGS_ALERTS_TOGGLE = f"{CB_PREFIX_SETTINGS}alert_toggle"
CB_SETTINGS_LANG = f"{CB_PREFIX_SETTINGS}lang"
CB_ACTION_SET_PERIOD = f"{CB_PREFIX_SETTINGS}set_period_"
CB_ACTION_SHOW_PREMIUM = CMD_PREMIUM

# Язык
CB_ACTION_SET_LANG = f"{CB_PREFIX_LANG}set_"

# Watchlist
CB_ACTION_DEL_WATCH = f"{CB_PREFIX_WATCH_DEL}del_"

# Алерты: Добавление
CB_ACTION_CHOOSE_ALERT_TYPE = f"{CB_PREFIX_ALERT_ADD}type_"
CB_ALERT_COND_GT = f"{CB_PREFIX_ALERT_ADD}cond_gt"
CB_ALERT_COND_LT = f"{CB_PREFIX_ALERT_ADD}cond_lt"
CB_ACTION_SET_RSI_COND = f"{CB_PREFIX_ALERT_ADD}rsi_"
CB_ACTION_QUICK_ADD_ALERT = f"{CB_PREFIX_QUICK_ALERT}add_"

# Алерты: Удаление
CB_ACTION_DEL_ALERT = f"{CB_PREFIX_ALERT_DEL}del_"
CB_ACTION_DEL_ALERT_CONFIRMED = f"{CB_PREFIX_ALERT_DEL}yes_"
CB_ACTION_DEL_ALERT_CANCELLED = f"{CB_PREFIX_ALERT_DEL}no_"

# Алерты: Редактирование
CB_ACTION_EDIT_ALERT_START = f"{CB_PREFIX_ALERT_EDIT}start_"
CB_ACTION_EDIT_ALERT_SET_COND = f"{CB_PREFIX_ALERT_EDIT}set_cond_"
CB_ACTION_EDIT_ALERT_SET_VAL = f"{CB_PREFIX_ALERT_EDIT}set_val_"
CB_ACTION_EDIT_ALERT_SAVE = f"{CB_PREFIX_ALERT_EDIT}save_"
CB_ACTION_EDIT_ALERT_CANCEL_SINGLE = f"{CB_PREFIX_ALERT_EDIT}cancel_s_"

# Отчет
CB_ACTION_REPORT_GRAPH = f"{CB_PREFIX_REPORT_GRAPH}show_"
CB_REPORT_NAV_MAIN = f"{CB_PREFIX_REPORT_NAV}main"
CB_REPORT_NAV_CRYPTO = f"{CB_PREFIX_REPORT_NAV}crypto"
CB_REPORT_NAV_MACRO = f"{CB_PREFIX_REPORT_NAV}macro"
CB_REPORT_NAV_ONCHAIN = f"{CB_PREFIX_REPORT_NAV}onchain"
CB_REPORT_NAV_GRAPHS = f"{CB_PREFIX_REPORT_NAV}graphs"


# =============================================================================
# --- Состояния ConversationHandler ---
# =============================================================================
# Диалог добавления алерта
ASK_ASSET, CHOOSE_TYPE, ASK_PRICE_CONDITION, ASK_PRICE_VALUE, ASK_RSI_CONDITION = range(5)

# Диалог редактирования алерта
EDIT_ALERT_CHOICE, EDIT_ALERT_CONDITION, EDIT_ALERT_VALUE, EDIT_ALERT_CONFIRM = range(5, 9)

# Диалог аналитического чата
ANALYTICS_CHAT_STATE_START = 0


# =============================================================================
# --- Типы и Константы для Алертов ---
# =============================================================================
ALERT_TYPE_PRICE = "price"
ALERT_TYPE_RSI = "rsi"
ALERT_ID_REGEX_PART = r"(\d+)"
ALERT_QUICK_ADD_REGEX = r"^\s*([A-Z0-9]+)\s*([<>])\s*([0-9,.]+)\s*$"
ALERT_QUICK_ADD_PERCENT_REGEX = r"^\s*([A-Z0-9]+)\s*([+-])\s*([0-9,.]+)\s*%\s*$"


# =============================================================================
# --- Константы Базы Данных ---
# =============================================================================
DB_TABLE_USER_SETTINGS = "user_settings"
DB_TABLE_WATCHLIST = "user_watchlist"
DB_TABLE_PRICE_ALERTS = "user_price_alerts"
DB_TABLE_USER_PORTFOLIO = "user_portfolio"
DB_FIELD_IS_PREMIUM = "is_premium"


# =============================================================================
# --- Активы и API Идентификаторы ---
# =============================================================================
# Типы активов
ASSET_CRYPTO = "crypto"
ASSET_FOREX = "forex"
ASSET_INDEX = "index"

# Идентификаторы FRED
FRED_CPI = "CPIAUCNS"
FRED_M2 = "M2SL"
FRED_DXY = "DTWEXBGS"
FRED_FEDFUNDS = "FEDFUNDS"
FRED_UNRATE = "UNRATE"

# Идентификаторы CoinGecko
CG_BTC = "bitcoin"
CG_ETH = "ethereum"
CG_SOL = "solana"
CG_DOGE = "dogecoin"
CG_ADA = "cardano"
CG_XRP = "ripple"
CG_DOT = "polkadot"
CG_LINK = "chainlink"
CG_LTC = "litecoin"
CG_BNB = "binancecoin"
CG_GLOBAL_DATA = "global"

# Идентификаторы Glassnode
GLASSNODE_BTC_NET_TRANSFER_EXCHANGES = "transfers/net_transfer_volume_to_from_exchanges_sum"

# Идентификаторы Forex и Индексов
FOREX_PAIRS = ["EURUSD", "GBPUSD", "USDJPY"]
INDEX_SPY = "SPY"
INDEX_QQQ = "QQQ"

# Сводные словари и списки
MARKET_INDEX_SYMBOLS = {INDEX_SPY: "S&P 500", INDEX_QQQ: "Nasdaq 100"}
SUPPORTED_MACRO_FOR_REPORT = {
    "CPI": FRED_CPI,
    "M2": FRED_M2,
    "DXY": FRED_DXY,
    "FFR": FRED_FEDFUNDS,
    "UNRATE": FRED_UNRATE,
}

SUPPORTED_ASSETS = {
    "BTC": (ASSET_CRYPTO, CG_BTC),
    "ETH": (ASSET_CRYPTO, CG_ETH),
    "SOL": (ASSET_CRYPTO, CG_SOL),
    "DOGE": (ASSET_CRYPTO, CG_DOGE),
    "ADA": (ASSET_CRYPTO, CG_ADA),
    "XRP": (ASSET_CRYPTO, CG_XRP),
    "DOT": (ASSET_CRYPTO, CG_DOT),
    "LINK": (ASSET_CRYPTO, CG_LINK),
    "LTC": (ASSET_CRYPTO, CG_LTC),
    "BNB": (ASSET_CRYPTO, CG_BNB),
    "EURUSD": (ASSET_FOREX, "EURUSD"),
    "GBPUSD": (ASSET_FOREX, "GBPUSD"),
    "USDJPY": (ASSET_FOREX, "USDJPY"),
    INDEX_SPY: (ASSET_INDEX, INDEX_SPY),
    INDEX_QQQ: (ASSET_INDEX, INDEX_QQQ),
}
REVERSE_ASSET_MAP = {v[1]: k for k, v in SUPPORTED_ASSETS.items()}


# =============================================================================
# --- Конфигурационные Константы ---
# =============================================================================
MIN_POINTS_FOR_SIGNAL = 5
MIN_POINTS_FOR_GRAPH = 3


# =============================================================================
# --- Ключи для Локализации (Сообщения) ---
# =============================================================================

# Общие и системные сообщения
MSG_WELCOME = "MSG_WELCOME"
MSG_HELP = "MSG_HELP"
ERROR_ADDWATCH_LIMIT = "ERROR_ADDWATCH_LIMIT"
ERROR_ADDALERT_LIMIT = "ERROR_ADDALERT_LIMIT"
MSG_PREMIUM_AD_TEXT = "MSG_PREMIUM_AD_TEXT"
MSG_ERROR_FETCH = "error_fetch"
MSG_ERROR_FETCH_SHORT = "error_fetch_short"
MSG_ERROR_ANALYSIS = "error_analysis"
MSG_ERROR_GENERAL = "error_general"
MSG_ERROR_DB = "error_db"
MSG_LOADING = "loading_data"
MSG_DATA_STALE_FETCHING = "data_stale_fetching"
MSG_NO_DATA_AVAILABLE = "no_data_available"
MSG_NO_DATA_SHORT = "no_data_short"
MSG_CONFIG_ERROR = "config_error"
MSG_INVALID_API_KEY = "invalid_api_key"
MSG_RATE_LIMIT_ERROR = "rate_limit_error"
MSG_TIMEOUT_ERROR = "timeout_error"
MSG_DIALOG_CANCELLED_BY_COMMAND = "dialog_cancelled_by_command"
TEXT_ASSET_DEFAULT = "text_asset_default"
MSG_ERROR_DATA_FORMAT_DETAIL = "error_data_format_detail"

# Меню и Навигация
MSG_MENU_MAIN_HEADER = "menu_main_header"
MSG_MENU_CATEGORY_ANALYTICS = "menu_category_analytics"
MSG_MENU_CATEGORY_MARKET_DATA = "menu_category_market_data"
MSG_MENU_CATEGORY_LISTS = "menu_category_lists"
MSG_MENU_BTN_BACK = "menu_btn_back"

# Настройки
MSG_SETTINGS_MENU = "settings_menu"
MSG_SELECT_PERIOD = "settings_select_period"
MSG_PERIOD_SET = "settings_period_set"
MSG_ALERTS_ON = "settings_alerts_on"
MSG_ALERTS_OFF = "settings_alerts_off"
MSG_ALERT_TOGGLE_CONFIRM = "settings_alert_toggle_confirm"
MSG_LANG_SELECT = "lang_select"
MSG_LANG_SET = "lang_set"
MSG_RESTART_NEEDED = "restart_needed"

# Отчет (/report) и Графики
MSG_REPORT_DASHBOARD_HEADER = "report_dashboard_header"
MSG_REPORT_DATA_EXPIRED = "report_data_expired"
MSG_REPORT_BTN_CRYPTO = "report_btn_crypto"
MSG_REPORT_BTN_MACRO = "report_btn_macro"
MSG_REPORT_BTN_ONCHAIN = "report_btn_onchain"
MSG_REPORT_BTN_GRAPHS = "report_btn_graphs"
MSG_REPORT_BTN_BACK = "report_btn_back"
MSG_REPORT_GRAPH_PROMPT = "report_graph_prompt"
MSG_CRYPTO_HEADER = "report_crypto_header"
MSG_MACRO_HEADER = "report_macro_header"
MSG_INDEX_HEADER = "report_index_header"
MSG_FOREX_HEADER = "report_forex_header"
MSG_REPORT_ONCHAIN_HEADER = "report_onchain_header"
MSG_REPORT_ONCHAIN_NETFLOW = "report_onchain_netflow"
MSG_REPORT_ONCHAIN_INFLOW = "report_onchain_inflow"
MSG_REPORT_ONCHAIN_OUTFLOW = "report_onchain_outflow"
MSG_GRAPH_LOADING = "graph_loading"
MSG_GRAPH_ERROR_GENERAL = "graph_error_general"
MSG_NO_DATA_FOR_GRAPH = "error_no_graph_data"
MSG_GRAPH_DATA_INSUFFICIENT = "graph_data_insufficient"
MSG_ERROR_GRAPH_ONLY_CRYPTO = "error_graph_only_crypto"
MSG_REPORT_TA_HEADER = "report_ta_header"
MSG_REPORT_TA_RSI = "report_ta_rsi"
MSG_REPORT_TA_SMA_POSITION = "report_ta_sma_position"
MSG_REPORT_TA_PRICE_ABOVE_SMA = "report_ta_price_above_sma"
MSG_REPORT_TA_PRICE_BELOW_SMA = "report_ta_price_below_sma"
MSG_REPORT_TA_PRICE_BETWEEN_SMAS = "report_ta_price_between_smas"
MSG_REPORT_TA_NOT_ENOUGH_DATA = "report_ta_not_enough_data"

# Сигнал (/signal)
MSG_SIGNAL_HEADER = "signal_header"
MSG_SIGNAL_ERROR_POINTS = "signal_error_points"
MSG_SIGNAL_ERROR_NO_COMMON_DATES = "signal_error_no_common_dates"
MSG_SIGNAL_FNG_INFO = "signal_fng_info"
MSG_SIGNAL_BTC_DOMINANCE_CURRENT = "signal_btc_dominance_current"
SIGNAL_FINAL_SIGNAL = "signal_final_signal"
SIGNAL_DETAILS_HEADER = "signal_details_header"
SIGNAL_FACTOR_BTC_TREND = "signal_factor_btc_trend"
SIGNAL_FACTOR_CPI_TREND = "signal_factor_cpi_trend"
SIGNAL_FACTOR_DXY_TREND = "signal_factor_dxy_trend"
SIGNAL_FACTOR_FNG = "signal_factor_fng"
SIGNAL_FACTOR_CORR = "signal_factor_corr"
SIGNAL_FACTOR_BTC_DOMINANCE = "signal_factor_btc_dominance"
SIGNAL_DETAIL_BTC_UP = "signal_detail_btc_up"
SIGNAL_DETAIL_BTC_DOWN = "signal_detail_btc_down"
SIGNAL_DETAIL_BTC_FLAT = "signal_detail_btc_flat"
SIGNAL_DETAIL_CPI_UP = "signal_detail_cpi_up"
SIGNAL_DETAIL_CPI_DOWN = "signal_detail_cpi_down"
SIGNAL_DETAIL_CPI_FLAT = "signal_detail_cpi_flat"
SIGNAL_DETAIL_CPI_NA = "signal_detail_cpi_na"
SIGNAL_DETAIL_CPI_ONE_POINT = "signal_detail_cpi_one_point"
SIGNAL_DETAIL_CPI_ACCELERATES = "signal_detail_cpi_accelerates"
SIGNAL_DETAIL_CPI_DECELERATES = "signal_detail_cpi_decelerates"
SIGNAL_DETAIL_DXY_UP = "signal_detail_dxy_up"
SIGNAL_DETAIL_DXY_DOWN = "signal_detail_dxy_down"
SIGNAL_DETAIL_DXY_FLAT = "signal_detail_dxy_flat"
SIGNAL_DETAIL_CORR_NEG_DXY_UP = "signal_detail_corr_neg_dxy_up"
SIGNAL_DETAIL_CORR_NEG_DXY_DOWN = "signal_detail_corr_neg_dxy_down"
SIGNAL_DETAIL_CORR_NONE = "signal_detail_corr_none"
SIGNAL_DETAIL_FNG_EXTREME_FEAR = "signal_detail_fng_extreme_fear"
SIGNAL_DETAIL_FNG_FEAR = "signal_detail_fng_fear"
SIGNAL_DETAIL_FNG_NEUTRAL = "signal_detail_fng_neutral"
SIGNAL_DETAIL_FNG_GREED = "signal_detail_fng_greed"
SIGNAL_DETAIL_FNG_EXTREME_GREED = "signal_detail_fng_extreme_greed"
SIGNAL_DETAIL_FNG_NA = "signal_detail_fng_na"
SIGNAL_DETAIL_BTC_DOMINANCE_NA = "signal_detail_btc_dominance_na"
SIGNAL_DETAIL_BTC_DOMINANCE_RISING = "signal_detail_btc_dominance_rising"
SIGNAL_DETAIL_BTC_DOMINANCE_FALLING = "signal_detail_btc_dominance_falling"
SIGNAL_STRONG_BUY = "signal_strong_buy"
SIGNAL_WEAK_BUY = "signal_weak_buy"
SIGNAL_HOLD = "signal_hold"
SIGNAL_WEAK_SELL = "signal_weak_sell"
SIGNAL_STRONG_SELL = "signal_strong_sell"

# Алерты (/alerts, /addalert)
MSG_ALERTS_HEADER = "alerts_header"
MSG_ALERT_ITEM_PRICE = "alert_item_price"
MSG_ALERT_ITEM_RSI = "alert_item_rsi"
MSG_ALERT_EMPTY = "alert_empty"
ADDALERT_PROMPT_ASSET = "addalert_prompt_asset"
MSG_ADDALERT_PROMPT_TYPE = "addalert_prompt_type"
MSG_ADDALERT_PROMPT_CONDITION_BASE = "addalert_prompt_condition_base"
MSG_ADDALERT_CURRENT_PRICE = "addalert_current_price"
MSG_ADDALERT_PROMPT_VALUE = "addalert_prompt_value"
MSG_ADDALERT_PROMPT_RSI_CONDITION = "addalert_prompt_rsi_condition"
MSG_ADDALERT_SUCCESS = "addalert_success"
MSG_ADDALERT_SUCCESS_RSI = "addalert_success_rsi"
MSG_ADDALERT_QUICK_SUCCESS = "addalert_quick_success"
MSG_ADDALERT_PERCENT_SUCCESS = "addalert_percent_success"
MSG_ADDALERT_FAIL_LIMIT = "addalert_fail_limit"
MSG_ADDALERT_FAIL_INVALID_ASSET = "addalert_fail_invalid_asset"
MSG_ADDALERT_FAIL_RSI_ONLY_CRYPTO = "addalert_fail_rsi_only_crypto"
MSG_ADDALERT_FAIL_INVALID_VALUE = "addalert_fail_invalid_value"
MSG_ADDALERT_FAIL_PERCENT_VALUE = "addalert_fail_percent_value"
MSG_ADDALERT_FAIL_FETCH_PRICE = "addalert_fail_fetch_price"
MSG_ADDALERT_FAIL_GENERIC = "addalert_fail_generic"
MSG_ADDALERT_CANCEL = "addalert_cancel"
MSG_DELALERT_PROMPT = "delalert_prompt"
MSG_DELALERT_SUCCESS = "delalert_success"
MSG_DELALERT_FAIL_NOTFOUND = "delalert_fail_notfound"
MSG_DELALERT_FAIL_INVALID_ID = "delalert_fail_invalid_id"
MSG_DELALERT_CONFIRM = "delalert_confirm"
MSG_PRICE_ALERT_TRIGGERED = "price_alert_triggered"
MSG_RSI_ALERT_TRIGGERED = "rsi_alert_triggered"
MSG_ERROR_VALUE_MUST_BE_POSITIVE = "error_value_must_be_positive"
MSG_EDIT_ALERT_PROMPT_CHOICE = "edit_alert_prompt_choice"
MSG_EDIT_ALERT_PROMPT_NEW_CONDITION = "edit_alert_prompt_new_condition"
MSG_EDIT_ALERT_PROMPT_NEW_VALUE = "edit_alert_prompt_new_value"
MSG_EDIT_ALERT_PROMPT_CONFIRM = "edit_alert_prompt_confirm"
MSG_EDIT_ALERT_NO_CHANGES = "edit_alert_no_changes"
MSG_EDIT_ALERT_SUCCESS = "edit_alert_success"
MSG_EDIT_ALERT_CANCELLED = "edit_alert_cancelled"
BTN_ALERT_TYPE_PRICE = "btn_alert_type_price"
BTN_ALERT_TYPE_RSI = "btn_alert_type_rsi"
BTN_RSI_COND_GT70 = "btn_rsi_cond_gt70"
BTN_RSI_COND_LT30 = "btn_rsi_cond_lt30"
BTN_EDIT = "btn_edit"
BTN_DELETE = "btn_delete"
BTN_CANCEL = "btn_cancel"
BTN_YES = "btn_yes"
BTN_NO = "btn_no"
BTN_CONDITION_GT = "btn_condition_gt"
BTN_CONDITION_LT = "btn_condition_lt"
BTN_EDIT_CONDITION = "btn_edit_condition"
BTN_EDIT_VALUE = "btn_edit_value"

# Watchlist (/watchlist)
MSG_WATCHLIST_HEADER = "watchlist_header"
MSG_WATCHLIST_ITEM = "watchlist_item"
MSG_WATCHLIST_EMPTY = "watchlist_empty"
ADDWATCH_PROMPT = "addwatch_prompt"
MSG_ADDWATCH_SUCCESS = "addwatch_success"
MSG_ADDWATCH_FAIL_LIMIT = "addwatch_fail_limit"
MSG_ADDWATCH_FAIL_EXISTS = "addwatch_fail_exists"
MSG_ADDWATCH_FAIL_INVALID = "addwatch_fail_invalid"
MSG_DELWATCH_PROMPT = "delwatch_prompt"
MSG_DELWATCH_SUCCESS = "delwatch_success"
MSG_DELWATCH_FAIL_NOTFOUND = "delwatch_fail_notfound"

# Portfolio (/portfolio)
MSG_PORTFOLIO_HEADER = "portfolio_header"
MSG_PORTFOLIO_ITEM = "portfolio_item"
MSG_PORTFOLIO_ITEM_NO_PRICE = "portfolio_item_no_price"
MSG_PORTFOLIO_TOTAL = "portfolio_total"
MSG_PORTFOLIO_EMPTY = "portfolio_empty"
MSG_P_ADD_PROMPT = "p_add_prompt"
MSG_P_ADD_SUCCESS = "p_add_success"
MSG_P_ADD_FAIL_INVALID_FORMAT = "p_add_fail_invalid_format"
MSG_P_DEL_PROMPT = "p_del_prompt"
MSG_P_DEL_SUCCESS = "p_del_success"
MSG_P_DEL_FAIL_NOTFOUND = "p_del_fail_notfound"

# Технический Анализ (/ta)
MSG_TA_HEADER = "ta_header"
MSG_TA_PROMPT = "ta_prompt"
MSG_TA_INVALID_ASSET = "ta_invalid_asset"
MSG_TA_DATA_ERROR = "ta_data_error"
MSG_TA_RSI_LINE = "ta_rsi_line"
MSG_TA_RSI_DESC_OVERBOUGHT = "ta_rsi_desc_overbought"
MSG_TA_RSI_DESC_OVERSOLD = "ta_rsi_desc_oversold"
MSG_TA_RSI_DESC_NEUTRAL = "ta_rsi_desc_neutral"
MSG_TA_MACD_LINE = "ta_macd_line"
MSG_TA_MACD_DESC_BULLISH = "ta_macd_desc_bullish"
MSG_TA_MACD_DESC_BEARISH = "ta_macd_desc_bearish"
MSG_TA_BBANDS_LINE = "ta_bbands_line"
MSG_TA_BBANDS_DESC_ABOVE = "ta_bbands_desc_above"
MSG_TA_BBANDS_DESC_BELOW = "ta_bbands_desc_below"
MSG_TA_BBANDS_DESC_INSIDE = "ta_bbands_desc_inside"

# Команды с рыночными данными (Misc)
MSG_VOLATILITY_HEADER = "volatility_header"
MSG_VOLATILITY_GAINER = "volatility_gainer"
MSG_VOLATILITY_LOSER = "volatility_loser"
MSG_VOLATILITY_SEPARATOR = "volatility_separator"
MSG_MARKETCAP_HEADER = "marketcap_header"
MSG_MARKETCAP_ITEM = "marketcap_item"
MSG_MARKETCAP_INVALID_N = "marketcap_invalid_n"
MSG_FNG_HEADER = "fng_header"
MSG_FNG_DATA = "fng_data"
MSG_GAS_HEADER = "gas_header"
MSG_GAS_DATA = "gas_data"
MSG_GAS_API_KEY_MISSING = "gas_api_key_missing"
MSG_TRENDING_HEADER = "trending_header"
MSG_TRENDING_ITEM = "trending_item"
MSG_FUNDING_HEADER = "funding_header"
MSG_FUNDING_HIGH_RATES_HEADER = "funding_high_rates_header"
MSG_FUNDING_LOW_RATES_HEADER = "funding_low_rates_header"
MSG_FUNDING_ITEM = "funding_item"
MSG_TVL_PROMPT = "tvl_prompt"
MSG_TVL_HEADER = "tvl_header"
MSG_TVL_NOT_FOUND = "tvl_not_found"
MSG_EVENTS_HEADER = "events_header"
MSG_EVENTS_ITEM_FORMAT = "events_item_format"
MSG_EVENTS_NO_EVENTS = "events_no_events"
MSG_EVENTS_PERIOD_TODAY = "events_period_today"
MSG_EVENTS_PERIOD_TOMORROW = "events_period_tomorrow"
MSG_EVENTS_PERIOD_THIS_WEEK = "events_period_this_week"
MSG_MENU_CMD_TA = "menu_cmd_ta"
MSG_MENU_CMD_GAS = "menu_cmd_gas"
MSG_MENU_CMD_TRENDING = "menu_cmd_trending"
MSG_MENU_CMD_FUNDING = "menu_cmd_funding"
MSG_MENU_CMD_TVL = "menu_cmd_tvl"

# Новости (/news, /cryptonews)
MSG_CRYPTO_NEWS_HEADER = "crypto_news_header"
MSG_CRYPTO_NEWS_ITEM = "crypto_news_item"
MSG_CRYPTO_NEWS_API_KEY_MISSING = "crypto_news_api_key_missing"
MSG_CRYPTO_NEWS_NO_NEWS = "crypto_news_no_news"
MSG_NEWS_PROMPT = "news_prompt"
MSG_NEWS_HEADER = "news_header"
MSG_NEWS_ITEM_WITH_SOURCE = "news_item_with_source"
MSG_NEWS_API_KEY_MISSING = "news_api_key_missing"
MSG_NEWS_NO_RESULTS = "news_no_results"
MSG_NEWS_QUERY_TOO_SHORT = "news_query_too_short"

# Прочие команды (/explain, /feedback, /premium)
MSG_EXPLAIN_PROMPT = "explain_prompt"
MSG_EXPLAIN_NOT_FOUND = "explain_not_found"
KEY_EXPLAIN_TERMS_COLLECTION = "explain_terms_collection"
MSG_PREMIUM_INFO = "premium_info"
MSG_PREMIUM_STATUS_ACTIVE = "premium_status_active"
MSG_PREMIUM_STATUS_INACTIVE = "premium_status_inactive"
MSG_PREMIUM_AD_TEXT = "premium_ad_text"
MSG_FEEDBACK_PROMPT = "feedback_prompt"
MSG_FEEDBACK_SENT = "feedback_sent"
MSG_FEEDBACK_ERROR = "feedback_error"

# Аналитический Чат (/chat)
MSG_ANALYTICS_CHAT_WELCOME = "analytics_chat_welcome"
MSG_ANALYTICS_CHAT_PROMPT_ASSET = "analytics_chat_prompt_asset"
MSG_ANALYTICS_CHAT_PROMPT_COMMAND = "analytics_chat_prompt_command"
MSG_ANALYTICS_CHAT_INVALID_ASSET = "analytics_chat_invalid_asset"
MSG_ANALYTICS_CHAT_UNKNOWN_COMMAND = "analytics_chat_unknown_command"
MSG_ANALYTICS_CHAT_CANCELLED = "analytics_chat_cancelled"
MSG_ANALYTICS_CHAT_COMMAND_LIST = "analytics_chat_command_list"
MSG_ANALYTICS_CHAT_NO_ASSET_SELECTED = "analytics_chat_no_asset_selected"
MSG_ANALYTICS_CHAT_ASSET_CHANGED = "analytics_chat_asset_changed"
MSG_ANALYTICS_CHAT_COMMAND_NOT_SUPPORTED_FOR_ASSET = (
    "analytics_chat_command_not_supported_for_asset"
)

# Статусы API
API_STATUS_OK = "api_status_ok"
API_STATUS_NO_DATA = "api_status_no_data"
API_STATUS_API_ERROR = "api_status_api_error"
API_STATUS_TIMEOUT = "api_status_timeout"
API_STATUS_NETWORK_ERROR = "api_status_network_error"
API_STATUS_RATE_LIMIT = "api_status_rate_limit"
API_STATUS_NOT_FOUND = "api_status_not_found"
API_STATUS_FORBIDDEN = "api_status_forbidden"
API_STATUS_INVALID_KEY = "api_status_invalid_key"
API_STATUS_BAD_REQUEST = "api_status_bad_request"
API_STATUS_CONFIG_ERROR = "api_status_config_error"
API_STATUS_FORMAT_ERROR = "api_status_format_error"
API_STATUS_UNKNOWN_ERROR = "api_status_unknown_error"

# Дайджест
MSG_DAILY_DIGEST_HEADER = "digest_header"
MSG_DIGEST_CRYPTO_NEWS = "digest_crypto_news"
MSG_DAILY_DIGEST_VOLATILITY = "digest_volatility"
MSG_DAILY_DIGEST_WATCHLIST = "digest_watchlist"
MSG_DAILY_DIGEST_WATCHLIST_ITEM_FORMAT = "digest_watchlist_item_format"
MSG_DAILY_DIGEST_NO_WATCHLIST = "digest_no_watchlist"
MSG_DIGEST_FNG_LINE = "digest_fng_line"
MSG_DAILY_DIGEST_FOOTER = "digest_footer"

# Графики
GRAPH_TITLE_BTC_TREND = "graph_title_btc_trend"
GRAPH_TITLE_DXY_TREND = "graph_title_dxy_trend"
GRAPH_TITLE_CPI_TREND = "graph_title_cpi_trend"
TITLE_WATCHLIST = "TITLE_WATCHLIST"
