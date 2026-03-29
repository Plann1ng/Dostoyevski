# =============================================================================
# BREAKER + THREE TAP v1.0 — LIVE BOT CONFIGURATION
# =============================================================================
# 
# SETUP INSTRUCTIONS:
# 1. Go to https://demo.binance.com
# 2. Log in with your Binance account
# 3. Go to API Management: https://demo.binance.com/en/my/settings/api-management
# 4. Create a new API key (name it "breaker_bot_v1" or similar)
# 5. Copy the API Key and Secret below
# 6. IMPORTANT: This is DEMO MODE — no real money at risk
#
# DO NOT commit this file to git with real API keys.
# =============================================================================

# --- Binance Demo Mode API Credentials ---
# Replace these with your actual demo API keys
API_KEY = "YOUR_DEMO_API_KEY_HERE"
API_SECRET = "YOUR_DEMO_API_SECRET_HERE"

# --- Binance Demo Mode Endpoints ---
# These are DIFFERENT from live Binance endpoints
REST_BASE_URL = "https://demo-api.binance.com"
WS_STREAM_URL = "wss://demo-stream.binance.com/ws"
WS_API_URL = "wss://demo-ws-api.binance.com/ws-api/v3"

# --- Trading Parameters (committee-validated) ---
TIMEFRAME = "1h"                    # Validated timeframe
SYMBOLS = [                         # Top performers from backtest
    "BTCUSDT",
    "SOLUSDT",
    "TRXUSDT",
    "DOTUSDT",
    "ETCUSDT",
    "DOGEUSDT",
    "ETHUSDT",
    "BNBUSDT",
    "LTCUSDT",
    "ADAUSDT",
]

# --- Risk Management (committee rules) ---
RISK_PER_TRADE = 0.005              # 0.5% of account per trade (conservative start)
MIN_RR = 1.5                        # Minimum risk:reward ratio
MAX_POSITIONS_PER_SYMBOL = 1        # Never more than 1 position per coin
MIN_BARS_BETWEEN_ENTRIES = 6        # Minimum 6 candles between entries on same coin
STOP_BUFFER_ATR = 0.3               # Stop buffer in ATR multiples

# --- Engine Parameters (locked from backtest) ---
SWING_LOOKBACK = 5
ATR_PERIOD = 14
MIN_SWING_ATR = 0.5
EQUAL_THRESHOLD_ATR = 0.1
BREAKER_MAX_RETEST = 48
BREAKER_MIN_DISPLACEMENT = 1.0
TAP_SWEEP_MIN = 0.2
TAP_SWEEP_MAX = 2.0
TAP_MAX_RETEST = 48
QUALITY_GATE = 50
SMA_PERIOD = 200                    # Fetched from daily via API

# --- Rolling Window (production optimization) ---
ROLLING_WINDOW = 2000               # Last 2000 bars for structure analysis

# --- Logging ---
LOG_FILE = "trading_bot.log"
TRADE_JOURNAL = "trade_journal.csv"
