# Breaker + Three Tap v1.0 — Live Demo Bot Setup Guide

## Quick start (5 minutes)

### Step 1: Install dependencies

```bash
pip install requests websocket-client numpy
```

That's it — no heavy frameworks. The bot uses raw REST calls for maximum transparency.

### Step 2: Get Binance Demo API keys

1. Go to https://demo.binance.com
2. Log in with your regular Binance account
3. Navigate to API Management: https://demo.binance.com/en/my/settings/api-management
4. Click "Create API" and name it (e.g., "breaker_bot_v1")
5. Copy both the API Key and Secret Key

Important: This is DEMO MODE. It uses virtual funds with realistic market prices. No real money is involved.

### Step 3: Configure the bot

Open `config.py` and replace the placeholder keys:

```python
API_KEY = "paste_your_demo_api_key_here"
API_SECRET = "paste_your_demo_api_secret_here"
```

All other settings are pre-configured with the committee-validated parameters. Don't change them unless you know what you're doing.

### Step 4: Run the bot

```bash
python live_bot.py
```

Or with specific symbols:

```bash
python live_bot.py --symbols BTCUSDT SOLUSDT TRXUSDT
```

The bot will:
1. Connect to Binance Demo Mode
2. Load 2,000 historical 1H candles per symbol
3. Compute the 200-day SMA for each symbol
4. Start monitoring for new candle closes (checks every 30 seconds)
5. When a setup triggers, log the signal and execute on the next candle open

### Step 5: Monitor

The bot logs everything to both console and `trading_bot.log`. Every trade is recorded in `trade_journal.csv`.

Press Ctrl+C to stop the bot gracefully.

---

## What happens when the bot runs

Every 30 seconds, the bot checks if a new 1H candle has closed. When one closes:

1. If there's a pending signal from the previous candle, it executes at the new candle's open price (market order for entry, OCO order for stop/target)
2. It runs the full strategy pipeline on the updated candle data
3. If a Breaker or Three Tap setup triggers AND passes all filters (quality >= 50, R:R >= 1.5, 200 SMA alignment, stop safety), it stores the signal
4. The signal executes at the NEXT candle's open (not the current close)

This is the corrected execution model that eliminates same-bar entry bias.

---

## Important notes for demo trading

### Spot mode = long only
Binance Spot Demo does not support true shorting. The bot will:
- Execute LONG signals with market buy + OCO exit
- LOG SHORT signals to the journal but not execute them

The long-only portion produced +297.5R in the backtest (PF 1.34) — this alone is a viable system.

To trade shorts, you would need Binance Futures Demo Mode. The strategy logic supports it; only the order execution layer needs to change.

### Order flow
- Entry: Market order (guarantees fill, ~0.1% commission)
- Exit: OCO order (take-profit limit + stop-loss trigger placed together)
- When either the TP or SL fills, the other is automatically cancelled

### Position sizing
At 0.5% risk per trade with a $10,000 demo balance:
- Risk amount = $50 per trade
- If BTC stop distance is $500, position size = $50 / $500 = 0.1 BTC
- The bot calculates this automatically based on current balance and stop distance

---

## File structure

```
trading_system/
├── config.py                  # API keys + all parameters
├── live_bot.py                # Main bot (run this)
├── binance_client.py          # Exchange API wrapper
├── market_structure.py        # Core: swing detection, MSB, regimes
├── liquidity_mapper.py        # Core: FTA targets, liquidity levels
├── breaker_detector.py        # Core: Breaker setup detection
├── additional_setups.py       # Core: Three Tap + archived setups
├── backtest_engine.py         # Backtest: results tracking
├── strategy_v1_pine.txt       # TradingView Pine Script
├── SYSTEM_DOCUMENTATION_v1.md # Full system documentation
├── SETUP_GUIDE.md             # This file
│
├── trading_bot.log            # Generated: runtime logs
└── trade_journal.csv          # Generated: every trade recorded
```

---

## Troubleshooting

**"Failed to connect to Binance Demo"**
- Check that your API keys in config.py are from demo.binance.com (not the live site)
- Verify the keys have spot trading permission enabled

**"Quantity too small"**
- The demo account may not have enough USDT balance
- Go to demo.binance.com → Assets → Reset to get fresh virtual funds

**No signals after hours of running**
- This is normal. On 1H timeframe, the system generates ~25 trades per year per coin
- That's roughly 1 trade every 2 weeks per coin
- With 10 coins, expect 1-2 signals per week on average

**Bot shows "SHORT signal logged" but doesn't trade**
- This is by design. Spot mode can only go long
- Short signals are logged for tracking. Their performance validates the strategy even if not executed

---

## Monitoring checklist (daily)

1. Check `trading_bot.log` for errors
2. Check `trade_journal.csv` for new entries
3. Verify on demo.binance.com that positions match the journal
4. After 2 weeks: compare signals against TradingView Pine Script on the same charts — they should match

---

## Scaling to real money

After 1-3 months of demo trading with consistent results:

1. Compare demo results against the backtest expectations (+0.196R per trade)
2. If results are within range, create API keys on live Binance (not demo)
3. Change `config.py`:
   - REST_BASE_URL = "https://api.binance.com"
   - WS_STREAM_URL = "wss://stream.binance.com/ws"
   - API_KEY / API_SECRET = your live keys
4. Start with RISK_PER_TRADE = 0.005 (0.5%) and 2-3 coins only
5. Scale gradually over months, never exceeding 1% risk per trade
