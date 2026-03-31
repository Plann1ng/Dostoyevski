"""
BREAKER + THREE TAP v1.0 — LIVE DEMO TRADING BOT
===================================================

This is the production bot. It:
1. Connects to Binance Demo Mode (paper trading with real market data)
2. Fetches historical candles to build market structure
3. Monitors for new candle closes on 1H timeframe
4. Runs the locked Breaker + Three Tap pipeline
5. Places market orders for entries, OCO orders for exits
6. Logs every decision to a trade journal

USAGE:
  1. Edit config.py with your Binance Demo API keys
  2. pip install requests websocket-client numpy
  3. python live_bot.py

  Or with custom symbols:
  python live_bot.py --symbols BTCUSDT SOLUSDT ETHUSDT

IMPORTANT:
  This runs on Binance DEMO MODE — no real money.
  Prices are realistic but fills are simulated.
  The bot will run continuously until you stop it (Ctrl+C).

Requirements:
  pip install requests websocket-client numpy
"""

import argparse
import copy
import csv
import json
import logging
import os
import sys
import time
import threading
from datetime import datetime, timezone
from typing import Dict, List, Optional

import numpy as np

# Add project directory to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from config import (
    API_KEY, API_SECRET, REST_BASE_URL, WS_STREAM_URL,
    TIMEFRAME, SYMBOLS, RISK_PER_TRADE, MIN_RR,
    MAX_POSITIONS_PER_SYMBOL, MIN_BARS_BETWEEN_ENTRIES,
    STOP_BUFFER_ATR, SWING_LOOKBACK, ATR_PERIOD, MIN_SWING_ATR,
    EQUAL_THRESHOLD_ATR, BREAKER_MAX_RETEST, BREAKER_MIN_DISPLACEMENT,
    QUALITY_GATE, SMA_PERIOD, ROLLING_WINDOW, LOG_FILE, TRADE_JOURNAL,
)
from binance_client import BinanceDemoClient, fetch_full_klines
from market_structure import MarketStructureEngine
from breaker_detector import BreakerDetector, BreakerType, SignalStatus
from liquidity_mapper import LiquidityMapper
from additional_setups import ThreeTapDetector, TradeSetup, SetupType

# =============================================================================
# Logging
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("live_bot")


# =============================================================================
# Symbol State — tracks candles + positions for each coin
# =============================================================================

class SymbolState:
    def __init__(self, symbol: str):
        self.symbol = symbol
        self.candles: List[dict] = []       # Historical candles
        self.last_entry_bar: int = -100     # Bar index of last entry
        self.position: Optional[dict] = None  # Active position info
        self.pending_signal: Optional[dict] = None  # Signal waiting for next bar

    @property
    def n(self) -> int:
        return len(self.candles)

    def arrays(self):
        """Convert candle list to numpy arrays for the engine."""
        o = np.array([c["open"] for c in self.candles])
        h = np.array([c["high"] for c in self.candles])
        l = np.array([c["low"] for c in self.candles])
        c = np.array([c_["close"] for c_ in self.candles])
        v = np.array([c_["volume"] for c_ in self.candles])
        return o, h, l, c, v


# =============================================================================
# Core Analysis — runs the locked pipeline
# =============================================================================

class StrategyEngine:
    """
    Encapsulates the committee-validated strategy.
    All parameters are from config.py (locked from backtest).
    """

    def __init__(self):
        self.structure = MarketStructureEngine(
            swing_lookback=SWING_LOOKBACK,
            atr_period=ATR_PERIOD,
            min_swing_atr_multiple=MIN_SWING_ATR,
            equal_threshold_atr=EQUAL_THRESHOLD_ATR,
        )
        self.breaker_det = BreakerDetector(
            max_bars_to_retest=BREAKER_MAX_RETEST,
            zone_buffer_atr=0.1,
            min_move_from_zone_atr=BREAKER_MIN_DISPLACEMENT,
        )
        self.liq_mapper = LiquidityMapper(
            equal_tolerance_atr=0.15,
            untapped_lookback=200,
            max_levels=20,
        )
        self.tap_det = ThreeTapDetector()

    def analyze(self, sym_state: SymbolState, daily_sma: float) -> Optional[dict]:
        """
        Run the full pipeline on the current candle data.
        Returns a signal dict if a setup triggers, None otherwise.

        This is called when a NEW candle closes. The signal, if any,
        will be executed at the OPEN of the NEXT candle (next-bar-open model).
        """
        if sym_state.n < ROLLING_WINDOW:
            return None

        # Use rolling window for structure analysis
        start = max(0, sym_state.n - ROLLING_WINDOW)
        o, h, l, c, v = sym_state.arrays()
        o_w, h_w, l_w, c_w, v_w = o[start:], h[start:], l[start:], c[start:], v[start:]

        current_price = c[-1]
        current_bar = sym_state.n - 1

        # Structure analysis on rolling window
        state = self.structure.analyze(o_w, h_w, l_w, c_w, v_w)
        atr = self.structure._calculate_atr(h_w, l_w, c_w)

        if len(atr) == 0:
            return None

        atr_val = atr[-1]
        if atr_val <= 0:
            return None

        # Collect setups that trigger on the LAST bar of the window
        last_bar_idx = len(c_w) - 1
        setups = []

        # Breaker setups
        breakers = self.breaker_det.detect(state, h_w, l_w, c_w, atr, htf_bias="neutral")
        for bk in breakers:
            if bk.status != SignalStatus.TRIGGERED or bk.trigger_index is None:
                continue
            if bk.trigger_index != last_bar_idx:
                continue

            direction = "short" if bk.breaker_type == BreakerType.BEARISH else "long"
            stop = (bk.zone_high + atr_val * STOP_BUFFER_ATR
                    if bk.breaker_type == BreakerType.BEARISH
                    else bk.zone_low - atr_val * STOP_BUFFER_ATR)

            levels = self.liq_mapper.map_liquidity(state, h_w, l_w, c_w, atr, current_bar=last_bar_idx)
            fta = self.liq_mapper.find_fta(levels, bk.entry_price, direction, stop)

            if fta and fta.rr_with_stop and fta.rr_with_stop >= MIN_RR:
                target = fta.level.price
            else:
                risk = abs(bk.entry_price - stop)
                if risk <= 0:
                    continue
                target = (bk.entry_price - risk * MIN_RR if direction == "short"
                          else bk.entry_price + risk * MIN_RR)

            setups.append({
                "type": "breaker",
                "direction": direction,
                "signal_price": bk.entry_price,
                "stop": stop,
                "target": target,
                "quality": bk.quality_score,
                "atr": atr_val,
            })

        # Three Tap setups
        taps = self.tap_det.detect(state, h_w, l_w, c_w, atr)
        for tap in taps:
            if tap.trigger_index != last_bar_idx:
                continue
            setups.append({
                "type": "three_tap",
                "direction": tap.direction,
                "signal_price": tap.entry_price,
                "stop": tap.stop_price,
                "target": tap.target_price,
                "quality": tap.quality_score,
                "atr": atr_val,
            })

        if not setups:
            return None

        # Sort by quality, pick best
        setups.sort(key=lambda s: s["quality"], reverse=True)

        for setup in setups:
            # Quality gate
            if setup["quality"] < QUALITY_GATE:
                continue

            # R:R gate
            risk = abs(setup["signal_price"] - setup["stop"])
            reward = abs(setup["target"] - setup["signal_price"])
            if risk <= 0 or reward / risk < MIN_RR:
                continue

            # SMA filter (longs only above, shorts only below)
            if daily_sma > 0:
                if current_price > daily_sma and setup["direction"] == "short":
                    continue
                if current_price < daily_sma and setup["direction"] == "long":
                    continue

            # Min bars between entries
            bars_since_last = current_bar - sym_state.last_entry_bar
            if bars_since_last < MIN_BARS_BETWEEN_ENTRIES:
                continue

            # Max positions
            if sym_state.position is not None:
                continue

            # Stop safety check
            if setup["type"] == "breaker":
                levels = self.liq_mapper.map_liquidity(state, h_w, l_w, c_w, atr, current_bar=last_bar_idx)
                dangerous = self.liq_mapper.check_liquidity_near_stop(
                    levels, setup["stop"], atr_val, setup["direction"]
                )
                if len(dangerous) >= 2:
                    continue

            return setup

        return None


# =============================================================================
# Trade Journal
# =============================================================================

def init_journal():
    """Create the trade journal CSV if it doesn't exist."""
    if not os.path.exists(TRADE_JOURNAL):
        with open(TRADE_JOURNAL, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "timestamp", "symbol", "setup_type", "direction",
                "entry_price", "stop_price", "target_price",
                "quantity", "risk_r", "quality", "status", "notes"
            ])


def log_trade(symbol, setup_type, direction, entry_price, stop_price,
              target_price, quantity, quality, status, notes=""):
    with open(TRADE_JOURNAL, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            datetime.now(timezone.utc).isoformat(),
            symbol, setup_type, direction,
            f"{entry_price:.8f}", f"{stop_price:.8f}", f"{target_price:.8f}",
            f"{quantity:.8f}", "1R", f"{quality:.1f}", status, notes,
        ])


# =============================================================================
# Main Bot Loop
# =============================================================================

class TradingBot:
    def __init__(self, symbols: List[str]):
        self.client = BinanceDemoClient(API_KEY, API_SECRET, REST_BASE_URL)
        self.engine = StrategyEngine()
        self.symbols = symbols
        self.states: Dict[str, SymbolState] = {}
        self.daily_smas: Dict[str, float] = {}
        self.running = False

    def initialize(self):
        """Load historical data and compute SMA for all symbols."""
        logger.info("=" * 60)
        logger.info("BREAKER + THREE TAP v1.0 — INITIALIZING")
        logger.info(f"Mode: Binance DEMO (paper trading)")
        logger.info(f"Symbols: {', '.join(self.symbols)}")
        logger.info(f"Timeframe: {TIMEFRAME}")
        logger.info(f"Risk per trade: {RISK_PER_TRADE*100:.1f}%")
        logger.info("=" * 60)

        # Check account balance
        try:
            balance = self.client.get_balance("USDT")
            logger.info(f"Demo account USDT balance: {balance:.2f}")
        except Exception as e:
            logger.error(f"Failed to connect to Binance Demo: {e}")
            logger.error("Check your API keys in config.py")
            return False

        # Load historical candles for each symbol
        for symbol in self.symbols:
            logger.info(f"Loading {ROLLING_WINDOW} candles for {symbol}...")
            try:
                klines = fetch_full_klines(
                    self.client, symbol, TIMEFRAME, total_bars=ROLLING_WINDOW
                )
                state = SymbolState(symbol)
                state.candles = klines
                self.states[symbol] = state
                logger.info(f"  {symbol}: {len(klines)} candles loaded "
                           f"({klines[0]['timestamp']} → {klines[-1]['timestamp']})")

                # Fetch daily SMA
                daily_klines = fetch_full_klines(
                    self.client, symbol, "1d", total_bars=SMA_PERIOD + 10
                )
                if len(daily_klines) >= SMA_PERIOD:
                    closes = [k["close"] for k in daily_klines[-SMA_PERIOD:]]
                    self.daily_smas[symbol] = sum(closes) / len(closes)
                    logger.info(f"  {symbol}: 200d SMA = {self.daily_smas[symbol]:.2f}")
                else:
                    self.daily_smas[symbol] = 0.0
                    logger.warning(f"  {symbol}: Not enough daily data for SMA")

            except Exception as e:
                logger.error(f"  {symbol}: Failed to load — {e}")
                continue

        init_journal()
        logger.info(f"\nInitialized {len(self.states)} symbols. Bot ready.")
        return True

    def process_new_candle(self, symbol: str, candle: dict):
        """
        Called when a new 1H candle closes.
        This is the core trading loop for one symbol.
        """
        state = self.states.get(symbol)
        if state is None:
            return

        # Execute any pending signal from the PREVIOUS candle
        # (next-bar-open execution model)
        if state.pending_signal is not None:
            self._execute_signal(symbol, state, candle["open"])
            state.pending_signal = None

        # Add the new candle
        state.candles.append(candle)

        # Trim to rolling window
        if len(state.candles) > ROLLING_WINDOW + 100:
            state.candles = state.candles[-ROLLING_WINDOW:]

        # Run the strategy pipeline
        sma = self.daily_smas.get(symbol, 0.0)
        signal = self.engine.analyze(state, sma)

        if signal is not None:
            logger.info(f"\n{'='*50}")
            logger.info(f"SIGNAL: {signal['type'].upper()} {signal['direction'].upper()} on {symbol}")
            logger.info(f"  Signal price: {signal['signal_price']:.2f}")
            logger.info(f"  Stop: {signal['stop']:.2f}")
            logger.info(f"  Target: {signal['target']:.2f}")
            risk = abs(signal['signal_price'] - signal['stop'])
            reward = abs(signal['target'] - signal['signal_price'])
            rr = reward / risk if risk > 0 else 0
            logger.info(f"  R:R: {rr:.2f}:1")
            logger.info(f"  Quality: {signal['quality']:.1f}")
            logger.info(f"  → Will execute at NEXT candle open")
            logger.info(f"{'='*50}")

            # Store signal for execution on next candle open
            state.pending_signal = signal

    def _execute_signal(self, symbol: str, state: SymbolState, open_price: float):
        """Execute a pending signal at the current candle's open price."""
        signal = state.pending_signal
        if signal is None:
            return

        direction = signal["direction"]
        setup_type = signal["type"]

        # Rebuild stop/target from the executed entry price
        original_risk = abs(signal["signal_price"] - signal["stop"])
        original_rr = abs(signal["target"] - signal["signal_price"]) / original_risk if original_risk > 0 else MIN_RR

        if direction == "long":
            entry = open_price
            stop = entry - original_risk
            target = entry + original_risk * original_rr
        else:
            entry = open_price
            stop = entry + original_risk
            target = entry - original_risk * original_rr

        # Validate geometry
        if direction == "long" and not (stop < entry < target):
            logger.warning(f"  {symbol}: Geometry invalid after open adjustment — skipping")
            return
        if direction == "short" and not (target < entry < stop):
            logger.warning(f"  {symbol}: Geometry invalid after open adjustment — skipping")
            return

        risk = abs(entry - stop)
        if risk <= 0:
            return

        # Calculate position size
        try:
            balance = self.client.get_balance("USDT")
        except Exception as e:
            logger.error(f"  Failed to get balance: {e}")
            return

        risk_amount = balance * RISK_PER_TRADE
        quantity = risk_amount / risk

        # Round to exchange precision
        try:
            quantity = self.client.round_quantity(symbol, quantity)
            entry = self.client.round_price(symbol, entry)
            stop = self.client.round_price(symbol, stop)
            target = self.client.round_price(symbol, target)
        except Exception as e:
            logger.error(f"  Failed to get symbol info: {e}")
            return

        if quantity <= 0:
            logger.warning(f"  {symbol}: Quantity too small — skipping")
            return

        # === SPOT TRADING NOTE ===
        # Binance Spot Demo doesn't support true shorting.
        # For SHORT signals: we log the signal but don't execute.
        # For LONG signals: buy with market order, set OCO exit.
        #
        # To trade shorts, you'd need Margin or Futures demo mode.
        # The committee recommends starting with long-only on spot,
        # which produced +297.5R in the backtest (PF 1.34).

        if direction == "short":
            logger.info(f"  {symbol}: SHORT signal logged (spot mode — long only)")
            log_trade(symbol, setup_type, direction, entry, stop, target,
                     quantity, signal["quality"], "LOGGED_SHORT",
                     "Spot demo mode — shorts logged only")
            return

        # Execute LONG entry
        try:
            logger.info(f"  {symbol}: Executing MARKET BUY {quantity} @ ~{entry:.2f}")
            order = self.client.place_market_order(symbol, "BUY", quantity)
            fill_price = float(order.get("fills", [{}])[0].get("price", entry))
            logger.info(f"  {symbol}: FILLED at {fill_price:.2f}")

            # Place OCO exit (take profit + stop loss)
            stop_limit = self.client.round_price(symbol, stop * 0.999)  # Slightly below stop trigger
            logger.info(f"  {symbol}: Placing OCO exit — TP={target:.2f}, SL={stop:.2f}")

            oco = self.client.place_oco_order(
                symbol=symbol,
                side="SELL",
                quantity=quantity,
                price=target,           # Take profit
                stop_price=stop,        # Stop loss trigger
                stop_limit_price=stop_limit,  # Stop loss limit
            )
            logger.info(f"  {symbol}: OCO placed — orders active")

            # Record position
            state.position = {
                "direction": direction,
                "entry_price": fill_price,
                "stop_price": stop,
                "target_price": target,
                "quantity": quantity,
                "setup_type": setup_type,
                "entry_time": datetime.now(timezone.utc).isoformat(),
            }
            state.last_entry_bar = state.n - 1

            log_trade(symbol, setup_type, direction, fill_price, stop, target,
                     quantity, signal["quality"], "EXECUTED",
                     f"OCO exit placed. Order ID: {oco.get('orderListId', 'N/A')}")

        except Exception as e:
            logger.error(f"  {symbol}: Order failed — {e}")
            log_trade(symbol, setup_type, direction, entry, stop, target,
                     quantity, signal["quality"], "FAILED", str(e))

    def run_poll_mode(self):
        """
        Simple polling mode: check for new candles every 30 seconds.
        More reliable than WebSocket for initial deployment.
        """
        logger.info("\nStarting polling mode (checks every 30 seconds)...")
        logger.info("Press Ctrl+C to stop.\n")

        # Track the last closed candle timestamp per symbol
        last_candle_time = {}
        for symbol in self.states:
            if self.states[symbol].candles:
                last_candle_time[symbol] = self.states[symbol].candles[-1]["close_time"]
            else:
                last_candle_time[symbol] = 0

        self.running = True
        cycle = 0

        while self.running:
            try:
                cycle += 1
                now = datetime.now(timezone.utc)

                for symbol in list(self.states.keys()):
                    try:
                        # Fetch the latest 2 candles
                        klines = self.client.get_klines(symbol, TIMEFRAME, limit=2)

                        for k in klines:
                            close_time = k[6]
                            # Check if this candle has closed and we haven't processed it
                            is_closed = int(time.time() * 1000) > close_time
                            is_new = close_time > last_candle_time.get(symbol, 0)

                            if is_closed and is_new:
                                candle = {
                                    "timestamp": k[0],
                                    "open": float(k[1]),
                                    "high": float(k[2]),
                                    "low": float(k[3]),
                                    "close": float(k[4]),
                                    "volume": float(k[5]),
                                    "close_time": close_time,
                                }
                                logger.info(f"New candle: {symbol} close={candle['close']:.2f} "
                                          f"vol={candle['volume']:.1f}")
                                self.process_new_candle(symbol, candle)
                                last_candle_time[symbol] = close_time

                    except Exception as e:
                        logger.error(f"Error processing {symbol}: {e}")

                # Status heartbeat every 10 cycles (~5 minutes)
                if cycle % 10 == 0:
                    active = sum(1 for s in self.states.values() if s.position)
                    pending = sum(1 for s in self.states.values() if s.pending_signal)
                    logger.info(f"[Heartbeat] {now.strftime('%H:%M')} | "
                              f"Positions: {active} | Pending signals: {pending} | "
                              f"Cycle: {cycle}")

                time.sleep(30)

            except KeyboardInterrupt:
                logger.info("\nShutting down gracefully...")
                self.running = False
            except Exception as e:
                logger.error(f"Main loop error: {e}")
                time.sleep(5)

        logger.info("Bot stopped.")


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Breaker + Three Tap v1.0 — Binance Demo Trading Bot"
    )
    parser.add_argument(
        "--symbols", nargs="*", default=None,
        help="Override symbols (e.g., --symbols BTCUSDT SOLUSDT)"
    )
    args = parser.parse_args()

    symbols = args.symbols if args.symbols else SYMBOLS

    print(r"""
    ╔══════════════════════════════════════════════════╗
    ║  BREAKER + THREE TAP v1.0                       ║
    ║  Binance Demo Mode — Paper Trading              ║
    ║  Committee-validated: +506R / 2,582 trades      ║
    ╚══════════════════════════════════════════════════╝
    """)

    bot = TradingBot(symbols)

    if not bot.initialize():
        logger.error("Initialization failed. Check config.py and API keys.")
        sys.exit(1)

    bot.run_poll_mode()


if __name__ == "__main__":
    main()
