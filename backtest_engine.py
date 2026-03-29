"""
Module 6: Backtest Engine
=========================

The moment of truth. This module loads real crypto OHLCV data and runs
our complete pipeline across multiple coins and market regimes.

Trader 4: "This is where we find out if the system works or if we need
to go back to the drawing board. Every metric gets tracked."

Trader 5: "We need: win rate, expectancy per trade, max drawdown,
profit factor, average R per trade, and per-regime breakdowns."

Data adaptation note:
The uploaded data is DAILY, not 1H. The system architecture is
timeframe-agnostic — we adjust the interpretation, not the logic.
- swing_lookback=5 on daily = 5 days each side (solid daily swing detection)
- ATR(14) on daily = 14-day ATR (standard)
- max_bars_to_retest=30 on daily = ~1 month for retest window
"""

import numpy as np
import csv
import os
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
import sys

sys.path.insert(0, "/home/claude/trading_system")

from market_structure import MarketStructureEngine, MarketRegime, StructureState
from breaker_detector import (
    BreakerDetector, BreakerTradeManager, BreakerZone,
    BreakerType, SignalStatus,
)
from liquidity_mapper import LiquidityMapper


# =============================================================================
# Data Loader
# =============================================================================

def load_crypto_csv(filepath: str) -> Optional[Dict]:
    """
    Load a crypto CSV file and return arrays ready for the engine.

    Returns dict with: opens, highs, lows, closes, volumes, timestamps, name
    Skips rows with zero volume (unreliable data).
    """
    rows = []
    with open(filepath, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            vol = float(row["Volume"])
            # Skip zero-volume rows (no reliable price discovery)
            if vol <= 0:
                continue
            rows.append(row)

    if len(rows) < 50:
        return None  # Not enough data

    n = len(rows)
    name = rows[0]["Name"]
    symbol = rows[0]["Symbol"]

    opens = np.array([float(r["Open"]) for r in rows])
    highs = np.array([float(r["High"]) for r in rows])
    lows = np.array([float(r["Low"]) for r in rows])
    closes = np.array([float(r["Close"]) for r in rows])
    volumes = np.array([float(r["Volume"]) for r in rows])
    timestamps = [r["Date"] for r in rows]

    return {
        "opens": opens,
        "highs": highs,
        "lows": lows,
        "closes": closes,
        "volumes": volumes,
        "timestamps": timestamps,
        "name": name,
        "symbol": symbol,
    }


# =============================================================================
# Performance Tracker
# =============================================================================

@dataclass
class TradeRecord:
    """One completed trade for the journal."""
    coin: str
    direction: str          # "long" or "short"
    entry_price: float
    stop_price: float
    target_price: float
    exit_price: float
    entry_date: str
    exit_date: str
    pnl_r: float            # P&L in R multiples
    outcome: str             # "win", "loss", "active"
    rr_ratio: float          # Risk:Reward at entry
    quality_score: float
    regime: str              # Market regime at entry
    fta_type: str            # What type of liquidity was the target
    breaker_type: str        # "bullish" or "bearish"


@dataclass
class BacktestResults:
    """Comprehensive backtest results."""
    trades: List[TradeRecord] = field(default_factory=list)
    coins_tested: List[str] = field(default_factory=list)

    @property
    def total_trades(self) -> int:
        return len(self.trades)

    @property
    def wins(self) -> int:
        return sum(1 for t in self.trades if t.outcome == "win")

    @property
    def losses(self) -> int:
        return sum(1 for t in self.trades if t.outcome == "loss")

    @property
    def win_rate(self) -> float:
        completed = self.wins + self.losses
        return self.wins / completed if completed > 0 else 0.0

    @property
    def total_r(self) -> float:
        return sum(t.pnl_r for t in self.trades if t.outcome in ("win", "loss"))

    @property
    def avg_r_per_trade(self) -> float:
        completed = [t for t in self.trades if t.outcome in ("win", "loss")]
        return self.total_r / len(completed) if completed else 0.0

    @property
    def avg_win_r(self) -> float:
        winners = [t.pnl_r for t in self.trades if t.outcome == "win"]
        return np.mean(winners) if winners else 0.0

    @property
    def avg_loss_r(self) -> float:
        losers = [t.pnl_r for t in self.trades if t.outcome == "loss"]
        return np.mean(losers) if losers else 0.0

    @property
    def profit_factor(self) -> float:
        gross_profit = sum(t.pnl_r for t in self.trades if t.pnl_r > 0)
        gross_loss = abs(sum(t.pnl_r for t in self.trades if t.pnl_r < 0))
        return gross_profit / gross_loss if gross_loss > 0 else float("inf")

    @property
    def max_drawdown_r(self) -> float:
        """Maximum drawdown measured in R."""
        if not self.trades:
            return 0.0
        equity = [0.0]
        for t in self.trades:
            if t.outcome in ("win", "loss"):
                equity.append(equity[-1] + t.pnl_r)
        peak = equity[0]
        max_dd = 0.0
        for e in equity:
            peak = max(peak, e)
            dd = peak - e
            max_dd = max(max_dd, dd)
        return max_dd

    @property
    def expectancy(self) -> float:
        """Expected R per trade = (win_rate * avg_win) - (loss_rate * avg_loss)"""
        if self.wins + self.losses == 0:
            return 0.0
        wr = self.win_rate
        return (wr * self.avg_win_r) - ((1 - wr) * abs(self.avg_loss_r))

    def by_regime(self) -> Dict[str, "BacktestResults"]:
        """Break down results by market regime."""
        regime_map = {}
        for t in self.trades:
            if t.regime not in regime_map:
                regime_map[t.regime] = BacktestResults()
            regime_map[t.regime].trades.append(t)
        return regime_map

    def by_direction(self) -> Dict[str, "BacktestResults"]:
        """Break down results by long/short."""
        dir_map = {}
        for t in self.trades:
            if t.direction not in dir_map:
                dir_map[t.direction] = BacktestResults()
            dir_map[t.direction].trades.append(t)
        return dir_map

    def by_coin(self) -> Dict[str, "BacktestResults"]:
        """Break down results by coin."""
        coin_map = {}
        for t in self.trades:
            if t.coin not in coin_map:
                coin_map[t.coin] = BacktestResults()
            coin_map[t.coin].trades.append(t)
        return coin_map

    def print_summary(self, label: str = "OVERALL"):
        """Print a formatted summary."""
        completed = self.wins + self.losses
        print(f"\n{'='*55}")
        print(f"  {label}")
        print(f"{'='*55}")
        print(f"  Trades completed:  {completed}")
        print(f"  Wins:              {self.wins}")
        print(f"  Losses:            {self.losses}")
        print(f"  Win rate:          {self.win_rate:.1%}")
        print(f"  Total R:           {self.total_r:+.2f}R")
        print(f"  Avg R/trade:       {self.avg_r_per_trade:+.3f}R")
        print(f"  Avg winner:        {self.avg_win_r:+.2f}R")
        print(f"  Avg loser:         {self.avg_loss_r:.2f}R")
        print(f"  Profit factor:     {self.profit_factor:.2f}")
        print(f"  Expectancy:        {self.expectancy:+.3f}R")
        print(f"  Max drawdown:      {self.max_drawdown_r:.2f}R")
        if completed > 0:
            # Consecutive losses
            streak = 0
            max_streak = 0
            for t in self.trades:
                if t.outcome == "loss":
                    streak += 1
                    max_streak = max(max_streak, streak)
                else:
                    streak = 0
            print(f"  Max losing streak: {max_streak}")


# =============================================================================
# Backtest Runner
# =============================================================================

class BacktestRunner:
    """
    Runs the complete pipeline on real data.

    Panel parameters for daily timeframe:
    - swing_lookback=5 (5 days each side = ~2 week confirmation)
    - ATR period=14 (standard)
    - min_swing_atr=0.5 (filter noise)
    - max_bars_to_retest=30 (1 month window for daily)
    - min_rr=1.5 (lower than 2.0 since FTA targets are precise)
    - risk_per_trade=2% (conservative)

    Trader 3: "On daily, breakers take longer to form and retest.
    30 bars = 1 month is reasonable."
    Trader 5: "We lower min R:R to 1.5 because FTA targets are
    data-driven, not arbitrary. A 1.5:1 with 60%+ win rate is profitable."
    """

    def __init__(self):
        self.structure_engine = MarketStructureEngine(
            swing_lookback=5,
            atr_period=14,
            min_swing_atr_multiple=0.5,
            equal_threshold_atr=0.1,
        )
        self.breaker_detector = BreakerDetector(
            max_bars_to_retest=30,
            zone_buffer_atr=0.1,
            min_move_from_zone_atr=0.8,
        )
        self.trade_manager = BreakerTradeManager(
            risk_percent=0.02,
            min_rr=1.5,
            stop_buffer_atr=0.3,
        )
        self.liquidity_mapper = LiquidityMapper(
            equal_tolerance_atr=0.15,
            untapped_lookback=100,
            max_levels=20,
        )

    def run_single_coin(
        self,
        data: Dict,
        results: BacktestResults,
    ) -> int:
        """
        Run backtest on a single coin. Returns number of trades found.
        """
        o = data["opens"]
        h = data["highs"]
        l = data["lows"]
        c = data["closes"]
        v = data["volumes"]
        ts = data["timestamps"]
        coin = data["symbol"]
        n = len(c)

        if n < 60:
            return 0

        # Run structure analysis
        state = self.structure_engine.analyze(o, h, l, c, v, ts)
        atr = self.structure_engine._calculate_atr(h, l, c)

        # Detect breakers
        breakers = self.breaker_detector.detect(
            state, h, l, c, atr, htf_bias="neutral"
        )

        trades_found = 0

        for bk in breakers:
            if bk.status != SignalStatus.TRIGGERED:
                continue
            if bk.trigger_index is None or bk.entry_price is None:
                continue

            entry_bar = bk.trigger_index
            if entry_bar >= n - 1:
                continue

            direction = "short" if bk.breaker_type == BreakerType.BEARISH else "long"
            atr_val = atr[min(entry_bar, len(atr) - 1)]

            # Calculate stop
            if bk.breaker_type == BreakerType.BEARISH:
                stop = bk.zone_high + (atr_val * self.trade_manager.stop_buffer_atr)
            else:
                stop = bk.zone_low - (atr_val * self.trade_manager.stop_buffer_atr)

            # Find FTA target using liquidity mapper
            levels = self.liquidity_mapper.map_liquidity(
                state, h, l, c, atr, current_bar=entry_bar
            )
            fta = self.liquidity_mapper.find_fta(
                levels, bk.entry_price, direction, stop
            )

            # Determine target
            fta_type = "default"
            if fta and fta.rr_with_stop and fta.rr_with_stop >= self.trade_manager.min_rr:
                target = fta.level.price
                fta_type = fta.level.liquidity_type.value
            else:
                # Default: use minimum R:R
                risk = abs(bk.entry_price - stop)
                if direction == "short":
                    target = bk.entry_price - (risk * self.trade_manager.min_rr)
                else:
                    target = bk.entry_price + (risk * self.trade_manager.min_rr)

            # Validate R:R
            risk = abs(bk.entry_price - stop)
            reward = abs(target - bk.entry_price)
            if risk <= 0:
                continue
            rr = reward / risk
            if rr < self.trade_manager.min_rr:
                continue

            # Check for dangerous liquidity near stop
            dangerous = self.liquidity_mapper.check_liquidity_near_stop(
                levels, stop, atr_val, direction
            )
            if len(dangerous) >= 2:
                # Too much liquidity near stop — high risk of being hunted
                continue

            # Set trade params and simulate
            bk.stop_price = stop
            bk.target_price = target

            bk = self.trade_manager.simulate_outcome(bk, h, l, c)

            if bk.status not in (SignalStatus.TARGET_HIT, SignalStatus.STOPPED):
                continue  # Trade didn't resolve

            # Record the trade
            outcome = "win" if bk.status == SignalStatus.TARGET_HIT else "loss"
            pnl_r = bk.pnl_r if bk.pnl_r is not None else 0.0

            entry_date = ts[entry_bar] if entry_bar < len(ts) else "unknown"
            exit_bar = bk.exit_index if bk.exit_index else entry_bar
            exit_date = ts[exit_bar] if exit_bar < len(ts) else "unknown"

            record = TradeRecord(
                coin=coin,
                direction=direction,
                entry_price=bk.entry_price,
                stop_price=stop,
                target_price=target,
                exit_price=bk.exit_price or 0.0,
                entry_date=entry_date,
                exit_date=exit_date,
                pnl_r=pnl_r,
                outcome=outcome,
                rr_ratio=rr,
                quality_score=bk.quality_score,
                regime=bk.regime_at_formation.value if bk.regime_at_formation else "unknown",
                fta_type=fta_type,
                breaker_type=bk.breaker_type.value,
            )
            results.trades.append(record)
            trades_found += 1

        return trades_found


def run_full_backtest():
    """Run backtest across all available coins."""
    data_dir = "/mnt/user-data/uploads"
    files = sorted([
        f for f in os.listdir(data_dir) if f.startswith("coin_") and f.endswith(".csv")
    ])

    print("=" * 65)
    print("FULL BACKTEST — Real Crypto Data")
    print("Breaker Setup on Daily Timeframe")
    print("=" * 65)
    print(f"\nCoins available: {len(files)}")

    runner = BacktestRunner()
    results = BacktestResults()

    for filename in files:
        filepath = os.path.join(data_dir, filename)
        data = load_crypto_csv(filepath)
        if data is None:
            print(f"  {filename}: skipped (insufficient data)")
            continue

        coin = data["symbol"]
        n = len(data["closes"])
        start = data["timestamps"][0][:10]
        end = data["timestamps"][-1][:10]

        trades = runner.run_single_coin(data, results)
        results.coins_tested.append(coin)

        status = f"{trades} trades" if trades > 0 else "no trades"
        print(f"  {coin:6s} | {n:5d} days | {start} → {end} | {status}")

    # === RESULTS ===
    results.print_summary("OVERALL BREAKER BACKTEST")

    # By direction
    print("\n" + "-" * 55)
    print("  BREAKDOWN BY DIRECTION")
    for direction, sub in results.by_direction().items():
        sub.print_summary(f"Direction: {direction.upper()}")

    # By regime
    print("\n" + "-" * 55)
    print("  BREAKDOWN BY MARKET REGIME")
    for regime, sub in sorted(results.by_regime().items()):
        if sub.total_trades >= 3:
            sub.print_summary(f"Regime: {regime}")

    # By coin (top coins by trade count)
    print("\n" + "-" * 55)
    print("  BREAKDOWN BY COIN (top 10 by trade count)")
    coin_results = results.by_coin()
    sorted_coins = sorted(
        coin_results.items(), key=lambda x: x[1].total_trades, reverse=True
    )
    for coin, sub in sorted_coins[:10]:
        completed = sub.wins + sub.losses
        if completed >= 2:
            sub.print_summary(f"Coin: {coin}")

    # Trade log (first 20 for inspection)
    print("\n" + "-" * 55)
    print("  SAMPLE TRADE LOG (first 20 resolved trades)")
    print(f"  {'Coin':6s} {'Dir':5s} {'Entry':>9s} {'Stop':>9s} "
          f"{'Target':>9s} {'R:R':>5s} {'P&L':>6s} {'Result':7s} {'Date':12s}")
    print("  " + "-" * 80)

    resolved = [t for t in results.trades if t.outcome in ("win", "loss")]
    for t in resolved[:20]:
        print(f"  {t.coin:6s} {t.direction:5s} {t.entry_price:9.2f} "
              f"{t.stop_price:9.2f} {t.target_price:9.2f} "
              f"{t.rr_ratio:5.2f} {t.pnl_r:+6.2f} "
              f"{'WIN' if t.outcome=='win' else 'LOSS':7s} {t.entry_date[:10]:12s}")

    # === Final Panel Assessment ===
    print("\n" + "=" * 65)
    print("PANEL ASSESSMENT")
    print("=" * 65)

    if results.total_trades == 0:
        print("  No trades generated. Parameters may need loosening.")
        return results

    wr = results.win_rate
    exp = results.expectancy
    pf = results.profit_factor

    if exp > 0:
        print(f"  POSITIVE EXPECTANCY: {exp:+.3f}R per trade")
        print(f"  Over {results.wins + results.losses} trades, this system is profitable.")
    else:
        print(f"  NEGATIVE EXPECTANCY: {exp:.3f}R per trade")
        print(f"  System needs refinement before live deployment.")

    if wr >= 0.55:
        print(f"  Win rate {wr:.1%} exceeds 55% threshold — GOOD")
    elif wr >= 0.45:
        print(f"  Win rate {wr:.1%} is marginal — needs R:R to compensate")
    else:
        print(f"  Win rate {wr:.1%} is low — review entry criteria")

    if pf >= 1.5:
        print(f"  Profit factor {pf:.2f} — STRONG")
    elif pf >= 1.0:
        print(f"  Profit factor {pf:.2f} — marginal, needs improvement")
    else:
        print(f"  Profit factor {pf:.2f} — LOSING SYSTEM")

    return results


if __name__ == "__main__":
    results = run_full_backtest()
