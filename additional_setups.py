"""
Additional Setup Detectors
============================

Three setups to complement the Breaker, covering regimes where it underperforms.

=== RANGE + MSB (RektProof pp.4-7, 61% hit rate) ===
Covers: consolidation/ranging markets
Mechanics:
  1. Range forms (two swing points define high/low boundary)
  2. Price sweeps/deviates beyond one boundary
  3. MSB occurs (structure breaks opposite to the sweep direction)
  4. Formed S/D zone is tested (entry)
  5. Target: opposite range boundary (untapped)

=== THREE TAP (RektProof pp.16-17) ===
Covers: range boundary reversals
Mechanics:
  1. Swing point forms at range boundary
  2. Price sweeps/deviates beyond the swing point (traps breakout traders)
  3. Price retests the swept level (third touch = entry)
  Key: "Those expecting a break will enter a breakout trade, ultimately
        getting trapped as we return to the level"

=== OTE REVERSAL (CryptoCred pp.36-40) ===
Covers: trend pullback entries after confirmed MSB
Mechanics:
  1. MSB occurs (shift in market structure confirmed)
  2. Impulse move runs, then starts retracing
  3. Apply Fibonacci from origin to extreme of impulse
  4. Enter between .618 and .786 retracement IF S/R structure exists there
  Critical: "I never trade this setup purely based on the retracement levels.
            I need there to be some sort of support/resistance structure at
            that level." — CryptoCred p.40
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, List
from enum import Enum

from market_structure import (
    MarketStructureEngine, StructureState, SwingPoint,
    SwingType, SwingLabel, MarketRegime, MarketStructureBreak,
)
from breaker_detector import SignalStatus
from liquidity_mapper import LiquidityMapper, LiquidityLevel, LevelRole


class SetupType(Enum):
    BREAKER = "breaker"
    RANGE_MSB = "range_msb"
    THREE_TAP = "three_tap"
    OTE = "ote_reversal"


@dataclass
class TradeSetup:
    """Universal trade setup container used by all detectors."""
    setup_type: SetupType
    direction: str              # "long" or "short"
    entry_price: float
    stop_price: float
    target_price: float
    trigger_index: int          # Bar where entry is triggered
    formation_index: int        # Bar where the pattern started forming
    quality_score: float = 0.0
    status: SignalStatus = SignalStatus.TRIGGERED
    exit_index: Optional[int] = None
    exit_price: Optional[float] = None
    pnl_r: Optional[float] = None
    fta_type: str = "default"

    @property
    def risk(self) -> float:
        return abs(self.entry_price - self.stop_price)

    @property
    def reward(self) -> float:
        return abs(self.target_price - self.entry_price)

    @property
    def rr(self) -> float:
        return self.reward / self.risk if self.risk > 0 else 0


# =========================================================================
# RANGE + MSB DETECTOR
# =========================================================================

class RangeMSBDetector:
    """
    Detects Range + MSB setups.

    RektProof pp.4-7:
    "Range forms → Price sweep/deviation → MSB → Test of formed SD → TP"

    Parameters adapted for 1H (committee):
    - min_range_bars: 30 (range must persist ~30 hours minimum)
    - max_range_atr_width: 6.0 (range can't be wider than 6 ATR)
    - min_range_atr_width: 1.5 (range must be at least 1.5 ATR wide)
    - sweep_threshold_atr: 0.3 (deviation beyond range must be meaningful)
    """

    def __init__(self, min_range_bars=30, max_range_atr_width=6.0,
                 min_range_atr_width=1.5, sweep_threshold_atr=0.3,
                 max_bars_to_entry=36):
        self.min_range_bars = min_range_bars
        self.max_range_atr_width = max_range_atr_width
        self.min_range_atr_width = min_range_atr_width
        self.sweep_threshold_atr = sweep_threshold_atr
        self.max_bars_to_entry = max_bars_to_entry

    def detect(self, state: StructureState, highs: np.ndarray,
               lows: np.ndarray, closes: np.ndarray,
               atr: np.ndarray) -> List[TradeSetup]:
        setups = []
        swings = state.swing_points
        if len(swings) < 6:
            return setups

        # Look for ranges: consecutive swings that stay within bounds
        for i in range(2, len(swings) - 3):
            sh = [s for s in swings[max(0,i-4):i+4] if s.swing_type == SwingType.HIGH]
            sl = [s for s in swings[max(0,i-4):i+4] if s.swing_type == SwingType.LOW]

            if len(sh) < 2 or len(sl) < 2:
                continue

            range_high = max(s.price for s in sh)
            range_low = min(s.price for s in sl)
            range_width = range_high - range_low

            first_idx = min(s.index for s in sh + sl)
            last_idx = max(s.index for s in sh + sl)

            if last_idx - first_idx < self.min_range_bars:
                continue

            atr_val = atr[min(last_idx, len(atr)-1)]
            if atr_val <= 0:
                continue

            width_in_atr = range_width / atr_val
            if width_in_atr > self.max_range_atr_width or width_in_atr < self.min_range_atr_width:
                continue

            # Look for sweep beyond range boundaries AFTER range forms
            sweep_start = last_idx + 1
            sweep_end = min(sweep_start + self.max_bars_to_entry, len(closes))

            for bar in range(sweep_start, sweep_end):
                sweep_threshold = atr_val * self.sweep_threshold_atr

                # Sweep above range high (bearish setup)
                if highs[bar] > range_high + sweep_threshold:
                    # Look for MSB below — close below range low
                    for msb_bar in range(bar + 1, min(bar + self.max_bars_to_entry, len(closes))):
                        if closes[msb_bar] < range_low:
                            # Look for retest of range low area (now resistance)
                            for entry_bar in range(msb_bar + 1, min(msb_bar + self.max_bars_to_entry, len(closes))):
                                if highs[entry_bar] >= range_low and closes[entry_bar] < range_low:
                                    entry = closes[entry_bar]
                                    stop = range_high + atr_val * 0.3
                                    target = range_low - range_width  # Project range width below
                                    
                                    setup = TradeSetup(
                                        setup_type=SetupType.RANGE_MSB,
                                        direction="short", entry_price=entry,
                                        stop_price=stop, target_price=target,
                                        trigger_index=entry_bar,
                                        formation_index=first_idx,
                                    )
                                    setup.quality_score = self._score(setup, atr_val, width_in_atr)
                                    if setup.rr >= 1.5:
                                        setups.append(setup)
                                    break
                            break
                    break

                # Sweep below range low (bullish setup)
                if lows[bar] < range_low - sweep_threshold:
                    for msb_bar in range(bar + 1, min(bar + self.max_bars_to_entry, len(closes))):
                        if closes[msb_bar] > range_high:
                            for entry_bar in range(msb_bar + 1, min(msb_bar + self.max_bars_to_entry, len(closes))):
                                if lows[entry_bar] <= range_high and closes[entry_bar] > range_high:
                                    entry = closes[entry_bar]
                                    stop = range_low - atr_val * 0.3
                                    target = range_high + range_width

                                    setup = TradeSetup(
                                        setup_type=SetupType.RANGE_MSB,
                                        direction="long", entry_price=entry,
                                        stop_price=stop, target_price=target,
                                        trigger_index=entry_bar,
                                        formation_index=first_idx,
                                    )
                                    setup.quality_score = self._score(setup, atr_val, width_in_atr)
                                    if setup.rr >= 1.5:
                                        setups.append(setup)
                                    break
                            break
                    break

        return setups

    def _score(self, setup: TradeSetup, atr_val: float, range_width_atr: float) -> float:
        score = 40.0  # Base for having all criteria met
        if 2.0 <= range_width_atr <= 4.0:
            score += 20  # Clean, well-defined range
        if setup.rr >= 2.0:
            score += 15
        if setup.rr >= 3.0:
            score += 10
        return min(score, 100.0)


# =========================================================================
# THREE TAP DETECTOR
# =========================================================================

class ThreeTapDetector:
    """
    Detects Three Tap setups.

    RektProof pp.16-17:
    "1. Swing Point forms
     2. Swing Point Swept/Deviate
     3. Retest"

    "Those expecting a break will enter a breakout trade, ultimately
     getting trapped as we return to the level."

    The three taps are:
    Tap 1: Initial swing point forms
    Tap 2: Price sweeps beyond swing (deviation) — traps breakout traders
    Tap 3: Price returns to the original swing level (entry)
    """

    def __init__(self, sweep_min_atr=0.2, sweep_max_atr=2.0,
                 max_bars_to_retest=48):
        self.sweep_min_atr = sweep_min_atr
        self.sweep_max_atr = sweep_max_atr
        self.max_bars_to_retest = max_bars_to_retest

    def detect(self, state: StructureState, highs: np.ndarray,
               lows: np.ndarray, closes: np.ndarray,
               atr: np.ndarray) -> List[TradeSetup]:
        setups = []
        swings = state.swing_points

        for i, swing in enumerate(swings):
            if swing.index >= len(closes) - 10:
                continue

            atr_val = atr[min(swing.index, len(atr)-1)]
            if atr_val <= 0:
                continue

            # TAP 1: The swing point itself
            tap1_price = swing.price
            tap1_idx = swing.index

            # Look for TAP 2: sweep/deviation beyond the swing
            sweep_start = tap1_idx + 1
            sweep_end = min(tap1_idx + self.max_bars_to_retest, len(closes))

            for bar in range(sweep_start, sweep_end):
                if swing.swing_type == SwingType.LOW:
                    # Bullish three tap: low forms, price sweeps below, returns
                    sweep_depth = tap1_price - lows[bar]
                    if sweep_depth < atr_val * self.sweep_min_atr:
                        continue
                    if sweep_depth > atr_val * self.sweep_max_atr:
                        break  # Too deep, not a sweep

                    # TAP 2 found — price went below the low
                    # Now check if price recovered (closed back above)
                    if closes[bar] > tap1_price:
                        continue  # Not a wick sweep, just volatile

                    # Look for TAP 3: retest of the original low area
                    for retest_bar in range(bar + 1, min(bar + self.max_bars_to_retest, len(closes))):
                        # Price must have moved away first
                        if highs[retest_bar] < tap1_price + atr_val * 0.5:
                            continue  # Hasn't moved away enough

                        # Now look for the actual retest
                        for tap3_bar in range(retest_bar, min(retest_bar + self.max_bars_to_retest, len(closes))):
                            if lows[tap3_bar] <= tap1_price + atr_val * 0.1 and closes[tap3_bar] > tap1_price:
                                # TAP 3: price touched the low area and bounced
                                entry = closes[tap3_bar]
                                stop = min(lows[bar], lows[tap3_bar]) - atr_val * 0.2
                                # Target: the high formed between tap 2 and tap 3
                                interim_high = max(highs[bar:tap3_bar+1])
                                target = interim_high

                                if abs(entry - stop) <= 0:
                                    break

                                setup = TradeSetup(
                                    setup_type=SetupType.THREE_TAP,
                                    direction="long", entry_price=entry,
                                    stop_price=stop, target_price=target,
                                    trigger_index=tap3_bar,
                                    formation_index=tap1_idx,
                                )
                                setup.quality_score = self._score(setup, atr_val, sweep_depth)
                                if setup.rr >= 1.5:
                                    setups.append(setup)
                                break
                        break
                    break

                elif swing.swing_type == SwingType.HIGH:
                    # Bearish three tap: high forms, price sweeps above, returns
                    sweep_depth = highs[bar] - tap1_price
                    if sweep_depth < atr_val * self.sweep_min_atr:
                        continue
                    if sweep_depth > atr_val * self.sweep_max_atr:
                        break

                    if closes[bar] < tap1_price:
                        continue

                    for retest_bar in range(bar + 1, min(bar + self.max_bars_to_retest, len(closes))):
                        if lows[retest_bar] > tap1_price - atr_val * 0.5:
                            continue

                        for tap3_bar in range(retest_bar, min(retest_bar + self.max_bars_to_retest, len(closes))):
                            if highs[tap3_bar] >= tap1_price - atr_val * 0.1 and closes[tap3_bar] < tap1_price:
                                entry = closes[tap3_bar]
                                stop = max(highs[bar], highs[tap3_bar]) + atr_val * 0.2
                                interim_low = min(lows[bar:tap3_bar+1])
                                target = interim_low

                                if abs(entry - stop) <= 0:
                                    break

                                setup = TradeSetup(
                                    setup_type=SetupType.THREE_TAP,
                                    direction="short", entry_price=entry,
                                    stop_price=stop, target_price=target,
                                    trigger_index=tap3_bar,
                                    formation_index=tap1_idx,
                                )
                                setup.quality_score = self._score(setup, atr_val, sweep_depth)
                                if setup.rr >= 1.5:
                                    setups.append(setup)
                                break
                        break
                    break

        return setups

    def _score(self, setup: TradeSetup, atr_val: float, sweep_depth: float) -> float:
        score = 40.0
        # Clean sweep (not too deep, not too shallow)
        sweep_atr = sweep_depth / atr_val if atr_val > 0 else 0
        if 0.3 <= sweep_atr <= 1.0:
            score += 20  # Clean wick sweep
        elif sweep_atr < 0.3:
            score += 5
        # Good R:R
        if setup.rr >= 2.0:
            score += 15
        if setup.rr >= 3.0:
            score += 10
        return min(score, 100.0)


# =========================================================================
# OTE REVERSAL DETECTOR
# =========================================================================

class OTEDetector:
    """
    Detects Optimal Trade Entry (OTE) pullback setups.

    CryptoCred pp.36-40:
    "I use the Fibonacci retracement tool when there is a shift in
     market structure. I choose the origin of the breakout as anchor 1
     and the highest high/lowest low made as anchor 2."

    "If I'm buying a dip/selling a rally following a break in market
     structure, I look for a reaction between the 0.618 and 0.786
     retracement levels."

    "I never trade this setup purely based on the retracement levels.
     I need there to be some sort of support/resistance structure at
     that level in order for me to make a trade."
    """

    def __init__(self, fib_low=0.618, fib_high=0.786,
                 max_bars_to_retrace=36, min_impulse_atr=2.0):
        self.fib_low = fib_low
        self.fib_high = fib_high
        self.max_bars_to_retrace = max_bars_to_retrace
        self.min_impulse_atr = min_impulse_atr

    def detect(self, state: StructureState, highs: np.ndarray,
               lows: np.ndarray, closes: np.ndarray, atr: np.ndarray,
               liquidity_levels: List[LiquidityLevel] = None) -> List[TradeSetup]:
        setups = []

        for msb in state.msb_events:
            if msb.index >= len(closes) - 10:
                continue

            atr_val = atr[min(msb.index, len(atr)-1)]
            if atr_val <= 0:
                continue

            # Find the impulse move after the MSB
            origin_idx = msb.broken_swing.index
            origin_price = msb.broken_swing.price

            if msb.break_type == "bullish_msb":
                # Bullish: MSB broke a lower high → looking for longs on pullback
                # Find the highest point after MSB (impulse top)
                search_end = min(msb.index + self.max_bars_to_retrace, len(highs))
                if search_end <= msb.index:
                    continue

                impulse_high_idx = msb.index + np.argmax(highs[msb.index:search_end])
                impulse_high = highs[impulse_high_idx]

                impulse_size = impulse_high - origin_price
                if impulse_size < atr_val * self.min_impulse_atr:
                    continue

                # Calculate OTE zone
                ote_high = impulse_high - impulse_size * self.fib_low  # .618 retrace
                ote_low = impulse_high - impulse_size * self.fib_high   # .786 retrace

                # Wait for price to retrace INTO the OTE zone
                for bar in range(impulse_high_idx + 1, min(impulse_high_idx + self.max_bars_to_retrace, len(closes))):
                    if lows[bar] <= ote_high and lows[bar] >= ote_low and closes[bar] > ote_low:
                        # Check for S/R structure at this level (CryptoCred requirement)
                        has_structure = self._check_structure(ote_low, ote_high, liquidity_levels)

                        entry = closes[bar]
                        stop = ote_low - atr_val * 0.3
                        target = impulse_high  # Target: retest of the impulse high

                        setup = TradeSetup(
                            setup_type=SetupType.OTE,
                            direction="long", entry_price=entry,
                            stop_price=stop, target_price=target,
                            trigger_index=bar,
                            formation_index=msb.index,
                        )
                        setup.quality_score = self._score(setup, atr_val, has_structure, impulse_size)
                        if setup.rr >= 1.5:
                            setups.append(setup)
                        break

            elif msb.break_type == "bearish_msb":
                # Bearish: MSB broke a higher low → looking for shorts on rally
                search_end = min(msb.index + self.max_bars_to_retrace, len(lows))
                if search_end <= msb.index:
                    continue

                impulse_low_idx = msb.index + np.argmin(lows[msb.index:search_end])
                impulse_low = lows[impulse_low_idx]

                impulse_size = origin_price - impulse_low
                if impulse_size < atr_val * self.min_impulse_atr:
                    continue

                ote_low = impulse_low + impulse_size * self.fib_low
                ote_high = impulse_low + impulse_size * self.fib_high

                for bar in range(impulse_low_idx + 1, min(impulse_low_idx + self.max_bars_to_retrace, len(closes))):
                    if highs[bar] >= ote_low and highs[bar] <= ote_high and closes[bar] < ote_high:
                        has_structure = self._check_structure(ote_low, ote_high, liquidity_levels)

                        entry = closes[bar]
                        stop = ote_high + atr_val * 0.3
                        target = impulse_low

                        setup = TradeSetup(
                            setup_type=SetupType.OTE,
                            direction="short", entry_price=entry,
                            stop_price=stop, target_price=target,
                            trigger_index=bar,
                            formation_index=msb.index,
                        )
                        setup.quality_score = self._score(setup, atr_val, has_structure, impulse_size)
                        if setup.rr >= 1.5:
                            setups.append(setup)
                        break

        return setups

    def _check_structure(self, zone_low: float, zone_high: float,
                         levels: List[LiquidityLevel] = None) -> bool:
        """
        CryptoCred p.40: "I need there to be some sort of support/resistance
        structure at that level in order for me to make a trade."
        """
        if not levels:
            return False
        for level in levels:
            if zone_low <= level.price <= zone_high:
                return True
        return False

    def _score(self, setup: TradeSetup, atr_val: float,
               has_structure: bool, impulse_size: float) -> float:
        score = 35.0
        if has_structure:
            score += 25  # CryptoCred's main confluence requirement
        impulse_atr = impulse_size / atr_val if atr_val > 0 else 0
        if impulse_atr >= 3.0:
            score += 15  # Strong impulse
        if setup.rr >= 2.0:
            score += 10
        if setup.rr >= 3.0:
            score += 10
        return min(score, 100.0)


# =========================================================================
# TRADE SIMULATOR (shared by all setups)
# =========================================================================

def simulate_trade(setup: TradeSetup, highs: np.ndarray,
                   lows: np.ndarray, closes: np.ndarray) -> TradeSetup:
    """Walk forward from entry to determine outcome."""
    if setup.trigger_index is None:
        return setup

    start = setup.trigger_index + 1
    risk = setup.risk

    for i in range(start, len(closes)):
        if setup.direction == "short":
            if highs[i] >= setup.stop_price:
                setup.status = SignalStatus.STOPPED
                setup.exit_index = i
                setup.exit_price = setup.stop_price
                setup.pnl_r = -1.0
                return setup
            if lows[i] <= setup.target_price:
                setup.status = SignalStatus.TARGET_HIT
                setup.exit_index = i
                setup.exit_price = setup.target_price
                setup.pnl_r = setup.reward / risk if risk > 0 else 0
                return setup
        else:
            if lows[i] <= setup.stop_price:
                setup.status = SignalStatus.STOPPED
                setup.exit_index = i
                setup.exit_price = setup.stop_price
                setup.pnl_r = -1.0
                return setup
            if highs[i] >= setup.target_price:
                setup.status = SignalStatus.TARGET_HIT
                setup.exit_index = i
                setup.exit_price = setup.target_price
                setup.pnl_r = setup.reward / risk if risk > 0 else 0
                return setup

    setup.status = SignalStatus.ACTIVE
    return setup
