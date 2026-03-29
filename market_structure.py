"""
Module 1: Market Structure Engine
=================================

This module implements the foundational market structure analysis that ALL
other modules depend on. It answers the most important question in trading:
"What is the current market structure, and has it changed?"

Sources & Cross-References:
- CryptoCred Study Guide, Lesson 1 & 3: Higher time frame bias, order flow,
  impulse/retrace model, HH/HL uptrend, LH/LL downtrend
- RektProof Setups, pp.21-23: Valid market structure follows orderflow with
  unbroken lows. Valid MSB = key low broken in uptrend / key high in downtrend
- EmperorBTC Manual, Ch.4: S/R flips confirm trend, more tests = weaker level
- EmperorBTC Manual, Ch.13: Price market psychology - enthusiasm matters

Panel Decisions:
- Trader 1 (Structure Purist): Swing detection uses fractal lookback method
  with N=5 for 1H timeframe. Swings MUST alternate (no consecutive HH without LL).
- Trader 3 (RektProof): Only "key" swing points define structure. Minor
  internal swings during re-accumulation/re-distribution do NOT invalidate.
- Trader 5 (Quant): ATR-based minimum swing size filter removes noise.
  Swing must move at least 0.5 * ATR(14) to be "significant."
- All 5 agree: Multi-timeframe analysis is mandatory. 1H structure is only
  valid when aligned with 4H and Daily context.

Algorithm:
1. Detect raw swing highs and swing lows (fractal method, lookback=5)
2. Filter insignificant swings (ATR-based minimum size)
3. Ensure swing alternation (must alternate H-L-H-L)
4. Classify each swing relative to the previous: HH, HL, LH, LL
5. Determine market regime: Uptrend, Downtrend, Ranging, Transition
6. Detect Market Structure Breaks (MSBs) — the critical signal
"""

import numpy as np
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class SwingType(Enum):
    HIGH = "swing_high"
    LOW = "swing_low"


class SwingLabel(Enum):
    """Classification relative to previous swing of same type"""
    HH = "higher_high"     # Current SH > Previous SH
    LH = "lower_high"      # Current SH < Previous SH
    EH = "equal_high"      # Current SH ≈ Previous SH (within tolerance)
    HL = "higher_low"      # Current SL > Previous SL
    LL = "lower_low"       # Current SL < Previous SL
    EL = "equal_low"       # Current SL ≈ Previous SL (within tolerance)
    FIRST = "first"        # No previous swing to compare


class MarketRegime(Enum):
    """
    Market regime classification.

    From CryptoCred Lesson 1:
    - Uptrend = HH and HL (retraces are for buying)
    - Downtrend = LH and LL (retraces are for selling)

    From RektProof p.21:
    - Trend → Range (accumulate/distribute) is the natural cycle
    - Valid structure follows orderflow with unbroken key lows/highs
    """
    STRONG_UPTREND = "strong_uptrend"       # Consecutive HH + HL
    UPTREND = "uptrend"                      # Most recent = HH + HL
    WEAK_UPTREND = "weak_uptrend"            # HH but LL (divergent)
    STRONG_DOWNTREND = "strong_downtrend"    # Consecutive LH + LL
    DOWNTREND = "downtrend"                  # Most recent = LH + LL
    WEAK_DOWNTREND = "weak_downtrend"        # LL but HH (divergent)
    RANGING = "ranging"                      # Price between defined bounds
    TRANSITION = "transition"                # Structure shifting
    UNKNOWN = "unknown"                      # Insufficient data


@dataclass
class SwingPoint:
    """A confirmed swing high or swing low in the price data."""
    index: int                          # Bar index in the OHLCV array
    price: float                        # The high (for SH) or low (for SL)
    swing_type: SwingType               # HIGH or LOW
    label: SwingLabel = SwingLabel.FIRST # HH/HL/LH/LL classification
    is_key_level: bool = False          # Is this a "key" structural level?
    atr_at_point: float = 0.0           # ATR value at this swing for context
    broken: bool = False                # Has this level been broken?
    broken_at_index: Optional[int] = None  # When was it broken?
    timestamp: Optional[str] = None     # Human-readable time


@dataclass
class MarketStructureBreak:
    """
    A Market Structure Break (MSB) — the single most important signal.

    From RektProof p.21-23:
    - Valid MSB in uptrend: key higher-low is broken to the downside
    - Valid MSB in downtrend: key lower-high is broken to the upside
    - MSB shifts orderflow from trend to range (accumulate/distribute)

    From CryptoCred Lesson 3:
    - Markets react violently when traders are trapped on wrong side
    - The MSB is what creates the "trap" — breakout traders enter,
      then price reverses and traps them
    """
    index: int                          # Bar where the break occurred
    break_type: str                     # "bullish_msb" or "bearish_msb"
    broken_swing: SwingPoint            # The swing point that was broken
    break_price: float                  # Exact price of the break
    prior_regime: MarketRegime          # What was the regime before?
    demand_zone: Optional[tuple] = None # (high, low) of formed demand
    supply_zone: Optional[tuple] = None # (high, low) of formed supply


@dataclass
class StructureState:
    """Complete snapshot of current market structure."""
    regime: MarketRegime
    swing_points: list                   # All confirmed SwingPoints
    key_levels: list                     # Only key structural levels
    msb_events: list                     # All MSB events detected
    last_hh: Optional[SwingPoint] = None
    last_hl: Optional[SwingPoint] = None
    last_lh: Optional[SwingPoint] = None
    last_ll: Optional[SwingPoint] = None
    current_trend_swings: int = 0        # How many swings in current trend


class MarketStructureEngine:
    """
    The core engine that analyzes OHLCV data and determines market structure.

    Parameters:
    -----------
    swing_lookback : int
        Number of bars on each side required to confirm a swing point.
        Panel decision: 5 for 1H timeframe.
        - Trader 1: "5 gives us 10 hours of context on 1H, filters noise"
        - Trader 5: "Matches standard fractal indicator settings"

    atr_period : int
        Period for ATR calculation used in swing filtering.
        Panel decision: 14 (standard)

    min_swing_atr_multiple : float
        Minimum swing size as a multiple of ATR.
        Panel decision: 0.5
        - Trader 5: "Below 0.5 ATR, a swing is just noise"
        - Trader 3: "Agreed — minor swings during re-accumulation aren't structural"

    equal_threshold_atr : float
        How close two levels need to be (in ATR multiples) to be "equal."
        Panel decision: 0.1
        - Trader 1: "Equal highs/lows are critical liquidity pools"
        - CryptoCred Lesson 3: Equal highs/lows attract orders
    """

    def __init__(
        self,
        swing_lookback: int = 5,
        atr_period: int = 14,
        min_swing_atr_multiple: float = 0.5,
        equal_threshold_atr: float = 0.1,
    ):
        self.swing_lookback = swing_lookback
        self.atr_period = atr_period
        self.min_swing_atr_multiple = min_swing_atr_multiple
        self.equal_threshold_atr = equal_threshold_atr

    def analyze(
        self,
        opens: np.ndarray,
        highs: np.ndarray,
        lows: np.ndarray,
        closes: np.ndarray,
        volumes: np.ndarray,
        timestamps: Optional[list] = None,
    ) -> StructureState:
        """
        Main analysis pipeline. Takes OHLCV arrays and returns complete
        market structure state.

        Pipeline:
        1. Calculate ATR for volatility context
        2. Detect raw swing points (fractal method)
        3. Filter by minimum ATR size
        4. Enforce alternation
        5. Classify swings (HH/HL/LH/LL)
        6. Identify key structural levels
        7. Detect MSBs
        8. Determine market regime
        """
        n = len(closes)
        if n < self.atr_period + self.swing_lookback * 2:
            return StructureState(
                regime=MarketRegime.UNKNOWN,
                swing_points=[],
                key_levels=[],
                msb_events=[],
            )

        # Step 1: Calculate ATR
        atr = self._calculate_atr(highs, lows, closes)

        # Step 2: Detect raw swing points
        raw_swings = self._detect_raw_swings(highs, lows, timestamps)

        # Step 3: Filter by minimum swing size
        filtered_swings = self._filter_by_atr(raw_swings, atr)

        # Step 4: Enforce alternation (must go H-L-H-L)
        alternating_swings = self._enforce_alternation(filtered_swings)

        # Step 5: Classify each swing (HH/HL/LH/LL)
        classified_swings = self._classify_swings(alternating_swings, atr)

        # Step 6: Identify key structural levels
        key_levels = self._identify_key_levels(classified_swings)

        # Step 7: Detect MSBs
        msb_events = self._detect_msb(
            classified_swings, key_levels, highs, lows, closes, atr
        )

        # Step 8: Determine market regime
        regime = self._determine_regime(classified_swings, msb_events)

        # Build state
        state = StructureState(
            regime=regime,
            swing_points=classified_swings,
            key_levels=key_levels,
            msb_events=msb_events,
        )

        # Populate convenience references
        self._populate_latest_swings(state, classified_swings)

        return state

    # =========================================================================
    # Step 1: ATR Calculation
    # =========================================================================

    def _calculate_atr(
        self, highs: np.ndarray, lows: np.ndarray, closes: np.ndarray
    ) -> np.ndarray:
        """
        Average True Range — measures volatility.

        Used for:
        - Filtering insignificant swings (Trader 5)
        - Determining "equal" levels (Trader 1)
        - Context for stop placement (Trader 2, EmperorBTC Ch.7)

        True Range = max(H-L, |H-Cprev|, |L-Cprev|)
        ATR = SMA(TR, period)
        """
        n = len(closes)
        tr = np.zeros(n)
        tr[0] = highs[0] - lows[0]

        for i in range(1, n):
            hl = highs[i] - lows[i]
            hc = abs(highs[i] - closes[i - 1])
            lc = abs(lows[i] - closes[i - 1])
            tr[i] = max(hl, hc, lc)

        atr = np.zeros(n)
        # Use SMA for initial period, then EMA-style for responsiveness
        if n >= self.atr_period:
            atr[self.atr_period - 1] = np.mean(tr[: self.atr_period])
            for i in range(self.atr_period, n):
                atr[i] = (
                    atr[i - 1] * (self.atr_period - 1) + tr[i]
                ) / self.atr_period
        # Backfill early values
        first_valid = self.atr_period - 1
        for i in range(first_valid):
            atr[i] = atr[first_valid] if first_valid < n else tr[i]

        return atr

    # =========================================================================
    # Step 2: Raw Swing Detection (Fractal Method)
    # =========================================================================

    def _detect_raw_swings(
        self,
        highs: np.ndarray,
        lows: np.ndarray,
        timestamps: Optional[list] = None,
    ) -> list:
        """
        Fractal-based swing detection.

        A swing high at bar i is confirmed when:
          highs[i] > highs[j] for all j in [i-N, i+N] where j != i

        A swing low at bar i is confirmed when:
          lows[i] < lows[j] for all j in [i-N, i+N] where j != i

        Panel note (Trader 1):
        "This introduces a lag of N bars. On 1H with N=5, we confirm
        a swing 5 hours after it forms. This is acceptable — we need
        confirmation, not prediction. CryptoCred explicitly states:
        wait for confirmation."

        Panel note (Trader 3):
        "The lag is a feature, not a bug. RektProof's highest hit-rate
        setup (76% breaker) requires the MSB to be CONFIRMED before
        entry. Pre-confirmation = 29% hit rate."
        """
        n = len(highs)
        lb = self.swing_lookback
        swings = []

        for i in range(lb, n - lb):
            # Check swing high
            is_sh = True
            for j in range(i - lb, i + lb + 1):
                if j == i:
                    continue
                if highs[j] >= highs[i]:
                    is_sh = False
                    break

            if is_sh:
                ts = timestamps[i] if timestamps else None
                swings.append(
                    SwingPoint(
                        index=i,
                        price=highs[i],
                        swing_type=SwingType.HIGH,
                        timestamp=ts,
                    )
                )

            # Check swing low
            is_sl = True
            for j in range(i - lb, i + lb + 1):
                if j == i:
                    continue
                if lows[j] <= lows[i]:
                    is_sl = False
                    break

            if is_sl:
                ts = timestamps[i] if timestamps else None
                swings.append(
                    SwingPoint(
                        index=i,
                        price=lows[i],
                        swing_type=SwingType.LOW,
                        timestamp=ts,
                    )
                )

        # Sort by bar index
        swings.sort(key=lambda s: s.index)
        return swings

    # =========================================================================
    # Step 3: ATR-Based Swing Filter
    # =========================================================================

    def _filter_by_atr(
        self, swings: list, atr: np.ndarray
    ) -> list:
        """
        Remove swings that are too small relative to current volatility.

        Panel decision (Trader 5):
        "A swing that moves less than 0.5 * ATR from the previous opposite
        swing is noise, not structure. In low-volatility environments this
        auto-adapts — we don't need separate parameters for different regimes."

        Panel agreement (Trader 3):
        "This is exactly what RektProof means by 'minor swings during
        re-accumulation don't invalidate structure.' Small wiggles inside
        a range aren't structural moves."
        """
        if len(swings) < 2:
            return swings

        filtered = [swings[0]]

        for i in range(1, len(swings)):
            current = swings[i]
            prev_opposite = None

            # Find the most recent swing of the OPPOSITE type
            for j in range(len(filtered) - 1, -1, -1):
                if filtered[j].swing_type != current.swing_type:
                    prev_opposite = filtered[j]
                    break

            if prev_opposite is None:
                filtered.append(current)
                continue

            # Calculate the move size from previous opposite swing
            move_size = abs(current.price - prev_opposite.price)
            atr_at_current = atr[current.index]
            min_move = atr_at_current * self.min_swing_atr_multiple

            if move_size >= min_move:
                filtered.append(current)
            # else: swing is too small, skip it

        return filtered

    # =========================================================================
    # Step 4: Enforce Alternation
    # =========================================================================

    def _enforce_alternation(self, swings: list) -> list:
        """
        Ensure swings strictly alternate: H-L-H-L or L-H-L-H.

        When two consecutive swings are the same type, keep the more
        extreme one (higher high or lower low).

        Panel note (Trader 1):
        "CryptoCred's Lesson 1 diagram shows this clearly — an uptrend
        is impulse-retrace-impulse. Each impulse creates a swing high,
        each retrace creates a swing low. They MUST alternate."

        Panel note (from useThinkScript research):
        "It's not really a swing high if it is followed by another swing
        high without an intervening low." — This is correct and we enforce it.
        """
        if len(swings) < 2:
            return swings

        result = [swings[0]]

        for i in range(1, len(swings)):
            current = swings[i]
            last = result[-1]

            if current.swing_type != last.swing_type:
                # Different type — good, alternation holds
                result.append(current)
            else:
                # Same type — keep the more extreme one
                if current.swing_type == SwingType.HIGH:
                    if current.price > last.price:
                        result[-1] = current  # Replace with higher high
                else:  # LOW
                    if current.price < last.price:
                        result[-1] = current  # Replace with lower low

        return result

    # =========================================================================
    # Step 5: Classify Swings (HH/HL/LH/LL)
    # =========================================================================

    def _classify_swings(
        self, swings: list, atr: np.ndarray
    ) -> list:
        """
        Compare each swing to the previous swing of the SAME type
        and label it as HH/LH/EH or HL/LL/EL.

        Panel note (Trader 1):
        "This is the core of market structure reading.
        HH + HL = uptrend (CryptoCred Lesson 1)
        LH + LL = downtrend
        The sequence of these labels tells us everything."

        Panel note (Trader 5):
        "Equal levels (within ATR tolerance) are critically important.
        CryptoCred Lesson 3: 'Equal highs/lows attract liquidity.'
        We tag them separately because they're prime liquidity pools."
        """
        prev_high = None
        prev_low = None

        for swing in swings:
            swing.atr_at_point = atr[swing.index]
            threshold = atr[swing.index] * self.equal_threshold_atr

            if swing.swing_type == SwingType.HIGH:
                if prev_high is None:
                    swing.label = SwingLabel.FIRST
                else:
                    diff = swing.price - prev_high.price
                    if abs(diff) <= threshold:
                        swing.label = SwingLabel.EH
                    elif diff > 0:
                        swing.label = SwingLabel.HH
                    else:
                        swing.label = SwingLabel.LH
                prev_high = swing

            else:  # LOW
                if prev_low is None:
                    swing.label = SwingLabel.FIRST
                else:
                    diff = swing.price - prev_low.price
                    if abs(diff) <= threshold:
                        swing.label = SwingLabel.EL
                    elif diff > 0:
                        swing.label = SwingLabel.HL
                    else:
                        swing.label = SwingLabel.LL
                prev_low = swing

        return swings

    # =========================================================================
    # Step 6: Identify Key Structural Levels
    # =========================================================================

    def _identify_key_levels(self, swings: list) -> list:
        """
        Not all swings are equal. "Key" levels are the ones that define
        the current trend structure.

        From RektProof p.22:
        "Valid market structure follows orderflow with unbroken lows as
        price exhausts and enters re-accumulation."

        Translation for code:
        - In an uptrend, the HIGHER LOWS are key levels. If a HL breaks,
          structure shifts.
        - In a downtrend, the LOWER HIGHS are key levels. If a LH breaks,
          structure shifts.
        - Equal highs/lows are ALWAYS key (liquidity pools)

        Panel decision (Trader 3):
        "A key level is one whose break would change the trend narrative.
        In an uptrend, the most recent HL is 'the floor.' Break it and
        bulls are in trouble. That's the key level."
        """
        key_levels = []

        for swing in swings:
            is_key = False

            # Equal levels are always key (liquidity pools)
            if swing.label in (SwingLabel.EH, SwingLabel.EL):
                is_key = True

            # Trend-defining swings are key
            elif swing.label in (SwingLabel.HH, SwingLabel.HL):
                # In a potential uptrend, both HH and HL define structure
                is_key = True

            elif swing.label in (SwingLabel.LH, SwingLabel.LL):
                # In a potential downtrend, both LH and LL define structure
                is_key = True

            if is_key:
                swing.is_key_level = True
                key_levels.append(swing)

        return key_levels

    # =========================================================================
    # Step 7: Detect Market Structure Breaks
    # =========================================================================

    def _detect_msb(
        self,
        swings: list,
        key_levels: list,
        highs: np.ndarray,
        lows: np.ndarray,
        closes: np.ndarray,
        atr: np.ndarray,
    ) -> list:
        """
        Detect Market Structure Breaks — THE critical signal.

        From RektProof pp.2-3 (Breaker Setup):
        "Demand fails to hold and you have a MSB to the downside."

        From RektProof p.23 (Valid MSB):
        "When market structure breaks a key low as price trends.
        Price goes range bound into distribution."

        From CryptoCred Lesson 3 (Order Flow):
        "Markets tend to react violently when traders are trapped on
        the wrong side of attractive reference points."

        Rules (Panel consensus):
        1. Bullish MSB: Price closes above the most recent LOWER HIGH
           in a downtrend → potential shift to bullish
        2. Bearish MSB: Price closes below the most recent HIGHER LOW
           in an uptrend → potential shift to bearish
        3. The CLOSE must be beyond the level, not just a wick
           (EmperorBTC: "fake breakout" wicks don't count)
        4. After MSB, the origin candle(s) form the demand/supply zone

        Panel note (Trader 2):
        "The MSB is where the Breaker setup begins. Once demand/supply
        fails and structure breaks, the failed zone becomes the breaker
        zone for the retest entry."
        """
        msb_events = []

        if len(swings) < 4:
            return msb_events

        # We need to track the "current" regime as we scan forward
        # to know which breaks are meaningful
        regime_context = MarketRegime.UNKNOWN

        # Collect swing highs and lows separately for tracking
        swing_highs = [s for s in swings if s.swing_type == SwingType.HIGH]
        swing_lows = [s for s in swings if s.swing_type == SwingType.LOW]

        # For each confirmed swing, check if subsequent price breaks it
        for i, swing in enumerate(swings):
            if swing.broken:
                continue

            # Determine what bars to check (from after the swing to end)
            start_check = swing.index + 1
            end_check = len(closes)

            # But only check until the NEXT swing of the same type
            # (after which structure has evolved)
            next_same = None
            for j in range(i + 1, len(swings)):
                if swings[j].swing_type == swing.swing_type:
                    next_same = swings[j]
                    break

            if next_same:
                end_check = min(end_check, next_same.index + 1)

            for bar_idx in range(start_check, end_check):
                if bar_idx >= len(closes):
                    break

                # Bearish MSB: Close below a Higher Low
                if (
                    swing.swing_type == SwingType.LOW
                    and swing.label == SwingLabel.HL
                    and closes[bar_idx] < swing.price
                ):
                    # Find the demand zone (candles that created the move
                    # up from this HL)
                    demand_zone = self._find_zone_around_swing(
                        swing, highs, lows, "demand"
                    )

                    msb = MarketStructureBreak(
                        index=bar_idx,
                        break_type="bearish_msb",
                        broken_swing=swing,
                        break_price=closes[bar_idx],
                        prior_regime=MarketRegime.UPTREND,
                        demand_zone=demand_zone,
                    )
                    msb_events.append(msb)
                    swing.broken = True
                    swing.broken_at_index = bar_idx
                    break

                # Bullish MSB: Close above a Lower High
                if (
                    swing.swing_type == SwingType.HIGH
                    and swing.label == SwingLabel.LH
                    and closes[bar_idx] > swing.price
                ):
                    supply_zone = self._find_zone_around_swing(
                        swing, highs, lows, "supply"
                    )

                    msb = MarketStructureBreak(
                        index=bar_idx,
                        break_type="bullish_msb",
                        broken_swing=swing,
                        break_price=closes[bar_idx],
                        prior_regime=MarketRegime.DOWNTREND,
                        supply_zone=supply_zone,
                    )
                    msb_events.append(msb)
                    swing.broken = True
                    swing.broken_at_index = bar_idx
                    break

        return msb_events

    def _find_zone_around_swing(
        self,
        swing: SwingPoint,
        highs: np.ndarray,
        lows: np.ndarray,
        zone_type: str,
    ) -> tuple:
        """
        Find the supply/demand zone created around a swing point.

        From RektProof p.2 (Breaker Setup):
        "Formed Demand in grey" — the demand zone is the candle body
        area around the swing low that initiated the move.

        We use the candle at the swing point and 1 candle on each side
        to define the zone boundaries.
        """
        idx = swing.index
        start = max(0, idx - 1)
        end = min(len(highs) - 1, idx + 1)

        zone_high = max(highs[start : end + 1])
        zone_low = min(lows[start : end + 1])

        return (zone_high, zone_low)

    # =========================================================================
    # Step 8: Determine Market Regime
    # =========================================================================

    def _determine_regime(
        self, swings: list, msb_events: list
    ) -> MarketRegime:
        """
        Classify the current market regime based on swing sequence.

        From CryptoCred Lesson 1:
        - Uptrend: HH and HL → retraces are for buying
        - Downtrend: LH and LL → retraces are for selling

        From RektProof p.21:
        - Orderflow: Trend → Range → Trend → Range
        - After MSB, expect range/transition before new trend

        Panel decision (all 5):
        "We look at the LAST 4 swings minimum (2 highs, 2 lows) to
        classify. If ALL recent swings agree, it's strong. If only
        the most recent agrees, it's regular. If they conflict,
        it's transition or ranging."
        """
        if len(swings) < 4:
            return MarketRegime.UNKNOWN

        # Get the last few classified swings
        recent_highs = [
            s for s in swings[-8:]
            if s.swing_type == SwingType.HIGH and s.label != SwingLabel.FIRST
        ]
        recent_lows = [
            s for s in swings[-8:]
            if s.swing_type == SwingType.LOW and s.label != SwingLabel.FIRST
        ]

        if not recent_highs or not recent_lows:
            return MarketRegime.UNKNOWN

        # Check the most recent high and low labels
        last_high_label = recent_highs[-1].label if recent_highs else None
        last_low_label = recent_lows[-1].label if recent_lows else None

        # Check if there was a recent MSB (indicates transition)
        recent_msb = None
        if msb_events:
            recent_msb = msb_events[-1]

        # STRONG UPTREND: Last 2+ highs are HH AND last 2+ lows are HL
        if len(recent_highs) >= 2 and len(recent_lows) >= 2:
            all_hh = all(
                h.label in (SwingLabel.HH, SwingLabel.EH)
                for h in recent_highs[-2:]
            )
            all_hl = all(
                l.label in (SwingLabel.HL, SwingLabel.EL)
                for l in recent_lows[-2:]
            )
            if all_hh and all_hl:
                return MarketRegime.STRONG_UPTREND

            all_lh = all(
                h.label in (SwingLabel.LH, SwingLabel.EH)
                for h in recent_highs[-2:]
            )
            all_ll = all(
                l.label in (SwingLabel.LL, SwingLabel.EL)
                for l in recent_lows[-2:]
            )
            if all_lh and all_ll:
                return MarketRegime.STRONG_DOWNTREND

        # REGULAR UPTREND: Most recent = HH + HL
        if last_high_label in (SwingLabel.HH, SwingLabel.EH) and last_low_label in (
            SwingLabel.HL,
            SwingLabel.EL,
        ):
            return MarketRegime.UPTREND

        # REGULAR DOWNTREND: Most recent = LH + LL
        if last_high_label in (SwingLabel.LH, SwingLabel.EH) and last_low_label in (
            SwingLabel.LL,
            SwingLabel.EL,
        ):
            return MarketRegime.DOWNTREND

        # WEAK/DIVERGENT: HH but LL, or LH but HL
        if last_high_label == SwingLabel.HH and last_low_label == SwingLabel.LL:
            return MarketRegime.WEAK_UPTREND  # Expanding range

        if last_high_label == SwingLabel.LH and last_low_label == SwingLabel.HL:
            return MarketRegime.RANGING  # Contracting range

        # TRANSITION: Recent MSB suggests structure is shifting
        if recent_msb:
            return MarketRegime.TRANSITION

        return MarketRegime.TRANSITION

    # =========================================================================
    # Utility Methods
    # =========================================================================

    def _populate_latest_swings(
        self, state: StructureState, swings: list
    ):
        """Fill in convenience references to latest HH/HL/LH/LL."""
        for swing in reversed(swings):
            if swing.swing_type == SwingType.HIGH:
                if swing.label == SwingLabel.HH and state.last_hh is None:
                    state.last_hh = swing
                elif swing.label == SwingLabel.LH and state.last_lh is None:
                    state.last_lh = swing
            else:
                if swing.label == SwingLabel.HL and state.last_hl is None:
                    state.last_hl = swing
                elif swing.label == SwingLabel.LL and state.last_ll is None:
                    state.last_ll = swing

            # Stop when we've found all
            if all([state.last_hh, state.last_hl, state.last_lh, state.last_ll]):
                break

    def get_swing_sequence(self, swings: list, last_n: int = 8) -> str:
        """
        Return a human-readable string of the last N swing labels.
        Useful for debugging and pattern matching.

        Example: "HH-HL-HH-HL" = clean uptrend
                 "HH-HL-LH-LL" = trend reversal
                 "HH-LL-LH-HL" = ranging/choppy
        """
        recent = swings[-last_n:] if len(swings) >= last_n else swings
        labels = []
        for s in recent:
            if s.label == SwingLabel.FIRST:
                continue
            labels.append(s.label.value.upper().replace("_", ""))
        return "-".join(labels)


# =============================================================================
# Multi-Timeframe Structure Analyzer
# =============================================================================

class MultiTimeframeStructure:
    """
    Combines market structure analysis across multiple timeframes.

    From CryptoCred Lesson 1:
    "Higher time frames (Monthly/Weekly/Daily) dictate macro market
    flow/direction. Lower time frames can be used to refine higher
    time frame structures for more precise trade parameters."

    From CryptoCred Lesson 3:
    "If HTF bearish, then a move above old highs on intraday basis
    is more likely to be a 'trap' before continuation lower."

    Panel configuration for crypto intraday:
    - Entry timeframe: 1H (swing_lookback=5)
    - Context timeframe 1: 4H (swing_lookback=5, effectively 20H context)
    - Context timeframe 2: Daily (swing_lookback=3, effectively 3 days)

    Panel note (Trader 1):
    "We don't need ALL timeframes to agree. The entry TF determines
    the setup. The higher TFs provide BIAS — are we looking for longs
    or shorts? The best setups occur when the entry TF setup aligns
    with the higher TF bias."
    """

    def __init__(self):
        self.entry_engine = MarketStructureEngine(swing_lookback=5)
        self.context_4h = MarketStructureEngine(swing_lookback=5)
        self.context_daily = MarketStructureEngine(swing_lookback=3)

    def analyze_all(
        self,
        ohlcv_1h: dict,
        ohlcv_4h: dict,
        ohlcv_daily: dict,
    ) -> dict:
        """
        Analyze all three timeframes and return combined state.

        Returns dict with:
        - entry_state: 1H StructureState
        - context_4h_state: 4H StructureState
        - context_daily_state: Daily StructureState
        - htf_bias: "bullish", "bearish", "neutral"
        - alignment: "aligned", "conflicting", "neutral"
        """
        entry_state = self.entry_engine.analyze(**ohlcv_1h)
        context_4h = self.context_4h.analyze(**ohlcv_4h)
        context_daily = self.context_daily.analyze(**ohlcv_daily)

        # Determine HTF bias
        htf_bias = self._determine_htf_bias(context_4h, context_daily)

        # Determine alignment
        alignment = self._check_alignment(entry_state, htf_bias)

        return {
            "entry_state": entry_state,
            "context_4h_state": context_4h,
            "context_daily_state": context_daily,
            "htf_bias": htf_bias,
            "alignment": alignment,
        }

    def _determine_htf_bias(
        self,
        state_4h: StructureState,
        state_daily: StructureState,
    ) -> str:
        """
        Determine higher timeframe directional bias.

        Panel decision (all 5):
        "If Daily is in uptrend/strong uptrend, bias is bullish.
        If Daily is in downtrend/strong downtrend, bias is bearish.
        If Daily conflicts or is ranging, look at 4H as tiebreaker.
        If both are unclear, bias is neutral — we can trade both directions
        but with reduced size."
        """
        bullish_regimes = {
            MarketRegime.STRONG_UPTREND,
            MarketRegime.UPTREND,
        }
        bearish_regimes = {
            MarketRegime.STRONG_DOWNTREND,
            MarketRegime.DOWNTREND,
        }

        daily_bull = state_daily.regime in bullish_regimes
        daily_bear = state_daily.regime in bearish_regimes
        h4_bull = state_4h.regime in bullish_regimes
        h4_bear = state_4h.regime in bearish_regimes

        if daily_bull:
            return "bullish"
        if daily_bear:
            return "bearish"

        # Daily unclear — use 4H as tiebreaker
        if h4_bull:
            return "bullish"
        if h4_bear:
            return "bearish"

        return "neutral"

    def _check_alignment(
        self, entry_state: StructureState, htf_bias: str
    ) -> str:
        """
        Check if entry timeframe structure aligns with HTF bias.

        From CryptoCred:
        "If you're bearish higher time frame, an intraday move up
        is (broadly) an opportunity to sell."

        Panel interpretation:
        - Aligned: Entry TF trend matches HTF bias → full conviction
        - Conflicting: Entry TF trend opposes HTF bias → reduced size or skip
        - Neutral: HTF has no clear bias → trade both directions cautiously
        """
        if htf_bias == "neutral":
            return "neutral"

        bullish_regimes = {
            MarketRegime.STRONG_UPTREND,
            MarketRegime.UPTREND,
            MarketRegime.WEAK_UPTREND,
        }
        bearish_regimes = {
            MarketRegime.STRONG_DOWNTREND,
            MarketRegime.DOWNTREND,
            MarketRegime.WEAK_DOWNTREND,
        }

        if htf_bias == "bullish" and entry_state.regime in bullish_regimes:
            return "aligned"
        if htf_bias == "bearish" and entry_state.regime in bearish_regimes:
            return "aligned"

        if htf_bias == "bullish" and entry_state.regime in bearish_regimes:
            return "conflicting"
        if htf_bias == "bearish" and entry_state.regime in bullish_regimes:
            return "conflicting"

        return "neutral"
