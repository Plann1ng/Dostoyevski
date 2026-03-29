"""
Module 3: Breaker Setup Detector
=================================

The Breaker is RektProof's highest hit-rate setup at 76%.
This module detects both bullish and bearish breaker formations.

Source: RektProof Setups pp.2-3, cross-referenced with:
- RektProof Reversal Pattern pp.13-15 (same mechanics, bullish side)
- CryptoCred Lesson 3 (order flow: why trapped traders fuel the move)
- EmperorBTC Ch.4 (S/R flip: "resistance becomes support")
- EmperorBTC Ch.7 (stop loss placement around invalidation)

=== BEARISH BREAKER MECHANICS (Short Setup) ===

RektProof p.2: "High forms, Price breaks the high/deviates above
and prints a market structure break. Formed Demand in grey."

Step 1: A swing high forms in the market
Step 2: Price pushes above this high (deviation/sweep)
        → This creates a DEMAND zone (the candles that pushed price up)
Step 3: "Ideally you want this demand to hold to confirm a valid shift"
        → If demand holds, it's a valid bullish shift — NO breaker
Step 4: "Demand fails to hold and you have a MSB to the downside"
        → The demand zone FAILS — price closes below it
        → This is the BEARISH MSB
        → The failed demand zone becomes the BEARISH BREAKER
Step 5: "Price retests the newly formed breaker"
        → ENTRY: Short when price retests the breaker zone from below
Step 6: "Price proceeds to the next area of liquidity (Equal Lows)"
        → TARGET: Next area of liquidity below

=== BULLISH BREAKER MECHANICS (Long Setup) ===
(Mirror of above, from RektProof Reversal Pattern p.13)

Step 1: A swing low forms
Step 2: Price pushes below this low (deviation/sweep)
        → Creates a SUPPLY zone
Step 3: Supply should hold for valid bearish shift
Step 4: Supply FAILS to hold → MSB to the upside
        → Failed supply zone becomes the BULLISH BREAKER
Step 5: Entry: Long when price retests the breaker zone from above
Step 6: Target: Next area of liquidity above

=== WHY IT WORKS (CryptoCred Lesson 3 Cross-Reference) ===

"Markets tend to react violently when traders are trapped on the
wrong side of attractive reference points."

The breaker works because:
1. When demand forms, longs enter expecting higher prices
2. When demand fails, those longs are TRAPPED — holding underwater
3. When price retests the failed demand (now breaker), trapped longs
   EXIT at breakeven → their selling = fuel for the move down
4. Smart traders short the retest knowing this exit pressure exists

Panel Notes:
- Trader 3: "This is my bread and butter. The key is WAITING for the
  retest. Don't short the MSB itself — wait for the breaker retest."
- Trader 1: "The breaker zone must be precise. Use the candle bodies
  that formed the original demand/supply, plus wicks for the zone."
- Trader 2: "Stop goes above the breaker zone high (for shorts) or
  below the breaker zone low (for longs). If it reclaims the zone,
  the thesis is dead."
- Trader 5: "We need to filter: the MSB must be a CLEAN close through
  the level, not just a wick. EmperorBTC confirms: wicks can be fakeouts."
- Trader 4: "Every signal gets logged. We track: hit rate, average R,
  average time to target, and which market regimes produce the best setups."
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List
from enum import Enum

from market_structure import (
    MarketStructureEngine,
    StructureState,
    SwingPoint,
    SwingType,
    SwingLabel,
    MarketRegime,
    MarketStructureBreak,
)


class BreakerType(Enum):
    BULLISH = "bullish_breaker"   # Failed supply → long on retest
    BEARISH = "bearish_breaker"   # Failed demand → short on retest


class SignalStatus(Enum):
    FORMING = "forming"          # Breaker zone identified, waiting for retest
    TRIGGERED = "triggered"      # Price entered the breaker zone
    ACTIVE = "active"            # Trade is live
    TARGET_HIT = "target_hit"    # Take profit reached
    STOPPED = "stopped"          # Stop loss hit
    INVALIDATED = "invalidated"  # Zone broken through — thesis dead
    EXPIRED = "expired"          # Too much time passed without trigger


@dataclass
class BreakerZone:
    """
    A breaker zone — a failed demand or supply zone that becomes
    the entry point for the breaker trade.
    """
    zone_high: float              # Upper boundary of the zone
    zone_low: float               # Lower boundary of the zone
    zone_mid: float               # Midpoint (common entry target)
    breaker_type: BreakerType     # Bullish or bearish
    formation_index: int          # Bar where the MSB confirmed this as breaker
    original_swing: SwingPoint    # The swing that was broken (the failed S/D)
    msb_event: MarketStructureBreak  # The MSB that confirmed the break

    # Trade parameters (filled in by risk engine, but defined here)
    entry_price: Optional[float] = None
    stop_price: Optional[float] = None
    target_price: Optional[float] = None

    # Status tracking
    status: SignalStatus = SignalStatus.FORMING
    trigger_index: Optional[int] = None   # Bar where retest occurred
    exit_index: Optional[int] = None      # Bar where trade ended
    exit_price: Optional[float] = None
    pnl_r: Optional[float] = None         # P&L in R multiples

    # Quality score (higher = more confluent)
    quality_score: float = 0.0

    # Context
    regime_at_formation: Optional[MarketRegime] = None
    htf_bias: Optional[str] = None


class BreakerDetector:
    """
    Detects Breaker setups from market structure analysis.

    Parameters:
    -----------
    max_bars_to_retest : int
        Maximum bars to wait for the retest after MSB.
        Panel decision: 50 bars on 1H (roughly 2 trading days)
        - Trader 3: "If it doesn't come back in 2 days, the move
          is too extended. The trapped traders have already exited."
        - Trader 5: "Statistically, retests within 24-48 hours have
          higher hit rates than later retests."

    zone_buffer_atr : float
        How much to extend the zone beyond the strict candle boundaries.
        Panel decision: 0.1 * ATR
        - Trader 1: "Price doesn't have to hit the zone exactly.
          Give it a small buffer."

    min_move_from_zone_atr : float
        Minimum distance price must move away from zone before
        a retest is considered valid.
        Panel decision: 1.0 * ATR
        - Trader 5: "If price never leaves the zone, there's no
          retest — it just chopped through it."

    require_htf_alignment : bool
        Whether to require HTF bias alignment for the signal.
        Panel decision: True for production, False for backtesting
        - Trader 1: "In production, ONLY take breakers aligned with
          HTF bias. In backtesting, log both to compare."
    """

    def __init__(
        self,
        max_bars_to_retest: int = 50,
        zone_buffer_atr: float = 0.1,
        min_move_from_zone_atr: float = 1.0,
        require_htf_alignment: bool = False,
    ):
        self.max_bars_to_retest = max_bars_to_retest
        self.zone_buffer_atr = zone_buffer_atr
        self.min_move_from_zone_atr = min_move_from_zone_atr
        self.require_htf_alignment = require_htf_alignment

    def detect(
        self,
        structure_state: StructureState,
        highs: np.ndarray,
        lows: np.ndarray,
        closes: np.ndarray,
        atr: np.ndarray,
        htf_bias: str = "neutral",
    ) -> List[BreakerZone]:
        """
        Scan for breaker setups from MSB events in the structure state.

        Pipeline:
        1. For each MSB event, identify the failed demand/supply zone
        2. Calculate zone boundaries with buffer
        3. Check if price moved away from zone (required for valid retest)
        4. Scan for retest of the zone
        5. Score the setup quality
        6. Return all detected breaker zones with their status
        """
        breakers = []

        for msb in structure_state.msb_events:
            breaker = self._process_msb(
                msb, structure_state, highs, lows, closes, atr, htf_bias
            )
            if breaker is not None:
                breakers.append(breaker)

        return breakers

    def _process_msb(
        self,
        msb: MarketStructureBreak,
        state: StructureState,
        highs: np.ndarray,
        lows: np.ndarray,
        closes: np.ndarray,
        atr: np.ndarray,
        htf_bias: str,
    ) -> Optional[BreakerZone]:
        """
        Process a single MSB event into a potential breaker setup.

        From RektProof p.2-3:
        - Bearish MSB: demand fails → demand zone becomes bearish breaker
        - The breaker zone is the original demand zone that failed

        From RektProof p.13-15 (Reversal Pattern):
        - Bullish MSB: supply fails → supply zone becomes bullish breaker
        """
        if msb.index >= len(closes):
            return None

        atr_at_msb = atr[msb.index] if msb.index < len(atr) else atr[-1]

        if msb.break_type == "bearish_msb":
            # The DEMAND zone that failed becomes the BEARISH breaker
            # Entry: short when price retests this zone from below
            if msb.demand_zone is None:
                return None

            zone_high = msb.demand_zone[0] + (atr_at_msb * self.zone_buffer_atr)
            zone_low = msb.demand_zone[1] - (atr_at_msb * self.zone_buffer_atr)
            breaker_type = BreakerType.BEARISH

        elif msb.break_type == "bullish_msb":
            # The SUPPLY zone that failed becomes the BULLISH breaker
            # Entry: long when price retests this zone from above
            if msb.supply_zone is None:
                return None

            zone_high = msb.supply_zone[0] + (atr_at_msb * self.zone_buffer_atr)
            zone_low = msb.supply_zone[1] - (atr_at_msb * self.zone_buffer_atr)
            breaker_type = BreakerType.BULLISH

        else:
            return None

        zone_mid = (zone_high + zone_low) / 2

        breaker = BreakerZone(
            zone_high=zone_high,
            zone_low=zone_low,
            zone_mid=zone_mid,
            breaker_type=breaker_type,
            formation_index=msb.index,
            original_swing=msb.broken_swing,
            msb_event=msb,
            regime_at_formation=msb.prior_regime,
            htf_bias=htf_bias,
        )

        # Now scan for the retest
        self._scan_for_retest(
            breaker, highs, lows, closes, atr
        )

        # Score the quality
        breaker.quality_score = self._score_quality(
            breaker, state, atr, htf_bias
        )

        return breaker

    def _scan_for_retest(
        self,
        breaker: BreakerZone,
        highs: np.ndarray,
        lows: np.ndarray,
        closes: np.ndarray,
        atr: np.ndarray,
    ):
        """
        Scan price data after the MSB for a retest of the breaker zone.

        The retest must follow this sequence:
        1. Price moves AWAY from the zone (confirming the MSB)
        2. Price RETURNS to the zone (the retest)
        3. Price REJECTS from the zone (confirmation of entry)

        Panel notes:
        - Trader 3: "The move away is critical. If price never leaves
          the zone, there's no setup — just chop."
        - Trader 5: "We need at least 1 ATR of displacement from the
          zone before a return counts as a retest."
        - Trader 2: "For the entry, I want to see the candle close
          showing rejection. Not just a wick into the zone."
        """
        start = breaker.formation_index + 1
        end = min(
            start + self.max_bars_to_retest,
            len(closes)
        )

        if start >= len(closes):
            breaker.status = SignalStatus.EXPIRED
            return

        atr_val = atr[breaker.formation_index]
        min_displacement = atr_val * self.min_move_from_zone_atr

        # Phase 1: Wait for price to move away from the zone
        displaced = False
        displacement_bar = None

        for i in range(start, end):
            if breaker.breaker_type == BreakerType.BEARISH:
                # For bearish breaker, price should move DOWN away from zone
                if lows[i] < breaker.zone_low - min_displacement:
                    displaced = True
                    displacement_bar = i
                    break
            else:
                # For bullish breaker, price should move UP away from zone
                if highs[i] > breaker.zone_high + min_displacement:
                    displaced = True
                    displacement_bar = i
                    break

        if not displaced:
            breaker.status = SignalStatus.EXPIRED
            return

        # Phase 2: Scan for the retest (price returns to the zone)
        retest_start = displacement_bar + 1
        retest_end = min(
            breaker.formation_index + self.max_bars_to_retest,
            len(closes)
        )

        for i in range(retest_start, retest_end):
            if breaker.breaker_type == BreakerType.BEARISH:
                # Bearish breaker retest: price rallies back UP into the zone
                if highs[i] >= breaker.zone_low and lows[i] <= breaker.zone_high:
                    # Price has entered the zone — check for rejection
                    # Rejection = candle closes BELOW the zone midpoint
                    if closes[i] < breaker.zone_mid:
                        breaker.status = SignalStatus.TRIGGERED
                        breaker.trigger_index = i
                        breaker.entry_price = closes[i]
                        return
                    # Price closes above zone = potential invalidation
                    if closes[i] > breaker.zone_high:
                        breaker.status = SignalStatus.INVALIDATED
                        return

            else:  # BULLISH
                # Bullish breaker retest: price dips back DOWN into the zone
                if lows[i] <= breaker.zone_high and highs[i] >= breaker.zone_low:
                    # Check for rejection — closes ABOVE zone midpoint
                    if closes[i] > breaker.zone_mid:
                        breaker.status = SignalStatus.TRIGGERED
                        breaker.trigger_index = i
                        breaker.entry_price = closes[i]
                        return
                    # Price closes below zone = potential invalidation
                    if closes[i] < breaker.zone_low:
                        breaker.status = SignalStatus.INVALIDATED
                        return

        # If we get here, no retest occurred within the window
        breaker.status = SignalStatus.EXPIRED

    def _score_quality(
        self,
        breaker: BreakerZone,
        state: StructureState,
        atr: np.ndarray,
        htf_bias: str,
    ) -> float:
        """
        Score the quality of a breaker setup on a 0-100 scale.

        Higher score = more confluence = higher probability.

        Scoring factors (Panel consensus):

        1. HTF Alignment (+25 points)
           - CryptoCred: "Most successful setups occur following stop runs
             in sync with the higher time frame flow"
           - Bullish breaker + bullish HTF bias = aligned
           - Bearish breaker + bearish HTF bias = aligned
           - Neutral = partial credit

        2. Clean MSB (+20 points)
           - The MSB candle should be decisive (large body, small wicks)
           - Trader 5: "A strong candle close through the level shows
             conviction. A doji or spinning top = weak break."

        3. Zone Clarity (+20 points)
           - Tight zone (small distance between high and low) = cleaner
           - Wide/sloppy zone = less reliable

        4. Displacement Quality (+20 points)
           - How far price moved from zone before retest
           - More displacement = more trapped traders = stronger setup

        5. Trend Strength (+15 points)
           - Strong trend regime at formation = more fuel
           - Transition/ranging = less conviction
        """
        score = 0.0

        # Factor 1: HTF Alignment
        if htf_bias == "neutral":
            score += 12  # Partial credit
        elif (
            (breaker.breaker_type == BreakerType.BULLISH and htf_bias == "bullish")
            or (breaker.breaker_type == BreakerType.BEARISH and htf_bias == "bearish")
        ):
            score += 25  # Full alignment
        else:
            score += 0  # Counter-trend — no points

        # Factor 2: Clean MSB (using zone tightness as proxy)
        atr_at_formation = atr[min(breaker.formation_index, len(atr) - 1)]
        zone_size = breaker.zone_high - breaker.zone_low
        if atr_at_formation > 0:
            zone_ratio = zone_size / atr_at_formation
            if zone_ratio < 0.5:
                score += 20  # Very tight zone
            elif zone_ratio < 1.0:
                score += 15  # Reasonable zone
            elif zone_ratio < 1.5:
                score += 10  # Wide zone
            else:
                score += 5   # Very wide — messy

        # Factor 3: Zone Clarity
        # Same metric as above — tight zone = clear demand/supply area
        if zone_ratio < 0.8:
            score += 20
        elif zone_ratio < 1.2:
            score += 12
        else:
            score += 5

        # Factor 4: Displacement Quality
        # Measured by how far price moved from zone relative to ATR
        # (This is checked during retest scan, but we approximate here)
        if breaker.trigger_index and breaker.formation_index:
            bars_to_retest = breaker.trigger_index - breaker.formation_index
            if bars_to_retest >= 10:
                score += 20  # Good displacement time
            elif bars_to_retest >= 5:
                score += 12
            else:
                score += 5   # Very quick retest

        # Factor 5: Trend Strength at Formation
        regime = breaker.regime_at_formation
        if regime in (MarketRegime.STRONG_UPTREND, MarketRegime.STRONG_DOWNTREND):
            score += 15
        elif regime in (MarketRegime.UPTREND, MarketRegime.DOWNTREND):
            score += 10
        else:
            score += 3

        return min(score, 100.0)


class BreakerTradeManager:
    """
    Manages active breaker trades — determines stops, targets, and outcomes.

    This is where Module 3 (Breaker) meets Module 4 (Risk) and Module 5 (Exit).

    Panel decisions on trade management:

    Stop Loss (Trader 2 leads):
    - Bearish breaker: stop above zone_high + buffer
    - Bullish breaker: stop below zone_low - buffer
    - CryptoCred Lesson 5: "Place stop above/below the candle that
      broke the level. Check it doesn't fall into liquidity."
    - EmperorBTC Ch.7: "Stop = invalidation level + buffer for wicks"

    Target (Trader 1 leads, FTA concept from CryptoCred Lesson 5):
    - Primary target: First Trouble Area (next S/R level)
    - For bearish: next swing low / equal lows / demand zone below
    - For bullish: next swing high / equal highs / supply zone above

    Position Sizing (Trader 2, EmperorBTC Ch.5):
    - Risk per trade: 1-2% of account
    - Position size = Risk Amount / Distance to Stop
    - R:R must be >= 2:1 or trade is rejected
    """

    def __init__(
        self,
        risk_percent: float = 0.02,    # 2% risk per trade
        min_rr: float = 2.0,           # Minimum 2:1 R:R
        stop_buffer_atr: float = 0.2,  # Stop buffer beyond zone
    ):
        self.risk_percent = risk_percent
        self.min_rr = min_rr
        self.stop_buffer_atr = stop_buffer_atr

    def calculate_trade_params(
        self,
        breaker: BreakerZone,
        atr: np.ndarray,
        account_balance: float,
        next_liquidity_target: Optional[float] = None,
    ) -> bool:
        """
        Calculate stop, target, and position size for a breaker trade.
        Returns True if the trade meets minimum R:R, False otherwise.

        Panel note (Trader 2):
        "If the R:R doesn't work, we DON'T TAKE THE TRADE. Period.
        A 76% hit rate means nothing if your losers are 3x your winners."
        """
        if breaker.entry_price is None:
            return False

        idx = breaker.trigger_index or breaker.formation_index
        atr_val = atr[min(idx, len(atr) - 1)]

        if breaker.breaker_type == BreakerType.BEARISH:
            # Short trade
            breaker.stop_price = breaker.zone_high + (atr_val * self.stop_buffer_atr)
            if next_liquidity_target:
                breaker.target_price = next_liquidity_target
            else:
                # Default: 2x the risk distance below entry
                risk_dist = breaker.stop_price - breaker.entry_price
                breaker.target_price = breaker.entry_price - (risk_dist * self.min_rr)

        else:  # BULLISH
            # Long trade
            breaker.stop_price = breaker.zone_low - (atr_val * self.stop_buffer_atr)
            if next_liquidity_target:
                breaker.target_price = next_liquidity_target
            else:
                risk_dist = breaker.entry_price - breaker.stop_price
                breaker.target_price = breaker.entry_price + (risk_dist * self.min_rr)

        # Calculate R:R
        risk = abs(breaker.entry_price - breaker.stop_price)
        reward = abs(breaker.target_price - breaker.entry_price)

        if risk == 0:
            return False

        rr = reward / risk

        if rr < self.min_rr:
            breaker.status = SignalStatus.INVALIDATED
            return False

        return True

    def simulate_outcome(
        self,
        breaker: BreakerZone,
        highs: np.ndarray,
        lows: np.ndarray,
        closes: np.ndarray,
    ) -> BreakerZone:
        """
        Walk forward from entry to determine trade outcome.
        Used in backtesting to measure actual performance.

        Rules:
        - If price hits stop first → STOPPED, pnl_r = -1R
        - If price hits target first → TARGET_HIT, pnl_r = +R:R
        - Check bar-by-bar (high/low of each bar)
        """
        if breaker.trigger_index is None or breaker.entry_price is None:
            return breaker
        if breaker.stop_price is None or breaker.target_price is None:
            return breaker

        start = breaker.trigger_index + 1
        risk = abs(breaker.entry_price - breaker.stop_price)

        for i in range(start, len(closes)):
            if breaker.breaker_type == BreakerType.BEARISH:
                # Short: stop is above entry, target is below
                if highs[i] >= breaker.stop_price:
                    breaker.status = SignalStatus.STOPPED
                    breaker.exit_index = i
                    breaker.exit_price = breaker.stop_price
                    breaker.pnl_r = -1.0
                    return breaker
                if lows[i] <= breaker.target_price:
                    breaker.status = SignalStatus.TARGET_HIT
                    breaker.exit_index = i
                    breaker.exit_price = breaker.target_price
                    if risk > 0:
                        breaker.pnl_r = abs(
                            breaker.entry_price - breaker.target_price
                        ) / risk
                    return breaker

            else:  # BULLISH
                # Long: stop is below entry, target is above
                if lows[i] <= breaker.stop_price:
                    breaker.status = SignalStatus.STOPPED
                    breaker.exit_index = i
                    breaker.exit_price = breaker.stop_price
                    breaker.pnl_r = -1.0
                    return breaker
                if highs[i] >= breaker.target_price:
                    breaker.status = SignalStatus.TARGET_HIT
                    breaker.exit_index = i
                    breaker.exit_price = breaker.target_price
                    if risk > 0:
                        breaker.pnl_r = abs(
                            breaker.target_price - breaker.entry_price
                        ) / risk
                    return breaker

        # Trade still open (data ended before resolution)
        breaker.status = SignalStatus.ACTIVE
        return breaker
