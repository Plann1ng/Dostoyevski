"""
Module 2: Liquidity Mapper
===========================

Maps the liquidity landscape — where are the resting orders, untapped levels,
and the next areas price is attracted to? This module provides TWO critical
functions:

1. LIQUIDITY POOLS: Where are stops clustered? (Entry context)
2. FTA — FIRST TROUBLE AREA: Where will our trade run into difficulty? (Targets)

Sources & Cross-References:
- CryptoCred Lesson 3 (Order Flow), p.14-17:
  "Orders accumulate around attractive reference points — old highs/lows,
   equal highs/lows, swing highs/lows, range boundaries"
  "Structures that attract liquidity: equal highs/lows, deep swing points,
   range boundaries, popular candlestick pattern highs/lows"
  "The more attractive and outstanding a structure is, especially if visible
   on a high time frame (D1+), the more significant it is."

- CryptoCred Lesson 5 (Horizontal S/R), p.31-35:
  "My target for a trade is the FTA in the form of a level on the same
   time frame as the structure I am trading."
  "At its core, I exit long trades at the resistance closest to price
   and exit short trades at the support closest to price."

- RektProof p.3 (Breaker Setup):
  "Price proceeds to the next area of liquidity (Equal Lows) before reversing"

- RektProof p.24-25 (Supply/Demand vs Liquidity):
  "Price moves from liquidity pocket to liquidity pocket (Swing Point to
   Swing Point). Supply/Demand that forms between 2 liquidity pockets
   is irrelevant."
  "Price is seeking liquidity rather than looking to fill iceberg orders
   using SD levels."

- EmperorBTC Ch.4 (S/R), p.34-35:
  "The more times a resistance is tested, the weaker it becomes."

Panel Decisions:
- Trader 1: "FTA is the primary target method. Simple, replicable, consistent."
- Trader 3: "Liquidity = equal highs/lows and untapped swing points. Price is
  drawn to these. The breaker target should be the NEXT liquidity below/above."
- Trader 2: "Liquidity near our stop is dangerous. If there's a pool just above
  our stop on a short, we should skip the trade or widen the stop."
- Trader 5: "Equal levels within 0.1 ATR tolerance. Quantifiable, not subjective."
- Trader 4: "Every level gets a 'strength' score so we can rank targets."
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, List, Tuple
from enum import Enum

from market_structure import (
    StructureState,
    SwingPoint,
    SwingType,
    SwingLabel,
)


class LiquidityType(Enum):
    EQUAL_HIGHS = "equal_highs"       # Multiple swing highs at same price
    EQUAL_LOWS = "equal_lows"         # Multiple swing lows at same price
    UNTAPPED_HIGH = "untapped_high"   # Swing high price hasn't returned to
    UNTAPPED_LOW = "untapped_low"     # Swing low price hasn't returned to
    RANGE_HIGH = "range_high"         # Upper boundary of a range
    RANGE_LOW = "range_low"           # Lower boundary of a range


class LevelRole(Enum):
    """Whether this level acts as support or resistance relative to current price."""
    SUPPORT = "support"
    RESISTANCE = "resistance"


@dataclass
class LiquidityLevel:
    """A mapped liquidity level — a place where orders are likely resting."""
    price: float                      # The price level
    liquidity_type: LiquidityType     # What kind of liquidity
    role: LevelRole                   # Support or resistance relative to current price
    strength: float                   # 0-100 strength score
    source_swings: list               # Which swing points define this level
    first_formed_index: int           # Bar when this level first appeared
    last_tested_index: int            # Last bar price was near this level
    test_count: int                   # How many times price has tested this level
    is_untapped: bool                 # Has price returned to this level since formation?
    distance_from_price: float        # Absolute distance from current price
    distance_atr: float               # Distance in ATR units


@dataclass
class FTAResult:
    """
    First Trouble Area — the target for a trade.

    CryptoCred p.33-35:
    "If you're long, FTA is the first resistance above your entry.
     If you're short, FTA is the first support below your entry."
    """
    level: LiquidityLevel             # The actual level
    direction: str                    # "above" or "below" entry
    distance: float                   # Distance from entry to FTA
    distance_atr: float               # Same in ATR units
    rr_with_stop: Optional[float]     # R:R if this FTA is the target


class LiquidityMapper:
    """
    Maps the liquidity landscape around current price.

    Parameters:
    -----------
    equal_tolerance_atr : float
        How close two swing prices must be (in ATR) to be "equal."
        Panel decision: 0.15
        - Trader 5: "0.1 was too tight in testing. 0.15 catches the
          clusters that manual traders would identify."
        - CryptoCred p.17: "Equal highs/lows" — they don't need to be
          pixel-perfect, just visually clustered.

    untapped_lookback : int
        How many bars back to check if a level has been revisited.
        Panel decision: 200 bars on 1H (~8 trading days)
        - Trader 3: "Untapped levels from more than a week ago lose
          relevance. Fresh untapped levels are the magnets."

    max_levels : int
        Maximum number of levels to track.
        Panel decision: 20
        - Trader 1: "More than 20 levels on a 1H chart is noise.
          Keep the most significant ones."

    test_proximity_atr : float
        How close price must come to a level to count as a "test."
        Panel decision: 0.3 ATR
    """

    def __init__(
        self,
        equal_tolerance_atr: float = 0.15,
        untapped_lookback: int = 200,
        max_levels: int = 20,
        test_proximity_atr: float = 0.3,
    ):
        self.equal_tolerance_atr = equal_tolerance_atr
        self.untapped_lookback = untapped_lookback
        self.max_levels = max_levels
        self.test_proximity_atr = test_proximity_atr

    def map_liquidity(
        self,
        structure_state: StructureState,
        highs: np.ndarray,
        lows: np.ndarray,
        closes: np.ndarray,
        atr: np.ndarray,
        current_bar: Optional[int] = None,
    ) -> List[LiquidityLevel]:
        """
        Main method: map all liquidity levels from the current structure.

        Pipeline:
        1. Find equal highs and equal lows
        2. Find untapped swing points
        3. Identify range boundaries (if ranging)
        4. Score each level
        5. Sort by strength, trim to max_levels
        6. Assign support/resistance roles relative to current price
        """
        if current_bar is None:
            current_bar = len(closes) - 1

        current_price = closes[current_bar]
        current_atr = atr[min(current_bar, len(atr) - 1)]

        levels = []

        # Step 1: Equal highs and lows
        equal_levels = self._find_equal_levels(
            structure_state.swing_points, atr
        )
        levels.extend(equal_levels)

        # Step 2: Untapped swing points
        untapped = self._find_untapped_levels(
            structure_state.swing_points, highs, lows, current_bar
        )
        levels.extend(untapped)

        # Step 3: Deduplicate (levels near each other merge)
        levels = self._deduplicate(levels, current_atr)

        # Step 4: Score each level
        for level in levels:
            level.strength = self._score_level(level, current_atr)
            level.distance_from_price = abs(level.price - current_price)
            level.distance_atr = (
                level.distance_from_price / current_atr
                if current_atr > 0 else 0
            )

            # Assign role
            if level.price > current_price:
                level.role = LevelRole.RESISTANCE
            else:
                level.role = LevelRole.SUPPORT

        # Step 5: Sort by strength and trim
        levels.sort(key=lambda l: l.strength, reverse=True)
        levels = levels[: self.max_levels]

        return levels

    def find_fta(
        self,
        levels: List[LiquidityLevel],
        entry_price: float,
        direction: str,
        stop_price: Optional[float] = None,
    ) -> Optional[FTAResult]:
        """
        Find the First Trouble Area for a trade.

        CryptoCred p.33:
        "If you're long, FTA is the first resistance above your entry.
         If you're short, FTA is the first support below your entry."

        Parameters:
        -----------
        levels : list of LiquidityLevel
        entry_price : float
        direction : "long" or "short"
        stop_price : optional, for R:R calculation
        """
        if direction == "long":
            # Find the closest RESISTANCE above entry
            candidates = [
                l for l in levels
                if l.price > entry_price and l.role == LevelRole.RESISTANCE
            ]
            candidates.sort(key=lambda l: l.price)  # Nearest first
        elif direction == "short":
            # Find the closest SUPPORT below entry
            candidates = [
                l for l in levels
                if l.price < entry_price and l.role == LevelRole.SUPPORT
            ]
            candidates.sort(key=lambda l: l.price, reverse=True)  # Nearest first
        else:
            return None

        if not candidates:
            return None

        fta_level = candidates[0]
        distance = abs(fta_level.price - entry_price)

        rr = None
        if stop_price:
            risk = abs(entry_price - stop_price)
            if risk > 0:
                rr = distance / risk

        return FTAResult(
            level=fta_level,
            direction="above" if direction == "long" else "below",
            distance=distance,
            distance_atr=fta_level.distance_atr,
            rr_with_stop=rr,
        )

    def check_liquidity_near_stop(
        self,
        levels: List[LiquidityLevel],
        stop_price: float,
        atr_value: float,
        direction: str,
    ) -> List[LiquidityLevel]:
        """
        Check if there's dangerous liquidity near our stop loss.

        CryptoCred p.32:
        "If there's a liquidity pool above my stop (especially very clean
         highs/lows or deep swing points) I'll often sit out of the obvious
         level trade and wait for a stop run setup instead."

        Trader 2: "This is the stop-safety check. If equal highs sit
        just above our short stop, we're likely to get hunted."

        Returns list of dangerous levels within 1 ATR of the stop.
        """
        danger_zone = atr_value * 1.0  # Within 1 ATR of stop

        dangerous = []
        for level in levels:
            dist = abs(level.price - stop_price)
            if dist <= danger_zone:
                if direction == "short" and level.price > stop_price:
                    # Liquidity above our short stop — dangerous
                    dangerous.append(level)
                elif direction == "long" and level.price < stop_price:
                    # Liquidity below our long stop — dangerous
                    dangerous.append(level)

        return dangerous

    # =========================================================================
    # Internal Methods
    # =========================================================================

    def _find_equal_levels(
        self,
        swings: List[SwingPoint],
        atr: np.ndarray,
    ) -> List[LiquidityLevel]:
        """
        Find clusters of swing points at approximately the same price.

        CryptoCred Lesson 3, p.17:
        "Structures that attract liquidity: Equal highs/lows"

        RektProof p.3:
        "Price proceeds to the next area of liquidity (Equal Lows)"

        Method: Group swings of the same type that are within
        tolerance of each other. 2+ swings at same price = equal level.
        """
        levels = []

        # Separate highs and lows
        swing_highs = [s for s in swings if s.swing_type == SwingType.HIGH]
        swing_lows = [s for s in swings if s.swing_type == SwingType.LOW]

        # Find equal highs
        eq_highs = self._cluster_by_price(swing_highs, atr)
        for cluster in eq_highs:
            if len(cluster) >= 2:
                avg_price = np.mean([s.price for s in cluster])
                levels.append(LiquidityLevel(
                    price=avg_price,
                    liquidity_type=LiquidityType.EQUAL_HIGHS,
                    role=LevelRole.RESISTANCE,  # Will be reassigned
                    strength=0.0,  # Will be scored
                    source_swings=cluster,
                    first_formed_index=min(s.index for s in cluster),
                    last_tested_index=max(s.index for s in cluster),
                    test_count=len(cluster),
                    is_untapped=False,  # Equal levels have been tapped by definition
                    distance_from_price=0.0,
                    distance_atr=0.0,
                ))

        # Find equal lows
        eq_lows = self._cluster_by_price(swing_lows, atr)
        for cluster in eq_lows:
            if len(cluster) >= 2:
                avg_price = np.mean([s.price for s in cluster])
                levels.append(LiquidityLevel(
                    price=avg_price,
                    liquidity_type=LiquidityType.EQUAL_LOWS,
                    role=LevelRole.SUPPORT,
                    strength=0.0,
                    source_swings=cluster,
                    first_formed_index=min(s.index for s in cluster),
                    last_tested_index=max(s.index for s in cluster),
                    test_count=len(cluster),
                    is_untapped=False,
                    distance_from_price=0.0,
                    distance_atr=0.0,
                ))

        return levels

    def _cluster_by_price(
        self,
        swings: List[SwingPoint],
        atr: np.ndarray,
    ) -> List[List[SwingPoint]]:
        """Group swings that are within ATR tolerance of each other."""
        if not swings:
            return []

        # Sort by price
        sorted_swings = sorted(swings, key=lambda s: s.price)
        clusters = []
        current_cluster = [sorted_swings[0]]

        for i in range(1, len(sorted_swings)):
            s = sorted_swings[i]
            prev = current_cluster[-1]
            atr_val = atr[min(s.index, len(atr) - 1)]
            tolerance = atr_val * self.equal_tolerance_atr

            if abs(s.price - prev.price) <= tolerance:
                current_cluster.append(s)
            else:
                clusters.append(current_cluster)
                current_cluster = [s]

        clusters.append(current_cluster)
        return clusters

    def _find_untapped_levels(
        self,
        swings: List[SwingPoint],
        highs: np.ndarray,
        lows: np.ndarray,
        current_bar: int,
    ) -> List[LiquidityLevel]:
        """
        Find swing points that price hasn't returned to since formation.

        These are "magnets" — price is drawn to them because the liquidity
        sitting there hasn't been taken yet.

        RektProof p.24:
        "Price always has an objective. When you have 2 swing points /
         liquidity pockets we are moving from one end to the other."

        An untapped high means the sell-side liquidity above it hasn't
        been collected. An untapped low means buy-side liquidity below
        it hasn't been collected.
        """
        levels = []
        lookback_start = max(0, current_bar - self.untapped_lookback)

        for swing in swings:
            if swing.index < lookback_start:
                continue
            if swing.index >= current_bar:
                continue

            # Check if price has returned to this level since formation
            tapped = False
            check_start = swing.index + 1
            check_end = min(current_bar + 1, len(highs))

            for bar in range(check_start, check_end):
                if swing.swing_type == SwingType.HIGH:
                    if highs[bar] >= swing.price:
                        tapped = True
                        break
                else:  # LOW
                    if lows[bar] <= swing.price:
                        tapped = True
                        break

            if not tapped:
                liq_type = (
                    LiquidityType.UNTAPPED_HIGH
                    if swing.swing_type == SwingType.HIGH
                    else LiquidityType.UNTAPPED_LOW
                )
                levels.append(LiquidityLevel(
                    price=swing.price,
                    liquidity_type=liq_type,
                    role=LevelRole.RESISTANCE if swing.swing_type == SwingType.HIGH else LevelRole.SUPPORT,
                    strength=0.0,
                    source_swings=[swing],
                    first_formed_index=swing.index,
                    last_tested_index=swing.index,
                    test_count=1,
                    is_untapped=True,
                    distance_from_price=0.0,
                    distance_atr=0.0,
                ))

        return levels

    def _deduplicate(
        self,
        levels: List[LiquidityLevel],
        current_atr: float,
    ) -> List[LiquidityLevel]:
        """
        Merge levels that are very close to each other.
        Keep the one with more source swings (more significant).
        """
        if len(levels) < 2:
            return levels

        levels.sort(key=lambda l: l.price)
        merged = [levels[0]]
        merge_tolerance = current_atr * 0.2

        for i in range(1, len(levels)):
            if abs(levels[i].price - merged[-1].price) <= merge_tolerance:
                # Merge: keep the one with more sources
                if len(levels[i].source_swings) > len(merged[-1].source_swings):
                    merged[-1] = levels[i]
                # Absorb the test count
                merged[-1].test_count += levels[i].test_count
            else:
                merged.append(levels[i])

        return merged

    def _score_level(
        self,
        level: LiquidityLevel,
        current_atr: float,
    ) -> float:
        """
        Score a liquidity level's significance.

        Scoring factors (Panel consensus):

        1. Type bonus (+30 max)
           - Equal highs/lows: +30 (strongest liquidity magnet)
           - Untapped levels: +20 (fresh, uncollected liquidity)
           - Range boundaries: +25

        2. Test count (+25 max)
           - More tests = more orders resting there
           - But EmperorBTC: "more tests = weaker" for S/R holds
           - Resolution: more tests = MORE liquidity but LESS likely to hold
           - For TARGET purposes, more tests = higher score (more attraction)

        3. Untapped bonus (+20)
           - Untapped = all the liquidity is still there

        4. Recency (+15 max)
           - Levels formed recently are more relevant

        5. HTF visibility (+10)
           - Levels visible on higher timeframes score higher
           - Proxy: large swing labels (HH, LL) vs minor swings
        """
        score = 0.0

        # Factor 1: Type
        type_scores = {
            LiquidityType.EQUAL_HIGHS: 30,
            LiquidityType.EQUAL_LOWS: 30,
            LiquidityType.RANGE_HIGH: 25,
            LiquidityType.RANGE_LOW: 25,
            LiquidityType.UNTAPPED_HIGH: 20,
            LiquidityType.UNTAPPED_LOW: 20,
        }
        score += type_scores.get(level.liquidity_type, 10)

        # Factor 2: Test count / cluster size
        cluster_size = len(level.source_swings)
        score += min(cluster_size * 8, 25)

        # Factor 3: Untapped bonus
        if level.is_untapped:
            score += 20

        # Factor 4: Recency (not easily calculated without current_bar,
        # so we use a simpler proxy)
        score += 10  # Base recency — refined when distance is known

        # Factor 5: HTF visibility — key labels score higher
        for swing in level.source_swings:
            if swing.label in (SwingLabel.HH, SwingLabel.LL):
                score += 5
                break
            elif swing.label in (SwingLabel.EH, SwingLabel.EL):
                score += 10  # Equal levels are very visible
                break

        return min(score, 100.0)

    def get_levels_summary(
        self, levels: List[LiquidityLevel], current_price: float
    ) -> str:
        """Human-readable summary of mapped liquidity levels."""
        above = [l for l in levels if l.price > current_price]
        below = [l for l in levels if l.price <= current_price]

        above.sort(key=lambda l: l.price)
        below.sort(key=lambda l: l.price, reverse=True)

        lines = [f"Current price: {current_price:.2f}"]
        lines.append(f"\nResistance levels above ({len(above)}):")
        for l in above[:5]:
            tag = "UNTAPPED" if l.is_untapped else f"{l.test_count}x tested"
            lines.append(
                f"  {l.price:.2f} [{l.liquidity_type.value}] "
                f"str={l.strength:.0f} ({tag})"
            )

        lines.append(f"\nSupport levels below ({len(below)}):")
        for l in below[:5]:
            tag = "UNTAPPED" if l.is_untapped else f"{l.test_count}x tested"
            lines.append(
                f"  {l.price:.2f} [{l.liquidity_type.value}] "
                f"str={l.strength:.0f} ({tag})"
            )

        return "\n".join(lines)
