"""
Walk-forward, no-lookahead version of final_backtest_v2_exchange.py.

Key differences vs the original:
- Rebuilds market structure on data[:t+1] at each bar t
- Only allows setups that trigger on the current last bar
- Executes at next bar open, not same-bar close
- Rebuilds stop/target from the actual executed entry
- No separate audit pass needed to test lookahead; this runner itself is causal
"""

import argparse
import copy
import os
import sys
from typing import List, Optional

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from market_structure import MarketStructureEngine, StructureState
from breaker_detector import BreakerDetector, BreakerType, SignalStatus
from liquidity_mapper import LiquidityMapper, LiquidityLevel
from additional_setups import ThreeTapDetector, TradeSetup, SetupType
from backtest_engine import BacktestResults, TradeRecord
from exchange_data import build_exchange_dataset


def calculate_sma(arr: np.ndarray, period: int) -> np.ndarray:
    out = np.full(len(arr), np.nan)
    if len(arr) < period:
        return out
    csum = np.cumsum(np.insert(arr.astype(float), 0, 0.0))
    vals = (csum[period:] - csum[:-period]) / period
    out[period - 1:] = vals
    return out


def classify_period(date_str: str) -> str:
    if "2021-05-01" <= date_str <= "2021-07-31":
        return "2021_crash_may_jul"
    if "2021-08-01" <= date_str <= "2021-12-31":
        return "2021_bull_Q3Q4"
    if "2022-01-01" <= date_str <= "2022-06-30":
        return "2022_bear_H1"
    if "2022-07-01" <= date_str <= "2022-12-31":
        return "2022_bear_H2_ftx"
    if "2023-01-01" <= date_str <= "2023-12-31":
        return "2023_recovery"
    if "2024-01-01" <= date_str <= "2024-06-30":
        return "2024_etf_rally"
    if "2024-07-01" <= date_str <= "2024-12-31":
        return "2024_H2"
    return "2025_2026"


def calculate_adx(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, period: int = 14) -> np.ndarray:
    n = len(closes)
    adx = np.full(n, np.nan)
    if n < period * 3:
        return adx

    tr = np.zeros(n)
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)

    for i in range(1, n):
        tr[i] = max(highs[i] - lows[i], abs(highs[i] - closes[i - 1]), abs(lows[i] - closes[i - 1]))
        up_move = highs[i] - highs[i - 1]
        down_move = lows[i - 1] - lows[i]
        plus_dm[i] = up_move if (up_move > down_move and up_move > 0) else 0
        minus_dm[i] = down_move if (down_move > up_move and down_move > 0) else 0

    atr_s = np.zeros(n)
    plus_dm_s = np.zeros(n)
    minus_dm_s = np.zeros(n)
    atr_s[period] = np.sum(tr[1:period + 1])
    plus_dm_s[period] = np.sum(plus_dm[1:period + 1])
    minus_dm_s[period] = np.sum(minus_dm[1:period + 1])

    for i in range(period + 1, n):
        atr_s[i] = atr_s[i - 1] - (atr_s[i - 1] / period) + tr[i]
        plus_dm_s[i] = plus_dm_s[i - 1] - (plus_dm_s[i - 1] / period) + plus_dm[i]
        minus_dm_s[i] = minus_dm_s[i - 1] - (minus_dm_s[i - 1] / period) + minus_dm[i]

    plus_di = np.zeros(n)
    minus_di = np.zeros(n)
    dx = np.zeros(n)

    for i in range(period, n):
        if atr_s[i] > 0:
            plus_di[i] = 100 * plus_dm_s[i] / atr_s[i]
            minus_di[i] = 100 * minus_dm_s[i] / atr_s[i]
        di_sum = plus_di[i] + minus_di[i]
        if di_sum > 0:
            dx[i] = 100 * abs(plus_di[i] - minus_di[i]) / di_sum

    first_adx = period * 2
    if first_adx < n:
        adx[first_adx] = np.mean(dx[period + 1:first_adx + 1])
        for i in range(first_adx + 1, n):
            adx[i] = (adx[i - 1] * (period - 1) + dx[i]) / period

    return adx


class OTEv2Detector:
    def __init__(self, fib_low=0.618, fib_high=0.786, max_bars_to_retrace=30, min_impulse_atr=3.0, min_adx=25):
        self.fib_low = fib_low
        self.fib_high = fib_high
        self.max_bars_to_retrace = max_bars_to_retrace
        self.min_impulse_atr = min_impulse_atr
        self.min_adx = min_adx

    def detect(
        self,
        state: StructureState,
        highs: np.ndarray,
        lows: np.ndarray,
        closes: np.ndarray,
        atr: np.ndarray,
        adx: np.ndarray,
        liquidity_levels: List[LiquidityLevel],
    ) -> List[TradeSetup]:
        setups: List[TradeSetup] = []

        for msb in state.msb_events:
            if msb.index >= len(closes) - 10:
                continue

            atr_val = atr[min(msb.index, len(atr) - 1)]
            if atr_val <= 0:
                continue

            origin_price = msb.broken_swing.price

            if msb.break_type == "bullish_msb":
                search_end = min(msb.index + self.max_bars_to_retrace, len(highs))
                if search_end <= msb.index:
                    continue

                impulse_high_idx = msb.index + np.argmax(highs[msb.index:search_end])
                impulse_high = highs[impulse_high_idx]
                impulse_size = impulse_high - origin_price
                if impulse_size < atr_val * self.min_impulse_atr:
                    continue

                ote_high = impulse_high - impulse_size * self.fib_low
                ote_low = impulse_high - impulse_size * self.fib_high
                structure_level = next((lvl for lvl in liquidity_levels if ote_low <= lvl.price <= ote_high), None)
                if structure_level is None:
                    continue

                for bar in range(impulse_high_idx + 1, min(impulse_high_idx + self.max_bars_to_retrace, len(closes))):
                    if bar < len(adx) and not np.isnan(adx[bar]) and adx[bar] < self.min_adx:
                        continue

                    if lows[bar] <= ote_high and lows[bar] >= ote_low and closes[bar] > ote_low:
                        setup = TradeSetup(
                            setup_type=SetupType.OTE,
                            direction="long",
                            entry_price=closes[bar],
                            stop_price=ote_low - atr_val * 0.3,
                            target_price=impulse_high,
                            trigger_index=bar,
                            formation_index=msb.index,
                            fta_type=structure_level.liquidity_type.value,
                        )
                        score = 50.0
                        if structure_level.strength > 50:
                            score += 20
                        if impulse_size > atr_val * 4:
                            score += 15
                        if setup.rr >= 2.0:
                            score += 10
                        setup.quality_score = min(score, 100.0)
                        if setup.rr >= 1.5:
                            setups.append(setup)
                        break

            elif msb.break_type == "bearish_msb":
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
                structure_level = next((lvl for lvl in liquidity_levels if ote_low <= lvl.price <= ote_high), None)
                if structure_level is None:
                    continue

                for bar in range(impulse_low_idx + 1, min(impulse_low_idx + self.max_bars_to_retrace, len(closes))):
                    if bar < len(adx) and not np.isnan(adx[bar]) and adx[bar] < self.min_adx:
                        continue

                    if highs[bar] >= ote_low and highs[bar] <= ote_high and closes[bar] < ote_high:
                        setup = TradeSetup(
                            setup_type=SetupType.OTE,
                            direction="short",
                            entry_price=closes[bar],
                            stop_price=ote_high + atr_val * 0.3,
                            target_price=impulse_low,
                            trigger_index=bar,
                            formation_index=msb.index,
                            fta_type=structure_level.liquidity_type.value,
                        )
                        score = 50.0
                        if structure_level.strength > 50:
                            score += 20
                        if impulse_size > atr_val * 4:
                            score += 15
                        if setup.rr >= 2.0:
                            score += 10
                        setup.quality_score = min(score, 100.0)
                        if setup.rr >= 1.5:
                            setups.append(setup)
                        break

        return setups


def default_symbols() -> List[str]:
    return [
        "BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT", "XRP/USDT",
        "ADA/USDT", "DOGE/USDT", "AVAX/USDT", "LINK/USDT", "DOT/USDT",
        "MATIC/USDT", "LTC/USDT", "ATOM/USDT", "ETC/USDT", "UNI/USDT",
        "XLM/USDT", "ALGO/USDT", "NEAR/USDT", "FIL/USDT", "APT/USDT",
    ]


def simulate_trade_from_next_open(
    setup: TradeSetup,
    highs: np.ndarray,
    lows: np.ndarray,
    opens: np.ndarray,
    atr: np.ndarray,
    entry_bar: int,
) -> TradeSetup:
    """
    Causal execution model:
    - setup triggers on bar t
    - actual execution is at open[t+1]
    - stop/target are rebuilt from executed entry
    - simulation starts on entry_bar itself
    - conservative same-bar handling: stop first, then target
    """
    out = copy.deepcopy(setup)

    if entry_bar >= len(opens):
        out.status = SignalStatus.EXPIRED
        return out

    executed_entry = float(opens[entry_bar])
    atr_val = atr[min(entry_bar, len(atr) - 1)]

    if out.setup_type == SetupType.BREAKER:
        if out.direction == "short":
            risk = abs(setup.stop_price - setup.entry_price)
            out.entry_price = executed_entry
            out.stop_price = executed_entry + risk
            out.target_price = executed_entry - risk * setup.rr
        else:
            risk = abs(setup.entry_price - setup.stop_price)
            out.entry_price = executed_entry
            out.stop_price = executed_entry - risk
            out.target_price = executed_entry + risk * setup.rr
    else:
        if out.direction == "short":
            risk = abs(setup.stop_price - setup.entry_price)
            out.entry_price = executed_entry
            out.stop_price = executed_entry + risk
            out.target_price = executed_entry - risk * setup.rr
        else:
            risk = abs(setup.entry_price - setup.stop_price)
            out.entry_price = executed_entry
            out.stop_price = executed_entry - risk
            out.target_price = executed_entry + risk * setup.rr

    if out.risk <= 0:
        out.status = SignalStatus.INVALIDATED
        return out

    out.trigger_index = entry_bar - 1

    for i in range(entry_bar, len(opens)):
        if out.direction == "short":
            if highs[i] >= out.stop_price:
                out.status = SignalStatus.STOPPED
                out.exit_index = i
                out.exit_price = out.stop_price
                out.pnl_r = -1.0
                return out
            if lows[i] <= out.target_price:
                out.status = SignalStatus.TARGET_HIT
                out.exit_index = i
                out.exit_price = out.target_price
                out.pnl_r = out.reward / out.risk if out.risk > 0 else 0.0
                return out
        else:
            if lows[i] <= out.stop_price:
                out.status = SignalStatus.STOPPED
                out.exit_index = i
                out.exit_price = out.stop_price
                out.pnl_r = -1.0
                return out
            if highs[i] >= out.target_price:
                out.status = SignalStatus.TARGET_HIT
                out.exit_index = i
                out.exit_price = out.target_price
                out.pnl_r = out.reward / out.risk if out.risk > 0 else 0.0
                return out

    out.status = SignalStatus.ACTIVE
    return out


def get_triggered_setups_for_last_bar(
    state: StructureState,
    brk_det: BreakerDetector,
    tap_det: ThreeTapDetector,
    ote_v2: OTEv2Detector,
    liq: LiquidityMapper,
    opens: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    volumes: np.ndarray,
    timestamps: List[str],
    atr: np.ndarray,
    adx: np.ndarray,
    current_bar: int,
) -> List[TradeSetup]:
    """
    Build only setups whose trigger occurs on the current last bar.
    This is the key no-lookahead rule.
    """
    levels = liq.map_liquidity(state, highs, lows, closes, atr)
    out: List[TradeSetup] = []

    breakers = brk_det.detect(state, highs, lows, closes, atr, htf_bias="neutral")
    for bk in breakers:
        if bk.status != SignalStatus.TRIGGERED or bk.trigger_index is None or bk.entry_price is None:
            continue
        if bk.trigger_index != current_bar:
            continue

        atr_val = atr[min(current_bar, len(atr) - 1)]
        direction = "short" if bk.breaker_type == BreakerType.BEARISH else "long"
        stop = bk.zone_high + atr_val * 0.3 if bk.breaker_type == BreakerType.BEARISH else bk.zone_low - atr_val * 0.3

        lvls = liq.map_liquidity(state, highs, lows, closes, atr, current_bar=current_bar)
        fta = liq.find_fta(lvls, bk.entry_price, direction, stop)
        fta_type = "default"

        if fta and fta.rr_with_stop and fta.rr_with_stop >= 1.5:
            target = fta.level.price
            fta_type = fta.level.liquidity_type.value
        else:
            risk = abs(bk.entry_price - stop)
            if risk <= 0:
                continue
            target = bk.entry_price - risk * 1.5 if direction == "short" else bk.entry_price + risk * 1.5

        out.append(
            TradeSetup(
                setup_type=SetupType.BREAKER,
                direction=direction,
                entry_price=bk.entry_price,
                stop_price=stop,
                target_price=target,
                trigger_index=current_bar,
                formation_index=bk.formation_index,
                quality_score=bk.quality_score,
                fta_type=fta_type,
            )
        )

    for s in tap_det.detect(state, highs, lows, closes, atr):
        if s.trigger_index == current_bar:
            out.append(s)

    for s in ote_v2.detect(state, highs, lows, closes, atr, adx, levels):
        if s.trigger_index == current_bar:
            out.append(s)

    return out


def run_final_backtest(
    symbols: List[str],
    exchange_id: str,
    timeframe: str,
    start_date: str,
    min_bars: int,
    max_bars: Optional[int],
    market_type: str,
) -> BacktestResults:
    all_data = build_exchange_dataset(
        symbols=symbols,
        timeframe=timeframe,
        exchange_id=exchange_id,
        start_date=start_date,
        min_bars=min_bars,
        max_bars=max_bars,
        market_type=market_type,
    )

    print("=" * 70)
    print("FINAL PRODUCTION BACKTEST v2 WALK-FORWARD")
    print("Breaker + Three Tap + OTEv2 (S/R confluence)")
    print("No-lookahead replay | next-bar-open execution")
    print("200d SMA filter | ADX >= 20 chop filter | Q >= 50")
    print("%d coins | %s -> latest via %s (%s)" % (len(all_data), start_date, exchange_id, market_type))
    print("=" * 70)

    engine = MarketStructureEngine(swing_lookback=5, atr_period=14, min_swing_atr_multiple=0.5, equal_threshold_atr=0.1)
    brk_det = BreakerDetector(max_bars_to_retest=48, zone_buffer_atr=0.1, min_move_from_zone_atr=1.0)
    liq = LiquidityMapper(equal_tolerance_atr=0.15, untapped_lookback=200, max_levels=20)
    tap_det = ThreeTapDetector()
    ote_v2 = OTEv2Detector(min_impulse_atr=3.0, min_adx=25)

    results = BacktestResults()

    for data in all_data:
        o = data["opens"]
        h = data["highs"]
        l = data["lows"]
        c = data["closes"]
        v = data["volumes"]
        ts = data["timestamps"]
        coin = data["symbol"]
        n = len(c)

        sma = calculate_sma(c, 4800)
        atr_full = engine._calculate_atr(h, l, c)
        adx_full = calculate_adx(h, l, c, 14)

        last_entry_bar = -10
        coin_trade_count = 0

        start_bar = max(4800, engine.atr_period + engine.swing_lookback * 2 + 1)

        for t in range(start_bar, n - 1):
            if t - last_entry_bar < 6:
                continue

            o_s = o[: t + 1]
            h_s = h[: t + 1]
            l_s = l[: t + 1]
            c_s = c[: t + 1]
            v_s = v[: t + 1]
            ts_s = ts[: t + 1]
            atr_s = atr_full[: t + 1]
            adx_s = adx_full[: t + 1]

            state = engine.analyze(o_s, h_s, l_s, c_s, v_s, ts_s)
            setups = get_triggered_setups_for_last_bar(
                state=state,
                brk_det=brk_det,
                tap_det=tap_det,
                ote_v2=ote_v2,
                liq=liq,
                opens=o_s,
                highs=h_s,
                lows=l_s,
                closes=c_s,
                volumes=v_s,
                timestamps=ts_s,
                atr=atr_s,
                adx=adx_s,
                current_bar=t,
            )

            if not setups:
                continue

            setups.sort(key=lambda s: (s.quality_score, s.rr), reverse=True)

            chosen = None
            for setup in setups:
                if setup.quality_score < 50 or setup.rr < 1.5:
                    continue

                if not np.isnan(sma[t]):
                    if c[t] > sma[t] and setup.direction == "short":
                        continue
                    if c[t] < sma[t] and setup.direction == "long":
                        continue

                if t < len(adx_full) and not np.isnan(adx_full[t]) and adx_full[t] < 20:
                    continue

                atr_val = atr_full[min(t, len(atr_full) - 1)]
                if setup.setup_type == SetupType.BREAKER:
                    levels_t = liq.map_liquidity(state, h_s, l_s, c_s, atr_s, current_bar=t)
                    dangerous = liq.check_liquidity_near_stop(levels_t, setup.stop_price, atr_val, setup.direction)
                    if len(dangerous) >= 2:
                        continue

                chosen = setup
                break

            if chosen is None:
                continue

            executed = simulate_trade_from_next_open(
                setup=chosen,
                highs=h,
                lows=l,
                opens=o,
                atr=atr_full,
                entry_bar=t + 1,
            )

            if executed.status not in (SignalStatus.TARGET_HIT, SignalStatus.STOPPED):
                continue

            last_entry_bar = t + 1
            coin_trade_count += 1

            outcome = "win" if executed.status == SignalStatus.TARGET_HIT else "loss"
            pnl_r = executed.pnl_r or 0.0
            entry_bar = t + 1
            exit_bar = executed.exit_index if executed.exit_index is not None else entry_bar

            entry_date = ts[entry_bar][:19] if entry_bar < len(ts) else "?"
            exit_date = ts[exit_bar][:19] if exit_bar < len(ts) else "?"
            exit_price = executed.exit_price if executed.exit_price is not None else 0.0

            results.trades.append(
                TradeRecord(
                    coin=coin,
                    direction=executed.direction,
                    entry_price=executed.entry_price,
                    stop_price=executed.stop_price,
                    target_price=executed.target_price,
                    exit_price=exit_price,
                    entry_date=entry_date,
                    exit_date=exit_date,
                    pnl_r=pnl_r,
                    outcome=outcome,
                    rr_ratio=executed.rr,
                    quality_score=executed.quality_score,
                    regime=classify_period(ts[t][:10]),
                    fta_type=executed.fta_type,
                    breaker_type=executed.setup_type.value,
                )
            )

        print("  %-6s | %3d trades" % (coin, coin_trade_count))

    results.print_summary("FINAL WALK-FORWARD: NO-LOOKAHEAD + NEXT-OPEN EXECUTION")
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run walk-forward exchange-fed crypto backtest")
    parser.add_argument("--exchange", default="binance")
    parser.add_argument("--market-type", default="spot")
    parser.add_argument("--timeframe", default="1h")
    parser.add_argument("--start-date", default="2021-01-01")
    parser.add_argument("--min-bars", type=int, default=5000)
    parser.add_argument("--max-bars", type=int, default=None)
    parser.add_argument("--symbols", nargs="*", default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    symbols = args.symbols if args.symbols else default_symbols()

    run_final_backtest(
        symbols=symbols,
        exchange_id=args.exchange,
        timeframe=args.timeframe,
        start_date=args.start_date,
        min_bars=args.min_bars,
        max_bars=args.max_bars,
        market_type=args.market_type,
    )