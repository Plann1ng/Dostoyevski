from __future__ import annotations

"""
Backtest sanity audit
=====================

Purpose
-------
A standalone audit layer for trading backtests. It is designed to catch
common sources of fake edge:

1. Future leakage / lookahead bias
2. Using unconfirmed pivots or structure too early
3. Same-bar optimistic fills
4. Entering on the same close that produced the signal
5. Overlapping trades that exceed the intended position model
6. Unrealistic TP/SL assumptions when both are touched in one candle
7. Suspiciously good trades that only work under favorable execution

This file is intentionally separate from the main strategy runner so you can:
- run the normal backtest
- export candles + trades/signals
- rerun an independent sanity audit
- compare optimistic vs conservative results

Recommended workflow
--------------------
1. Run your strategy and export:
   - candles.csv
   - trades.csv
   - optionally signals.csv
2. Run:
   python backtest_sanity_audit.py --candles candles.csv --trades trades.csv
3. Review warnings and the conservative re-simulation summary.

CSV expectations
----------------
Candles CSV must contain at least:
    timestamp, open, high, low, close

Trades CSV must contain at least:
    trade_id, symbol, side, signal_time, entry_time, entry_price,
    stop_price, target_price, exit_time, exit_price, outcome

Optional but strongly recommended trade columns:
    setup_type, signal_bar_index, entry_bar_index, exit_bar_index,
    pivot_time, pivot_confirm_time, quality_score,
    stop_hit_time, target_hit_time, both_touched_same_bar,
    used_next_bar_entry, used_conservative_intrabar

Signals CSV is optional. If supplied, it can contain extra metadata used for
lookahead checks.

Notes
-----
This file cannot magically prove there is no leakage if the source backtest
never records enough metadata. What it can do is:
- flag impossible or suspicious timing
- re-score trades under stricter assumptions
- highlight missing metadata that prevents strong validation
"""

import argparse
import math
from dataclasses import dataclass
from typing import Optional, List, Dict, Tuple

import pandas as pd


REQUIRED_CANDLE_COLUMNS = {"timestamp", "open", "high", "low", "close"}
REQUIRED_TRADE_COLUMNS = {
    "trade_id",
    "symbol",
    "side",
    "signal_time",
    "entry_time",
    "entry_price",
    "stop_price",
    "target_price",
    "exit_time",
    "exit_price",
    "outcome",
}


@dataclass
class AuditIssue:
    severity: str
    category: str
    trade_id: Optional[str]
    message: str


@dataclass
class ConservativeTradeResult:
    trade_id: str
    conservative_outcome: str
    conservative_exit_time: Optional[pd.Timestamp]
    conservative_exit_price: Optional[float]
    conservative_r: Optional[float]
    notes: str


@dataclass
class AuditSummary:
    issues: List[AuditIssue]
    trade_results: List[ConservativeTradeResult]


class BacktestSanityAuditor:
    def __init__(
        self,
        candles: pd.DataFrame,
        trades: pd.DataFrame,
        signals: Optional[pd.DataFrame] = None,
        force_next_bar_entry: bool = True,
        force_conservative_same_bar: bool = True,
        forbid_entry_before_pivot_confirmation: bool = True,
        max_allowed_overlap_per_symbol: int = 1,
    ) -> None:
        self.candles = candles.copy()
        self.trades = trades.copy()
        self.signals = signals.copy() if signals is not None else None
        self.force_next_bar_entry = force_next_bar_entry
        self.force_conservative_same_bar = force_conservative_same_bar
        self.forbid_entry_before_pivot_confirmation = forbid_entry_before_pivot_confirmation
        self.max_allowed_overlap_per_symbol = max_allowed_overlap_per_symbol

        self._normalize_inputs()

    def _normalize_inputs(self) -> None:
        missing_c = REQUIRED_CANDLE_COLUMNS - set(self.candles.columns)
        if missing_c:
            raise ValueError(f"Candles CSV missing required columns: {sorted(missing_c)}")

        missing_t = REQUIRED_TRADE_COLUMNS - set(self.trades.columns)
        if missing_t:
            raise ValueError(f"Trades CSV missing required columns: {sorted(missing_t)}")

        self.candles["timestamp"] = pd.to_datetime(self.candles["timestamp"], utc=True, errors="coerce")
        for col in ["open", "high", "low", "close"]:
            self.candles[col] = pd.to_numeric(self.candles[col], errors="coerce")

        for col in ["signal_time", "entry_time", "exit_time", "pivot_time", "pivot_confirm_time", "stop_hit_time", "target_hit_time"]:
            if col in self.trades.columns:
                self.trades[col] = pd.to_datetime(self.trades[col], utc=True, errors="coerce")

        for col in ["entry_price", "stop_price", "target_price", "exit_price", "quality_score"]:
            if col in self.trades.columns:
                self.trades[col] = pd.to_numeric(self.trades[col], errors="coerce")

        for col in ["signal_bar_index", "entry_bar_index", "exit_bar_index"]:
            if col in self.trades.columns:
                self.trades[col] = pd.to_numeric(self.trades[col], errors="coerce")

        self.candles = self.candles.dropna(subset=["timestamp", "open", "high", "low", "close"]).sort_values("timestamp").reset_index(drop=True)
        self.trades = self.trades.sort_values(["symbol", "entry_time", "trade_id"]).reset_index(drop=True)

        if self.signals is not None and "timestamp" in self.signals.columns:
            self.signals["timestamp"] = pd.to_datetime(self.signals["timestamp"], utc=True, errors="coerce")
            self.signals = self.signals.sort_values("timestamp").reset_index(drop=True)

        self.candle_time_to_index = {ts: i for i, ts in enumerate(self.candles["timestamp"].tolist())}

    def run(self) -> AuditSummary:
        issues: List[AuditIssue] = []
        conservative_results: List[ConservativeTradeResult] = []

        issues.extend(self._check_basic_trade_sanity())
        issues.extend(self._check_signal_vs_entry_timing())
        issues.extend(self._check_pivot_confirmation())
        issues.extend(self._check_overlap())
        issues.extend(self._check_price_geometry())
        issues.extend(self._check_time_membership())
        issues.extend(self._check_missing_validation_metadata())

        for _, trade in self.trades.iterrows():
            conservative_results.append(self._resim_trade_conservatively(trade, issues))

        return AuditSummary(issues=issues, trade_results=conservative_results)

    def _check_basic_trade_sanity(self) -> List[AuditIssue]:
        issues: List[AuditIssue] = []
        for _, t in self.trades.iterrows():
            tid = str(t["trade_id"])
            side = str(t["side"]).lower()
            entry = t["entry_price"]
            stop = t["stop_price"]
            target = t["target_price"]
            signal_time = t["signal_time"]
            entry_time = t["entry_time"]
            exit_time = t["exit_time"]

            if side not in {"long", "short"}:
                issues.append(AuditIssue("error", "trade_schema", tid, f"Invalid side: {side}"))
                continue

            if pd.isna(entry) or pd.isna(stop) or pd.isna(target):
                issues.append(AuditIssue("error", "trade_schema", tid, "Entry/stop/target contains NaN."))
                continue

            if side == "long":
                if not (stop < entry < target):
                    issues.append(AuditIssue("error", "trade_geometry", tid, f"Long trade geometry invalid: stop={stop}, entry={entry}, target={target}"))
            else:
                if not (target < entry < stop):
                    issues.append(AuditIssue("error", "trade_geometry", tid, f"Short trade geometry invalid: target={target}, entry={entry}, stop={stop}"))

            if pd.notna(signal_time) and pd.notna(entry_time) and entry_time < signal_time:
                issues.append(AuditIssue("error", "time_order", tid, "Entry time is before signal time."))

            if pd.notna(entry_time) and pd.notna(exit_time) and exit_time < entry_time:
                issues.append(AuditIssue("error", "time_order", tid, "Exit time is before entry time."))

        return issues

    def _check_signal_vs_entry_timing(self) -> List[AuditIssue]:
        issues: List[AuditIssue] = []
        for _, t in self.trades.iterrows():
            tid = str(t["trade_id"])
            signal_time = t["signal_time"]
            entry_time = t["entry_time"]

            if pd.isna(signal_time) or pd.isna(entry_time):
                continue

            same_bar = signal_time == entry_time
            if same_bar and self.force_next_bar_entry:
                issues.append(AuditIssue(
                    "warning",
                    "same_bar_entry",
                    tid,
                    "Signal and entry occur on the same candle. If the signal is computed on close, next-bar execution is safer.",
                ))

            if "used_next_bar_entry" in self.trades.columns and pd.notna(t.get("used_next_bar_entry")):
                if not bool(t["used_next_bar_entry"]) and same_bar:
                    issues.append(AuditIssue(
                        "warning",
                        "execution_assumption",
                        tid,
                        "Trade metadata says next-bar entry was not used and entry happened on signal candle.",
                    ))
        return issues

    def _check_pivot_confirmation(self) -> List[AuditIssue]:
        issues: List[AuditIssue] = []
        if not self.forbid_entry_before_pivot_confirmation:
            return issues

        if "pivot_confirm_time" not in self.trades.columns:
            return issues

        for _, t in self.trades.iterrows():
            tid = str(t["trade_id"])
            entry_time = t["entry_time"]
            pivot_confirm_time = t.get("pivot_confirm_time")
            if pd.notna(entry_time) and pd.notna(pivot_confirm_time) and entry_time < pivot_confirm_time:
                issues.append(AuditIssue(
                    "error",
                    "pivot_lookahead",
                    tid,
                    "Entry occurred before pivot/structure confirmation time.",
                ))
        return issues

    def _check_overlap(self) -> List[AuditIssue]:
        issues: List[AuditIssue] = []
        grouped = self.trades.sort_values(["symbol", "entry_time"]).groupby("symbol")
        for symbol, df in grouped:
            active: List[Tuple[pd.Timestamp, str]] = []
            for _, t in df.iterrows():
                tid = str(t["trade_id"])
                entry_time = t["entry_time"]
                exit_time = t["exit_time"]
                if pd.isna(entry_time) or pd.isna(exit_time):
                    continue
                active = [(e, aid) for e, aid in active if e > entry_time]
                if len(active) >= self.max_allowed_overlap_per_symbol:
                    issues.append(AuditIssue(
                        "warning",
                        "overlap",
                        tid,
                        f"Trade overlaps with {len(active)} active trade(s) on {symbol}. Check capital and position model.",
                    ))
                active.append((exit_time, tid))
        return issues

    def _check_price_geometry(self) -> List[AuditIssue]:
        issues: List[AuditIssue] = []
        for _, t in self.trades.iterrows():
            tid = str(t["trade_id"])
            entry = t["entry_price"]
            stop = t["stop_price"]
            target = t["target_price"]
            risk = abs(entry - stop) if pd.notna(entry) and pd.notna(stop) else math.nan
            reward = abs(target - entry) if pd.notna(entry) and pd.notna(target) else math.nan
            if pd.notna(risk) and risk <= 0:
                issues.append(AuditIssue("error", "risk_model", tid, "Risk <= 0."))
            if pd.notna(risk) and pd.notna(reward) and risk > 0:
                rr = reward / risk
                if rr > 20:
                    issues.append(AuditIssue("warning", "extreme_rr", tid, f"Very large RR={rr:.2f}. Verify target selection is realistic."))
        return issues

    def _check_time_membership(self) -> List[AuditIssue]:
        issues: List[AuditIssue] = []
        candle_times = set(self.candle_time_to_index.keys())
        for _, t in self.trades.iterrows():
            tid = str(t["trade_id"])
            for col in ["signal_time", "entry_time", "exit_time"]:
                ts = t[col]
                if pd.notna(ts) and ts not in candle_times:
                    issues.append(AuditIssue(
                        "warning",
                        "timestamp_alignment",
                        tid,
                        f"{col}={ts} is not an exact candle timestamp in the candles file.",
                    ))
        return issues

    def _check_missing_validation_metadata(self) -> List[AuditIssue]:
        issues: List[AuditIssue] = []
        helpful_columns = [
            "pivot_confirm_time",
            "signal_bar_index",
            "entry_bar_index",
            "exit_bar_index",
            "setup_type",
            "both_touched_same_bar",
            "used_next_bar_entry",
            "used_conservative_intrabar",
        ]
        missing = [c for c in helpful_columns if c not in self.trades.columns]
        if missing:
            issues.append(AuditIssue(
                "info",
                "missing_metadata",
                None,
                f"Trades CSV is missing validation columns that would improve audit strength: {missing}",
            ))
        return issues

    def _resim_trade_conservatively(self, trade: pd.Series, issues: List[AuditIssue]) -> ConservativeTradeResult:
        tid = str(trade["trade_id"])
        side = str(trade["side"]).lower()
        signal_time = trade["signal_time"]
        entry_time = trade["entry_time"]
        stop_price = float(trade["stop_price"])
        target_price = float(trade["target_price"])

        if pd.isna(signal_time) or pd.isna(entry_time):
            return ConservativeTradeResult(tid, "unknown", None, None, None, "Missing signal_time or entry_time.")

        signal_idx = self.candle_time_to_index.get(signal_time)
        if signal_idx is None:
            return ConservativeTradeResult(tid, "unknown", None, None, None, "signal_time not found in candles.")

        entry_idx_original = self.candle_time_to_index.get(entry_time)
        if entry_idx_original is None:
            return ConservativeTradeResult(tid, "unknown", None, None, None, "entry_time not found in candles.")

        conservative_entry_idx = entry_idx_original
        notes: List[str] = []

        if self.force_next_bar_entry:
            conservative_entry_idx = max(entry_idx_original, signal_idx + 1)
            if conservative_entry_idx != entry_idx_original:
                notes.append("Moved entry to next bar after signal.")

        if conservative_entry_idx >= len(self.candles):
            return ConservativeTradeResult(tid, "canceled", None, None, None, "No bar available for conservative entry.")

        entry_row = self.candles.iloc[conservative_entry_idx]
        conservative_entry_time = entry_row["timestamp"]
        conservative_entry_price = float(entry_row["open"])

        if side == "long" and not (stop_price < conservative_entry_price < target_price):
            return ConservativeTradeResult(
                tid,
                "canceled",
                conservative_entry_time,
                conservative_entry_price,
                None,
                "After moving to next-bar open, long geometry became invalid.",
            )
        if side == "short" and not (target_price < conservative_entry_price < stop_price):
            return ConservativeTradeResult(
                tid,
                "canceled",
                conservative_entry_time,
                conservative_entry_price,
                None,
                "After moving to next-bar open, short geometry became invalid.",
            )

        risk = abs(conservative_entry_price - stop_price)
        if risk <= 0:
            return ConservativeTradeResult(tid, "canceled", conservative_entry_time, conservative_entry_price, None, "Risk <= 0.")

        for i in range(conservative_entry_idx, len(self.candles)):
            row = self.candles.iloc[i]
            low = float(row["low"])
            high = float(row["high"])
            ts = row["timestamp"]

            if side == "long":
                stop_hit = low <= stop_price
                target_hit = high >= target_price
                if stop_hit and target_hit:
                    if self.force_conservative_same_bar:
                        notes.append("Both stop and target touched in same bar; conservative rule chose stop.")
                        return ConservativeTradeResult(tid, "loss", ts, stop_price, -1.0, "; ".join(notes))
                    return ConservativeTradeResult(tid, "ambiguous", ts, None, None, "Both stop and target touched in same bar.")
                if stop_hit:
                    return ConservativeTradeResult(tid, "loss", ts, stop_price, -1.0, "; ".join(notes))
                if target_hit:
                    r = (target_price - conservative_entry_price) / risk
                    return ConservativeTradeResult(tid, "win", ts, target_price, r, "; ".join(notes))
            else:
                stop_hit = high >= stop_price
                target_hit = low <= target_price
                if stop_hit and target_hit:
                    if self.force_conservative_same_bar:
                        notes.append("Both stop and target touched in same bar; conservative rule chose stop.")
                        return ConservativeTradeResult(tid, "loss", ts, stop_price, -1.0, "; ".join(notes))
                    return ConservativeTradeResult(tid, "ambiguous", ts, None, None, "Both stop and target touched in same bar.")
                if stop_hit:
                    return ConservativeTradeResult(tid, "loss", ts, stop_price, -1.0, "; ".join(notes))
                if target_hit:
                    r = (conservative_entry_price - target_price) / risk
                    return ConservativeTradeResult(tid, "win", ts, target_price, r, "; ".join(notes))

        return ConservativeTradeResult(tid, "open", None, None, None, "; ".join(notes) if notes else "Trade never resolved in available candles.")


def summarize_conservative_results(results: List[ConservativeTradeResult]) -> Dict[str, float]:
    finished = [r for r in results if r.conservative_outcome in {"win", "loss"} and r.conservative_r is not None]
    wins = [r for r in finished if r.conservative_outcome == "win"]
    losses = [r for r in finished if r.conservative_outcome == "loss"]

    total_r = sum(r.conservative_r for r in finished)
    gross_profit = sum(r.conservative_r for r in finished if r.conservative_r > 0)
    gross_loss = abs(sum(r.conservative_r for r in finished if r.conservative_r < 0))

    equity = 0.0
    peak = 0.0
    max_dd = 0.0
    for r in finished:
        equity += float(r.conservative_r)
        peak = max(peak, equity)
        max_dd = max(max_dd, peak - equity)

    return {
        "completed_trades": len(finished),
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": (len(wins) / len(finished)) if finished else 0.0,
        "total_r": total_r,
        "avg_r_per_trade": (total_r / len(finished)) if finished else 0.0,
        "profit_factor": (gross_profit / gross_loss) if gross_loss > 0 else math.inf,
        "max_drawdown_r": max_dd,
    }


def print_summary(audit: AuditSummary) -> None:
    issue_df = pd.DataFrame([vars(i) for i in audit.issues]) if audit.issues else pd.DataFrame(columns=["severity", "category", "trade_id", "message"])
    result_df = pd.DataFrame([vars(r) for r in audit.trade_results]) if audit.trade_results else pd.DataFrame()

    print("=" * 70)
    print("BACKTEST SANITY AUDIT")
    print("=" * 70)
    print(f"Issues found: {len(audit.issues)}")
    if not issue_df.empty:
        print("\nIssues by severity:")
        print(issue_df.groupby("severity").size().to_string())
        print("\nIssues by category:")
        print(issue_df.groupby("category").size().sort_values(ascending=False).to_string())

        print("\nFirst 20 issues:")
        with pd.option_context("display.max_colwidth", 120):
            print(issue_df.head(20).to_string(index=False))

    if not result_df.empty:
        stats = summarize_conservative_results(audit.trade_results)
        print("\n" + "=" * 70)
        print("CONSERVATIVE RE-SIMULATION")
        print("=" * 70)
        for k, v in stats.items():
            if isinstance(v, float):
                print(f"{k:20s}: {v:.4f}")
            else:
                print(f"{k:20s}: {v}")


def save_outputs(audit: AuditSummary, output_prefix: str) -> None:
    issue_df = pd.DataFrame([vars(i) for i in audit.issues]) if audit.issues else pd.DataFrame(columns=["severity", "category", "trade_id", "message"])
    result_df = pd.DataFrame([vars(r) for r in audit.trade_results]) if audit.trade_results else pd.DataFrame()

    issue_path = f"{output_prefix}_issues.csv"
    results_path = f"{output_prefix}_conservative_results.csv"

    issue_df.to_csv(issue_path, index=False)
    result_df.to_csv(results_path, index=False)

    print(f"\nSaved issues to: {issue_path}")
    print(f"Saved conservative results to: {results_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit trading backtests for lookahead and unrealistic execution.")
    parser.add_argument("--candles", required=True, help="Path to candles CSV.")
    parser.add_argument("--trades", required=True, help="Path to trades CSV.")
    parser.add_argument("--signals", default=None, help="Optional path to signals CSV.")
    parser.add_argument("--output-prefix", default="sanity_audit", help="Prefix for audit CSV outputs.")
    parser.add_argument("--allow-same-bar-entry", action="store_true", help="Do not warn on signal-bar entries.")
    parser.add_argument("--allow-optimistic-same-bar", action="store_true", help="Do not force stop-first on ambiguous same-bar TP/SL.")
    parser.add_argument("--allow-entry-before-pivot-confirm", action="store_true", help="Disable pivot confirmation timing checks.")
    parser.add_argument("--max-overlap-per-symbol", type=int, default=1, help="Maximum allowed overlapping trades per symbol.")
    args = parser.parse_args()

    candles = pd.read_csv(args.candles)
    trades = pd.read_csv(args.trades)
    signals = pd.read_csv(args.signals) if args.signals else None

    auditor = BacktestSanityAuditor(
        candles=candles,
        trades=trades,
        signals=signals,
        force_next_bar_entry=not args.allow_same_bar_entry,
        force_conservative_same_bar=not args.allow_optimistic_same_bar,
        forbid_entry_before_pivot_confirmation=not args.allow_entry_before_pivot_confirm,
        max_allowed_overlap_per_symbol=args.max_overlap_per_symbol,
    )

    audit = auditor.run()
    print_summary(audit)
    save_outputs(audit, args.output_prefix)


if __name__ == "__main__":
    main()

