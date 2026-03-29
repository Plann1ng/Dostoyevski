# Breaker + Dostoyevski Three Tap Trading System v1.0

## Release Documentation — Ready for Live Deployment

**System**: Breaker + Three Tap with 200-day SMA macro filter
**Validated on**: 20 crypto assets, 1H timeframe, Jan 2021 — Mar 2026 (5.25 years)
**Execution model**: Next-bar-open entry, stop-first on ambiguous bars
**Source material**: RektProof Setups, CryptoCred Study Guide, EmperorBTC Trading Manual

---

## 1. Corrected performance summary

All numbers below use the corrected execution model: signal on bar T, entry at open of bar T+1, stop checked before target when both are hit in the same candle. These are the honest, audited numbers.

| Metric | Value |
|---|---|
| Total trades | 2,582 |
| Wins | 659 |
| Losses | 1,923 |
| Win rate | 25.5% |
| Total profit | +506.05 R |
| Avg profit per trade | +0.196 R |
| Avg winning trade | +3.69 R |
| Avg losing trade | -1.00 R |
| Profit factor | 1.26 |
| Expectancy | +0.196 R per trade |
| Max drawdown | 46.71 R |
| Max losing streak | 20 consecutive losses |
| Trades per year per coin | ~24.6 |

At 1% risk per trade on a $10,000 account, this translates to approximately +$5,060 profit over the test period, with a worst-case drawdown of roughly $4,670 (46.7%).

At 0.5% risk per trade (recommended for initial live deployment), the max drawdown drops to ~23% of account while expected profit is ~$2,530 over the same period.

---

## 2. Performance by setup type

| Setup | Trades | Win rate | Total R | Profit factor | Expectancy |
|---|---|---|---|---|---|
| Breaker | 2,194 | 23.7% | +426.8 R | 1.25 | +0.195 R |
| Three Tap | 388 | 36.1% | +79.3 R | 1.32 | +0.204 R |

The Breaker carries the volume (85% of all trades). The Three Tap has a higher win rate and slightly better profit factor, but fires less frequently.

---

## 3. Performance by direction

| Direction | Trades | Win rate | Total R | Profit factor |
|---|---|---|---|---|
| Long | 1,147 | 23.4% | +297.5 R | 1.34 |
| Short | 1,435 | 27.2% | +208.5 R | 1.20 |

Both directions are profitable. Longs have a higher profit factor; shorts have a higher win rate. The 200-day SMA filter ensures longs are only taken above the macro trend and shorts only below it.

---

## 4. Performance by market regime

| Period | Description | Trades | Win rate | Total R | Profit factor |
|---|---|---|---|---|---|
| 2021 Q3-Q4 | Bull market recovery | 194 | 26.3% | +26.1 R | 1.18 |
| 2022 H1 | Bear market (Luna) | 225 | 24.4% | +11.0 R | 1.06 |
| 2022 H2 | Bear market (FTX) | 195 | 24.6% | +4.1 R | 1.03 |
| 2023 | Choppy recovery | 508 | 20.5% | +24.6 R | 1.06 |
| 2024 H1 | ETF rally | 268 | 21.3% | +30.0 R | 1.14 |
| 2024 H2 | Consolidation | 196 | 24.0% | +54.4 R | 1.37 |
| 2025-2026 | Current period | 996 | 29.8% | +355.8 R | 1.51 |

The system was profitable in ALL seven tested market regimes. The weakest periods (2022 bear, 2023 chop) were marginally positive rather than negative — the 200 SMA filter prevented the deep losses seen in unfiltered testing. The strongest performance came in 2025-2026 (+355.8R, PF 1.51).

---

## 5. Performance by coin (top 10)

| Coin | Trades | Win rate | Total R | Profit factor |
|---|---|---|---|---|
| TRX | 141 | 32.6% | +112.7 R | 2.19 |
| FIL | 119 | 27.7% | +92.5 R | 2.08 |
| SOL | 134 | 21.6% | +83.0 R | 1.79 |
| ATOM | 140 | 22.9% | +47.0 R | 1.44 |
| DOT | 132 | 33.3% | +45.5 R | 1.52 |
| ETC | 122 | 34.4% | +36.7 R | 1.46 |
| LTC | 130 | 22.3% | +33.8 R | 1.33 |
| DOGE | 141 | 29.8% | +31.5 R | 1.32 |
| BNB | 146 | 24.7% | +30.9 R | 1.28 |
| APT | 76 | 34.2% | +30.7 R | 1.61 |

All 20 coins were tested. Top performers show PF above 1.4. No coin had a catastrophic loss — the system degrades gracefully on coins where the edge is weaker.

---

## 6. What was tested and rejected

The committee tested the following additions. All were rejected because backtest data showed they either hurt performance or added no value:

| Component | Result | Why rejected |
|---|---|---|
| ADX chop filter (ADX < 20) | -181 R vs no filter | Removed profitable bear market setups along with noise |
| OTE Reversal (Fib .618-.786) | -751 R on 2,415 trades | Fires on every retrace, 82% loss rate even with S/R confluence |
| OTE v2 (redesigned with strict S/R) | -39 R on 132 trades | Better than v1 but still net negative |
| Range + MSB detector | 0 trades on 1H | Criteria too strict for 1H — never triggered |
| Structure-based HTF filter | +64.5 R (vs +72.5 SMA) | Underperformed the simpler 200 SMA |
| Combined filter (SMA + structure) | +58.6 R | Reduced opportunities without improving quality |

Key learning: the edge comes from the pattern itself (market structure + trapped traders), not from stacking indicators. The source material (RektProof, CryptoCred, EmperorBTC) was right — simplicity wins.

---

## 7. Known weaknesses and limitations

1. **Low win rate (25.5%)**: You will lose roughly 3 out of 4 trades. The system profits because winners average +3.69R while losers are capped at -1R. This is psychologically demanding.

2. **Max losing streak of 20**: At 1% risk, this is a 20% drawdown. At 0.5% risk, it's 10%. You must be capitalized to survive these streaks.

3. **Choppy markets degrade performance**: 2022-2023 produced thin margins (PF 1.03-1.06). The system doesn't lose money in chop, but it barely profits either. No tested filter solved this without hurting trending period profits.

4. **Execution model dependency**: The corrected next-bar-open model is essential. Same-bar entry inflates results by approximately 4% on 1H and up to 50% on lower timeframes (5m).

5. **Backtest ≠ live performance**: Slippage, spread, exchange fees, and order book depth are not fully modeled. Expect live performance to be 10-20% worse than backtest.

---

## 8. Deployment configuration

### TradingView (Pine Script v1.0)

File: `strategy_v1.pine` (325 lines)

Default parameters (all validated):
- Swing lookback: 5
- SMA period: 200 (fetched from daily timeframe)
- Breaker max bars to retest: 48
- Breaker min displacement: 1.0 ATR
- Three Tap sweep range: 0.2 - 2.0 ATR
- Minimum R:R: 1.5
- Stop buffer: 0.3 ATR
- Position size: 2% of equity

### Recommended live settings

- Timeframe: 1H on crypto USDT pairs
- Risk per trade: 0.5% for first 3 months, increase to 1% after validating on live data
- Maximum simultaneous positions: 1 per coin
- Coins: start with TRX, SOL, DOT, ETC, DOGE (top performers)
- Alerts: set TradingView alerts on Long Entry, Short Entry, and Position Closed

---

## 9. System architecture

The system consists of 16 files totaling ~7,500 lines of code:

### Core engine (4 files, ~2,810 lines)
- `market_structure.py` — Swing detection, HH/HL/LH/LL classification, 8 market regimes, MSB detection
- `liquidity_mapper.py` — Equal highs/lows, untapped levels, FTA target selection, stop-safety checks
- `breaker_detector.py` — RektProof Breaker setup detection, quality scoring, zone management
- `additional_setups.py` — Three Tap detector, OTE (archived), Range+MSB (archived), trade simulator

### Backtests (6 files, ~2,700 lines)
- `backtest_engine.py` — Daily data loader and results tracker
- `htf_filter_backtest.py` — Comparative filter testing (proved 200 SMA is best)
- `backtest_1h.py` — 1H hourly backtest runner
- `backtest_combined.py` — Multi-setup combined runner
- `final_backtest_v2.py` — ADX + OTEv2 testing (proved both hurt)
- `corrected_final_backtest.py` — v1.0 release validation with corrected execution

### Validation (3 files, ~1,600 lines)
- `test_market_structure.py` — 6/6 tests pass
- `test_pipeline.py` — End-to-end pipeline validation
- `test_integration.py` — Structure + liquidity + breaker chain

### Audit & exchange (2 files, ~1,130 lines)
- `backtest_sanity_audit.py` — Independent audit layer for lookahead/execution bias
- `final_backtest_v2_exchange.py` — Walk-forward exchange-fed runner

### Deployment (1 file, 325 lines)
- `strategy_v1.pine` — TradingView Pine Script v5, ready to paste

---

## 10. Risk management rules

Derived from EmperorBTC Ch.5-7 and committee consensus:

1. Never risk more than 1% of account per trade (0.5% recommended initially)
2. Maximum 1 position per coin at any time
3. Minimum 6 bars (6 hours) between entries on the same coin
4. R:R must be at least 1.5:1 or the trade is rejected
5. If 2+ liquidity pools exist near the stop level, skip the trade
6. Only long above the 200-day SMA, only short below it
7. After 10 consecutive losses, reduce position size to 0.25% for the next 5 trades
8. Monthly review: if the month is -15R or worse, pause for 48 hours and review

---

## 11. Decision audit trail

Every parameter in this system traces back to either source material or backtest evidence:

| Parameter | Value | Source |
|---|---|---|
| Swing lookback | 5 | Standard fractal detection; CryptoCred confirms wait for confirmation |
| 200 SMA filter | On | EmperorBTC Ch.16 + backtest: PF 2.48 vs 1.69 without |
| ATR period | 14 | Standard; used universally in the source material |
| Min R:R | 1.5 | CryptoCred Lesson 5: FTA target must provide actionable R:R |
| Stop buffer | 0.3 ATR | CryptoCred: stop above/below the candle that broke the level + buffer |
| Breaker max retest | 48 bars | Committee: 2 days on 1H. Trader 3: retests beyond this lose relevance |
| Three Tap sweep | 0.2-2.0 ATR | Committee: too shallow = noise, too deep = not a sweep |
| Quality gate | Q >= 50 | Backtest: Q>=50 balances trade count vs quality |
| No ADX filter | — | Backtest proved it hurt: -181R vs no filter |
| No OTE setup | — | Backtest proved it loses: -751R on 2,415 trades |
