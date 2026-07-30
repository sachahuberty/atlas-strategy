# ATLAS — All-Techniques Layered Allocation Strategy

Weekly-rebalanced, regime-aware, multi-signal global asset allocation strategy.
Built as a personal portfolio project on the IE MBDS Algorithmic Trading course material.

**Architecture in one line:** every course technique produces a *view* (regime posture from HMM/K-Means,
mean-reversion z-scores, technical levels, news sentiment) → Black-Litterman fuses views with equilibrium
returns → one SLSQP optimizer produces the portfolio → an autoencoder anomaly score can override to
defensive → everything validated weekly out-of-sample with walk-forward, transaction costs, and an
ablation study per signal.

See `PROJECT_STRUCTURE.md` for the full design document.

## Status

- [x] Stage 1 — scaffold, data layer, universe screener, metrics
- [x] Stage 2 — classical allocations + weekly backtester (baseline)
- [x] Stage 3 — regime detection (HMM primary) + posture switching
- [x] Stage 4 — autoencoder anomaly override
- [x] Stage 5 — mean-reversion views
- [x] Stage 6 — technical/options views + execution timing
- [x] Stage 7 — sentiment views
- [x] Stage 8 — Black-Litterman fusion
- [x] Stage 9 — full walk-forward validation
- [x] Stage 10 — Monte Carlo + stress testing
- [x] Stage 11 — ablation study

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
pip install -e .
set FRED_API_KEY=your_key_here  # free key: https://fred.stlouisfed.org/docs/api/api_key.html
pytest
```

## Results

Out-of-sample, cost-inclusive, 2022-01 to 2026-07 (notebook 09), frozen
config after stage 9's IS-only grid search (`meanreversion.lookback_days=20`,
`rebalance.no_trade_band=0.01`, `rebalance.max_weekly_turnover=0.01`),
re-run after the stage-11 V2 unit-mismatch fix:

| Strategy | Sharpe | Max drawdown | Ann. vol |
| --- | --- | --- | --- |
| permanent | 1.07 | -12.6% | 8.3% |
| hrp | 0.85 | -9.2% | 3.6% |
| black_litterman (frozen) | 0.83 | -10.2% | 5.2% |
| risk_parity | 0.81 | -16.6% | 9.8% |
| gmv | 0.75 | -8.1% | 3.2% |
| max_sharpe_static | 0.60 | -13.6% | 5.5% |
| sixty_forty | 0.56 | -22.6% | 11.5% |

The naive `permanent` benchmark still posts the best OOS Sharpe, as it
has since stage 8. Grid-searching mean-reversion and turnover
parameters IS-only, selected for cross-fold stability rather than
peak IS performance, did not improve OOS Sharpe over stage 8's
untuned defaults (0.83 vs. 0.895) -- reported as found, per this
project's overfitting-defense discipline. **Fixing the V2 unit-
mismatch bug (below) did not improve the frozen strategy's OOS Sharpe
either (0.8348, down slightly from 0.8564 pre-fix)** -- once properly
scaled, mean-reversion's net contribution in this OOS window is still
marginally negative, consistent with stage 5's own event study finding
mixed reversion hit rates across the universe. The fix was made on its
own merits (a real unit bug, not a calibration choice) and reported
honestly regardless of outcome. Full per-quarter breakdown and
methodology notes in `notebooks/09_strategy_backtest.ipynb`.

**Stage 10 forward-looking risk report** (notebook 10), on the current
book as of 2026-07-24: historical stress through 2008/2020/2022 shows
-17.6% / -16.9% / -10.4% cumulative return respectively (GFC 2008 the
worst, -22.6% max drawdown); a reverse stress scan of the full
2007-2026 history independently rediscovers the COVID crash
(2020-03-16, -4.8%) as the single worst realized day, cross-validating
the hand-picked scenario windows. A 1,000-path GBM Monte Carlo on the
strategy's own realized OOS drift/vol (4.4% / 5.3%) gives P(positive)
of 80% at 1 year rising to 95% at 4 years.

**Stage 11 ablation study** (notebook 11), the project's designated
referee for module worth: V1 (regime view) is the strategy's clearest
positive contributor (+0.147 marginal OOS Sharpe); V3 (technical view)
is actively hurting performance (-0.265 marginal Sharpe, the largest
effect in the study, in the wrong direction); V2 (mean-reversion) is
essentially inert (-0.013), consistent with a diagnosed unit-scale bug
now being fixed; the anomaly override barely moves the needle (+0.002,
matching stage 4's original finding); sentiment (V4) is a structural
zero since it was never wired into the backtested pipeline. A
turnover-cap sensitivity check found relaxing the frozen 1% cap to the
S4-convention 2% raises OOS Sharpe from 0.806 to 0.875 with no
drawdown cost -- see below for the resulting config changes.
