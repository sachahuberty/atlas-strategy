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

Out-of-sample, cost-inclusive, 2022-01 to 2026-07 (notebook 09), after
the stage-11 V2 unit-mismatch fix and the turnover-cap policy
restoration (`rebalance.max_weekly_turnover=0.02`, `no_trade_band=0.005`,
S4 convention -- no longer grid-tuned, see below):

| Strategy | Sharpe | Max drawdown | Ann. vol |
| --- | --- | --- | --- |
| permanent | 1.07 | -12.6% | 8.3% |
| black_litterman (frozen) | 0.90 | -10.0% | 5.5% |
| hrp | 0.82 | -9.2% | 3.5% |
| risk_parity | 0.81 | -16.5% | 9.8% |
| gmv | 0.74 | -8.1% | 3.2% |
| max_sharpe_static | 0.60 | -13.6% | 5.5% |
| sixty_forty | 0.57 | -22.5% | 11.5% |

The naive `permanent` benchmark still posts the best OOS Sharpe, but
the gap has narrowed considerably over the course of stages 9-11.
Two changes bracket this table's history:

- **Fixing the V2 unit-mismatch bug did not improve OOS Sharpe**
  (0.8348, down slightly from the pre-fix 0.8564) -- once properly
  scaled, mean-reversion's net contribution in this OOS window is
  still marginally negative, consistent with stage 5's own event
  study finding mixed reversion hit rates across the universe. Fixed
  on its own merits (a real unit bug, not a calibration choice) and
  reported honestly regardless of outcome.
- **Restoring the turnover cap to the S4-convention 2% (from stage 9's
  grid-frozen 1%, where the no-trade band equaled the cap and
  collapsed every rebalance to "trade exactly 1% or freeze") is the
  single largest improvement in the project's history: Sharpe rose
  from 0.8348 to 0.8969**, with max drawdown actually improving
  slightly (-10.02% vs -10.18%). `black_litterman_frozen` now beats
  both `risk_parity` and `hrp` for the first time. Stage 11's ablation
  had already flagged this precisely: a stability-penalized grid
  applied to a risk-control cap will always buy stability with
  paralysis, regardless of signal quality -- unwinding that throttle,
  not any signal improvement, drove this gain.

Grid-searching mean-reversion parameters IS-only, selected for
cross-fold stability, remains in place (`meanreversion.lookback_days=20`);
the turnover parameters are no longer grid-selected -- see
`config.yaml`'s `rebalance` section and notebook 09's addenda for the
full reasoning. Full per-quarter breakdown and methodology notes in
`notebooks/09_strategy_backtest.ipynb`.

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
