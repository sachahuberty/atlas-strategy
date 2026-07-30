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
all three of stage 11's priority-2 fixes (V2 unit mismatch, turnover
cap restored to the S4 policy, cash excluded from `max_sharpe`'s
universe). **Development-set OOS, not virgin OOS** -- see the caveat
below the table; with ~4.5 years of daily data the standard error on
a Sharpe estimate is roughly ±0.22, so most rows here are within one
SE of each other and should not be read as a confident ranking.

| Strategy | Sharpe | Max drawdown | Ann. vol |
| --- | --- | --- | --- |
| permanent | 1.07 ± 0.22 | -12.6% | 8.3% |
| hrp | 0.82 ± 0.22 | -9.2% | 3.5% |
| risk_parity | 0.81 ± 0.22 | -16.5% | 9.8% |
| black_litterman (frozen) | 0.73 ± 0.22 | -8.7% | 4.5% |
| gmv | 0.74 ± 0.22 | -8.1% | 3.2% |
| max_sharpe_static | 0.60 ± 0.22 | -13.6% | 5.5% |
| sixty_forty | 0.57 ± 0.22 | -22.5% | 11.5% |

The naive `permanent` benchmark still posts the best OOS Sharpe. Three
changes were made in sequence, each re-validated by re-running the
full frozen OOS walk-forward, and each reported honestly regardless of
which direction it moved the number:

- **V2 unit-mismatch fix: did not improve Sharpe** (0.8348, vs. 0.8564
  pre-fix). Once mean-reversion's view is correctly scaled to be
  comparable to the other view families (it previously entered
  Black-Litterman at roughly 1/100th the intended magnitude), its net
  contribution in this OOS window is still marginally negative --
  consistent with stage 5's own event study, which found reversion
  hit rates above chance for some assets and below chance for others.
- **Turnover cap restored to the S4 convention (2%, from stage 9's
  grid-frozen 1% where the no-trade band equaled the cap): the single
  largest improvement in the project's history, Sharpe rising from
  0.8348 to 0.8969**, with max drawdown also improving slightly. Stage
  11's ablation had flagged this precisely in advance: a stability-
  penalized grid applied to a risk-control cap will always buy
  stability with paralysis, regardless of signal quality.
- **Cash excluded from `max_sharpe`'s universe: made performance
  meaningfully worse** (Sharpe 0.7265, down from 0.8969), the opposite
  of the initial hypothesis that this would "un-mute" risk-taking. A
  targeted diagnostic traced the mechanism: with cash available,
  `max_sharpe` could sometimes build a moderate, cash-anchored book
  that won the utility gate over GMV/Risk Parity outright; once
  excluded, that book lost those comparisons more often and the
  pipeline defaulted to GMV (a pure minimum-variance book with no
  return view) more frequently instead. The fix is kept regardless --
  the underlying degeneracy (rf=0 makes a near-zero-vol asset
  pathologically attractive in a Sharpe objective) is real independent
  of whether removing it helps this particular backtest -- but the
  result argues the fix should go further (e.g. re-deriving the
  equilibrium prior or utility gate with cash held out from the
  start), not stop here.

Grid-searching mean-reversion parameters IS-only, selected for
cross-fold stability, remains in place (`meanreversion.lookback_days=20`);
turnover parameters are no longer grid-selected -- see `config.yaml`'s
`rebalance` section and notebook 09's addenda for the full reasoning
and per-fix numbers.

**On statistical significance:** every Sharpe above carries an
estimated standard error of roughly ±0.22 (0.5/√4.5 years of daily
data) purely from sampling noise. `permanent`'s apparent lead over
every active strategy, and the differences between `black_litterman`,
`risk_parity`, `hrp`, and `gmv`, are not statistically distinguishable
at this sample size -- this table should be read as a set of plausible
outcomes clustered around a similar Sharpe, not a confident ranking.
Full per-quarter breakdown and methodology notes in
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
(since fixed, see above); the anomaly override barely moves the needle (+0.002,
matching stage 4's original finding); sentiment (V4) is a structural
zero since it was never wired into the backtested pipeline. A
turnover-cap sensitivity check found relaxing the frozen 1% cap to the
S4-convention 2% raises OOS Sharpe from 0.806 to 0.875 with no
drawdown cost -- see below for the resulting config changes.
