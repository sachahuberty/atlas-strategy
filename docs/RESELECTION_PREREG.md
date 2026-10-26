# Re-selection pre-registration

**Written before any result in this task exists.** Its own commit hash
is the evidence of that; see the bottom of this file. Nothing below may
be changed after Phase 2 results are computed except to correct a typo
that doesn't alter the rule itself (any such correction would need its
own commit, visibly after the first result's).

**Scope.** This file is the pre-registration for the task re-selecting
four OOS-chosen settings (gate risk-aversion `A`, turnover cap/band,
V3 on/off, V5 on/off) plus the never-done Phase C risk-off-target
choice, using 2010-2021 in-sample data only, on the corrected
Black-Litterman pipeline (`excess_return_prior: true`,
`ledoit_wolf_corr`, current posture mapping). `config/config.yaml`,
`config/regime_posture.yaml`, and everything under `live/` are not
touched by this task; the live book keeps trading the current config
throughout. No return on or after `general.oos_start_date`
(2022-01-01) is read until Phase 4, and Phase 4 runs the result
through OOS exactly once and does not feed back into selection.

## 1. IS folds

**Fold scheme: exactly the one notebook 09's existing meanreversion/
momentum grid searches already use** (`fold_periods =
is_prices.index.to_period("Y").unique()`), now given a tested, shared
home in `src/atlas/selection.py` (`annual_fold_periods`) instead of
being re-typed inline per notebook cell. Not a new fold scheme.

Computed directly from the frozen universe (`universe_20211231.csv`),
`general.start_date` (2010-01-01) through `general.is_end_date`
(2021-12-31), on real downloaded prices, strictly `prices.loc[:is_end]`
-- **12 annual folds**, confirmed by running `selection.
annual_fold_periods` on that slice before writing this file:

```
2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021
```
(`is_prices` itself spans 2010-01-04 to 2021-12-31.)

**Clarification of a terminology clash:** elsewhere in this project
(README, ANALYSIS_V4.md) "IS 2014-2021" names the *reported comparison
window* used when quoting headline ann. return/vol/Sharpe tables --
2010-2013 is a warm-up period excluded from THOSE tables because the
3Y covariance lookback (`optimization.lookback_days: 756`) and 5Y HMM
lookback (`regimes.hmm.lookback_days: 1260`) make early decisions
noisier, not because those years are out-of-sample. The GRID-SEARCH
fold scheme (what Phase 0 point 1 requires reusing exactly) has always
spanned the full 2010-2021 `is_prices` range, and continues to here.
Phase 2's result tables report both: per-fold (2010-2021, all 12 folds,
for the selection rule itself) and the trimmed 2014-2021 window
(matching this project's existing headline-table convention, for
readability only -- it plays no role in selection).

**No-lookahead-within-a-fold rule.** Every model fit a fold's
evaluation segment depends on (HMM, any scaler, the autoencoder,
Ledoit-Wolf covariance) is fit on `window = returns.loc[:as_of]` --
`backtest.run`'s own truncation, unchanged by this task -- so a fold's
decisions never see data from later in that same fold, let alone a
later fold. This is not new scaffolding; it is the existing, tested
no-lookahead property of `backtest.run`/`strategy.
black_litterman_strategy`, confirmed in `test_backtest.py`'s
`test_lookahead_canary_shifted_returns_change_decisions` and relied on
unchanged here.

## 2. Metric

**Excess Sharpe, rf = DTB3, net of 10 bp transaction costs, weekly
decisions, next-day execution -- exactly `backtest.run`'s own
accounting**, i.e.:

- weights decided Friday close on `window = returns.loc[:as_of]`,
  traded at Monday open, costs = `rebalance.transaction_cost_bps`
  (10 bp, unchanged across every candidate in this task) times
  executed one-way turnover, charged on the execution day
  (`backtest.run`'s existing `cost_today = cost_rate *
  pending_turnover`);
- per-fold metric = `metrics.sharpe(fold_daily_returns, rf=fold_rf)`
  where `fold_rf` is the same DTB3-derived annual-rate series
  (`rf_raw["DTB3"] / 100.0`, reindexed/ffilled onto the returns
  calendar) every other notebook in this project already uses,
  sliced to the fold's own date range -- NOT `backtest.walk_forward`'s
  built-in `sharpe` column, which defaults to `rf=0.0` (total-return
  Sharpe) and is therefore the wrong metric for this task. A new,
  tested helper in `src/atlas/reselection.py` (Phase 1) computes
  per-fold excess Sharpe on the SAME fold partition
  `selection.annual_fold_periods` returns, reusing `metrics.sharpe`
  itself, not reimplementing it.

## 3. Selection rule

**Stability score**, unchanged formula from notebook 09's existing
grid searches and `backtest.walk_forward`'s summary (`mean(fold
values) - stability_penalty * std(fold values, ddof=1)`,
`grid_search.stability_penalty: 1.0`), now in `selection.
stability_score`.

**Tie rule** (`selection.select_candidate_or_keep_current`): the
CURRENT config value is kept unless a candidate (a) beats it on
stability score AND (b) wins a STRICT majority of the folds the two
share (a per-fold tie counts for neither side). Current/simpler wins
every tie on either criterion.

**One-setting-at-a-time, not joint grid-argmax.** The rule above is
applied separately to each of the five settings below, holding the
other four at their CURRENT value, not to the full 144-cell grid's
joint argmax. Rationale: "the settings that were chosen" were each
chosen independently by the original OOS ablation (notebook 11 scored
gate A, turnover, and each module flag in separate tables), so
re-selecting them the same way -- one setting at a time against the
same current baseline -- is the correct like-for-like correction, and
is far less exposed to interaction-effect overfitting on only 12
non-trivial IS years than a 144-cell joint argmax would be. The full
144-cell grid is still computed and reported in Phase 2 for
diagnostics (interaction effects, gate census, sensitivity context)
but is explicitly NOT the selection mechanism -- stated here before any
cell of it is computed.

**Current (incumbent) values**, the baseline every challenger is
compared against:

| Setting | Current value |
|---|---|
| V3 (`modules.technical_view`) | off |
| V5 (`modules.momentum_view`) | off |
| Risk-off target (Phase C, new) | `legacy` (fixed_income + commodity) |
| Gate risk aversion `A` | 2 |
| Turnover cap / no-trade band | 2% / 0.5% |

(V2 `modules.meanreversion_view` and `modules.anomaly_override` stay
`true` throughout this entire task, at their current parameters --
not part of the re-selection, per the task's explicit instruction.)

## 4. Candidate grid and N_trials

| Setting | Candidates |
|---|---|
| View set | {V3 off, V3 on} x {V5 off, V5 on} -- 4 combinations |
| Risk-off target | {legacy, bonds_cash, bonds_gold} -- 3 |
| Gate risk aversion `A` | {1, 2, 3, 5} -- 4 |
| Turnover cap / band | {1%/0.5%, 2%/0.5%, 4%/0.5%} -- 3 |

**N_trials = 4 x 3 x 4 x 3 = 144.** This is the full Cartesian grid
size, not the (smaller) number of pairwise comparisons the
one-setting-at-a-time rule above actually uses for the adopt/keep
decision (that count is 9 non-current candidates + the current
baseline = 10). N_trials for the deflated Sharpe ratio (Phase 4) uses
the larger, more conservative 144: every cell is computed and reported,
so a skeptical reader should assume all 144 were implicitly available
to cherry-pick from, even though the pre-registered rule only acts on
a subset of those comparisons.

## 5. Efficient-by-construction harness (Phase 1) and the anomaly-cost concern

Per decision date, running the real `black_litterman_strategy` +
`with_anomaly_override` pipeline once per one of the 144 grid cells
would mean refitting the autoencoder (expensive: LSTM train with early
stopping, every `anomaly.refit_frequency_days`) up to 144x per decision
date across a ~600-decision IS period -- **this would be impractically
slow** (single-digit minutes per autoencoder refit observed elsewhere
in this project's own notebooks; 144x that per refit date is not
tractable in this task's time budget).

**Proposed solution, to be implemented in Phase 1, not silently
dropped:** only the VIEW SET and the risk-off target actually change
`mu_bl` (the Black-Litterman posterior) and therefore the max-Sharpe
candidate. Per decision date, the following are IDENTICAL across all
144 cells and are computed/cached exactly ONCE:
`rf`, `posture`, `cov`, `prior` (equilibrium returns), the raw V2/V3/V5
per-asset view SERIES (`meanrev_view`, `tech_view`, `mom_view` --
whether a module flag is on only changes whether `views.py` turns that
series into P/Q rows, not the series itself), the autoencoder's
reconstruction-error/anomaly flag, and the GMV/risk-parity candidate
books (both independent of `mu_bl`). Only 12 (view-set x risk-off-target)
variants need their own `mu_bl` / Black-Litterman-candidate / anomaly-
blended-weight computed and cached per decision date -- the other
132 cells reuse that same per-date cache. Gate `A` re-selects among a
variant's cached candidates with no re-run (`allocation.utility_select`
is cheap). Turnover cap/band re-run only `backtest.run`'s execution
layer over a `backtest.memoize_strategy`-wrapped lookup of the already-
computed target-weight series, not the signal pipeline. This turns 144
full pipeline evaluations into 12, each anomaly-fit-bearing, plus 132
cheap re-derivations.

## 6. Phase 4 commitment

Phase 4 runs the IS-selected config (Phase 2's output, mechanically
applied) through the real `backtest.run` over 2022-2026 and the full
2014-2026 window, using the CURRENT live config's V2/anomaly settings
(on), **exactly once each**. That OOS result is reported alongside the
current config and the Permanent Portfolio benchmark, with standard
errors, a paired significance test, and a deflated Sharpe ratio using
`N_trials = 144` from this document plus an honest count of every
prior OOS look this project has already taken (documented in Phase 4,
not estimated here). **The OOS result does not change, re-open, or
re-run the Phase 2 selection in any way.** If Phase 4 technically fails
and must be re-run, every re-run is logged in `docs/RESELECTION.md`;
the selection itself is never touched because of an OOS number.

---

*This document's own commit is the pre-registration timestamp. No
result in Phases 1-4 may alter section 3's rule, section 4's grid, or
section 6's commitment after that commit exists.*
