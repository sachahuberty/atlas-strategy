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

Out-of-sample, cost-inclusive, 2022-01 to 2026-07 (notebook 09),
current state after DIAGNOSTIC.md's Tier-1 fixes (V3 disabled, cash
exclusion reverted, a real risk-free rate). **Development-set OOS, not
virgin OOS** -- with ~4.5 years of daily data the standard error on a
Sharpe estimate is roughly ±0.22, so most rows here are within one SE
of each other and should not be read as a confident ranking.

| Strategy | Sharpe | Excess Sharpe (vs. T-bill) | Max drawdown | Ann. vol |
| --- | --- | --- | --- | --- |
| permanent | 1.07 ± 0.22 | 0.59 | -12.6% | 8.3% |
| risk_parity | 0.81 ± 0.22 | 0.41 | -16.5% | 9.8% |
| sixty_forty | 0.57 ± 0.22 | 0.22 | -22.5% | 11.5% |
| max_sharpe_static | 0.60 ± 0.22 | -0.12 | -13.6% | 5.5% |
| hrp | 0.82 ± 0.22 | -0.31 | -9.2% | 3.5% |
| **black_litterman (frozen)** | **0.74 ± 0.22** | **-0.51** | -8.1% | 4.5% |
| gmv | 0.74 ± 0.22 | -0.51 | -8.1% | 3.2% |

`black_litterman` and `gmv` are numerically identical -- the gate is
selecting GMV every week (see below). **On an excess-return basis
(net of the T-bill rate, which averaged ~3-4% through this window),
neither ATLAS nor plain GMV nor HRP beat cash at all** -- their
excess Sharpes are negative. Only `permanent`, `risk_parity`, and
`sixty_forty` cleared that bar. This is a materially different, more
sobering picture than the total-return Sharpes alone show, and it was
missing from this README until DIAGNOSTIC.md's audit flagged it
(§5.2 item 2).

### What changed, in order, each re-validated by re-running the full frozen OOS walk-forward

Every entry below is kept regardless of which direction it moved the
number -- **the measured V3 regression, the V2 result, and the
cash-exclusion regression are the most valuable evidence in this
project**, not failures to edit out.

1. **V2 unit-mismatch fix: did not improve Sharpe** (0.8348, vs. 0.8564
   pre-fix). Once mean-reversion's view is correctly scaled to be
   comparable to the other view families, its net contribution is
   still marginally negative -- consistent with stage 5's own event
   study, which found reversion hit rates above chance for some
   assets and below chance for others.
2. **Turnover cap restored to the S4 convention (2%, from stage 9's
   grid-frozen 1% where the no-trade band equaled the cap): the
   single largest improvement up to that point, Sharpe rising from
   0.8348 to 0.8969.** Stage 11's ablation had flagged this precisely
   in advance: a stability-penalized grid applied to a risk-control
   cap will always buy stability with paralysis.
3. **Cash excluded from `max_sharpe`'s universe (REVIEW.md's
   recommendation): made performance meaningfully worse** (Sharpe
   0.7265, down from 0.8969) -- the opposite of the hypothesis that
   this would "un-mute" risk-taking. A gate census proved the
   mechanism: cash wasn't just riding a Sharpe-ratio degeneracy, it
   was the *variance anchor* that let the max-Sharpe book win the
   utility gate at all. With cash allowed, it won 239/239 OOS weeks;
   excluded, 0/239 -- the whole pipeline silently became plain GMV.
4. **DIAGNOSTIC.md Tier 1 (independently re-derived from the repo's
   own cached data, not the notebooks): V3 disabled** (`modules.
   technical_view: false`, measured -0.265 marginal Sharpe, the
   largest effect of any module) **and the cash exclusion from step 3
   reverted** -- it was the wrong remedy; the right one is pricing
   cash properly.
5. **A real risk-free rate (FRED DTB3) replaces `rf=0` throughout:**
   threaded into `sharpe`, `utility`, `max_sharpe`, and overriding the
   cash bucket's equilibrium prior (which was ~0.1%/yr by construction
   for a near-zero-covariance asset, versus BIL's realized ~3.9%/yr).
   This fixes the cash degeneracy at its mathematical source instead
   of deleting the asset.

**Combined Tier-1 result: Sharpe 0.7382, essentially unchanged from
before any of these three fixes, and identical to plain GMV.** A
direct gate census with the fixed pipeline (16 sampled OOS weeks)
confirms why: **GMV still wins the utility gate 16/16 times.** Cash
is no longer degenerately attractive (its weight inside the max-Sharpe
book now ranges ~1-20% instead of being pinned to the cap), but the
max-Sharpe book's return advantage still doesn't overcome its higher
variance under the gate's quadratic risk-aversion penalty (A=5).
DIAGNOSTIC.md's own diagnosis anticipated this exactly: *"the gate is
not a safety net on the optimizer -- it is a second, stricter
optimizer that always prefers minimum variance... you have two
conflicting objective functions in series, and the second one wins
every time."* Tier 1 was worth doing on its own merits (V3 is a real,
measured negative contributor; cash is now priced correctly rather
than by mathematical accident), but it was **not sufficient** to move
the headline number -- the utility gate itself is the confirmed
bottleneck, and re-scoping it (DIAGNOSTIC.md Tier 2) is the next,
not-yet-applied step.

### The benchmark comparison itself is window-dependent

Re-running `permanent` and GMV over the **in-sample** period
(2013-2021, never used to design either) reframes the entire question
(DIAGNOSTIC.md §5.3):

| Strategy | 2013-2021 (IS) | 2022-2026 (OOS) |
| --- | --- | --- |
| permanent | **0.476** | 1.067 |
| GMV (= what ATLAS currently is) | **1.018** | 0.734 |

Permanent's OOS lead is not a stable property of the benchmark -- it
is a property of *this specific window* (a commodity spike, ~5% cash
yields, and an equity bull market all landed in 2022-2026). Over
2013-2021, the minimum-variance posture ATLAS has converged to scored
*better* than Permanent, 1.018 vs. 0.476. Neither number should be
read as "the" answer; judged against a single lucky or unlucky draw,
either strategy can look definitively better. This does not excuse
the utility-gate finding above, but it means the honest headline is
"ATLAS currently behaves like a minimum-variance book, which wins in
some regimes and loses in others" rather than "ATLAS loses to
Permanent."

Grid-searching mean-reversion parameters IS-only, selected for
cross-fold stability, remains in place (`meanreversion.lookback_days=20`);
turnover parameters are no longer grid-selected -- see `config.yaml`'s
`rebalance` section and notebook 09's addenda for the full history and
every intermediate number.

**On statistical significance:** every Sharpe above carries an
estimated standard error of roughly ±0.22 (0.5/√4.5 years of daily
data) purely from sampling noise. This table should be read as a set
of plausible outcomes clustered around a similar Sharpe, not a
confident ranking. Full per-quarter breakdown and methodology notes
in `notebooks/09_strategy_backtest.ipynb`; the full evidence-based
diagnosis is in `DIAGNOSTIC.md`.

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
