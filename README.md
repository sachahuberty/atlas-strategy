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
current state after all of DIAGNOSTIC.md's Tier-1, Tier-2, Tier-3, AND
ANALYSIS_V2.md's Step-1/Step-2 fixes (V3 disabled, cash exclusion
reverted, a real risk-free rate, the utility gate recalibrated from
A=5 to A=2, HMM `n_states` selected by BIC, V2 restricted to
per-asset hit-rate-eligible tickers, classical benchmarks capped at
the same 20% per-asset limit the strategy faces, V5 dual momentum
implemented and tried, then disabled once the stage-11 ablation
measured it). **Development-set OOS, not virgin OOS** -- with ~4.5
years of daily data the standard error on a Sharpe estimate is
roughly ±0.22, so most rows here are within one SE of each other and
should not be read as a confident ranking.

| Strategy | Sharpe | Excess Sharpe (vs. T-bill) | Max drawdown | Ann. vol |
| --- | --- | --- | --- | --- |
| permanent | 1.07 ± 0.22 | 0.62 | -13.4% | 8.8% |
| **black_litterman (frozen)** | **0.95 ± 0.22** | **0.68** | -16.7% | 14.7% |
| risk_parity | 0.86 ± 0.22 | 0.45 | -16.5% | 9.8% |
| black_litterman (bucket-level) | 0.82 ± 0.22 | 0.41 | -16.2% | 9.7% |
| hrp | 0.83 ± 0.22 | -0.29 | -9.2% | 3.5% |
| gmv | 0.74 ± 0.22 | -0.51 | -8.1% | 3.2% |
| max_sharpe_static | 0.65 ± 0.22 | -0.07 | -13.6% | 5.5% |
| sixty_forty | 0.63 ± 0.22 | 0.28 | -22.5% | 11.5% |

Momentum (V5) was implemented, IS-tuned, and tried in the fusion
(ANALYSIS_V2.md Step 1) -- and it was a real, measured regression
(`black_litterman (frozen)` fell from Tier 3's 0.90 to 0.72). The
formal stage-11 ablation (Step 2, excess Sharpe as the primary
metric) confirmed why on its own dedicated terms: V5's marginal
contribution was **-0.236 excess Sharpe**, the single largest
negative effect ever measured in this project's ablation study,
exceeding V3's original -0.265 in relative terms. `modules.
momentum_view` is now `false`, the same measured-negative -> disabled
treatment V3 got in Tier 1. With momentum off, `black_litterman
(frozen)` is back to clearing every classical benchmark except
`permanent` -- Sharpe 0.95, excess Sharpe 0.68, essentially recovering
(and modestly exceeding, mostly via ordinary calendar drift as a few
more trading days entered the dataset between runs) Tier 3's
pre-momentum reading. Momentum's module, tests, and full negative
result all stay in the codebase -- an implemented-and-rejected signal
family is exactly the kind of evidence this project's methodology is
built to produce. **On an excess-return basis (net of the T-bill
rate, which averaged ~3-4% through this window), `hrp` and `gmv`
still don't beat cash at all** -- their excess Sharpes are negative.

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
6. **DIAGNOSTIC.md Tier 2 item 4: the utility gate itself was
   ablated** (notebook 11: gate on/off, and risk-aversion A in
   {2, 5, 10}), confirming DIAGNOSTIC.md's diagnosis that the gate,
   not the fusion pipeline, was the dominant bottleneck. A=5 let GMV
   win the gate 236/236 sampled OOS weeks; **A=2 measured the single
   largest Sharpe improvement in the whole ablation** (0.8330 ->
   0.9863 on the ablation's own shorter harness). A `vol_floor`
   alternative gate (reject max-Sharpe only if its vol exceeds a
   band around GMV's) was also measured and did not help -- identical
   to A=5, since it produced the same GMV-every-week outcome.
   `black_litterman.risk_aversion_for_utility_gate` was recalibrated
   5 -> 2 in `config.yaml` on this evidence.
7. **DIAGNOSTIC.md Tier 2 item 5: a bucket-level allocation layer**
   (`src/atlas/buckets.py`, `strategy.
   bucket_black_litterman_strategy`) was built and wired in for
   comparison, on the reasoning that ~22 assets are really 4-5
   correlated bets and per-asset covariance/return estimation is
   mostly noise. Kept alongside, not instead of, the asset-level
   path (`modules.bucket_level` documents which is "current" without
   deleting either) -- see the Tier-2 result below for how it
   performed once actually measured on this repo's full pipeline.
8. **DIAGNOSTIC.md Tier 3 action 6: per-asset V2 hit-rate gate.**
   `meanreversion.hit_rate_eligible_tickers` restricts V2's view to
   only tickers whose own historical hit rate clears
   `meanreversion.min_hit_rate` (0.55, IS-only, frozen for OOS like
   `lookback_days`/`entry_z`). Found 8 of 22 tickers eligible (`ACWI,
   GLD, HYG, IWM, QQQ, SPY, VNQ, VTI`).
9. **DIAGNOSTIC.md Tier 3 action 7: HMM `n_states` selected by BIC.**
   `regimes.hmm_bic_curve` fits a candidate HMM at each `n_states` in
   {2..6} and scores by hmmlearn's own `.bic()`; the IS-only argmin is
   **4** (previously a fixed, untuned 3).
10. **DIAGNOSTIC.md Tier 3 action 8: benchmarks capped like the
    strategy.** `allocation.permanent`/`sixty_forty` gained an optional
    `cap` parameter (default `None`, unchanged behavior everywhere
    except where a caller opts in); notebook 09's benchmark comparison
    now passes `cap=constraints.per_asset_cap` so `permanent`'s
    single-ticker `cash` bucket (previously an uncapped 25%) faces the
    same 20% limit every optimizer in this project already respects.
11. **ANALYSIS_V2.md Step 1: V5 dual momentum added to the fusion --
    made performance meaningfully worse** (Sharpe 0.9029 -> 0.7172).
    `src/atlas/momentum.py` implements Antonacci-style dual momentum:
    a relative leg (12-1 cross-sectional rank within each bucket) and
    an absolute leg (a bucket only gets a positive relative view if
    its own trailing return beats the risk-free rate; otherwise every
    member gets a flat negative view). IS-only grid search (2010-2021,
    a pure vectorized rank-spread test, no model fitting) selected
    `lookback_days=189, skip_days=0, top_fraction=0.5`, frozen for
    OOS. A 12-week gate census (`momentum_view` on vs. off, same
    frozen pipeline otherwise) shows the mechanism: in most sampled
    weeks the same book still wins the utility gate, but its
    composition differs substantially (mean L1 weight difference
    0.69) since V5's view reshapes `mu_BL` every week it fires; in 2
    of 12 sampled weeks -- both inside 2022's rate-hike/bond-selloff
    stretch -- the gate's choice itself flips from the max-Sharpe book
    to the defensive `risk_parity` book, driven by the absolute leg's
    flat negative view firing across most buckets simultaneously
    during a broad selloff. This is a different failure mode from V2
    (measured ~zero, too weak to matter) or V3 (a clean, uniform
    negative): V5 is a believable, mechanically-explained signal that
    materially reshapes the book and occasionally flips the gate, and
    on this OOS window that reshaping cost return. Kept in the
    codebase and left `modules.momentum_view: true` pending the
    stage-11 ablation's formal marginal-contribution number (next
    step) rather than disabled on this one frozen-pipeline reading
    alone.
12. **ANALYSIS_V2.md Step 2: the stage-11 ablation, re-run with excess
    Sharpe as the primary metric and V5 included, confirmed item 11's
    reading and gave it a number -- momentum's marginal contribution
    is -0.236 excess Sharpe, the largest negative effect of any module
    ever measured in this study.** The ablation harness itself needed
    modernizing first (it predated Tier 1-3 entirely: no `rf_series`,
    no `meanrev_eligible`, untuned HMM `n_states`), which also
    revealed V1 remains the clearest positive contributor (+0.081
    excess Sharpe), V2 stays ~neutral, V3/V4 are now structural
    no-ops (V3 already off, V4 never wired in), and the anomaly
    override never fired at all in this exact window. On V5's
    measurement, `modules.momentum_view` is now `false` -- the same
    measured-negative-by-ablation -> disabled treatment V3 got in
    Tier 1, not a new rule.

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

**Combined Tier-2 result: Sharpe rose from 0.7382 to 0.9103, a large,
real improvement on the actual frozen pipeline** (excess-return Sharpe
0.6333) -- directionally consistent with notebook 11's ablation (0.83
-> 0.99) though not identical in magnitude, expected since the
ablation ran on a different, shorter/differently-scoped date range and
without the anomaly override / turnover-cap interactions the full
frozen pipeline has. `black_litterman (frozen)` now beats every
classical benchmark in the comparison except `permanent` -- the first
time in this project's history the BL pipeline has cleared that bar.
The cost is real, not free: annualized vol nearly tripled (4.5% ->
14.4%) and max drawdown widened past `permanent`'s (-16.4% vs -12.6%),
because the recalibrated gate now lets the higher-vol max-Sharpe book
win the large majority of weeks instead of defaulting to minimum
variance.

**The bucket-level allocation layer does NOT reproduce DIAGNOSTIC.md's
own counterfactual, and this is reported as measured rather than
reframed.** DIAGNOSTIC.md §5.3 found an isolated bucket-level
counterfactual at Sharpe 1.279 OOS, beating `permanent` (1.067) in
that window. Once actually built and wired into this repo's real
pipeline (`strategy.bucket_black_litterman_strategy`, with the real
gate, real risk-free rate, and real turnover cap), it scores **0.8602**
-- close to `risk_parity` and `hrp`, but behind both the asset-level
book (0.9103) and `permanent` (1.0841). The most likely explanation is
that DIAGNOSTIC.md's counterfactual was computed as an isolated
experiment whose rf handling, gate behavior, and turnover/anomaly
interactions were not necessarily identical to the full pipeline this
repo now runs, so the two numbers are not measuring quite the same
object despite the shared label. Kept in the codebase behind
`modules.bucket_level` for future comparison, not deleted, per this
project's discipline of reporting negative or unremarkable results
plainly.

**Permanent still leads the whole comparison (1.0841 Sharpe), the same
conclusion as every prior stage of this project.** Tier 2's real,
measured win is that the BL pipeline's own OOS Sharpe now clears every
classical benchmark it previously lost to except `permanent` itself --
achieved by fixing the gate's calibration, not any signal or view
logic, and consistent with DIAGNOSTIC.md §3's diagnosis that the gate
was the dominant bottleneck. It is not yet a result that beats the
benchmark this project is measured against.

**Combined Tier-3 result: essentially flat on the strategy's own
number, and a modestly fairer (harder) benchmark.**
`black_litterman (frozen)` moved from 0.9103 to **0.9029** -- within
noise of Tier 2's number, not a real change. `permanent`, by contrast,
moved from 1.0841 to **1.0571**: capping its `cash` bucket (BIL alone,
previously an uncapped 25%) to the same 20% limit the strategy already
respects pulled its Sharpe down measurably, since `cash` had been
riding an advantage no other book in the comparison had. `black_
litterman (bucket-level)` also moved (0.8602 -> 0.8070), driven by the
new `n_states=4` regime read (bucket-level V1 is still active) rather
than V2 (bucket-level has no V2 view at all). `gmv`, `risk_parity`,
`hrp`, and `max_sharpe_static` are unchanged -- none of them touch V2,
HMM `n_states`, or a capped `permanent`/`sixty_forty` call.
`sixty_forty`'s Sharpe is also unchanged despite now being capped too:
nothing in its equity/fixed_income split actually exceeded 20% in this
universe, so the cap only ever bound for `permanent`'s single-ticker
`cash` bucket. None of Tier 3's three items moved the headline number
the way Tier 2's gate recalibration did, and that absence is itself
informative: this pipeline's remaining, measurable headroom was in the
gate's calibration, not in the HMM's state count, V2's asset coverage,
or the benchmark's fairness (though the last of those was still worth
fixing on its own terms -- an uncapped benchmark was never a fair
comparison to begin with, regardless of how much it moved the number).

**Momentum (V5) result: a real, measured regression -- Sharpe fell
from 0.9029 to 0.7172 (excess Sharpe 0.6319 -> 0.4357; ann. return
12.96% -> 9.57%).** This does not match ANALYSIS_V2.md's own
standalone momentum harness, which found cross-sectional momentum
raising IS and OOS returns and the absolute-trend leg cutting
drawdown -- but that harness tested momentum in isolation (a naive
tilt on top of `permanent`, no Black-Litterman fusion, no utility
gate), not wired into the actual pipeline this repo runs. This is the
same shape of surprise the bucket-level layer produced (ANALYSIS_V2.md
§3): a signal that looks good standalone can behave differently once
it flows through BL fusion and a utility gate that reacts to the
resulting `mu_BL`, not just to the raw signal. `black_litterman
(frozen)` no longer beats every classical benchmark the way it did
after Tier 2/3 -- it now trails `risk_parity`, `hrp`, and even the
bucket-level book, though it still beats `gmv`, `max_sharpe_static`,
and `sixty_forty`. `modules.momentum_view` stays `true` for now: this
one frozen-pipeline reading is a real, honest data point, but the
project's own convention (matching how V3 was disabled) is to make
that call from the stage-11 ablation's formal marginal-contribution
number, not from a single before/after comparison -- that ablation
re-run, with excess Sharpe as the primary metric, is the next step.

**Momentum (V5) ablation result and final disposition: -0.236 excess
Sharpe, the largest negative marginal contribution ever measured in
this study, larger in relative terms than V3's original -0.265.**
Modernizing the stage-11 ablation harness first (it had never been
updated for `rf_series`, `meanrev_eligible`, or the BIC-selected HMM
`n_states` -- it was measuring a stale, pre-Tier-1 pipeline) also
reconfirmed the rest of the module picture on the current, correct
pipeline: V1 remains the clearest positive contributor (+0.081 excess
Sharpe); V2 stays ~neutral (-0.003); V3 and V4 are now structural
no-ops (V3 already disabled, V4 never wired into the backtest); the
anomaly override never fired at all in this exact OOS window. On V5's
measurement, `modules.momentum_view` was set `false` -- the same
measured-negative -> disabled treatment V3 got in Tier 1, applied
consistently rather than as a special case. Re-running the frozen OOS
walk-forward with momentum off recovers (and modestly exceeds, mostly
via ordinary calendar drift between runs) Tier 3's pre-momentum
number: **Sharpe 0.9491, excess Sharpe 0.6784** -- `black_litterman
(frozen)` again clears every classical benchmark except `permanent`.
Momentum's implementation, tests, and full negative result -- both
the frozen-pipeline reading and the formal ablation number -- all stay
in the codebase, unedited: a signal that was implemented, honestly
tried, and rejected on measured evidence is exactly the kind of result
this project's methodology exists to produce.

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

**This ablation was re-run on the current pipeline for stage-11 Step 2
(ANALYSIS_V2.md), ranked primarily by excess Sharpe.** V1 is still the
clearest positive contributor (+0.081 excess Sharpe); V2 still ~zero
(-0.003); V3/V4 are now structural no-ops (V3 already disabled since
Tier 1, V4 never wired in); the anomaly override never fired in this
exact window; and V5 (momentum, added and measured for the first time
here) came back at **-0.236 excess Sharpe, the largest negative
marginal contribution of any module measured across this whole
study** -- see the Results section above for the full writeup and the
resulting `modules.momentum_view: false`.
