# ATLAS — Comprehensive Project Review

**Reviewer perspective:** senior quantitative developer / software architect
**Scope reviewed:** all 16 `src/atlas` modules (~2,900 LOC), 148 tests (~2,400 LOC), notebooks 01–10, config, git history (stages 1–10), `PROJECT_STRUCTURE.md` conformance, and the algorithmic trading course material (S1–S14).
**Date:** 2026-07-28

---

## 1. Project Architecture & Design

### 1.1 What the architecture gets right

The layered design from `PROJECT_STRUCTURE.md` survived contact with implementation remarkably intact. The L0–L5 separation is real, not aspirational: `universe.py` (L0), `regimes.py`/`anomaly.py` (L1), `meanreversion.py`/`technicals.py`/`sentiment.py` (L2), `views.py`/`allocation.py` (L3), `risk.py` (L4), `strategy.py`/`backtest.py` (L5). Dependencies flow one way; no signal module imports the backtester; the `strategy_fn(as_of, window) -> weights` contract is honored by every stage and makes the whole system composable. The wrapper pattern (`with_anomaly_override`, `with_meanreversion_tilt`, …) let each build stage add exactly one concern without touching prior stages — visible in a clean 10-commit history that matches the build order one-to-one.

Notebooks are thin narrative drivers importing from `src/atlas`, exactly as the spec demanded. All logic lives in the package. Config-driven design is ~90% real (see 1.3 for the gaps).

### 1.2 Architectural deviations from PROJECT_STRUCTURE.md

| Spec promise | Reality | Severity |
|---|---|---|
| `plotting.py`: "every figure in one place" | Docstring-only stub; every notebook hand-rolls its own matplotlib code, with the equity-curve/drawdown plot duplicated across notebooks 02–09 | Medium — exactly the duplication the module was designed to prevent |
| `models/` per-fold persistence of scalers/HMM/AE | Empty; refits are in-memory caches inside `strategy.py` closures | Medium — no reproducibility of which model made which decision |
| `reports/figures/`, `reports/results/` populated (`fig01_...`, `results_summary.json`) | Both empty; figures inline in notebooks, results only in notebook state and README prose | Medium — deliverables not exportable/diffable |
| Universe re-screened per walk-forward fold (`rescreen_frequency: quarterly`) | Screened exactly once at 2021-12-31 and frozen for the whole 2022–2026 OOS run | Low-Medium — point-in-time correct (no lookahead) but the config key is dead and the spec behavior absent |
| "Unrestricted universe — any asset, any market worldwide," FX conversion, calendar alignment across exchanges | 23 hard-coded US-listed USD ETFs; `to_base_currency` exists but is never exercised by real multi-currency data | Medium — the flagship claim of the project is materially overstated relative to what was built |
| GMM/HMM agreement score → BL confidence scaler (§5 step 1d) | Agreement computed and reported in notebook 04 (ARI 0.028), never wired into `views.assemble`; confidences are static config constants | Low-Medium |
| Ablation study (stage 11) | Not yet built — the one component the spec calls "the referee" for every other module | High (it's the missing keystone, though honestly tracked as pending in README) |

### 1.3 Config hygiene

Dead keys that promise behavior that doesn't exist: `universe.min_avg_dollar_volume` (no volume data is ever downloaded — notebook 01 admits this), `universe.correlation_dedup_threshold` (the de-dup uses K-Means cluster count instead, and with `n_asset_clusters: 8` vs. bucket sizes of ≤ 10, de-dup was a near-no-op: 23 candidates → 22 survivors, keeping SPY *and* VTI *and* ACWI, which correlate ~0.99), `constraints.tracking_error_budget` (never read), `rebalance.frequency` (weekly is hard-coded in `_week_end_dates`).

`modules.*` flags are enforced only in `views.py`. `modules.anomaly_override` and `modules.execution_timing` are not consulted anywhere — wrappers are applied (or not) manually in notebooks. Consequence: in the canonical frozen OOS run (notebook 09, cell D), **`phase_flags_fn` was not passed to `backtest.run`, so execution timing was silently absent from the final result even though its config flag says `true`.** The config no longer describes the experiment that produced the headline number. This is the kind of drift the flags were designed to prevent.

---

## 2. Trading Strategy Analysis

### 2.1 The strategy that was designed vs. the strategy that runs

On paper: a regime-aware, multi-view Black-Litterman strategy that tilts aggressively risk-on/risk-off. In practice, the pipeline's compounded conservatism produces a quasi-static, heavily defensive book (notebook 10: top holdings BIL 14.7%, AGG 8.5%, SHY 6.0%, BSV 5.8%). Four mechanisms compound:

1. **The turnover throttle is doing most of the talking.** The grid froze `max_weekly_turnover = 0.01` and `no_trade_band = 0.01`. Band = cap means every rebalance either trades exactly 1% (one-way) or nothing — the parameterization has collapsed to "creep or freeze." At 1%/week, a full posture switch (say GMV → Max-Sharpe, ~40–60% turnover) takes roughly a year to complete. The entire L1/L2 signal apparatus can shout weekly; the execution layer lets it whisper. This is the single biggest reason the strategy's OOS behavior barely distinguishes itself from a static defensive mix — and it was *selected by* the stability-penalized grid, which structurally rewards the configuration that trades least. That is circular: a stability criterion applied to turnover parameters will almost always choose maximal damping, regardless of signal quality.
2. **The equilibrium prior + rf = 0 + cash in the optimizer is a degenerate attraction to BIL.** `equilibrium_returns` gives Π = δΣw with the Permanent Portfolio as "the market." Cash has near-zero covariance with everything, so Π(BIL) ≈ 0 with vol ≈ 0.2% — at rf = 0, its implied Sharpe in the max-Sharpe objective is enormous, so the optimizer piles into it up to whatever the cap allows. The strategy's defensiveness is partly an artifact of the objective, not a view.
3. **The utility gate (A = 5) frequently overrides BL toward GMV/Risk-Parity.** With μ_BL barely different from a conservative prior (see 2.2 — most views are tiny), the max-Sharpe book rarely wins the utility comparison. The gate was designed as a safety net; combined with 1 and 2 it becomes the default outcome.
4. **The HMM in practice found ~2 states, not 3.** Notebook 03 reports two states mapping nearest to risk_on and none to neutral in the fitted window — so the "primary switch" has less resolution than designed, further reducing signal variance.

None of these is individually a bug — each is defensible — but their product is a strategy whose realized OOS annualized return is 4.4% at 5.3% vol while its naive benchmark (Permanent, 8.3% vol) earns a higher Sharpe (1.10 vs 0.86). **The system's sophistication is currently spent suppressing its own signals.**

### 2.2 A real unit bug: the V2 mean-reversion view is denominated in the wrong time scale

`meanreversion.reversion_signal` sets `view = beta * detrended_deviation`. Beta comes from a *daily* OLS of Δx on x, so this view is an expected *next-day* log-price drift — typically a few tens of basis points. `views._per_asset_views` then adds it to the *annualized* equilibrium prior: `Q = prior_annualized + view_daily`. A 40 bp expected daily reversion — an enormous signal — enters BL as a 0.4% *annual* tilt, i.e., noise next to `regime_view_magnitude = 3%`. **V2 is effectively inert in the fusion, not by calibration decision but by unit mismatch.** The technical view (bounded ±2%, implicitly annual) and regime view (3%, annual) are consistent with each other; V2 is off by roughly two orders of magnitude. This likely explains why stage 5's tilt "didn't clearly improve" results — it barely participated. Fix: scale the OU view to an annualized (or holding-period) equivalent, e.g. multiply the expected per-day drift by min(252, horizon implied by half-life), or express Q as expected return over the reversion horizon and set Ω accordingly.

### 2.3 Coherence and internal consistency — the good parts

The decision pipeline order (universe → regime → anomaly → views → BL → optimize → gate → band/cap → execute t+1) matches the spec exactly and is implemented without lookahead. The BL machinery itself is correct (verified against a hand-computed two-asset example in tests; no-views ⇒ posterior = prior; Ω via the standard (P τΣ Pᵀ)/confidence construction). The honest live-only quarantine of sentiment (V4) and options positioning — refusing to fake a historical backtest for data that has no history — is a genuinely professional call that most student projects get wrong in the other direction.

### 2.4 Unnecessary complexity / missing pieces

- Five stacked wrapper strategies (stages 3–7) remain in `strategy.py` after stage 8 superseded them. Kept deliberately as historical record — defensible, but `strategy.py` is now 434 lines where ~60% is superseded code paths that new readers must distinguish from the live one.
- `exit_z` is config-documented as unused; fine, but it signals the stateless-view design quietly abandoned part of the S10 mean-reversion design (entry/exit hysteresis) without an explicit decision record.
- The missing piece that matters: **stage 11 ablation**. Every "did this module earn its complexity?" question in notebooks 03–08 is answered anecdotally (one OOS number vs. the previous stage's, on overlapping but non-identical windows). The spec's own answer to this — flip each `modules.*` flag on the frozen config — has not been run.

---

## 3. Algorithmic Trading Methodology (course fidelity)

| Course topic | Applied? | Assessment |
|---|---|---|
| S1/S3 MPT, efficient frontier, max Sharpe, CML, utility | ✅ | Correct SLSQP formulations; utility function used as a live decision gate (a nice extension beyond the course's illustrative use). Frontier/CML present in notebook 02 as IS diagnostics only. |
| S2 SciPy optimization, tracking error | ✅ / ⚠️ | Optimizers correct with bounds/constraints per the S2 checklist. `tracking_error_min` implemented + tested but orphaned — no TE budget wired (dead config key). |
| S4 GMV, Risk Parity, HRP, Black-Litterman, Permanent | ✅ | All five implemented correctly; HRP is a faithful López-de-Prado reduction (corr→distance→single linkage→quasi-diag→recursive bisection) with a sensible cap extension. BL math verified in tests. Equilibrium-from-Permanent is a documented, defensible hack — but see §2.1(2) for its side effect. |
| S4 turnover constraint (2% weekly) | ⚠️ | Implemented, then grid-tuned down to 1% — see §2.1(1) for why letting a stability criterion choose the cap subverts its purpose as a risk control. |
| S5 backtester (rebalance blocks, weight drift, weights charts) | ✅ | Weekly adaptation is correct and cleaner than the course version: proper position lagging, costs on execution day, drift between rebalances, first-rebalance funding exemption. |
| S7/S8 regimes: K-Means macro + elbow + PCA + profile heatmap; HMM + transition matrix | ✅ / ⚠️ | All present and methodologically correct (stationary transforms of FRED series, train-only scaler, profile-based state mapping — the exact leakage disciplines the course taught). ⚠️ Macro K-Means never feeds the live decision (context only — spec intended it as a confidence input); HMM effectively resolved 2 states while configured for 3, unexamined. |
| S8 autoencoder anomaly detection | ✅ | Faithful to the course pattern: sequences, chronological split, scaler on train only, threshold from *training* errors at the 99th percentile, early stopping. LSTM variant instead of dense — reasonable. Honest finding recorded: ARI 0.028 vs HMM, override moved Sharpe by −0.002. |
| S8 Monte Carlo / GBM | ✅ / ⚠️ | Correct discretization (drift − σ²/2, seeded, vectorized). ⚠️ Feeding the strategy's *own realized OOS* μ/σ back in as GBM parameters makes the "80% P(positive) at 1y" claim partly circular — it assumes the OOS drift persists. Course did the same, but a review should note it. |
| S10 mean reversion: ADF, OU half-life, z-scores, vol filter, event study, IS/OOS, grid search, walk-forward | ✅ / ❌ | Individually excellent — the ADF-on-raw-log-price insight (rolling detrend induces spurious stationarity) is *better* than the course notebook. ❌ But the view's time-scale bug (§2.2) means the correctly computed signal was mis-fused; and "walk-forward" here is fold *reporting* over one continuous run, not per-fold refits (honestly documented, but weaker than S10's method). |
| S12 pivots, K-Means S/R zones, event study, walls, gamma | ✅ | Correct and honestly quarantined (options = live-only). Event study surfaced the BIL/SHY spurious-zone problem — good diagnostic, but the signal still runs on those assets in production; the finding wasn't acted on (e.g., excluding near-cash tickers from V3). |
| S14 scraping + NLP (robots.txt, tokenize/lemmatize, VADER, LDA, wordcloud) | ✅ | Complete and careful (robots checked at request time, fail-closed; POS-aware lemmatization with documented rationale). Correctly excluded from backtest. |
| S6 fund-selection stats + clustering for universe | ⚠️ | Stat ratios + factor correlations + K-Means present, but the de-dup was a near-no-op at these pool sizes (§1.3) and the liquidity screen was never implemented — the screening layer mostly passed everything through. |

**Overall:** course techniques are implemented with above-course-level rigor at the module level; the misapplications are all at the *integration* level (unit consistency across views, tuned-away turnover control, signals computed but not acted on).

---

## 4. Code Quality

### 4.1 Strengths (genuinely high standard)

- **Contracts and composition.** One strategy interface everywhere; allocation functions all return normalized, capped `pd.Series`. The wrapper/closure pattern is idiomatic and testable.
- **Documentation of *why*, not *what*.** Module docstrings record empirical findings that justify design choices (the BLAS single-threading discovery with measured 9× impact; the rolling-detrend/ADF false-positive experiment; the html.parser lowercased-`<pubDate>` trap). This is rare and valuable.
- **Test suite: 148 tests, aimed at the right things.** Known-answer tests for metrics; constraint invariants for every optimizer; a hand-computed BL example; an injected-anomaly detection test; backtest accounting identities (costs = bps × turnover); the lookahead canary; memoization single-call guarantees; walk-forward partition checks. This is the CLAUDE.md test list, actually executed.
- **Defensive engineering.** Graceful degradation for hmmlearn EM collapses, TF fit failures, adfuller crashes, dead RSS feeds — each with a documented rationale and a safe fallback, never a bare `except`.
- **Reproducibility basics:** seeds everywhere (numpy, sklearn, TF), date-stamped data caches keyed by request range, deterministic GBM.

### 4.2 Code smells and refactoring opportunities

1. **`strategy.py` wrapper duplication.** `with_meanreversion_tilt` and `with_technical_view` are the same 30-line function modulo a signal callable and a magnitude key; the has-history/try-except/reindex/rescale/cap sequence appears four times. Extract one `_with_tilt(strategy_fn, signal_fn, magnitude, cap)` and a `_safe_signal` helper. Similarly, the HMM posture-detection block is copy-pasted between `regime_switching_strategy` and `black_litterman_strategy` — extract `_current_posture(window, cfg, posture_cfg)`.
2. **Prices-from-returns reconstruction `(1 + window).cumprod()` appears in four call sites** with the same subtle correctness argument re-explained in comments each time. Make it `data.implied_prices(returns)` with the rationale documented once.
3. **`plotting.py` should exist.** The equity-curve/drawdown/weights-area chart trio is re-implemented in at least four notebooks with cosmetic drift. This is the spec's own module; building it would delete ~150 lines of notebook code and standardize figures for stage 11.
4. **Dead config keys** (§1.3) violate the project's "config describes the experiment" principle — implement or delete, and make `modules.*` flags actually gate the wrappers (a 5-line change in `strategy.py` construction sites, or a small factory that reads the flags).
5. **Magic numbers in guards:** `vol_window + 10 * n_states`, `seq_len + 30`, `lookback + 20` — fine heuristics, but they're uncommented thresholds in exactly the style the project elsewhere refuses to allow.
6. **Metrics/backtest turnover duplication:** `metrics.turnover` exists but `backtest.run` re-derives one-way turnover inline; one should call the other.
7. **Not a smell but a risk:** `strategy.py` imports TensorFlow at module load solely for one exception type in a guard — pushes a heavyweight import onto everything that touches `strategy`. Import it lazily inside `with_anomaly_override`.

Readability, naming, typing, and docstring coverage are otherwise excellent — this codebase reads like a professional's.

---

## 5. Strategy Robustness

### 5.1 Leakage audit — what is clean

Genuine credit here; the classic sins are absent:

- **No lookahead in the engine:** decisions use `returns.loc[:date]`; trades apply the next trading day; the canary test enforces sensitivity to a one-day shift.
- **No scaling leakage:** every scaler (HMM features implicitly, AE explicitly, macro K-Means) fits on training windows only.
- **No threshold leakage:** AE anomaly threshold from training-error percentiles, never validation/eval errors.
- **Grid search stayed in-sample** (2010–2021), selected on cross-fold stability with a sample-size floor, and the frozen config ran OOS once — with the negative outcome (0.856 < 0.895 untuned) reported rather than re-tuned. Textbook discipline, rarer than it should be.
- **Point-in-time universe:** screened with data ≤ 2021-12-31 only.
- **Live-only signals quarantined** rather than pseudo-backtested.

### 5.2 Where the reported results should still be discounted

1. **The OOS window was observed throughout development.** Stages 3–8 each ran and recorded OOS Sharpes (0.75 → 0.75 → …→ 0.895), and stage 8's architecture (BL + anomaly + utility gate) is "current best" partly *because* it looked best on those OOS reads. The stage-9 freeze protects two parameter families; it does not retroactively un-see six OOS evaluations of architectural variants. The final 0.86 is best understood as *development-set* performance, not virgin OOS. This is the project's largest epistemic weakness, and the README currently presents the number without this caveat.
2. **One OOS window, one market environment (2022–2026).** All fold slicing happens within it; the strategy's defensive profile was never tested through a 2008- or 2020-style regime *out of sample* (historical stress ≠ walk-forward: notebook 10 applies today's static book to 2008 returns, which is a sensitivity report, not strategy performance).
3. **No statistical significance framing.** With ~4.5 years of daily data, the standard error of a Sharpe estimate is roughly 0.5/√4.5 ≈ 0.22 — the frozen strategy (0.86), risk parity (0.84), HRP (0.82), and even Permanent (1.10) are within ~1 SE of each other. The README's ranking table invites reading noise as signal; per-fold dispersion exists (walk-forward summary) but no confidence statement is made anywhere.
4. **Survivorship tint in the candidate pool.** The 23 built-in ETFs were chosen in 2026 and all exist today; a 2010-vintage selector might have included since-delisted or since-decayed funds. Impact is modest for broad index ETFs, but the "no survivorship peeking" claim in `universe.py`'s docstring is stronger than the pool construction supports.
5. **The Monte Carlo forward-risk numbers are partially circular** (OOS-realized drift as the GBM μ — see §3), and cost realism (10 bps × ~0.5%/wk turnover ⇒ ~3 bp/yr drag) is trivially small because the turnover throttle makes it so; spreads/slippage on a real 22-ETF weekly program would be modest but not zero.
6. **Missing validation:** the ablation study (the project's designated arbiter of module worth); any nested/holdout validation of *architectural* choices (as opposed to parameters); per-fold model persistence that would let a specific historical decision be reproduced and audited.

**Bottom line on trust:** the *engine* deserves high trust (mechanics verified by tests and honest cross-checks — notebook 10 independently recomputed notebook 09's result within 0.01 Sharpe). The *headline number* deserves moderate trust as a development-set figure with wide error bars, not a deployable expectation.

---

## 6. Overall Assessment

### 6.1 What went well

- **Process discipline that most professional teams don't achieve:** strict stage-by-stage build with per-stage commits; IS-only tuning with a stability criterion; a frozen-config single OOS run; negative results (grid search didn't help; anomaly override moved nothing; ARI 0.028; naive tilts didn't beat their base) reported plainly in six separate notebooks. The "honest findings" culture is the project's single most valuable asset.
- **Leakage engineering:** train-only scalers, training-error thresholds, decision/execution lagging, the lookahead canary test, live-only quarantine for archive-less data. §5.1 is a short list of things done right that each kill most student backtests.
- **Software quality:** clean layering, one strategy contract, 148 well-aimed tests, defensive numerical guards with documented failure modes, reproducible seeds and caches, docstrings that record empirical rationale.
- **Course coverage with genuine depth:** every session S1–S14 is present and, in several places (ADF-on-raw-price rationale, HRP cap extension, BL hand-verified tests, robots.txt fail-closed scraping), executed *above* course level.
- **The BL fusion earned its keep on its own evidence:** 0.895 vs 0.738 for the naive chain — the one stage where added structure clearly improved results, correctly identified as such.

### 6.2 What went wrong

- **The strategy defeats itself.** Four compounding conservatisms (1%-and-equal band/cap throttle, cash-attracted equilibrium prior at rf = 0, utility gate defaulting to GMV, an effectively 2-state HMM) produce a quasi-static defensive book that underperforms the naive Permanent benchmark it was built to beat (0.86 vs 1.10 Sharpe, development-set). The signal layers work; the integration mutes them.
- **A concrete integration bug:** V2's daily-scale view fused against annualized priors (§2.2) — the mean-reversion module effectively never participated in BL. Undetected because view *shapes* are tested but view *units* are not.
- **The self-imposed referee never showed up:** stage 11's ablation — the mechanism the whole "complexity is welcome" stance depends on — remains unbuilt, so no module's marginal contribution is actually known.
- **Spec drift at the edges:** dead config keys, unenforced module flags (execution timing silently absent from the canonical run), empty `models/`/`reports/`, `plotting.py` unbuilt, "global unrestricted universe" reduced in practice to 22 US ETFs with a near-no-op de-dup and no liquidity screen.
- **Results framing overstates certainty:** README ranks strategies by Sharpe differences that are within estimation noise, without flagging that the OOS window informed development for six stages.

### 6.3 How to improve — prioritized

1. **Run the ablation study (stage 11) before touching anything else.** It is the project's own designated arbiter, all wiring exists (`modules.*` flags in `views.py`; wrappers composable), and every subsequent decision (what to fix, drop, or deepen) should be made on its output, not on stage-by-stage anecdotes. Include a "turnover cap = 2%/4%" arm — treating the throttle as an ablatable module — to measure how much performance the execution layer is suppressing.
2. **Fix the V2 unit mismatch, IS-validate, then re-freeze.** Annualize or horizon-scale the OU view before it enters Q (e.g., expected reversion over min(half-life, 21) days, annualized), sanity-check magnitudes against `regime_view_magnitude` on IS data only, and re-run the frozen OOS once. Add a *units* test: every view family's |Q − prior| must land within an order of magnitude of the others on synthetic data.
3. **Stop letting the grid choose the risk control.** Set the turnover cap back to the S4-convention 2% (a policy, per PROJECT_STRUCTURE) and keep the no-trade band strictly below it (0.25–0.5%). A stability-penalized objective will always buy stability with paralysis; caps are constraints, not free parameters. Alternatively, make the grid objective cost- and lag-aware (penalize tracking distance to target weights), so damping has a price.
4. **Remove the cash degeneracy from the optimizer.** Either exclude the `cash` bucket from the max-Sharpe universe (hold it as the residual/defensive asset only), or optimize on excess returns over BIL. Re-derive the equilibrium prior accordingly. This single change will likely un-mute the risk-on side of the strategy more than any signal improvement.
5. **Reframe the results honestly in README:** label 2022–2026 as development-period OOS (observed during stages 3–8); add ±1 SE bands or per-fold dispersion next to every Sharpe; state that strategy-vs-benchmark differences are not statistically significant at this sample size. Credibility is this project's differentiator — spend it carefully.
6. **Establish a truly virgin test:** start the live/paper loop now (sentiment and options positioning finally get to participate) and designate everything after today as untouchable evaluation data. That is the only fully clean OOS this project can still get.
7. **Close the spec-drift list:** implement `plotting.py` and export `reports/figures` + `results_summary.json` (needed for stage 11 anyway); make `modules.*` flags gate wrapper application; implement or delete dead config keys; persist per-fold models to `models/`; either widen the candidate pool (with a real dollar-volume screen, correlation-threshold de-dup, FX conversion exercised) or amend PROJECT_STRUCTURE to describe the US-ETF scope actually built.
8. **Refactor opportunistically, not urgently:** extract the shared tilt-wrapper and posture-detection helpers; move superseded stage 3–7 strategies to a `legacy` module (kept, tested, out of the live file); lazy-import TensorFlow; unify turnover computation. None of this blocks correctness; all of it lowers the cost of stage 11 and beyond.
9. **Regime layer follow-ups (after ablation says it's worth it):** investigate the 2-effective-state HMM (BIC across n_states; or accept 2 states and simplify the posture map); wire the GMM/HMM agreement score into view confidence as §5 intended; consider decode-only weeks between scheduled refits to reduce label churn.

---

*Method note: this review is based on full reads of every `src/atlas` module and test file, the executed notebooks' markdown/code/outputs, config files, git history, and cross-reference against `PROJECT_STRUCTURE.md` and the course materials in the parent folder. Tests were inspected, not re-executed (the review environment lacks the project's TF toolchain); test names and assertions were verified against the modules they cover.*
