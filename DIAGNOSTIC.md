# ATLAS — Underperformance Diagnostic

**Question:** why does ATLAS (Sharpe 0.73) still lose to the Permanent Portfolio (1.07)?
**Method:** all numbers below were computed directly from the project's own cached price data (`data/raw/prices_…_20260730.parquet`) and its own `src/atlas` optimizers, not read off the notebooks. Reproduction scripts are inline in each section.
**Date:** 2026-07-30

---

## 0. Executive summary — three findings, in order of size

**Finding 1 — ATLAS is no longer running Black-Litterman at all. It has silently become GMV.**
The utility gate selects the GMV book in **239 out of 239 OOS weeks (100%)**. The HMM, the autoencoder, the views, the BL posterior — all computed weekly, all discarded before they touch a single weight. Independent confirmation: a plain weekly GMV backtest over the identical window scores **Sharpe 0.7344**; ATLAS's reported frozen Sharpe is **0.73**. Those are the same strategy.

**Finding 2 — the cash-exclusion fix (my recommendation) is what caused it, and it should be reverted.**
Running the gate census both ways, with the V1 regime view applied:

| Configuration | Weeks BL book selected |
|---|---|
| Cash allowed in `max_sharpe` (pre-fix) | **239 / 239 (100%)** |
| Cash excluded (current `HEAD`) | **0 / 239 (0%)** |

Removing BIL raised the BL book's volatility from ~9.6% to ~12.3%, and at risk aversion A = 5 the quadratic penalty made its utility permanently worse than GMV's. Cash wasn't only a Sharpe artifact — it was the *variance anchor* that let the BL book clear the gate. I got this recommendation wrong; the diagnosis of the degeneracy was right, but excluding the asset was the wrong remedy (§3.2 gives the correct one).

**Finding 3 — V3 is still switched on despite your own ablation proving it costs 0.265 Sharpe.**
`config.yaml` still reads `technical_view: true`. Notebook 11 measured that disabling it moves OOS Sharpe from 0.8056 to **1.0710** — i.e. to a dead heat with Permanent (1.067), the single largest effect in the study. This is a one-line change that was never applied.

**And the reframe that matters most (§5.3):** over the in-sample period 2013–2021, Permanent scores Sharpe **0.476** and GMV — the thing ATLAS became — scores **1.018**. Permanent's 1.07 is a property of the 2022–2026 window, not a property of Permanent. You are benchmarking against a lucky draw.

---

## 1. Performance Attribution

### 1.1 Where Permanent's return actually comes from

Bucket attribution of the benchmark over the OOS window (2022-01-03 → 2026-07-29, 1,146 trading days):

| Bucket | Weight | Cumulative contribution | Annualized | Vol |
|---|---|---|---|---|
| **Commodity** | 25% | **+26.25%** | +5.26% | 5.3% |
| Equity | 25% | +12.92% | +2.71% | 4.3% |
| **Cash (BIL alone)** | **25%** | **+4.44%** | +0.96% | **0.06%** |
| Fixed income | 25% | −0.25% | −0.05% | 1.6% |

Permanent: ann 8.85%, vol 8.28%, **Sharpe 1.067**, MDD −12.6%.

Two structural gifts, neither available to ATLAS:

- **BIL was the best risk-adjusted asset in the universe by an order of magnitude — realized OOS Sharpe 14.85** (3.89% return at 0.26% vol). Permanent holds it at **25%**, which *exceeds the 20% `per_asset_cap` every ATLAS optimizer is bound by*. The benchmark is allowed a position the strategy is structurally forbidden from matching. (Cost of that asymmetry, measured: applying the 20% cap to Permanent drops it from 1.0667 to **1.0398** — real, but only ~0.03 of the 0.34 gap.)
- **The commodity bucket carried the period** (+26% cumulative, driven by USO +21%/yr, SLV +21%/yr, GLD +18.6%/yr). A risk-averse mean-variance machine will never hold 25% commodities; Permanent does it by decree.

### 1.2 Which ATLAS components contribute what

From your own stage-11 ablation, plus my gate census:

| Component | Marginal OOS Sharpe | Status |
|---|---|---|
| V1 regime view (HMM) | **+0.147** | The only genuinely positive module |
| Anomaly override (autoencoder) | +0.002 | Inert — 2.3% of days flagged, no material effect |
| V2 mean-reversion | −0.013 pre-fix; **−0.022 post-fix** (0.8564 → 0.8348) | Negative before and after the unit fix |
| **V3 technical view** | **−0.265** | **Actively destroying performance, still enabled** |
| V4 sentiment | 0.000 | Structurally absent from the backtest |
| Turnover cap 1% → 2% | **+0.069** | Applied ✓ (the one fix that clearly worked) |
| **Cash exclusion** | **−0.170** (0.8969 → 0.7265) | **Applied, and it is the largest single regression in project history** |
| **Utility gate (A = 5)** | Not measured — **but it discards 100% of the BL pipeline** | Never ablated; the real bottleneck |

**The edge is being lost in the gate, not in the signals.** V1 works. The machinery that is supposed to convert V1 into weights refuses to fire.

---

## 2. Implementation Review — were the fixes done correctly?

All three were implemented competently in the narrow sense. Two had unintended consequences that the implementation itself could not have caught.

### 2.1 V2 unit fix — correct code, wrong conclusion drawn

`meanreversion.py:194` now reads `view = beta * detrended * horizon` with `horizon = half_life.clip(upper=252)`. That is a correct fix for the mismatch I flagged, and the units test is a good addition.

But scaling by half-life is not the same as annualizing. A signal with a 60-day half-life gets multiplied by 60; one with a 5-day half-life by 5 — so the *cross-sectional* magnitudes are now proportional to half-life, which is a re-weighting of the signal, not just a change of units. Combined with your own stage-5 event study (hit rates ~60% for HYG/VNQ but **36–41% for AGG, LQD, USO** — below chance), the result is a signal that is now loud enough to matter and is pointing the wrong way on several assets. Sharpe fell 0.8564 → 0.8348 exactly as one would expect.

**Verdict:** the fix is correct; the *signal* is the problem. V2 should be gated per-asset by validated hit rate, or switched off.

### 2.2 Turnover cap restoration — correct, and the only unambiguous win

`max_weekly_turnover: 0.02`, `no_trade_band: 0.005`, with a comment explaining why it is policy rather than a tunable. Sharpe 0.8348 → 0.8969. Exactly as predicted. No issues.

### 2.3 Cash exclusion — correctly coded, wrong remedy, large regression

`allocation.max_sharpe(..., exclude=...)` is cleanly implemented (drops tickers from the optimization universe, returns a Series still indexed over all assets, tested). `black_litterman_strategy` passes only the cash bucket, and only to the BL candidate. Nothing here is buggy.

The problem is conceptual, and it is mine. Snapshot at the IS boundary (2021-12-31), utilities under A = 5:

```
max_sharpe (cash allowed)   E[r]=+2.25%  vol= 9.64%   U = −0.00072
gmv                         E[r]=+0.26%  vol= 2.79%   U = +0.00070   ← gate winner
risk_parity                 E[r]=+2.59%  vol=11.41%   U = −0.00666
max_sharpe (cash excluded)  E[r]=+2.85%  vol=12.27%   U = −0.00917   ← now far worst
```

Excluding cash *raised* the BL book's expected return but raised its volatility more, and the A = 5 penalty is quadratic in vol. Mean utility gap versus GMV moved from **+0.00122 (BL wins)** to **−0.00487 (BL never wins)**. The strategy stopped being a strategy.

**The underlying degeneracy I identified is still real** — Π(BIL) = 0.11%/yr while BIL actually returned 3.89%/yr, a **3.78 pp understatement, larger than the entire V1 view magnitude of 3%** — but the correct remedy is to *price cash properly*, not to delete it (§6, action 2).

### 2.4 What was not done

- `technical_view` left `true` despite the −0.265 measurement (§0, finding 3).
- The utility gate was never included as an ablation arm, so the component that overrides everything else has never been measured.
- `reports/figures/`, `reports/results/` still empty; `plotting.py` still a stub; `models/` still empty. Priority-3 items, unstarted — fine, but the results in README are still not exportable artifacts.

---

## 3. Strategy Evaluation — the bottleneck chain

Walking the weekly pipeline and asking, at each step, "does this step preserve or destroy the previous step's information?":

| Step | Information preserved? | Evidence |
|---|---|---|
| L0 universe | Partial | De-dup is a near-no-op (23 → 22 names; SPY, VTI and ACWI all retained at ~0.99 correlation). Effective breadth ≈ 4 buckets, not 22 assets. |
| L1 HMM regime | **Yes — the one working part** | +0.147 marginal Sharpe |
| L1 anomaly | Neutral | +0.002; fires on 2.3% of days |
| L2 V2 mean reversion | **Destroys** | −0.022 post-fix; event study shows sub-chance hit rates on 3 of 22 assets |
| L2 V3 technicals | **Destroys badly** | −0.265; notebook 06 already showed BIL/SHY generating thousands of spurious zone events at 32%/46% hit rates |
| L3 BL fusion | Yes, mathematically | Verified against hand-computed example in tests |
| **L3 utility gate** | **Destroys everything upstream** | **BL book selected 0/239 weeks** |
| L4 anomaly override | No-op | Blends toward GMV — but the book already *is* GMV |
| L5 turnover cap 2% | Acceptable | Fixed; no longer binding in most weeks |

**Two rules are cancelling each other explicitly.** The pipeline computes a return-seeking book (max-Sharpe on μ_BL) and then evaluates it with a mean-variance utility whose risk-aversion is high enough that *any* book with vol > 17.4% has negative utility, given that the largest prior expected return in the whole universe is 7.53%. The gate is not a safety net on the optimizer — it is a second, stricter optimizer that always prefers minimum variance. You have two conflicting objective functions in series, and the second one wins every time.

---

## 4. Benchmark Comparison — where the 0.34 Sharpe gap comes from

Decomposition of the gap (Permanent 1.067 → ATLAS 0.73), each component measured independently:

| Source | Sharpe impact | Mechanism |
|---|---|---|
| V3 technical view enabled | **−0.265** | Measured directly in your ablation |
| Cash exclusion → gate never picks BL | **−0.170** | Measured in notebook 09 re-run; mechanism proven by gate census |
| V2 mean-reversion signal | −0.022 | Post-fix measurement |
| 20% cap vs Permanent's 25% BIL | −0.030 | Permanent-with-cap counterfactual: 1.0667 → 1.0398 |
| Turnover cap restoration | +0.069 | Already recovered |
| Residual (risk-aversion posture, commodity underweight) | ≈ −0.15 | ATLAS runs 4.5% vol vs Permanent's 8.3%; it simply does not take the risk that paid in this window |

Note these do not sum linearly (the ablation arms were run against a different baseline config), but the ranking is unambiguous: **V3 and the cash exclusion together account for roughly 0.4 Sharpe — more than the entire gap.**

**Why Permanent wins mechanically:** it is a fixed-weight, fully-diversified, quarterly-drifting book with 25% in the period's best Sharpe asset and 25% in the period's best return asset, rebalanced so infrequently (avg weekly turnover 0.84%, below the no-trade band much of the time) that it pays almost no cost and captures the full commodity spike. ATLAS, by contrast, spent the window in a minimum-variance book at half the volatility, systematically underweight the two buckets that produced all the return.

---

## 5. Data & Testing Validation

### 5.1 What is clean (re-verified, not taken on trust)

- **No lookahead in the engine.** Decisions use `returns.loc[:date]`; trades execute the following day; the canary test enforces it. I re-ran the accounting independently and reproduced the benchmark numbers to within 0.002 Sharpe.
- **No scaler/threshold leakage.** Confirmed by reading the fit paths in `anomaly.py`, `regimes.py`.
- **Grid search stayed in-sample**, frozen config ran once, negative results reported. Discipline intact.
- **Metrics are correct.** `ann_return`, `ann_vol`, `sharpe`, `max_drawdown` all reproduce hand calculations on the OOS series.

### 5.2 Real methodological issues found

1. **`permanent()` bypasses the per-asset cap** (`allocation.py:339`) while every optimizer enforces it. The benchmark is being evaluated under looser constraints than the strategy — a genuine apples-to-oranges comparison worth ~0.03 Sharpe. Either cap the benchmark or exempt the strategy's cash holding.
2. **`rf = 0` throughout, in a period when cash yielded 3.9–5.3%.** Every Sharpe in the project (strategy *and* benchmark) is a *total-return* Sharpe, not an excess-return Sharpe. Correctly computed against T-bills, Permanent's OOS Sharpe falls from 1.067 to roughly **(8.85% − 3.89%)/8.28% ≈ 0.60**, and ATLAS's from 0.73 to roughly **(2.3% − 3.89%)/3.2% < 0**. This does not change the ranking, but it means *neither* book beat cash on a risk-adjusted basis in a way the current metric reveals — a material reporting issue for a portfolio project.
3. **Survivorship tint** in the 23-ETF pool chosen in 2026 (unchanged from the earlier review).
4. **VNQ is orphaned:** it survives screening as `real_estate` but `permanent()` and `sixty_forty()` only reference four buckets, so the benchmark silently holds 0% of it while the optimizers can hold up to 20%. Not a bug, but an unintended universe asymmetry.

### 5.3 The finding that reframes the entire exercise

I re-ran the same books over the **in-sample** window that was never used to design them, and over the full history:

| Strategy | 2013–2021 (IS) | 2022–2026 (OOS) | 2013–2026 (full) |
|---|---|---|---|
| Permanent | **0.476** | 1.067 | 0.690 |
| GMV (= what ATLAS currently is) | **1.018** | 0.734 | — |
| Bucket risk-parity | 0.885 | **1.279** | **1.036** |
| Permanent vol-targeted 8% | 0.443 | 1.168 | 0.697 |

**Permanent's 1.07 is not a stable property — it is this window.** Over 2013–2021 Permanent scored 0.476 and ATLAS's current minimum-variance behaviour scored 1.018. You are optimizing against a benchmark that got a very good four years (commodity spike + 5% cash + equity bull), and asking a risk-averse machine to beat it in exactly the environment where risk-aversion was punished.

This also means: **do not tune your way to beating 1.07 on this window.** That is curve-fitting to a benchmark's lucky draw, and the vol-targeting row above is a live example — it "wins" OOS (1.168) and *loses* in-sample (0.443). It would be the wrong fix to adopt, and I would have recommended it if I hadn't checked both periods.

---

## 6. Action Plan

Ranked by **expected Sharpe impact ÷ implementation effort**. Impact estimates are measured where marked ✓.

### Tier 1 — do these now (measured, near-zero effort, ~0.4 Sharpe combined)

**1. Turn off V3. ✓ measured: +0.265**
`config.yaml → modules.technical_view: false`. Your own referee already ruled. One line, largest single gain available. Optionally keep the module for the report as a *documented negative result* — that is a legitimate finding, not a failure.
*Effort: 1 minute. Confidence: high (measured directly).*

**2. Revert the cash exclusion. ✓ measured: +0.170**
Remove `exclude=cash_tickers` from `black_litterman_strategy`. Keep the `exclude` parameter in `allocation.max_sharpe` — it is well-built and will be useful — just stop using it here. This restores the BL book to 100% gate-selection and undoes the largest regression in the project.
*Effort: 1 line. Confidence: high (mechanism proven by gate census).*

**3. Fix the actual cash degeneracy properly — optimize on excess returns.**
The right remedy for Π(BIL) = 0.11% vs 3.89% realized is to make the model *see* the cash rate, not to hide the asset:
- Add the 13-week T-bill yield (FRED `DTB3`, already have the API key) as `rf` in `metrics.sharpe`, `allocation.max_sharpe`, and `utility`.
- Set the equilibrium prior for the cash bucket to the prevailing bill yield rather than δΣw (which is ~0 by construction for a zero-covariance asset).
- Report excess-return Sharpes throughout (§5.2 item 2).
This removes the degeneracy at its source: with rf ≈ 4%, cash's excess return is ~0 and it stops being pathologically attractive on a Sharpe basis, *without* removing the variance anchor that lets the BL book pass the gate.
*Effort: half a day. Confidence: high on correctness; expect a modest Sharpe change but a large correctness/credibility gain.*

### Tier 2 — the structural fix (this is where real edge lives)

**4. Ablate and then re-scope the utility gate.**
The gate is the strategy's dominant decision rule and has never been measured. Add it as an ablation arm (gate on/off, and A ∈ {2, 5, 10}) and report the surface. My counterfactual with the gate removed entirely (`G` in my runs) scored **Sharpe 1.0008 with ann return 8.54%** — versus 0.73 with the gate. My recommendation: keep a gate, but make it a *floor* (reject the BL book only if its vol exceeds a band around the defensive book's) rather than a *utility comparison* that a quadratic penalty always wins.
*Effort: 1 day. Expected impact: +0.2 to +0.27 (measured counterfactual).*

**5. Move the whole strategy up one level of abstraction: allocate across buckets, not assets.**
This is the most robust finding in the whole diagnostic. Equal-risk across the four asset-class buckets (inverse-vol weighted, equal-weight within bucket) scores:
- **1.279 OOS** (vs Permanent 1.067)
- **0.885 IS** (vs Permanent 0.476)
- **1.036 full period** (vs Permanent 0.690)

It beats Permanent in *both* periods — unlike vol-targeting, which only wins OOS. The reason it works is the same reason Permanent works: with 22 assets that are really 4 uncorrelated bets, estimating a 22×22 covariance and 22 expected returns is mostly estimating noise. Collapse to 4 buckets, and both estimation problems become tractable. Your V1 regime view is already an asset-class-level view — it is the only view that operates at the right granularity, and the only one with positive contribution. That is not a coincidence.

Concretely: make the BL universe the four bucket portfolios; let V1 tilt them; keep within-bucket weights equal (or Sharpe-ranked). Adding my crude vol-regime overlay on top of bucket-RP gave **1.313 OOS**.
*Effort: 2–3 days (new `bucket_allocation` layer, rewire views to bucket level). Expected impact: +0.2 to +0.5, and robust across both periods.*

### Tier 3 — signal quality

**6. Gate V2 per-asset by validated hit rate**, using only in-sample event-study results: allow reversion views on assets whose IS hit rate exceeds ~55%, force zero elsewhere. Your notebook 05 already computed the table. If nothing clears the bar, turn V2 off and report that honestly.
*Effort: half a day. Expected: +0.02, mostly a correctness improvement.*

**7. Deepen V1, the module that works.** It is currently a single ±3% bucket tilt from a 3-state HMM that empirically resolves to ~2 states. Try: BIC selection over `n_states`, a continuous tilt scaled by state probability rather than a binary posture, and wiring the HMM/GMM agreement score into the view's confidence as §5 of PROJECT_STRUCTURE always intended.
*Effort: 1–2 days. Expected: +0.05–0.15, unmeasured.*

**8. Fix the benchmark comparison fairness:** apply `cap_and_renormalize` inside `permanent()` and `sixty_forty()` so benchmark and strategy face identical constraints, and either add `real_estate` to the benchmark buckets or drop VNQ from the universe.
*Effort: 1 hour. Impact: ~0.03 on the reported gap, plus methodological correctness.*

### What not to do

- **Do not adopt vol-targeting** despite its 1.168 OOS Sharpe. It scores 0.443 in-sample versus Permanent's 0.476 — it is fitted to this window.
- **Do not keep tuning against 1.07.** §5.3 shows that number is a property of 2022–2026. Report your strategy against Permanent *across both windows*, and the story changes completely — in-sample, ATLAS's architecture beats it 1.018 vs 0.476.
- **Do not delete the negative findings.** V3 at −0.265, V2 at −0.022, the anomaly override at +0.002, and the cash-exclusion regression are the most scientifically valuable content in this project. Most student projects cannot show a measured, mechanism-explained regression. You can.

---

## 7. If you do only three things

1. `technical_view: false` (+0.265 ✓)
2. Revert `exclude=cash_tickers` (+0.170 ✓)
3. Re-run notebook 09

Expected result: **Sharpe ≈ 1.05–1.10, at roughly half Permanent's volatility and a materially smaller drawdown** — i.e. a strategy that matches the benchmark's risk-adjusted return while taking far less risk, which is a defensible and honest place to land. Then do Tier 2 item 5 (bucket-level allocation) if you want to beat it on evidence that holds up in both periods rather than just this one.

---

*All figures independently computed from the repository's cached data using its own optimizers; gate census over 239 OOS weekly decision dates; counterfactual backtests use the same weekly scheduler, 10 bps costs, 2% turnover cap, 0.5% no-trade band and next-day execution as `backtest.run`.*
