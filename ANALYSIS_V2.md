# ATLAS — Deep Analysis After the Tier 1–3 Fixes

**Scope:** end-to-end re-reading of the post-fix codebase (3,838 LOC across 17 modules), config, notebooks, and an independent re-derivation of the results from the repo's own cached data.
**Date:** 2026-08-04

---

## 1. How the system now works, end to end

Each Friday close, `strategy.black_litterman_strategy` executes:

```
window (returns ≤ Friday)          rf = DTB3.asof(Friday)          [Tier 1 fix]
  │
  ├─ regimes.market_regime ──► HMM(3 states, 5y window) ──► posture by PROFILE
  │                                                          (risk_on/neutral/risk_off)
  ├─ cov = LedoitWolf(756d trailing) ── annualized
  ├─ prior Π = δΣw_permanent, with the cash bucket's Π overridden to rf   [Tier 1 fix]
  │
  ├─ views: V1 regime (bucket-level, ±3%)          ← the only positive contributor
  │         V2 mean-reversion (8 hit-rate-eligible tickers only)  [Tier 3 fix]
  │         V3 technical — DISABLED                 [Tier 1 fix, measured −0.265]
  │         V4 sentiment — structurally absent from backtest
  │
  ├─ μ_BL = BL(Π, Σ, P, Q, Ω, τ=0.05)
  ├─ candidates: max_sharpe(μ_BL, Σ, rf) │ gmv(Σ) │ risk_parity(Σ)
  ├─ utility gate: argmax (E[r]−rf) − ½·A·σ²,  A = 2               [Tier 2 fix, was 5]
  ├─ anomaly override: if AE reconstruction error > 99th pct → blend to GMV
  └─ backtest: no-trade band 0.5% → turnover cap 2% → execute Monday, 10bps
```

The architecture is now coherent and the fixes did what they were supposed to do. The gate no longer strangles the pipeline (A=2 lets the max-Sharpe book win most weeks instead of GMV winning 239/239), cash is priced rather than deleted, and the two measured-negative modules are off or restricted.

**The result, from your README: Sharpe 0.90, excess Sharpe 0.63, ann vol 14.7%, ann return ≈ 13.2%.** Against Permanent's 1.06 / 0.61 / 8.8% / 9.1%.

So on the question you actually asked — returns — **you already win: 13.2% vs 9.1% per year.** Section 2 is about whether that means what it looks like.

---

## 2. The headline finding: the strategy currently has ~zero alpha

Sharpe is leverage-invariant, so the honest test of whether ATLAS adds skill is: *lever the benchmark to ATLAS's volatility and compare returns.* Computed from the repo's own OOS data (2022-01-03 → 2026-07-29, BIL as the financing rate):

| | Ann. return | Ann. vol | Excess return | Excess Sharpe |
|---|---|---|---|---|
| Risk-free (BIL) | 3.90% | — | — | — |
| Permanent | 9.13% | 8.26% | 5.23% | **0.634** |
| **Permanent levered 1.78× to 14.7% vol** | **13.21%** | 14.70% | 9.31% | 0.634 |
| **ATLAS (BL frozen)** | **13.23%** | 14.70% | 9.33% | **0.630** |

**ATLAS's alpha over a risk-matched Permanent Portfolio is +0.02% per year.** The excess Sharpes differ by 0.004 — indistinguishable from zero at any sample size, let alone 4.5 years.

This is not a criticism of the fixes; they worked. It is a precise statement of where the project stands: **every incremental return ATLAS earns over Permanent is currently explained by the extra risk it takes, not by the HMM, the autoencoder, Black-Litterman, or any signal.** You have built a machine that reproduces `1.78 × Permanent`.

That reframes the goal. "Beat Permanent's return" is already done and is trivially achievable by any risk-scaling. The meaningful target is **excess Sharpe > 0.634** — earning more per unit of risk. Nothing in the current pipeline does that yet.

---

## 3. A correction I owe you

My DIAGNOSTIC.md Tier 2 item 5 recommended the bucket-level layer on the strength of "1.279 OOS / 0.885 IS / 1.036 full-period — the only change that beat Permanent in both windows." You implemented it faithfully, and it scored 0.81. Two reasons, both mine:

**(a) I measured the wrong thing.** My counterfactual was *plain inverse-volatility weighting across four buckets* — no Black-Litterman, no views, no utility gate, no max-Sharpe. You implemented BL-at-bucket-level, which is a different strategy that happens to operate on bucket returns. The thing that scored 1.279 was the *absence* of the optimizer, not the change of granularity. My write-up conflated them.

**(b) The metric I used was the one I had just criticized.** I ranked on total-return Sharpe while §5.2 of the same document argued that in a 4%-cash environment total-return Sharpe flatters low-volatility books. Re-running my own counterfactual on excess return:

| | Ann. return | Vol | Total Sharpe | **Excess Sharpe** |
|---|---|---|---|---|
| Permanent | 9.19% | 8.27% | 1.105 | **0.643** |
| Bucket inverse-vol (my "winner") | 4.73% | 3.05% | **1.533** | **0.278** |

It looks spectacular on the metric I told you not to trust and is less than half as good on the one I told you to use. It earns 4.7% against a 3.9% cash rate — it barely beats T-bills. **Discard that recommendation.** The bucket layer is still worth keeping in the repo as a comparison arm (it is well-built, and `buckets.py` is clean), but it is not the path forward, and `modules.bucket_level: false` is the correct setting.

---

## 4. Errors, incoherences, and loose ends found in the current code

Nothing here is fatal; several are the kind of drift that quietly makes the config lie about the experiment.

**4.1 `execution_timing: true` still does nothing.** `technical_phase_flags` and `phase_flags_fn` appear **zero times** in notebook 09. The canonical OOS run has never once used execution timing, across three review cycles, while the config has claimed it was on the whole time. Either pass `phase_flags_fn=strategy.technical_phase_flags(cfg)` into `backtest.run`, or set the flag false. (Note V3 is now disabled anyway, and the phasing logic keys off resistance zones from the same module — so with `technical_view: false`, execution timing is arguably incoherent by construction and should be off.)

**4.2 `sentiment_view: true` while V4 is structurally absent** from `black_litterman_strategy`'s `view_sets`. The ablation notebook documents this honestly, but the flag still reads as if the module were live. Set it false with a comment, or gate it behind a `live_mode` flag.

**4.3 BIC selection is a diagnostic, not a selection.** `regimes.hmm_bic_curve` exists and is correct, but `config.regimes.hmm.n_states` is still the hard-coded `3` and nothing reads the curve's argmin. If BIC's answer *was* 3, say so in the config comment with the BIC values; if it wasn't, the config is now inconsistent with your README's claim that `n_states` is "selected by BIC."

**4.4 Stale comment in `regime_posture.yaml`:** still says *"Stage 8 will swap this for the true BL posterior (max_sharpe_bl); until then this is a naive proxy…"* — stage 8 shipped six commits ago and these posture methods are now only used by the superseded `regime_switching_strategy`. Misleading to a reader.

**4.5 The `per_class_cap: 0.60` is nearly non-binding at bucket level.** With four buckets summing to 1, a 60% cap only ever binds in extreme corner solutions. Fine, but it isn't the risk control the comment implies.

**4.6 Superseded strategy wrappers still live in `strategy.py` (656 lines).** Stages 3–7's `with_*` wrappers are retained as historical record — a defensible choice I supported — but the file is now large enough that the live path is hard to find. Move them to `strategy_legacy.py`, keep the tests.

**4.7 `plotting.py` is still a 10-line stub**, `reports/figures/` and `reports/results/` are still empty, `models/` is still empty. Three review cycles on, the project still has no exportable artifacts. This matters for the final deliverable.

**4.8 Leakage re-audit: clean.** I re-checked the new code paths specifically — `rf_series.asof(as_of)` is point-in-time correct, the V2 hit-rate eligibility list is IS-derived and frozen, and `hit_rate_eligible_tickers` is computed on data ≤ `is_end_date`. No new leakage introduced by the fixes. The engine remains trustworthy.

---

## 5. Where the returns can actually come from — in light of the class material

Your ablation has now measured every signal you built: V1 +0.147, V2 ≈ 0, V3 −0.265, V4 absent, anomaly +0.002. That is a thorough negative result, and it points at a specific gap.

### 5.1 The missing factor: momentum (S4 — Factor Investing / Smart Beta)

Session 4 covered Fama-French factor extensions and smart beta explicitly; Session 10 covered mean reversion. **You implemented mean reversion and technicals — both of which measured negative — and never implemented momentum, which is the single most robust cross-asset factor in the literature and the natural complement to a Permanent Portfolio.** Your own event studies were telling you this: notebook 05 found extreme z-scores *continued* rather than reverted on AGG, LQD, USO (hit rates 36–41%, i.e. reliably wrong in the mean-reversion direction — which is a momentum signal wearing a disguise).

I tested it on your data, using your backtester's exact conventions (weekly, 10 bps, 2% cap, 0.5% band, next-day execution), reporting excess Sharpe, in **both** windows:

**Time-series (absolute) momentum — bucket in downtrend vs cash → move to cash:**

| Variant | OOS exSharpe | OOS MDD | IS exSharpe | IS MDD |
|---|---|---|---|---|
| Permanent | 0.643 | −12.6% | 0.407 | −18.5% |
| + trend 252d | **0.709** | **−7.4%** | 0.360 | −16.1% |
| + trend 126d | **0.727** | **−6.0%** | 0.400 | −16.2% |

**Cross-sectional momentum — hold top half by 12-1 return within each bucket:**

| | OOS ann ret | OOS exSharpe | IS ann ret | IS exSharpe |
|---|---|---|---|---|
| Permanent | 9.19% | 0.643 | 3.27% | 0.407 |
| + XS momentum 12-1 | **9.75%** | 0.570 | **4.23%** | **0.531** |

**Dual momentum (Antonacci: relative selection + absolute risk-off), 252d/top-half:**

| | Ann ret | Vol | exSharpe | MDD |
|---|---|---|---|---|
| OOS — Permanent | 9.19% | 8.27% | 0.643 | −12.6% |
| OOS — Dual momentum | **9.65%** | 9.46% | 0.617 | −12.9% |
| OOS — Dual, 126d trend | **10.09%** | 8.89% | **0.696** | −12.7% |
| IS — Permanent | 3.27% | 7.32% | 0.407 | −18.5% |
| IS — Dual momentum | 2.79% | 4.87% | **0.486** | **−12.9%** |
| IS — Dual, top-third | 3.09% | 5.31% | **0.505** | −14.4% |

**Honest reading:** momentum is not a silver bullet, and I am not going to oversell it the way I oversold the bucket layer. But it is the only signal family tested here that (a) raises *returns* in both windows in its cross-sectional form, (b) improves *excess* Sharpe in at least one window in every variant, and (c) consistently and substantially reduces drawdowns in both windows in its absolute form. Every ATLAS signal to date failed all three. Momentum is also the one gap where a course topic (S4 factor investing) was covered and never implemented — so it is defensible academically as well as empirically.

The pattern across variants is consistent and mechanically sensible: **relative momentum buys return, absolute momentum buys drawdown protection.** Neither is huge. Combined, they are roughly a +0.05 excess-Sharpe and −4pp drawdown effect — real, modest, and not obviously fitted.

### 5.2 The three levers on returns, ranked honestly

1. **Risk level (immediate, no skill required).** ATLAS runs 14.7% vol. If you want more return and are comfortable with the risk, lower `risk_aversion_for_utility_gate` further or explicitly target a volatility. But per §2 this buys return, not alpha, and it should be labelled as such rather than presented as a strategy improvement. Note the drawdown consequence: ATLAS's MDD (−16.7%) is already worse than Permanent's (−13.4%).
2. **A signal with genuine excess-Sharpe edge (the only real path).** Momentum is the best-supported candidate on your own data and course material. Implement it as `V5` — a per-asset BL view, exactly like V2/V3, so it flows through the machinery you already built and gets measured by the same ablation.
3. **Better estimation, not more signals.** μ from 756-day trailing means is the classic weakness of the whole approach (S3 warned about mean-variance's sensitivity to expected-return error; that's *why* the course taught Black-Litterman in S4). You currently only use BL's prior + views — you never exploit its other benefit, shrinking toward equilibrium when views are weak. Consider raising `tau` sensitivity analysis, or replacing the trailing-mean component entirely so μ comes *only* from Π + views.

### 5.3 What I would not do

- **Don't chase Permanent's total-return Sharpe of 1.06.** Per DIAGNOSTIC §5.3, Permanent scores 0.476 IS. You are chasing a window-specific number. Report both windows and the comparison becomes honest and much more favourable to you.
- **Don't add vol-targeting or the bucket layer** (§3, and DIAGNOSTIC §5.3) — both were artifacts of the total-return metric.
- **Don't re-enable V3 or broaden V2.** They are measured. Leave them documented as negative results; that is your best scientific content.

---

## 6. Recommended next steps, in order

| # | Action | Effort | Expected effect |
|---|---|---|---|
| 1 | **Implement momentum as V5**: `src/atlas/momentum.py` (12-1 cross-sectional score + 252d absolute trend gate), wired into `views.py` as a per-asset view with its own `modules.momentum_view` flag and confidence in `black_litterman.view_confidence`. Validate IS only, freeze, run OOS once. | 1 day | +0.05 excess Sharpe, −3 to −4 pp drawdown; ~+0.5 pp return |
| 2 | **Re-run the stage-11 ablation with V5 included and excess Sharpe as the metric.** Your current ablation ranks on total-return Sharpe, which §3 shows is misleading now that rf is priced. | 2 hrs | Correct ranking of every module |
| 3 | **Add the risk-matched-benchmark row to README** (levered Permanent at ATLAS's vol). It is the only fair comparison and it is currently missing; §2's table can be pasted in. | 30 min | Credibility — this is the single most professional thing you can add |
| 4 | **Fix the config/behaviour mismatches** in §4.1–4.4 (execution timing, sentiment flag, BIC, stale comment). | 1 hr | Config once again describes the experiment |
| 5 | **Build `plotting.py` and export `reports/figures` + `results_summary.json`.** | Half day | Deliverable-ready artifacts |
| 6 | Move superseded wrappers to `strategy_legacy.py`. | 1 hr | `strategy.py` readable again |

If you only do two: **(1) momentum as V5**, and **(3) the risk-matched benchmark row** — the first is the only credible path to real alpha, and the second is what turns "my strategy underperforms" into "my strategy matches a risk-matched benchmark and here is exactly how I proved it."

---

## 7. Standing back

The engineering here is now genuinely good: point-in-time correct, leakage-audited across three review cycles, 148+ tests, every module measured, every negative result retained. The reason ATLAS doesn't beat Permanent on a risk-adjusted basis is not that the code is wrong — it is that **four uncorrelated asset classes rebalanced to fixed weights is a very strong baseline**, and the signals implemented so far (mean reversion, support/resistance, autoencoder anomalies) have no measured edge over it. That is a real, publishable finding, and most quant projects never arrive at it because they stop when the backtest looks good.

The project's honest headline is: *"a full multi-signal, regime-aware, Black-Litterman allocation system, rigorously validated, reproduces a levered Permanent Portfolio to within 0.02% per year — here is the evidence, and here is which of eleven techniques contributed what."* That is a stronger result than a fitted equity curve.

---

*All figures independently computed from `data/raw/prices_…_2010-01-01_latest_*.parquet` using the repo's own `allocation`/`buckets` modules and a replication of `backtest.run`'s scheduler, costs, band, cap and next-day execution. Momentum variants use no parameter search beyond the two lookbacks reported.*
