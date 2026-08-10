# ATLAS — Analysis V3: State of the Project, and the Return Ceiling

**Method:** full re-read of the post-Step-5 codebase (20 modules), plus an independent re-run of the fusion pipeline from this repo's own cached data across 240 OOS and 418 in-sample weekly decision dates, testing eight configuration levers and two structural alternatives.
**Date:** 2026-08-07

---

## 1. Verdict up front

The project is in good shape and the last round of work was correct. Momentum was implemented properly, measured honestly, and rejected on evidence — that is the methodology working exactly as designed. The config now describes the experiment actually run. Deliverables exist.

**On raw returns: you are essentially at the ceiling of this design.** I tested every remaining knob — risk aversion, per-asset cap, turnover cap, view magnitude, two-fund separation, equilibrium calibration — on the real pipeline. Reproduction of your current config gives 14.2% annualized. Of eight alternatives, **seven are worse**, and the one that is better out-of-sample is worse in-sample (§4). Further return gains are not available from tuning this system; they require changing its constraints (§6).

**One claim in the README needs qualifying, and it is the important one.** The "+0.93%/yr alpha" is real in the OOS window and statistically strong on a paired test — but the same pipeline **underperforms Permanent on every metric in-sample, with a −32% drawdown versus Permanent's −18%.** §3.

---

## 2. What is genuinely good

- **The momentum rejection was handled correctly.** V5 was built to spec, IS-tuned, wired into the real fusion, measured at −0.236 excess Sharpe, disabled, and kept in the repo with the negative result documented. I predicted this could happen (a signal that works standalone can fail once diluted through Ω and the utility gate) and the project caught it without being told what answer to reach.
- **Config now matches reality.** `execution_timing: false`, `sentiment_view: false`, the stale posture comment removed. After three review cycles flagging drift, the config finally describes the experiment being run.
- **The metric discipline held.** Excess Sharpe is now the primary ranking metric, which is what exposed that HRP and GMV never beat cash at all (−0.29 and −0.51).
- **Deliverables exist:** `plotting.py`, four exported figures, `results_summary.json`, `strategy_legacy.py`. The repo is now presentable.
- **My reproduction matches your reported numbers.** Independent re-implementation gives 14.2% return / 0.703 excess Sharpe against your 13.65% / 0.673 — the gap is explained by my using a fixed risk-on posture and no anomaly override. The backtester is sound.

---

## 3. The alpha claim: strong in one window, absent in the other

Your README reports +0.93%/yr over a risk-matched levered Permanent. I tested this properly, using the Memmel (2003) paired test, which is the right test because the two series are highly correlated:

| | Excess Sharpe | ρ(ATLAS, Permanent) | z | p |
|---|---|---|---|---|
| ATLAS vs Permanent, OOS | 0.703 vs 0.644 | 0.930 | 4.48 | <0.0001 |

So the OOS gap is *not* noise in the way my earlier "±0.22 standard error" comment implied — that figure applies to a standalone Sharpe estimate, not to a paired difference between two 93%-correlated series. Credit where due: the claim is defensible on its own terms.

**But run the identical machinery over 2014–2021:**

| 2014–2021 (in-sample) | Ann. return | Vol | Excess Sharpe | Max drawdown |
|---|---|---|---|---|
| Permanent | 3.76% | 7.4% | **0.438** | **−18.5%** |
| ATLAS (δ=2.5, A=2) | 5.71% | 14.4% | 0.409 | **−32.3%** |

In-sample ATLAS earns more raw return (5.71% vs 3.76%) but loses on risk-adjusted terms and takes a **−32% drawdown against Permanent's −18%**. The alpha does not survive out of its window.

Two things follow. First, the honest framing is *"ATLAS beat a risk-matched Permanent in 2022–2026 and lost to it in 2014–2021"* — which is a much stronger and more publishable statement than an unqualified alpha claim, because it is the truth and it comes with a mechanism (§4). Second, the p-value should not be quoted without noting that this configuration (V3 off, A=2, V5 off, cash reverted, δ=2.5) was arrived at through roughly a dozen OOS evaluations. Under that much selection, a nominal p-value is not interpretable as evidence. The in-sample result is the real out-of-sample test, and it says no.

---

## 4. The deepest remaining problem: the equilibrium prior is mis-calibrated, and it only bites at high rates

This is the most interesting thing I found, and it is grounded directly in S1 (CAPM, risk premia) and S4 (Black-Litterman equilibrium returns).

`equilibrium_returns` computes Π = δ·Σ·w_mkt with **δ = 2.5**, the textbook default. But 2.5 is calibrated for an *equity* market portfolio. Your "market" is the Permanent Portfolio — only 25% equity, ~8.3% volatility. Applying an equity-calibrated δ to a low-volatility multi-asset market produces implausibly small risk premia:

| δ | Implied equity premium | Assets with μ > rf (of 22) |
|---|---|---|
| **2.5 (current)** | **2.11%** | **1** |
| 5.0 | 4.22% | 11 |
| 8.0 | 6.75% | 13 |

An implied equity premium of 2.11% is roughly half any credible historical estimate. Now combine that with the risk-free rate you correctly introduced in Tier 1. Sampling μ_BL − rf across the OOS window:

| Date | rf | Assets beating cash | Mean expected excess return |
|---|---|---|---|
| 2022-01 | 0.10% | 18 / 22 | +3.23% |
| 2022-10 | 3.65% | 13 / 22 | +0.47% |
| **2023-12** | **5.24%** | **1 / 22** | **−2.31%** |
| **2024-04** | **5.25%** | **1 / 22** | **−2.37%** |
| 2025-06 | 4.21% | 5 / 22 | −1.30% |

**For most of 2023–2025 the model believed that essentially nothing in the universe beat cash** — in December 2023 and April 2024, the only asset with non-negative expected excess return was cash itself. The strategy kept producing books anyway, because `max_sharpe` maximizes a *ratio* and will return something even when the numerator is negative for every asset. It was optimizing over expected returns it did not believe in.

This also explains three earlier puzzles cleanly:
- Why the utility gate is so hair-trigger (tiny expected returns against a quadratic vol penalty).
- Why lowering A from 2 to 1 or 0.5 **reduces** returns (14.23% → 13.77%): the gate already picks the BL book 218/240 weeks at A=2; lowering A only converts the 22 risk-parity weeks, and those weeks were helping.
- Why proper two-fund separation collapses (below).

**Why it didn't bite in-sample:** 2014–2021 was a zero-rate period. With rf ≈ 0, a 2.11% implied premium still leaves everything positive, so δ's mis-calibration was invisible. It only becomes destructive when rf ≈ 5%. Your model has a rate-regime dependency nobody designed in.

**And yet — do not simply set δ = 8.** I tested it:

| | OOS return | OOS excess Sharpe | OOS MDD | **IS excess Sharpe** |
|---|---|---|---|---|
| δ = 2.5 (current) | 14.23% | 0.703 | −16.7% | **0.409** |
| δ = 5.0 | 14.97% | 0.717 | −20.3% | **0.382** |
| δ = 8.0 | **15.29%** | **0.733** | −20.5% | not tested |

δ = 5 improves OOS and **degrades IS** — the same overfit signature as vol-targeting and the bucket layer. The theoretically-correct-sounding fix does not replicate. The right response is not to pick a number but to **make δ self-calibrating**: δ_t = (trailing market excess return)/σ²_market, recomputed weekly from the same trailing window everything else uses. That adapts across rate regimes without being fitted to either window, and it is defensible as the textbook reverse-optimization definition rather than a tuned constant. I would present it as a methodological correction and report whatever it does to performance.

---

## 5. Things I tested that do not work (so you don't have to)

| Lever | OOS return | OOS excess Sharpe | Verdict |
|---|---|---|---|
| **Current config (A=2, cap 20%, 2% turnover)** | **14.23%** | **0.703** | baseline |
| Risk aversion A = 1.0 | 13.77% | 0.642 | worse |
| Risk aversion A = 0.5 | 13.77% | 0.642 | worse (identical — gate saturates) |
| Per-asset cap 30% | 12.35% | 0.580 | worse |
| Per-asset cap 50% | 4.52% | 0.094 | **much** worse |
| Turnover cap 4% | 14.13% | 0.687 | marginally worse |
| Turnover cap 10% | 14.08% | 0.681 | marginally worse |
| V1 view 3% → 6% | 12.41% | 0.567 | worse |
| CML two-fund separation (A=2) | 3.59% | −0.064 | fails |

Three of these are worth understanding:

**The 20% cap is fully binding and is doing you a favour.** Your current book holds exactly five positions, all at the cap: BIL, IWM, QQQ, VEA, SLV (60% equity, 20% commodity, 20% cash, 0% bonds). The optimizer wants more concentration than it is allowed. But *loosening* the cap makes things dramatically worse — at 50% the return collapses to 4.5%, because the unconstrained optimizer concentrates into low-volatility short-duration bonds. The cap is functioning as a forced-risk-taking constraint, not just a diversification rule. Leave it at 20%.

**Two-fund separation fails — but for a diagnosable reason.** I tested the textbook S1/S3 structure: compute the tangency portfolio over risky assets only, then blend with the risk-free asset at the utility optimum k = (μ_tan − rf)/(A·σ²_tan). It returns 3.59% with a *negative* excess Sharpe. The reason is §4: the mean optimal risky weight came out at **k = 0.157**, because the mis-calibrated prior sees almost no risk premium to harvest. So this is not evidence against two-fund separation — it is another symptom of the δ problem, and worth re-testing if δ is ever made self-calibrating. (Note this also means your current design — cash sitting *inside* the max-Sharpe optimization — is not what CAPM prescribes, even though it currently performs better.)

**Risk aversion has saturated.** A=1.0 and A=0.5 produce byte-identical results: below A≈1 the gate picks the BL book in all 240 weeks and further reduction changes nothing. You cannot buy more return by turning that dial.

---

## 6. Where additional raw return can actually come from

Having exhausted the internal knobs, the honest answer is that **returns are now bounded by the constraint set, not the intelligence of the strategy.** Three real options, in order of how defensible they are:

1. **Widen the universe (highest impact, fully in the spirit of the project).** Your book is capped at five positions from 22 broad ETFs whose best OOS performers returned 21% (SLV, USO). The PROJECT_STRUCTURE.md promise was "any asset, any market worldwide," and the built-in pool of 23 US ETFs was always a placeholder — notebook 01 says as much, and the K-Means de-dup is a near-no-op at this size. Adding sector ETFs (XLE, XLK, XLF…), single-country funds, factor/smart-beta ETFs (S4's actual topic — MTUM, QUAL, USMV), and a real dollar-volume liquidity screen would give the optimizer something to select *from*. This is the one change that raises the ceiling rather than redistributing under it, and it is the largest unfulfilled promise in your own design document.

2. **Make δ self-calibrating (§4).** Principled, fixes a real defect, adapts across rate regimes. Report the performance effect honestly whatever it is — do not adopt it only if it helps.

3. **Explicit, disclosed leverage.** If the goal is raw return, say so and target a volatility with a stated leverage factor rather than achieving 14.7% vol as an accident of the gate. That is honest and controllable, but per §3 it produces no alpha — it is a preference, not an edge, and should be labelled that way.

**What I would not do:** more signal families. You have now measured five (V1–V5) and four were neutral or negative. The marginal value of a sixth is low, and every one you add increases the selection problem that already makes §3's p-value uninterpretable.

---

## 7. Remaining code issues

Minor; nothing structural.

- **`momentum.py` asymmetry worth noting in the write-up:** a bucket failing the absolute-trend gate gives *every* member a flat −3% view, which in 2022 (equities and bonds both in downtrends simultaneously) puts a strong negative view on most of the portfolio at once, after the drawdown. That mechanism is very likely why V5 measured −0.236 in the fusion despite working standalone. Worth one sentence in the notebook — it converts "momentum didn't work" into "here is precisely why a view-based implementation of momentum fails where a position-based one doesn't," which is a much better finding.
- **`regimes.hmm.n_states: 3` is still a hard-coded constant** while `hmm_bic_curve` exists. Either read the argmin or state the BIC values in the comment. Flagged in ANALYSIS_V2 §4.3, still open.
- **`universe.min_avg_dollar_volume` remains dead** (no volume data is ever downloaded). Would need to be implemented as part of option 1 above.
- **`per_class_cap: 0.60` is nearly non-binding** across four buckets summing to 1.
- Notebooks 02–08 still contain hand-rolled chart code that `plotting.py` now duplicates; worth a cleanup pass so the module is actually the single source.

---

## 8. Is the strategy "working as it should"?

Mechanically, yes: no leakage, point-in-time correct, costs and lags modelled, every module measured, negative results retained. I re-audited the new code paths and found nothing wrong.

Scientifically, the result is: **a full multi-signal, regime-aware, Black-Litterman allocation system, honestly validated, produces a portfolio that beat a risk-matched Permanent Portfolio in 2022–2026 and lost to it in 2014–2021, with four of five signal families measuring neutral or negative.** That is a real finding about the difficulty of adding value over a well-diversified static allocation — and the evidence trail behind it (every regression kept, every module ablated, every metric corrected mid-project) is worth more than a good-looking equity curve. Most projects of this type never discover their own strategy was silently GMV, or that their headline signal fails once fused; yours did, twice, and documented both.

If you want one sentence for the report: *the machinery is correct, the discipline is unusually rigorous, and the honest conclusion is that the edge over a naive four-bucket benchmark is regime-dependent rather than persistent.*

---

*All figures re-derived from `data/raw/prices_…_2010-01-01_latest_*.parquet` using this repo's own `allocation`/`views` modules, over 240 OOS and 418 IS weekly decision dates, with `backtest.run`'s exact scheduler, 10 bps costs, 0.5% no-trade band, 2% turnover cap and next-day execution. Paired Sharpe test: Memmel (2003) correction to Jobson–Korkie.*
