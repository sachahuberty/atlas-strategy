# ATLAS — Strategy Explanation

**A plain-language walkthrough of what this project does and how the strategy works.**

Author: Sacha Huberty · IE MBDS, Algorithmic Trading

---

## 1. What this project is in one paragraph

ATLAS is a **systematic multi-asset allocation strategy**. Every Friday after the close it decides how to split a portfolio across roughly twenty exchange-traded funds covering global equities, bonds, commodities and cash, and it places the resulting trades the following Monday. The decision is fully rules-based — no discretion, no human override. What makes it more than a simple rebalancing rule is that it combines several independent sources of opinion (a market-regime model, a mean-reversion signal, and others that were tested and rejected) into a single portfolio using **Black-Litterman**, the same framework institutional allocators use to blend house views with market equilibrium. The project also includes an unusually strict validation process, which turned out to matter more than the strategy itself.

---

## 2. The core idea

Most allocation strategies fall into one of two camps:

- **Static allocations** (60/40, the Permanent Portfolio) — fix the weights, rebalance periodically, ignore the news. Robust, cheap, hard to beat.
- **Tactical allocations** — try to shift weights based on forecasts. Higher potential, but forecasting is difficult and the errors compound through an optimizer.

ATLAS is tactical, but built to fail gracefully. The design principle is:

> **Start from a sensible neutral portfolio. Only move away from it to the extent you have a genuine, quantified reason — and size the move by how confident you are in that reason.**

This is exactly what Black-Litterman was invented to do, and it is why the whole architecture is organised around it. If every signal is silent in a given week, the portfolio simply sits at its neutral allocation. Signals do not *replace* the baseline; they *tilt* it.

---

## 3. The investment universe

Twenty-two liquid US-listed ETFs, grouped into five asset-class buckets:

| Bucket | Holdings |
|---|---|
| **Equity** | ACWI, EFA, IWM, QQQ, SPY, VEA, VTI, VWO |
| **Fixed income** | AGG, BSV, HYG, IEF, LQD, SHY, TIP, TLT |
| **Commodity** | DBC, GLD, SLV, USO |
| **Cash** | BIL (1–3 month T-bills) |
| **Real estate** | VNQ |

These are not hand-picked. A screening step filters a candidate pool on minimum history, data completeness, and correlation (keeping one representative per cluster of near-identical funds). The universe is fixed as of 31 December 2021 using only data available at that date, so nothing is selected with hindsight.

**Constraints, applied at all times:** long-only, fully invested (weights sum to 100%), and a **maximum 20% in any single asset**. The 20% cap turns out to matter a lot — see §9.

---

## 4. One week in the life of the strategy

This is the heart of the project. Everything below happens automatically each Friday, using only information available at that Friday's close.

### Step 1 — Read the market's mood

A **Hidden Markov Model** is fitted to the last five years of daily returns and volatility of a broad global equity index (ACWI). An HMM is a statistical model that assumes the market is always in one of a small number of unobservable "states" — calm and rising, volatile and falling, and so on — and infers which state we are most likely in today, along with the probability of switching between states.

Four states are used; the number was chosen by a statistical criterion (BIC), not by hand. Each fitted state is then labelled by its *characteristics* — average return and volatility — rather than by its arbitrary index number. This detail matters: the model is refitted regularly, and "state 2" might mean something completely different after a refit. Labelling by profile keeps the meaning stable.

The output is a single **posture**: risk-on, neutral, or risk-off.

*Why an HMM rather than simpler clustering:* an HMM models the *sequence*, so it has a built-in tendency to stay in a state rather than flip back and forth. That persistence matters when regime changes trigger trades, because every flip costs money.

### Step 2 — Establish the neutral starting point

Before forming any opinion, the strategy asks: *what returns would each asset need to offer for the neutral portfolio to already be optimal?*

This is **reverse optimisation**. Given a reference allocation (here, an equally-weighted split across the four main buckets) and the observed covariances between assets, you can back out the expected returns implied by that allocation. These are the **equilibrium returns** — the market's implicit view, before we add anything of our own.

**Correction (found 2026-10-18, see §7):** the equilibrium returns this step produces are *excess* returns — expected return *above* the cash rate, not above zero. For a year and a half this detail was missed: cash's own equilibrium return (which legitimately comes out near zero, since cash barely moves with anything else) was overridden with the actual T-bill yield instead, and then the portfolio-construction step in §5 subtracted the T-bill yield a *second* time from every other asset. The net effect was subtracting the cash rate twice from everything except cash itself — with T-bills near 4%, that pushed most assets' adjusted expected return below cash's, and the model ended up holding whatever was volatile enough to survive the double deduction, which is not what any of its signals were actually saying. Cash's equilibrium return is no longer overridden; it is used exactly as reverse optimisation produces it, and the T-bill rate is subtracted exactly once, at the portfolio-construction step in §5. See §7 for what this changes.

### Step 3 — Form opinions ("views")

Each signal module produces a **view**: a statement of the form *"asset X will return Y% over the coming period"*, accompanied by a confidence level.

Two views are active in the current configuration:

- **V1 — Regime view.** If the HMM says risk-on, equities are assigned an expected return 3 percentage points above equilibrium. If risk-off, bonds and commodities get the boost instead. If neutral, no view is expressed at all. This operates at the asset-class level, which is the natural granularity for a regime signal.

  **In practice, this has been "add 3% to equities when calm," not a three-way switch.** Two separate problems keep risk-off from firing out of sample. First, a labelling bug (fixed 2026-10-18): the model picks a posture by comparing each fitted state's risk profile against a fixed scale, but the comparison used to shrink toward the middle whenever only two of the model's states actually showed up in a given week — which made risk-off tie with neutral and lose by coin-flip-like tie-breaking, however extreme the volatile state really was. Fixed, so risk-off can now actually be reached. Second, even reachable, the HMM only recognises a crash from the volatility that follows it, by which point the 2%-per-week turnover cap makes a full defensive repositioning take the better part of a year — too slow to matter for a sharp, already-over shock, useful mainly for a slow-grinding one. Across the entire 2014–2026 test history, risk-off posture was active in essentially none of the out-of-sample weeks. The honest description of V1 today is a calm-market equity tilt, not the three-way regime switch the design called for.
- **V2 — Mean-reversion view.** For assets that have stretched unusually far from their recent trend, the model expects partial reversion. This is filtered heavily: the asset must pass a statistical stationarity test (Augmented Dickey-Fuller), have a reversion half-life under 60 days, not be in an unusual volatility spike, and — critically — belong to the subset of tickers that historically showed genuine reversion behaviour (8 of 22 qualified, tested on in-sample data only).

Three further view families were built, measured, and switched off. That story is §8.

### Step 4 — Blend views with the neutral baseline

**Black-Litterman** combines the equilibrium returns from Step 2 with the views from Step 3 into one revised set of expected returns.

The intuition is Bayesian. The equilibrium returns are the prior belief. Each view is new evidence with a stated precision. The output — the posterior — sits between the two, pulled toward each view in proportion to its confidence and away from it in proportion to its uncertainty. Practically:

- A high-confidence view about equities moves the equity expectation substantially, and — because assets are correlated — nudges related assets too.
- A low-confidence view barely registers.
- **No views at all means the posterior equals the prior**: the strategy holds the neutral portfolio.

That last property is the reason Black-Litterman was chosen over simply averaging signals or feeding them into a machine-learning model. Naive combinations of noisy forecasts produce wild, unstable portfolios when fed into an optimiser — a well-documented failure of classical mean-variance optimisation. Black-Litterman is structurally protected against it.

### Step 5 — Build the portfolio

Three candidate portfolios are constructed from the revised expected returns and the covariance matrix (estimated over three years, with Ledoit-Wolf shrinkage to reduce estimation noise):

1. **Maximum Sharpe** — the highest risk-adjusted return, using the blended expectations.
2. **Global Minimum Variance** — the lowest-risk portfolio, ignoring return forecasts entirely.
3. **Risk Parity** — each asset contributes equally to total portfolio risk.

All three are solved numerically (sequential quadratic programming) subject to the long-only, fully-invested, 20%-cap constraints.

A **utility gate** then picks one. Each candidate is scored on the classic mean-variance utility — expected excess return minus a penalty proportional to variance — and the winner is held. The penalty parameter (risk aversion) is set to 2. In the current configuration the maximum-Sharpe portfolio wins roughly 90% of weeks; risk parity takes the remainder.

*This gate has an interesting history.* It was originally set at a much higher risk aversion, which made the minimum-variance portfolio win **every single week for four and a half years** — the entire signal pipeline was being computed and then silently discarded. The strategy was, unknowingly, just a minimum-variance fund. Recalibrating this one parameter was the largest single improvement in the project.

### Step 6 — The emergency brake

Separately, an **autoencoder** — a neural network trained to compress and reconstruct recent market behaviour — monitors five-day windows of returns. When current market behaviour is so unusual that the network reconstructs it badly (reconstruction error above the 99th percentile of its training range), the portfolio is switched to the defensive minimum-variance allocation.

The logic: the network learns what "normal" looks like from history. When it can no longer reproduce what it is seeing, something structurally unusual is happening — a reason to de-risk regardless of what any other signal says. In practice this fires on about 2% of days and has had a negligible measured effect. It is retained because it is cheap and its failure mode is conservative.

### Step 7 — Trade slowly and cheaply

The target portfolio is not implemented immediately. Three execution rules apply:

- **No-trade band (0.5%)** — if the gap between current and target weights is trivial, do nothing. Avoids paying costs to chase noise.
- **Turnover cap (2% per week, one-way)** — no more than 2% of the portfolio may change hands in any week. A full repositioning therefore takes several weeks.
- **Next-day execution** — the decision uses Friday's close; the trade executes at Monday's open. Costs of 10 basis points are charged on every trade.

The turnover cap is set as a **policy**, not optimised. An earlier version let a parameter search choose it, and the search picked the most restrictive setting available — because a criterion that rewards stability will always reward doing nothing. That configuration throttled every upstream signal into irrelevance. Setting it by policy at the 2% convention taught in the course fixed it, and was the second-largest improvement in the project.

---

## 5. Summary of the weekly decision

```
Friday close
    │
 1. HMM reads market state ──────────────► posture (risk-on / neutral / risk-off)
    │
 2. Reverse-optimise neutral portfolio ──► equilibrium expected returns
    │                                       (excess returns; T-bill rate added
    │                                        back exactly once, at step 5)
    │
 3. Signals form views ──────────────────► V1 regime tilt, V2 mean-reversion
    │
 4. Black-Litterman blends 2 + 3 ────────► revised expected returns
    │
 5. Build 3 candidate portfolios ────────► max-Sharpe / min-variance / risk-parity
    │  utility gate picks one
    │
 6. Autoencoder anomaly check ───────────► if abnormal, switch to defensive
    │
 7. Apply no-trade band + 2% turnover cap
    │
Monday open — execute, pay 10 bps
```

---

## 6. How the strategy is prevented from fooling itself

This is where most of the project's effort went, and it is the part worth defending in an interview.

**Strict time separation.** Data from 2010–2021 is the design period. Everything — parameter choices, filters, model settings — was decided using only that window. The 2022–2026 period is the test. The finalised configuration was run through it once.

**No lookahead.** A decision dated Friday may only use data available at that Friday's close. Every model is fitted on trailing windows; every data transformation is checked for accidental forward-looking. There is an explicit unit test — shift the entire price history forward by one day and confirm the strategy's decisions change accordingly — that would fail if future information leaked in anywhere.

**Realistic costs.** Transaction costs, turnover limits, and next-day execution are all modelled. Returns are reported net.

**Walk-forward evaluation.** Performance is reported per quarter across the test period, not as a single aggregate, so a result driven by one lucky quarter is visible.

**Ablation study.** Every signal module can be switched off independently. The strategy is re-run with each one disabled to measure its individual contribution. This is the project's referee, and it overruled several design decisions.

**Excess-return reporting.** All performance is reported both gross and net of the risk-free rate. In a period when cash paid 4–5%, a strategy returning 4% is not a good strategy, and total-return Sharpe ratios hide this.

---

## 7. Results

**Update, 2026-10-18: the "+0.9 points a year of alpha" this section used to report was not real skill — it was the double-counted-T-bill-rate bug described in §4, and it has been corrected.** The table and reading below are kept as the original, honest record of what was measured before the bug was found (nothing here was faked or cherry-picked), followed by what changed once it was fixed.

Test period: January 2022 to July 2026, out of sample, net of costs.

| Strategy | Return | Volatility | Sharpe | Sharpe vs. cash | Max drawdown |
|---|---|---|---|---|---|
| Permanent Portfolio (benchmark) | 9.5% | 8.8% | 1.07 | 0.62 | −13.4% |
| **ATLAS** | **13.7%** | **14.7%** | **0.95** | **0.68** | **−16.7%** |
| Risk parity | — | 9.8% | 0.86 | 0.45 | −16.5% |
| Hierarchical risk parity | — | 3.5% | 0.83 | −0.29 | −9.2% |
| Minimum variance | — | 3.2% | 0.74 | −0.51 | −8.1% |
| 60/40 | 6.7% | 11.5% | 0.63 | 0.28 | −22.5% |

**Reading this table honestly:**

ATLAS earns the highest raw return of any book tested — 13.7% against the benchmark's 9.5%. But it does so at 14.7% volatility versus 8.8%, so the higher return is substantially compensation for higher risk rather than pure skill.

The fair comparison is to lever the benchmark up to the same volatility, financed at the T-bill rate. Doing that gives the Permanent Portfolio a 12.8% return — leaving ATLAS ahead by roughly **0.9 percentage points a year at matched risk**, with a shallower maximum drawdown (−16.7% vs −22.1%). That is the honest measure of what the strategy adds.

Two caveats, both important:

1. **The configuration was refined while looking at this test window.** Several decisions (which signals to disable, the risk-aversion setting) were informed by out-of-sample results. That is not the same as a clean out-of-sample test, and it should be described as a development period.
2. **The same machinery underperforms in the design period.** Run over 2014–2021, ATLAS scores worse than the benchmark on risk-adjusted terms and suffers a −32% drawdown against the benchmark's −18%. The edge is regime-dependent, not persistent.

Note also that minimum variance and hierarchical risk parity show **negative Sharpe ratios versus cash**: over this period they did not beat T-bills at all. Their attractive-looking total-return Sharpes are an artefact of measuring against a zero risk-free rate in a 5% rate environment.

### What actually explains the "+0.9 points a year," and the corrected numbers

Both caveats above turned out to share one cause. Tracing the model's own reasoning (why does it hold 42% of its risk in one silver ETF; why does it rate its own portfolio as barely beating cash) led to the double-counted T-bill rate in §4: the model was, to first order, "hold whatever is the most volatile asset allowed under the 20% cap," because only the highest-volatility assets' expected returns survived having the cash rate subtracted from them twice. That is not a view about markets; it is an artefact of the bug. Measuring each asset's own volatility against how often it was actually held across the whole 2014–2026 history confirms it directly: before the fix, the two track almost in lock-step (a rank correlation of about +0.7); after the fix, that relationship is gone (it even mildly reverses).

This also explains why the strategy looked good in 2022–2026 and bad in 2014–2021: commodities (the most volatile bucket, and the bug's favourite) lost money in the first period and gained over 20% a year in the second. The "edge" was a hidden, unsized bet on commodity volatility, placed by accident, that happened to pay off in the window being used for the headline number.

**Corrected, over the same two windows plus the full 12-year history:**

| Window | Strategy | Return | Volatility | Sharpe vs. cash | Max drawdown |
|---|---|---|---|---|---|
| 2014–2021 | Permanent | 3.9% | 7.9% | 0.43 | −19.7% |
| 2014–2021 | ATLAS (bug) | 4.2% | 13.5% | 0.32 | −31.5% |
| 2014–2021 | **ATLAS (corrected)** | **5.3%** | **10.0%** | **0.50** | **−23.5%** |
| 2022–2026 | Permanent | 9.9% | 8.8% | 0.66 | −13.4% |
| 2022–2026 | ATLAS (bug) | 14.2% | 14.4% | 0.72 | −16.6% |
| 2022–2026 | **ATLAS (corrected)** | **10.7%** | **10.7%** | **0.63** | **−15.9%** |
| 2014–2026 | Permanent | 6.0% | 8.3% | 0.52 | −19.7% |
| 2014–2026 | ATLAS (bug) | 7.8% | 13.8% | 0.47 | −31.5% |
| 2014–2026 | **ATLAS (corrected)** | **7.2%** | **10.2%** | **0.55** | **−23.5%** |

Fixed, ATLAS roughly **ties the Permanent Portfolio over the full twelve years** (0.55 vs 0.52) with a noticeably shallower drawdown than the buggy version (−23.5% vs −31.5%, though still deeper than Permanent's own −19.7%). The 2022–2026 "outperformance" shrinks and the 2014–2021 "underperformance" shrinks too — both move toward Permanent, because both were the same mispriced bet, not two separate, opposite findings. **The honest conclusion is that the model's own signals add little measurable edge over a simple 25/25/25/25 split; the earlier apparent edge was an accidental volatility bet, not evidence of skill.**

---

## 8. What was tested and rejected

Five signal families were built. Three were switched off after measurement:

| Signal | Idea | Measured effect | Status |
|---|---|---|---|
| **V1 Regime** | HMM posture tilts asset classes | **+0.147 Sharpe** | Active |
| **V2 Mean reversion** | Stretched assets revert | ≈ 0 | Active, heavily filtered |
| **V3 Technical levels** | Support/resistance zones from clustered price pivots | **−0.265 Sharpe** | Disabled |
| **V4 Sentiment** | News headline sentiment via NLP | Not backtestable | Live-only |
| **V5 Momentum** | Dual momentum (relative + absolute trend) | **−0.236 Sharpe** | Disabled |

Two of these deserve comment because they are counter-intuitive.

**Technical levels failed for a diagnosable reason.** The support/resistance detector generated thousands of spurious signals on near-cash instruments like BIL and SHY, whose prices barely move — any tiny fluctuation registers as "near a level". Its hit rate on those assets was below chance.

**Momentum failed despite working standalone.** Tested on its own, dual momentum improved returns in both periods and cut drawdowns meaningfully. Wired into the Black-Litterman pipeline, it measured strongly negative. The likely mechanism: when a bucket fails the trend test, every member receives a flat negative view — so in 2022, with equities *and* bonds both in downtrends simultaneously, the model placed strong negative views across most of the portfolio at once, after the losses had already occurred. A signal that works as a position rule does not necessarily work as a *return forecast*, which is what Black-Litterman consumes. This is a genuine finding about the architecture, not just a failed experiment.

All three rejected modules remain in the codebase with their tests and their measured results. A signal family that was implemented properly and rejected on evidence is a result, not a failure.

---

## 9. Things worth knowing about how it behaves

- **The portfolio is concentrated.** With a 20% cap and an optimiser that wants more concentration than that, the book typically holds exactly five positions, all at the cap. A recent snapshot: BIL, IWM, QQQ, VEA, SLV.
- **Loosening the cap makes it worse, not better.** At a 50% cap the return collapses to 4.5%, because the unconstrained optimiser piles into low-volatility short-duration bonds. The cap is effectively forcing the strategy to take risk, not just diversifying it.
- **Risk aversion has saturated.** Lowering the utility-gate parameter below 2 changes nothing — the max-Sharpe book already wins nearly every week. There is no dial left that buys more return.
- **A past misdiagnosis, corrected.** An earlier pass noticed that the model seemed to believe almost nothing in the universe beat cash for much of 2023–2025, and attributed it to the risk-aversion constant in the reverse-optimisation step being miscalibrated for a low-volatility multi-asset reference portfolio (implying an equity risk premium of only ~2.1%, roughly half any credible estimate). Re-deriving an asset-specific version of that constant from recent data each week was tried and made results worse in both the design and test periods. The symptom (almost nothing beating cash) was real; the diagnosis was wrong. The actual cause was the double-counted T-bill rate described in §4 — once that was fixed, the "nothing beats cash" symptom went away on its own, and the risk-aversion constant was left exactly as it was.

---

## 10. Honest summary

ATLAS is a complete, working implementation of institutional multi-asset allocation machinery: regime detection, Bayesian view blending, constrained optimisation, anomaly-based de-risking, and realistic execution — validated with a rigour that is unusual outside professional research.

The strategy delivers the highest raw return of any book tested and modestly beats a risk-matched benchmark over the test window — though §7's update found that edge was itself a mispriced bet, not skill, once corrected ATLAS roughly ties the benchmark over the full history instead. It does not beat that benchmark on risk-adjusted terms without leverage, and (even before the correction) the edge did not replicate in the earlier period.

The most valuable output is arguably not the equity curve but the evidence trail. Over the project's life the analysis discovered that the strategy had silently degenerated into a minimum-variance fund for four and a half years, that its execution layer was throttling every signal it produced, that two of its most sophisticated signals actively destroyed value, that one apparent improvement was an artefact of the wrong performance metric, and — the largest single finding — that the risk-free rate was being subtracted twice, which had been quietly driving the model into a volatility bet it never intended to place. Each was found by measurement, corrected, and documented — including the ones that made results worse.

The conclusion the evidence supports: **a well-diversified static allocation is a genuinely hard benchmark to beat, and the value of this kind of machinery lies less in the returns it generates than in the discipline required to know whether those returns are real.**

---

## Appendix — Glossary

| Term | Plain meaning |
|---|---|
| **Black-Litterman** | A method for blending market-implied expected returns with an investor's own views, weighted by confidence |
| **Equilibrium returns** | The expected returns implied by a reference portfolio being optimal ("reverse optimisation") |
| **Hidden Markov Model** | A model that infers which unobservable "state" a system is in, and how likely it is to switch |
| **Autoencoder** | A neural network trained to reproduce its input; large reproduction errors flag unusual conditions |
| **Sharpe ratio** | Return per unit of volatility. "Excess" Sharpe measures return above cash |
| **Ledoit-Wolf shrinkage** | A technique that stabilises a covariance matrix estimated from limited data |
| **Ablation study** | Re-running the system with one component removed to measure that component's contribution |
| **Walk-forward** | Evaluating performance across successive time slices rather than one aggregate figure |
| **Turnover** | Fraction of the portfolio traded; here capped at 2% per week to control costs |
| **Out-of-sample** | Data not used when designing the strategy — the only honest test of whether it works |
