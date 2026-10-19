# ATLAS — Analysis V4: The First Live Run, and the Root Cause Behind the Whole Project's Results

**Scope:** the first live run (decision 2026-10-09, run 2026-10-18), its `diagnostics.json`, `state.json`, `trades.csv`, and the rendered report; then a root-cause pass through the strategy using an independent harness built on the repo's own `allocation`, `views` and `regimes` modules, over 658 weekly decisions (2014-01 → 2026-08), both windows.
**Date:** 2026-10-18

---

## 0. Headline

The live pipeline works: it ran on a clean machine, funded the book, committed state, and produced a correct-looking report. **But the run surfaced a specification error that turns out to explain the behaviour of the strategy across the whole project.**

> **The risk-free rate is subtracted twice.** Black-Litterman's equilibrium returns Π = δΣw are *excess* returns (risk premia). ATLAS treats them as *total* returns — it sets cash's prior to the T-bill rate, then has max-Sharpe subtract the T-bill rate again. With T-bills at ~4%, only the most volatile assets, whose implied premia exceed 4%, survive. Every bond fund is pushed out.

Consequences, all measured below:

- With **no views at all**, the deployed optimiser turns a 25/25/25/25 anchor into **40% commodities, 40% equity, 0% bonds**. Correctly specified, it hands back the anchor.
- Across 2014–2026 the deployed strategy's holdings have a **+0.85 rank correlation with asset volatility**. It is, to first order, *"hold the most volatile assets at the 20% cap"*.
- That accidental high-volatility, commodity-heavy tilt **explains the project's IS/OOS reversal**: commodities returned −2.2%/yr in 2014–2021 (ATLAS lost to Permanent) and +21.0%/yr in 2022–2026 (ATLAS beat it).
- **Corrected, ATLAS ties the Permanent Portfolio over 12 years** (excess Sharpe 0.514 vs 0.515) with a drawdown of −21% instead of −31%.

The honest conclusion this supports: **the signals add little measurable edge over the anchor; the deployed version's out-of-sample outperformance was an unintended volatility/commodity bet created by a specification error, which happened to pay in 2022–2026.**

Five further issues were found (§2.2–2.7), including a one-week decision lag and a defensive regime posture that never fires.

---

## 1. The live run — what it did, and what checks out

| | |
|---|---|
| Decision date | 2026-10-09 (Fri) |
| Run time | 2026-10-18 17:35 UTC (manual dispatch) |
| Status | `ok`, funding run (cap-exempt, 50% one-way) |
| Book | BIL / IWM / QQQ / SLV / VEA, 20% each (all at cap) |
| Posture | `risk_on` (HMM state 2: +13.9%/yr, 13.3% vol; stay-probability 99.56%) |
| View fired | V1 regime: +3.0% on the equity bucket, confidence 1.0 |
| Gate | Black-Litterman book, utility margin 0.0057 |
| Anomaly | reconstruction error 0.44 vs threshold 5.31 — quiet |

**Checks that pass:**

- **Matches the validated strategy.** Same five-asset book as the August local dry run, and as my independent reproduction. Replay parity holds in practice, not just in tests.
- **The plumbing is correct.** Atomic state write, `initialized: true`, weights sum to 1 within tolerance, funding exemption applied, diagnostics serialised, report self-contained and rendering cleanly.
- **The regime read is internally sensible.** The two visited states are genuinely distinct (calm/rising vs. volatile/falling), and the transition matrix shows the persistence that justified choosing an HMM.

*Correction to my previous message:* I said no views fired. They did — the diagnostics key is `views_fired`, not `views`, and my lookup was wrong.

---

## 2. Bugs and inconsistencies, ranked by importance

### 2.1 Risk-free rate subtracted twice — **root cause, HIGH**

**Where:** `strategy.black_litterman_strategy` sets the cash bucket's prior to `rf`, then calls `allocation.max_sharpe(mu, cov, cfg, rf=rf)` and `utility_select(..., rf=rf)`, which compute `w @ mu − rf`.

**Why it's wrong:** `allocation.equilibrium_returns` computes Π = δΣw. In He–Litterman, Π is the vector of *implied equilibrium excess returns*: δ is calibrated as (E[r_m] − r_f)/σ²_m, so Π already has the risk-free rate removed. Treating Π as a total return and subtracting `rf` again makes every asset's expected excess return Π − rf.

**Evidence from the live book:**

| | μ_BL | Excess vs cash (4.06%) | Held |
|---|---|---|---|
| SLV | 7.60% | +3.54% | 20% |
| QQQ | 4.20% | +0.14% | 20% |
| IWM | 4.19% | +0.13% | 20% |
| VEA | 3.50% | **−0.56%** | 20% |
| BIL | 4.06% | 0.00% | 20% |

- **19 of 22 assets** are expected to return less than cash.
- **VEA is held at 20% while expected to underperform cash.** Only three risky assets clear the bar, the cap allows 20% each plus 20% cash, so the optimiser is forced to fill the last 20% with the "least bad" negative-excess asset.
- **All three candidate books have negative utility** (BL −0.018, risk parity −0.023, GMV −0.030). By its own objective the model prefers 100% cash to every portfolio it is allowed to hold, and the gate picks the least bad.
- **The book's ex-ante excess Sharpe is 0.04.** The model believes its own portfolio barely beats T-bills.
- **42% of portfolio risk sits in one silver ETF.** SLV is the top conviction because Π scales with covariance to the anchor, and silver is the most volatile asset with a 6.25% anchor weight — not because of any signal.

**Decisive test — no views at all:**

| Forward step, given the same Π and **no views** | Equity | Fixed income | Commodity | Cash |
|---|---|---|---|---|
| Anchor (Permanent proxy) | 25% | 25% | 25% | 25% |
| Mean-variance at the same δ (theory) | 25% | 25% | 25% | 25% |
| **ATLAS as deployed** | **40%** | **0%** | **40%** | 20% |
| **Corrected** (total = rf + Π) | 26% | 28% | 26% | 20%* |

*Cash capped at 20% by the per-asset limit.

**What it selects for:** across 2014–2026, Spearman correlation between an asset's volatility and the share of weeks it is held = **+0.85**. USO (36% vol) held 78% of weeks; SLV 69%; IWM 78%; QQQ 81%. Every bond fund ~1%.

**Why this explains the IS/OOS reversal:** average commodity weight is 24% (IS) and 31% (OOS) under the deployed strategy versus 1–2% under a market-cap prior. Commodities returned −2.2%/yr IS and +21.0%/yr OOS.

**Correction to ANALYSIS_V3 §4:** I diagnosed the "only 1 of 22 assets beats cash" symptom as δ = 2.5 being miscalibrated and recommended a self-calibrating δ. The symptom was right; the diagnosis was wrong. δ = 2.5 implies a ~2.1% equity *excess* premium against a low-volatility multi-asset anchor, which is low but not absurd. The real problem is subtracting the T-bill rate from an already-excess premium. This also explains why the self-calibrating-δ test made things worse in both windows.

### 2.2 One-week decision lag — HIGH (pure plumbing)

**Where:** `live.resolve_as_of` excludes the reference date's ISO week. ISO weeks run Monday → Sunday, so on Saturday — the cron day — the Friday that just closed is excluded as "in progress".

| Run day | Decides for | |
|---|---|---|
| Wed 14 Oct | Fri 9 Oct | correct |
| **Sat 17 Oct (cron)** | **Fri 9 Oct** | **7 days stale** |
| **Sun 18 Oct (your run)** | **Fri 9 Oct** | **7 days stale** |
| Mon 19 Oct | Fri 16 Oct | correct |

The diagnostics show `prices_as_of: 2026-10-16`: the right Friday's data was downloaded and ignored. Every scheduled run publishes a decision a week late, and the first run's state records an execution on 12 October for a decision computed on 18 October.

The decisions remain point-in-time correct (no lookahead — they use only data up to the decision date), so the paper NAV is still a faithful replay. But the report is not actionable, and the "live" record is published a week behind.

**Fix:** exclude the current ISO week only on weekdays — `if reference.weekday() < 5`. The existing freshness check already holds the run if Friday's close hasn't posted.

### 2.3 The defensive regime posture is effectively dead — HIGH (strategy logic)

**Measured posture usage:**

| Window | risk_on | neutral | risk_off |
|---|---|---|---|
| IS 2014–2021 (418 wks) | 79% | 19% | **2% (8 weeks)** |
| OOS 2022–2026 (241 wks) | 77% | 23% | **0%** |

Three independent causes:

**(a) Mapping bug.** `regimes.map_states_to_postures` compares empirical ranks on a 0…n−1 scale with posture targets on a fixed 0…2 scale. With two visited states, empirical scores are ±1; the risk-off target (+2) ties with neutral (0) and dictionary order picks neutral. **Risk-off is unreachable whenever only two states are visited — 31% of OOS fits**, however extreme the volatile state. The live run is an example: state 1 at −1.4%/yr and 28% vol is labelled *neutral*. HMM `n_states` is 4, yet never more than 3 states are visited, and 2% of fits degenerate to NaN and silently fall back to neutral.

**(b) The turnover cap makes the regime layer unactionable.** With the mapping fixed, risk-off fires on 28 Feb 2020 (two weeks earlier than the current code) and the target turns correctly defensive: long Treasuries, IG credit, cash. At the 18 March trough the book held **55% equity against a 7% target and 5% bonds against a 76% target**. Closing that gap at 2%/week takes **~41 weeks**. The book still held 7% USO on 20 April 2020, the day oil futures went negative.

**(c) Even when unblocked, the HMM is a lagging detector.** Lifting the cap on transitions into/out of risk-off improves IS (drawdown −31.5% → −25.0%, excess Sharpe 0.331 → 0.366) but hurts OOS (0.728 → 0.621). The OOS risk-off flags were April 2025 (the tariff shock — Session 7's own example) and April 2026, both confirmed **at the bottom**: ACWI rose +22.5% over the next three months while TLT fell 6.3%. A volatility-based regime model confirms a crash after the damage; acting on it quickly helps if the crash continues (2020) and hurts in a V-shaped shock (2025).

**Implication for the write-up:** `Strategy Explanation.md` §4 says "if risk-off, bonds and commodities get the boost instead". In practice that never happened out-of-sample. V1 is, in effect, "add 3% to equities when calm".

### 2.4 Ledoit-Wolf inflates defensive assets' volatility — MEDIUM

`allocation.covariance_matrix(method="ledoit_wolf")` uses sklearn's default target: average variance × identity. In a universe where variances differ ~3000× (BIL vs USO), even 2.8% shrinkage pulls every variance toward the same number:

| | Sample vol | LW vol | Inflation |
|---|---|---|---|
| BIL | 0.23% | 2.96% | **×12.8** |
| SHY | 1.63% | 3.36% | ×2.1 |
| BSV | 2.27% | 3.70% | ×1.6 |
| SLV | 42.2% | 41.7% | ×0.99 |

Defensive books (GMV, risk parity) look riskier than they are, which biases the utility gate against them. **Fix:** shrink the *correlation* matrix and rescale by sample volatilities, which preserves each asset's own variance. **Measured impact on returns: negligible** (±0.01 excess Sharpe both windows) — a correctness fix, not a performance lever.

### 2.5 The live anomaly detector is not the validated one — MEDIUM

Live reads `config.yaml` directly (`epochs: 50, patience: 5`, refit every run). The canonical OOS run in notebook 09 overrode these to `epochs: 10, patience: 3, refit_frequency_days: 126`. The deployed autoencoder is trained roughly five times longer than the one validated. Practical impact is small (the override fires on ~2% of days and measured +0.002), but deployed and validated strategies should be identical. Align them, one way or the other.

### 2.6 Operational fragility — MEDIUM, **time-sensitive**

- **`runs-on: ubuntu-latest` migrates to Ubuntu 26 from 19 October 2026** (the notice is in your run's annotations). Next Saturday's run will be on a new OS. Pin `ubuntu-24.04`.
- **`requirements.txt` is entirely unpinned.** Each weekly run installs whatever is newest. An `hmmlearn`, `scikit-learn` or `pandas` release can silently change HMM fits or covariance numerics, and parity was verified against your local versions, not the runner's. Commit a `requirements.lock.txt` from `pip freeze` and install from that in CI.
- **Node.js 20 deprecation** on `actions/checkout@v4`, `setup-python@v5`, `upload-artifact@v4`: harmless now, will eventually break. Bump action versions.
- **Cold data download every week.** `data/raw/` is gitignored, so each run pulls 16 years × 22 tickers from Yahoo. It worked; it is also the most likely next failure (rate-limits on cloud IPs). Consider caching `data/raw` with `actions/cache`.

### 2.7 Cosmetic — LOW

- `trades.csv`, `allocation.csv` and `state.json` carry SLSQP residuals such as `4.5e-17`. Round to 6 dp when writing (the HTML report already formats correctly).
- `history.csv` book returns are gross of costs (~9 bp/yr overstatement vs the backtest); label it on the chart or net the costs.
- The report header shows only the decision date. Show the run date too, so a lag like §2.2 is visible on the page.
- The Delta column shows 2 dp while Target/Current show 1 dp.

---

## 3. Is the result good and coherent?

**Coherent with the validated strategy: yes.** The live system faithfully reproduces what the backtest does.

**Economically sound: no.** The book is a product of §2.1. A portfolio manager looking at this week's report would ask why 42% of risk is in silver, why there are no bonds, why a fund expected to lose to cash is held at full weight, and why the model rates its own portfolio at an excess Sharpe of 0.04. None of these has a defensible answer under the current specification. All of them disappear once Π is treated as an excess return.

---

## 4. Measured impact of every variant tested

Independent harness on the repo's own modules: real weekly HMM, Black-Litterman, the three candidate books, the utility gate at A = 2, `backtest.run`'s exact scheduler, 10 bp costs, 0.5% band, 2% cap, next-day execution. It omits V2 (mean reversion) and the anomaly override, both measured near-zero contributors. It reproduces the deployed results within ~0.04 excess Sharpe of the README.

| Variant | IS ann. | IS exSharpe | IS MDD | OOS ann. | OOS exSharpe | OOS MDD |
|---|---|---|---|---|---|---|
| Permanent (benchmark) | 3.76% | 0.438 | −18.5% | 9.50% | 0.659 | −12.6% |
| **A. Deployed** | 4.35% | 0.331 | −31.5% | 14.36% | 0.728 | −16.6% |
| B. + posture-mapping fix | 4.26% | 0.328 | −31.1% | 14.12% | 0.714 | −16.6% |
| C. + correlation-shrinkage covariance | 4.29% | 0.328 | −31.4% | 14.50% | 0.739 | −16.6% |
| D. B + C | 4.20% | 0.325 | −31.0% | 14.29% | 0.727 | −16.6% |
| F. B + uncapped on risk-off transitions | 4.20% | 0.366 | −25.0% | 12.60% | 0.621 | −16.6% |
| G. B + market-cap prior | 9.20% | 0.677 | −25.9% | 9.46% | 0.415 | −24.6% |
| **X. B + excess-return fix** | 4.89% | **0.505** | **−21.1%** | 9.14% | 0.561 | −14.1% |

**Full period 2014–2026:** Permanent 5.75% / 0.515 / −18.5% · Deployed 7.80% / 0.475 / −31.5% · **X 6.30% / 0.514 / −21.1%** (vol 9.0% vs deployed 13.8%).

Reading:

- B, C, D: correctness fixes with no performance effect. Do them because the code should do what the documentation says.
- F and G: large swings in opposite directions across windows — the same regime-dependent pattern the project keeps finding. Not adopted.
- **X**: the only change that is (a) required by the theory, not chosen, and (b) more balanced across windows. It beats Permanent in-sample, trails it out-of-sample, and ties it over 12 years at much lower risk than the deployed version. What remains is "the anchor plus small tilts" — what Black-Litterman with weak views is supposed to produce.

---

## 5. What this means for the project's conclusions

- **The README's "+0.93%/yr alpha over a risk-matched Permanent" is not attributable to the signals.** It comes from a volatility/commodity tilt created by §2.1. This should be stated explicitly. It is a more valuable finding than the alpha claim was.
- **The IS/OOS reversal is now explained,** rather than merely observed: one accidental structural tilt, paid in one window and punished in the other.
- **ANALYSIS_V2's original finding stands:** ATLAS ≈ levered Permanent with ~zero alpha. Correctly specified, it converges to Permanent at similar risk.
- **The methodology still did its job.** Every symptom that pointed here — GMV winning 239/239 weeks at A = 5, cash acting as a "variance anchor", only 1 of 22 assets beating cash, self-calibrating δ failing in both windows — was recorded honestly along the way. The root cause sat in one line connecting them.

---

## 6. Improvement plan, prioritised

### Tier 0 — operational, this week (no strategy change)

1. **Pin `runs-on: ubuntu-24.04`** before Saturday's run.
2. **Pin dependencies:** generate `requirements.lock.txt`, install from it in CI.
3. **Fix `resolve_as_of`** (weekday rule, §2.2), with a test asserting a Saturday run decides for the Friday just closed.
4. Bump the GitHub Actions versions (Node 20 deprecation).

### Tier 1 — correctness (changes the strategy; re-validate)

5. **Fix the excess-return specification (§2.1).** Π is an excess return: drop the cash-prior override, run Black-Litterman on excess returns, and either optimise in excess space with rf = 0 or add rf back before max-Sharpe and the gate. Add a regression test: *with no views, the forward step returns the anchor* (within the effect of the 20% cap).
6. **Fix the posture mapping (§2.3a):** rescale empirical ranks to the 0–2 target scale. Test: with two visited states, the high-vol/low-return state maps to risk_off.
7. **Correlation-shrinkage covariance (§2.4).**
8. **Align the live and validated anomaly configs (§2.5).**
9. **Re-run notebook 09 and notebook 11** on the corrected pipeline, both windows, and rewrite the README results honestly, including §5.
10. **Restart the live track record.** It is one week old, so now is the cheapest moment there will ever be. Delete `live/state.json` and let the next run re-fund. With a 2% cap, trading from the old five-asset book to the corrected diversified book would otherwise take ~40 weeks.

### Tier 2 — portfolio quality (after Tier 1, IS-only discipline)

With Black-Litterman correctly specified, the levers on returns become explicit and defensible, instead of hidden inside a bug:

11. **View strength and confidence.** The regime tilt (3%) and τ were set while views were swamped by the double subtraction; they now carry real weight. Calibrate on IS only, freeze, run OOS once.
12. **Make any strategic tilt deliberate.** If you want commodity exposure, express it as an explicit, sized, justified view — not an accident of the prior. The 2022–2026 window shows such a tilt can pay; 2014–2021 shows the cost.
13. **Risk-contribution cap.** No single asset above ~25% of portfolio variance (the live book had 42% in silver). A risk-control policy, like the turnover cap — not a return lever.
14. **Separate "noise turnover" from "regime turnover"** only if the regime signal becomes leading rather than lagging. The current HMM on daily return/volatility confirms crashes after the fact (§2.3c). Leading candidates already in your stack: the FRED macro K-Means (yield-curve slope, unemployment trend), or credit spreads. Until then, accept V1 as a risk-on tilt and drop the defensive claim from the write-up.
15. **Widen the universe** — still the largest unfulfilled promise of `PROJECT_STRUCTURE.md` and the only change that raises the ceiling rather than redistributing under it.
16. **Drop USO from the universe or replace it with a broad commodity index.** It was held 78% of weeks by the deployed strategy and carries severe roll decay. Note it only became attractive *because of* §2.1.

### Tier 3 — engineering

17. Net-of-cost live NAV in `history.csv`.
18. Round weights at write time; show run date in the report header.
19. Cache `data/raw` in Actions to avoid a cold Yahoo download every week.
20. Implement the quarterly re-screen the config promises, or remove `rescreen_frequency`.

---

## 7. Validation-discipline note

The 2022–2026 window has now been observed so many times that no result on it is out-of-sample in any meaningful sense. **The live record is the only virgin test left.** Make the Tier 1 fixes now, freeze the config, restart the record, and do not tune anything on live weeks. A year of untouched live weeks will say more about this strategy than any further backtest.

---

*Harness: `allocation.equilibrium_returns`, `views.regime_view/assemble`, `allocation.black_litterman/max_sharpe/gmv/risk_parity/utility_select`, `regimes.market_regime`, over `data/raw/prices_…_2010-01-01_latest_20260811.parquet` and `macro_DTB3_…`, 658 week-end decision dates. Backtest loop replicates `backtest.run` (decision at week-end close, execution next trading day, 0.5% band, 2% cap with funding exemption, 10 bp on executed turnover).*
