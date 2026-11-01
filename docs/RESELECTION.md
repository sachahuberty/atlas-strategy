# Re-selection: results and recommendation

**Pre-registration:** `docs/RESELECTION_PREREG.md`, commit `25bdcf2`
(written and committed before any result in this task existed; frozen
and never edited, including by the correction task below).

**Scope:** re-select, IS-only (2010-2021), the four settings a
leakage audit found were originally chosen by looking at 2022-2026
performance (gate risk-aversion `A`, turnover cap/band, V3 on/off, V5
on/off), plus the never-done Phase C risk-off-target choice. One OOS
look at the end, exactly once, per the pre-registered commitment.
**`config/config.yaml`, `config/regime_posture.yaml`, and `live/` were
never touched by this task** -- the live book traded the current
config throughout, including two real cron runs that landed mid-task
(`7a11d75` Saturday funding, decision 2026-10-30; a later Monday
catch-up) both confirming `config_changed: false` in their
diagnostics.

**Correction notice (external review, after the first version of this
document):** the original Phase 4 deflated Sharpe ratio had a units
bug (an annualized Sharpe ratio mixed with a per-observation `n_obs`),
and the original recommendation leaned on the 2022-2026 result as its
stated reason, which the pre-registration's own spirit does not
support -- a decision reasoned from the window you are simultaneously
treating as contaminated is itself contaminated. Both are fixed below.
**Nothing from Phases 1-3 changed. The Phase 2 selection outcome
itself is unchanged.** Only the Phase 4 arithmetic and the stated
grounds for the final recommendation are corrected.

## Recommendation: **keep the current config.**

See Section 4 below for why -- **on IS grounds, not because of the
2022-2026 result.** This is a deviation from the pre-registered rule
(Section 3 found V5 and the 1% turnover cap both pass it), decided
after seeing the OOS look, and it is counted as such (Section 8,
`N_trials = 171`).

---

## 1. What was re-selected

| Setting | Current (unchanged) | IS-only decision (pre-registered rule) |
| --- | --- | --- |
| V3 (technical view) | off | **keep off** |
| V5 (momentum view) | off | **turn on** |
| Risk-off target (Phase C, new) | legacy (fixed_income + commodity) | **keep legacy** |
| Gate risk aversion `A` | 2 | **keep 2** |
| Turnover cap / no-trade band | 2% / 0.5% | **1% / 0.5%** |

Rule applied mechanically (docs/RESELECTION_PREREG.md Sec 3): keep the
current value unless a challenger beats it on `stability_score`
(`mean(fold values) - penalty*std(fold values)`, 12 annual IS folds,
2010-2021) AND wins a strict majority of those folds. One setting at a
time, holding the other four at their current value -- not the full
144-cell grid's joint argmax (see the caveat in Sec 2). **This
section is unchanged from the pre-registered rule's output -- the
rule still says adopt V5 and the 1% cap.** Section 4 explains why they
are not being adopted anyway.

## 2. IS evidence (Phase 2, commit `33074b2`)

Full grid: `reports/results/reselection_phase2_grid.csv` (144 rows);
per-cell fold Sharpe: `reselection_phase2_fold_sharpe.csv`; the
mechanical decisions: `reselection_phase2_decisions.json`.

| Setting | Current stability | Candidate stability | Fold wins | Verdict |
| --- | --- | --- | --- | --- |
| V3 on | -0.3327 | -0.1669 | 6/12 | current kept (not a majority) |
| V5 on | -0.3327 | **-0.1835** | **7/12** | rule says adopt (not adopted -- Sec 4) |
| risk_off bonds_cash | -0.3327 | -0.3334 | 3/12 | current kept |
| risk_off bonds_gold | -0.3327 | -0.3332 | 4/12 | current kept |
| gate A=1 | -0.3327 | -0.3156 | 3/12 | current kept |
| gate A=3 | -0.3327 | -0.2635 | 6/12 | current kept (not a majority) |
| gate A=5 | -0.3327 | -0.4009 | 6/12 | current kept (worse stability) |
| turnover 1%/0.5% | -0.3327 | **-0.3230** | **8/12** | rule says adopt (not adopted -- Sec 4) |
| turnover 4%/0.5% | -0.3327 | -0.3427 | 4/12 | current kept |

**Caveat, stated plainly:** the full grid's joint best cell (V3 on, V5
on, gate A=5, turnover 4%/0.5%) scores far better on IS stability
(-0.013) than any one-setting-at-a-time result. It was **not**
adopted -- exactly the interaction-effect overfitting risk the
pre-registration (Sec 3) flagged and ruled out in advance as the
selection mechanism, before this number existed.

## 3. IS vs. the original OOS-era ablation: how much did the bias matter?

| Setting | OOS-era choice | IS-only re-selection | Agree? |
| --- | --- | --- | --- |
| V3 | disabled (-0.265 marginal OOS Sharpe, stage-11 ablation) | keep off | **agree** |
| V5 | disabled (-0.236 marginal OOS excess Sharpe, ANALYSIS_V2 Step 2) | turn on | **disagree** |
| Gate `A` | recalibrated 5 -> 2 (notebook 11 gate ablation) | keep 2 | **agree** |
| Turnover | relaxed 1%/1% -> 2%/0.5% (stage 11, notebook 11 turnover sensitivity) | 1%/0.5% | **disagree** (same band, tighter cap) |
| Risk-off target | never set (Phase C is new) | keep legacy | n/a -- no prior decision to compare |

Two of four historical decisions hold up under an honest IS-only
process (V3, gate `A`). Two (V5, turnover) do not. Section 4 digs
into exactly *why* V5 and turnover pass the pre-registered rule, and
finds the rule itself has real weaknesses that this specific case
exposes.

## 4. Deviation from pre-registration (post-hoc, written after the OOS look)

**The decision is to keep the current config.** The stated grounds
are IS-only, from the post-hoc diagnostic below (`reports/results/
reselection_posthoc_is_diagnostic.csv`, `src/atlas/reselection.
fold_delta_diagnostic`) -- **not** the 2022-2026 result in Section 7.

| Challenger | Mean delta, all 12 folds | Wins/losses/ties, all | 2010-2013 wins | 2010-2013 mean delta | 2014-2021 wins (of 8) | 2014-2021 mean delta |
| --- | --- | --- | --- | --- | --- | --- |
| V5 on | **+0.049** | 7/4/1 | 3/4 | **+0.257** | **4/8** | **-0.056** |
| turnover 1% | +0.020 | 8/4/0 | 3/4 | +0.030 | 5/8 | +0.016 |
| V3 on | +0.138 | 6/6/0 | 1/4 | +0.063 | 5/8 | +0.175 |
| risk_off bonds_cash | -0.006 | 3/4/5 | 1/4 | -0.006 | 2/8 | -0.006 |
| risk_off bonds_gold | -0.000 | 4/3/5 | 1/4 | -0.001 | 3/8 | +0.000 |
| gate A=1 | +0.031 | 3/1/8 | 2/4 | +0.069 | 1/8 | +0.012 |
| gate A=3 | +0.058 | 6/3/3 | 3/4 | +0.081 | 3/8 | +0.047 |
| gate A=5 | +0.082 | 6/6/0 | 3/4 | +0.513 | 3/8 | -0.134 |
| turnover 4% | -0.033 | 4/7/1 | 1/4 | -0.025 | 3/8 | -0.038 |

**V5's entire IS win comes from the 2010-2013 cold-start years.** Over
2010-2013 turning V5 on adds +0.257 average excess Sharpe per fold
(3 of 4 years won); over 2014-2021 -- the years that are not a strategy
warm-up artifact -- it **loses** on average (-0.056, winning only 4 of
8). The all-years verdict (+0.049, 7/12) is driven almost entirely by
four unrepresentative years. The 1% turnover cap's all-years edge
(+0.020, "8/12 wins") is also small against the fold-to-fold noise
visible everywhere in this table (several challengers show swings of
±0.1-0.5 from one fold to the next) -- it survives into 2014-2021
(+0.016, 5/8) but the margin is noise-sized, not a result you would
bet the live book on.

The cold-start mechanism itself, documented precisely
(`reports/results/reselection_posthoc_cold_start.csv`,
`src/atlas/reselection.cold_start_diagnostic`, computed directly from
IS-only price data, no OOS access, no strategy backtest):

| Year | Decisions | Posture forced to neutral (insufficient HMM history) | Covariance window < `lookback_days` |
| --- | --- | --- | --- |
| 2010 | 52 | **12** | **52** |
| 2011 | 52 | 0 | **52** |
| 2012 | 52 | 0 | **52** |
| 2013 | 52 | 0 | 0 |
| 2014-2021 | 420 | 0 | 0 |

2010-2012 are not measuring the strategy that trades live: every
decision in those three years is estimated on a covariance window
shorter than `optimization.lookback_days` (756 days / ~3 years), and
2010 additionally has 12 of 52 weeks where the HMM lacked enough
trailing history and the posture was forced to neutral regardless of
what the market was actually doing. 2013 is the first year with a
full-length covariance window. The fold scheme (docs/
RESELECTION_PREREG.md Sec 1) includes 2010-2013 because that is the
same fold scheme notebook 09's own grid searches already used -- it
was the right choice to reuse exactly, as pre-registered, but it means
4 of this task's 12 "IS years" are, for V5 and the turnover cap,
measuring a cold-start artifact as much as a genuine signal.

**This deviation was decided after seeing 2022-2026.** A skeptical
reader should treat it as OOS-informed for exactly that reason, and it
is counted in `N_trials = 171` (Section 8) on that basis -- the keep-
current decision is itself one more configuration this project chose
with knowledge of OOS performance, even though the stated grounds
above are IS-only and would have been available before Phase 4 ran.

## 5. Weaknesses of the pre-registered rule

Honest, not defensive -- the rule (docs/RESELECTION_PREREG.md Sec 3)
has real gaps this specific case exposed:

- **No minimum size of improvement.** "Beats current on stability
  score" accepts a win of any size, including the noise-level +0.02
  the 1% turnover cap shows.
- **The strict-majority-of-folds test over 12 noisy annual folds
  admits noise-level wins and rejected the single largest IS
  improvement measured anywhere in this task.** V3's +0.138 mean delta
  (+0.175 over 2014-2021 alone) is larger than V5's or the turnover
  cap's, in both the full and the warm-years column -- but it only won
  6 of 12 folds (a tie with its losses), so the rule correctly keeps
  it off by its own strict-majority requirement. The rule is not
  biased toward adopting large improvements; it is sensitive to the
  SHAPE of the fold-by-fold win pattern, which can reject a bigger,
  more consistent-looking improvement (V3, 5/8 warm-year wins, losing
  years still modest) while accepting a smaller, more erratic one
  (V5, 4/8 warm-year wins net negative, offset by an extreme cold-start
  read) just because the overall count crossed the majority line.
- **Warm-up years (2010-2013) were included as folds on equal footing
  with 2014-2021**, despite Section 4's finding that they are
  measuring cold-start mechanics, not strategy skill, for at least
  three of the nine challengers tested.
- **Phase C had almost no statistical power.** 5 of the risk-off
  target's 12 folds are identical by construction (`risk_off` posture
  never fired in those years under ANY target, so `bonds_cash`/
  `bonds_gold` tie `legacy` exactly) -- an alternative needed
  near-unanimous wins on the remaining 7 folds to clear a strict
  majority of 12, which is a much harder bar than the rule's wording
  suggests at a glance.

**For a future re-selection:** restrict folds to 2014-2021 (or
explicitly exclude any year `cold_start_diagnostic` flags), and add a
materiality threshold (e.g. require the stability-score margin to
exceed some multiple of the cross-fold standard deviation), not just a
sign and a majority count.

## 6. V3 (technical view): open question, NOT adopted

V3's IS evidence is the most favourable of anything measured in this
task (+0.138 mean delta, +0.175 over 2014-2021 alone, the largest
clean -- non-cold-start-driven -- improvement in Section 4's table).
**It stays off anyway.** Adopting it now, after looking at this
specific number, would be exactly the kind of after-the-fact
cherry-picking this whole project exists to avoid -- the fact that it
is a *good* number does not make deciding to act on it post-hoc any
less contaminated than Section 4's keep-current decision. Any future
test of V3 must be pre-registered with the improved rule from Section
5 (2014-2021-only folds, a materiality threshold) before any result
exists, and only the live/forward record from 2026-10-23 onward can
evaluate it without this problem recurring.

## 7. Phase 4: the single OOS look -- corrected statistics

Run once each, `backtest.run` directly, current live config's
V2/anomaly settings ON (what is actually trading), no technical
failures or reruns needed. These non-DSR numbers are unchanged from
the original version of this document (recomputed from the same
committed pickle by `scripts/reselection_phase4_stats.py`, matching to
1e-6 -- see that script's own reproduction-check output).

| Window | Config | Ann. return | Ann. vol | Excess Sharpe | Max DD | Sharpe SE |
| --- | --- | --- | --- | --- | --- | --- |
| OOS 2022-2026 (4.82y) | current | 9.98% | 10.52% | 0.579 | -15.89% | 0.492 |
| OOS 2022-2026 | **phase2** | 8.13% | 9.34% | 0.458 | -12.71% | 0.479 |
| OOS 2022-2026 | permanent | 9.42% | 8.72% | 0.620 | -13.39% | 0.497 |
| Full 2014-2026 (12.82y) | current | 7.02% | 10.12% | 0.529 | -23.15% | 0.298 |
| Full 2014-2026 | **phase2** | 5.94% | 8.44% | 0.497 | -20.48% | 0.296 |
| Full 2014-2026 | permanent | 5.94% | 8.23% | 0.507 | -19.66% | 0.297 |

**Sharpe SE** uses `sqrt((1 + SR^2/2) / years)`, not README's simpler
`0.5/sqrt(years)` convention -- 0.48-0.50 (OOS), ~0.30 (full), large
relative to every gap above.

**Paired test (Newey-West HAC, weekly return differences), unchanged:**

| Window | Comparison | Mean diff (weekly) | t-stat | p-value |
| --- | --- | --- | --- | --- |
| OOS | phase2 - permanent | -0.00021 | -0.616 | 0.538 |
| OOS | current - phase2 | +0.00034 | 0.842 | 0.400 |
| Full | phase2 - permanent | +0.0000057 | 0.024 | 0.981 |
| Full | current - phase2 | +0.00022 | 0.829 | 0.407 |

No pairwise difference clears conventional significance.

### Erratum: deflated Sharpe ratio

**The DSR values in the original version of this document were wrong
-- a units bug, not a formula bug.** `metrics.deflated_sharpe_ratio`
mixed an ANNUALIZED Sharpe ratio (e.g. 0.458) with a PER-OBSERVATION
`n_obs` (252 weeks), which collapsed `sigma(SR_hat)` to a tiny value
and made a modest edge look like near-certainty (originally reported:
~0.985 OOS, ~1.0 full). Fixed in `metrics.py` (`periods_per_year` is
now a required argument on `deflated_sharpe_ratio`/
`probabilistic_sharpe_ratio`, so the old call pattern can no longer
silently mix units -- it raises `TypeError`). Corrected, using WEEKLY
units (the decision frequency -- the headline; daily is kept only as a
consistently-computed secondary figure, since daily returns under
weekly rebalancing are autocorrelated):

| Window | DSR, N=170 (as originally reported) | DSR, N=171 (headline, Sec 8) | PSR, no trial correction |
| --- | --- | --- | --- |
| OOS 2022-2026 | 0.6235 | **0.6233** | 0.8425 |
| Full 2014-2026 | 0.7428 | **0.7425** | 0.9624 |

(Superseded values, for the record, not deleted: daily-units-mixed
DSR was 0.9846390529 OOS / 0.9999955191 full -- see `reports/results/
reselection_phase4_statistics.json`'s `superseded_dsr_units_bug` key
for the exact old numbers and this same erratum.)

**Plain-language reading:** after accounting for ~171 configurations
this project has effectively compared using OOS data (Section 8), the
evidence that the strategy's 2022-2026 Sharpe ratio is reliably above
zero is weak (DSR ~0.62, barely above a coin flip once the multiple-
comparisons correction is applied) -- not the near-certainty the
original, buggy number implied. No individual setting examined in
this task has robust evidence either way once that correction and the
pre-registered rule's own limits (Section 5) are both accounted for.
**ATLAS is, on the evidence gathered across this whole task,
statistically indistinguishable from the Permanent Portfolio across
every configuration examined.**

## 8. What is clean

- **2022-2026 is selection-contaminated.** 26 prior OOS looks (Section
  9 below) plus this task's own post-hoc keep-current decision
  (Section 4) make 27 configurations compared against this window
  before this document was written, on top of Phase 2's own
  pre-registered 144-cell grid (`N_trials = 171` total, Section 9).
- **The live record from 2026-10-23 onward is the only untouched
  test.** It has not been looked at to make any decision in this
  project's history, this task included.

## 9. Historical prior-OOS-look count (for `N_trials`)

An honest tally of distinct configurations this project compared using
OOS data before this task, cited to source:

| Episode | Source | Trials |
| --- | --- | --- |
| Original notebook 11 build: module ablation (baseline + 5: regime/meanrev/technical/momentum/anomaly off; V4/sentiment is a structural no-op, not counted separately) | notebook 11, `module_results` dict | 6 |
| Original notebook 11 build: turnover sensitivity (2%, 4%; 1% is the shared baseline, not double-counted) | notebook 11, `turnover_results` dict | 2 |
| Original notebook 11 build: gate ablation (A=5, A=2, A=10, off, vol_floor) | notebook 11, `gate_results` dict | 5 |
| ANALYSIS_V2 Step 2: modernized ablation re-run (V1/V2/V3/V5/anomaly re-measured on the corrected harness; V4 structural no-op) | README changelog item 12 | 6 |
| ANALYSIS_V3 Sec 4/6.2: self-calibrating delta vs. fixed | README changelog item 17 | 2 |
| ANALYSIS_V4 Sec 2.1 Phase 2 (this project's EARLIER remediation, not this task): arms A/B/C/X/ALL | README changelog item 18, commit `2ee5944` | 5 |
| Subtotal (as originally reported, N=170 with Phase 2's 144) | | **26** |
| **This task's own post-hoc keep-current decision (Section 4)** -- decided after seeing 2022-2026, so it counts the same way every episode above does | docs/RESELECTION.md Sec 4 | **+1** |
| **Total** | | **27** |

`N_trials_total = 144 (Phase 2, pre-registered) + 27 (above) = 171`
(headline). The N=170 figure from the original version of this
document is kept alongside every corrected statistic in Section 7 for
comparison, not deleted. This count involves judgment calls (e.g.
whether a reused baseline counts again) and is deliberately
conservative (biased toward counting more, not fewer, trials, which
makes the DSR more conservative too).

## 10. Which README OOS numbers are selection-contaminated

Stated plainly: README's **"ANALYSIS_V4: the corrected numbers"**
table (the project's current headline, excess Sharpe 0.547
full-period) is built from a config whose gate risk-aversion, turnover
convention, and V3/V5 module flags were themselves chosen by looking
at OOS performance across the episodes in Section 9 above -- i.e.
**the headline number in README was not produced by a configuration
selected independently of the window it is measured on.** This task
does not change that number (Section 7's `current` row reproduces it,
V2/anomaly ON, matching what is actually live) -- it only establishes,
separately and pre-registered, that an IS-only re-derivation of the
same four settings does not recover a better OOS result either, and
(Section 4) that even the pre-registered rule's own output should not
be adopted once its cold-start sensitivity is accounted for.

## 11. Git log of this task

```
7138c1f reselection: extract the IS fold/stability-score primitives into selection.py
25bdcf2 reselection: pre-registration                              <- PRE-REGISTRATION COMMIT
c2cee45 reselection: Phase C risk-off target + efficient IS re-selection harness
36b0782 reselection: retry save_checkpoint through transient OneDrive locks
33074b2 reselection: Phase 2 IS-only re-selection results
931e91c reselection: add with_delta for Phase 3's delta sensitivity sweep
b37000c reselection: Phase 3 sensitivity analysis (diagnostic only)
e79209d metrics: add Phase 4 inference tools (SE, Newey-West, deflated Sharpe)
56503bb reselection: Phase 4 results -- the single OOS look
18e2d6a reselection: final report + executed notebook
fd7c57c metrics: fix deflated_sharpe_ratio units bug (external review)
6fec1f4 reselection: commit the Phase 4 generation script + reproducible stats
18ee93b reselection: post-hoc IS diagnostic (external review Step 3)
<this commit> reselection: correct the recommendation + README labelling (external review Steps 4-7)
```

(Several unrelated live-automation commits landed between these from
the normal Saturday/Monday cron while this task was running; none
touched `config/`, and each confirmed `config_changed: false`, i.e. no
drift from this task.)

## 12. What was NOT done, per instruction

`config/config.yaml`, `config/regime_posture.yaml`,
`docs/RESELECTION_PREREG.md`, and the live record were not modified.
Nothing here is adopted automatically -- this document and its
recommendation (top of this file) are for you to act on or not.
