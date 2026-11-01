# Re-selection: results and recommendation

**Pre-registration:** `docs/RESELECTION_PREREG.md`, commit `25bdcf2`
(written and committed before any result in this task existed).

**Scope:** re-select, IS-only (2010-2021), the four settings a
leakage audit found were originally chosen by looking at 2022-2026
performance (gate risk-aversion `A`, turnover cap/band, V3 on/off, V5
on/off), plus the never-done Phase C risk-off-target choice. One OOS
look at the end, exactly once, per the pre-registered commitment.
**`config/config.yaml`, `config/regime_posture.yaml`, and `live/` were
never touched by this task** -- the live book traded the current
config throughout, including one real Saturday-cron funding run
(`7a11d75`, decision 2026-10-30) that landed mid-task with
`config_changed: false` in its diagnostics, confirming no drift.

## Recommendation: **keep the current config.** Do not adopt Phase 2's selected changes.

Phase 2's IS-only re-selection (V5 on, turnover 1%/0.5%) looked better
than the current config on the 12 in-sample folds -- that is exactly
why it was selected. But the one OOS look that was supposed to validate
it shows the opposite: Phase 2's config **underperforms both the
current config and the Permanent Portfolio** in both the OOS
(2022-2026) and full (2014-2026) windows, on excess Sharpe, by a
noticeable margin in point estimates:

| Window | Current | Phase 2 | Permanent |
| --- | --- | --- | --- |
| OOS 2022-2026 | **0.579** | 0.458 | **0.620** |
| Full 2014-2026 | **0.529** | 0.497 | 0.507 |

Phase 2 is the worst of the three in both windows. None of the
pairwise differences are statistically significant (see Sec 5 below),
so this is not "Phase 2 is proven worse" -- it is "the IS-only
selection process, done carefully and pre-registered, did not produce
an OOS improvement, and on these point estimates trended the wrong
way." That is itself the honest, useful result: it shows the original
selection bias mattered for V5/turnover (Sec 3), and it shows that
correcting it with a cleaner IS-only process does not, by itself,
hand you a better live strategy -- a result this project's own
validation-discipline note (ANALYSIS_V4.md Sec 7) effectively
predicted.

---

## 1. What was re-selected

| Setting | Current (unchanged) | IS-only decision |
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
144-cell grid's joint argmax (see the caveat in Sec 2).

## 2. IS evidence (Phase 2, commit `33074b2`)

Full grid: `reports/results/reselection_phase2_grid.csv` (144 rows);
per-cell fold Sharpe: `reselection_phase2_fold_sharpe.csv`; the
mechanical decisions: `reselection_phase2_decisions.json`.

| Setting | Current stability | Candidate stability | Fold wins | Verdict |
| --- | --- | --- | --- | --- |
| V3 on | -0.3327 | -0.1669 | 6/12 | current kept (not a majority) |
| V5 on | -0.3327 | **-0.1835** | **7/12** | **adopted** |
| risk_off bonds_cash | -0.3327 | -0.3334 | 3/12 | current kept |
| risk_off bonds_gold | -0.3327 | -0.3332 | 4/12 | current kept |
| gate A=1 | -0.3327 | -0.3156 | 3/12 | current kept |
| gate A=3 | -0.3327 | -0.2635 | 6/12 | current kept (not a majority) |
| gate A=5 | -0.3327 | -0.4009 | 6/12 | current kept (worse stability) |
| turnover 1%/0.5% | -0.3327 | **-0.3230** | **8/12** | **adopted** |
| turnover 4%/0.5% | -0.3327 | -0.3427 | 4/12 | current kept |

**Caveat, stated plainly:** the full grid's joint best cell (V3 on, V5
on, gate A=5, turnover 4%/0.5%) scores far better on IS stability
(-0.013) than any one-setting-at-a-time result. It was **not**
adopted -- exactly the interaction-effect overfitting risk the
pre-registration (Sec 3) flagged and ruled out in advance as the
selection mechanism, before this number existed. Phase 4's result
below is a reminder of why that caution was right: even the much more
conservative one-setting-at-a-time winner did not survive contact with
OOS data.

## 3. IS vs. the original OOS-era ablation: how much did the bias matter?

| Setting | OOS-era choice | IS-only re-selection | Agree? |
| --- | --- | --- | --- |
| V3 | disabled (-0.265 marginal OOS Sharpe, stage-11 ablation) | keep off | **agree** |
| V5 | disabled (-0.236 marginal OOS excess Sharpe, ANALYSIS_V2 Step 2) | turn on | **disagree** |
| Gate `A` | recalibrated 5 -> 2 (notebook 11 gate ablation) | keep 2 | **agree** |
| Turnover | relaxed 1%/1% -> 2%/0.5% (stage 11, notebook 11 turnover sensitivity) | 1%/0.5% | **disagree** (same band, tighter cap) |
| Risk-off target | never set (Phase C is new) | keep legacy | n/a -- no prior decision to compare |

Two of four historical decisions hold up under an honest IS-only
process (V3, gate `A`) -- not everything in this project's history was
cherry-picked. Two (V5, turnover) do not -- these are exactly where
the original OOS-era selection bias most likely mattered. Phase 4
shows that "correcting" those two via IS-only re-selection does not,
on its own, produce a better OOS result either (Sec 5) -- the honest
conclusion is that this project's free parameters are not well
identified by either window alone, not that the IS-only numbers are
secretly the "true" ones.

## 4. Phase 3 sensitivity (diagnostic only, commit `b37000c`; nothing here was adopted)

Full table: `reports/results/reselection_phase3_sensitivity.csv`.

- **`tau` is mathematically inert in this implementation.** `views.
  assemble` builds Omega as `diag(P_i @ (tau*Sigma) @ P_i^T /
  confidence_i)`, so tau multiplies both terms of the Black-Litterman
  posterior precision identically and cancels exactly in closed form.
  Confirmed empirically: identical stability score and excess Sharpe
  (0.4886) at tau in {0.025, 0.05, 0.1}, to four decimal places. Not a
  "no sensitivity" finding in the usual sense -- `black_litterman.tau`
  currently does nothing, regardless of its configured value.
- **`delta`** and **`per_asset_cap`** each show the grid's largest
  single-step moves, flagged for review rather than auto-adopted:
  - `delta` 2.5 -> 3.5: excess Sharpe +0.044, but stability score
    *worsens* -- the two metrics disagree on direction across the
    whole grid, suggesting delta is weakly identified by this sample.
  - `per_asset_cap` 0.15 -> 0.20 (the step toward today's actual cap):
    excess Sharpe -0.034 (the single largest move in the table), while
    stability score improves in that same direction.
- `regime_view_magnitude` is smooth and monotonic in both metrics
  across {0.015, 0.03, 0.05} -- no cliff.

## 5. Phase 4: the single OOS look (commit `56503bb`)

Run once each, `backtest.run` directly, current live config's
V2/anomaly settings ON (what is actually trading), no technical
failures or reruns needed.

| Window | Config | Ann. return | Ann. vol | Excess Sharpe | Max DD | Sharpe SE |
| --- | --- | --- | --- | --- | --- | --- |
| OOS 2022-2026 (4.82y) | current | 9.98% | 10.52% | 0.579 | -15.89% | 0.492 |
| OOS 2022-2026 | **phase2** | 8.13% | 9.34% | 0.458 | -12.71% | 0.479 |
| OOS 2022-2026 | permanent | 9.42% | 8.72% | 0.620 | -13.39% | 0.497 |
| Full 2014-2026 (12.82y) | current | 7.02% | 10.12% | 0.529 | -23.15% | 0.298 |
| Full 2014-2026 | **phase2** | 5.94% | 8.44% | 0.497 | -20.48% | 0.296 |
| Full 2014-2026 | permanent | 5.94% | 8.23% | 0.507 | -19.66% | 0.297 |

**Sharpe SE** uses `sqrt((1 + SR^2/2) / years)` (requested explicitly
for this task), not README's simpler `0.5/sqrt(years)` convention --
notice how large it is (0.48-0.50 OOS, ~0.30 full) relative to every
gap in the table above. None of these point estimates should be read
as a confident ranking.

**Paired test (Newey-West HAC, weekly return differences):**

| Window | Comparison | Mean diff (weekly) | t-stat | p-value |
| --- | --- | --- | --- | --- |
| OOS | phase2 - permanent | -0.00021 | -0.616 | 0.538 |
| OOS | current - phase2 | +0.00034 | 0.842 | 0.400 |
| Full | phase2 - permanent | +0.0000057 | 0.024 | 0.981 |
| Full | current - phase2 | +0.00022 | 0.829 | 0.407 |

No pairwise difference clears conventional significance. Current
numerically beats phase2 in both windows; phase2 numerically trails
permanent OOS and is statistically indistinguishable from it over the
full period.

**Deflated Sharpe Ratio for phase2** (Bailey & Lopez de Prado 2014),
`N_trials = 144` (Phase 2's pre-registered grid) `+ 26` (an honest
count of this project's PRIOR OOS looks, Sec 6 below) `= 170`,
`sharpe_std_across_trials = 0.116` (the empirical cross-trial std of
excess Sharpe across Phase 2's own 144-cell grid):

| Window | DSR (daily n_obs) | DSR (weekly n_obs) |
| --- | --- | --- |
| OOS | 0.99999895 | **0.9846** |
| Full | 1.0 | **0.99999874** |

**Read this carefully -- it does not mean phase2 is validated.** DSR
answers "is phase2's own Sharpe ratio distinguishable from the best of
170 skill-less trials' expected Sharpe (`SR_0 ≈ 0.315`, derived from
the empirical trial variance)?" -- a much weaker claim than "does
phase2 beat Permanent." Permanent's own excess Sharpe (0.51-0.62) also
clears `SR_0` comfortably, by the same logic, so a high DSR for phase2
here mostly just confirms that a diversified multi-asset book reliably
beats cash by a real margin over a decade-plus -- unsurprising, and
not evidence that phase2's specific changes earn their complexity
relative to Permanent. The paired test above is the right tool for
that question, and it says no. The daily-`n_obs` DSR is reported only
for completeness; it should be discounted further still, since daily
returns from a weekly-rebalanced strategy are autocorrelated and the
DSR's own variance formula assumes i.i.d. observations at the
frequency given -- the weekly-`n_obs` version is the more defensible
of the two, and it is still this high.

## 6. Historical prior-OOS-look count (for `N_trials`)

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
| **Total** | | **26** |

This count involves judgment calls (e.g. whether a reused baseline
counts again) and is deliberately conservative (biased toward
counting more, not fewer, trials, which makes the DSR more
conservative too). `N_trials_total = 144 (Phase 2, pre-registered) +
26 (above) = 170`.

## 7. Which README OOS numbers are selection-contaminated

Stated plainly, per the task's instruction: README's **"ANALYSIS_V4:
the corrected numbers"** table (the project's current headline,
excess Sharpe 0.547 full-period) is built from a config whose gate
risk-aversion, turnover convention, and V3/V5 module flags were
themselves chosen by looking at OOS performance across the episodes in
Sec 6 above -- i.e. **the headline number in README was not produced
by a configuration selected independently of the window it is
measured on.** This task does not change that number (Sec 5's `current`
row reproduces it, V2/anomaly ON, matching what is actually live) --
it only establishes, separately and pre-registered, that an IS-only
re-derivation of the same four settings does not recover a better OOS
result either. Section 5's `current` excess Sharpe (0.529 full) is
close to but not identical to README's 0.547, because README's number
there is measured with V2/anomaly OFF (to match ANALYSIS_V4.md's own
harness) while this task's Phase 4 run keeps them ON, per this task's
explicit instruction, to match what is genuinely trading live.

## 8. Git log of this task

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
```

(Two unrelated live-automation commits, `8a84408` and `7a11d75`,
landed between these from the normal Saturday/Monday cron while this
task was running; neither touched `config/` and both confirm
`config_changed: false`, i.e. no drift from this task.)

## 9. What was NOT done, per instruction

`config/config.yaml`, `config/regime_posture.yaml`, and the live
record were not modified. The README results and Strategy
Explanation.md were not touched. Nothing here is adopted automatically
-- this document and its recommendation (Sec "Recommendation" above)
are for you to act on or not.
