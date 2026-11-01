"""Reselection task Phase 4 -- reproducible statistics (external
review correction).

Recomputes every number in `reports/results/reselection_phase4_
statistics.json` from ONLY the already-committed `reselection_phase4_
daily_returns.pkl` (the single OOS look, never re-run) and the cached
DTB3 risk-free series -- no backtest is executed here, and none of
this touches a date on or after `general.oos_start_date` beyond what
that pickle already recorded.

Fixes the units bug an external review found in the original
deflated-Sharpe calculation (an annualized Sharpe ratio was mixed with
a per-observation n_obs): `metrics.deflated_sharpe_ratio`/
`probabilistic_sharpe_ratio` now require `periods_per_year` explicitly.
Weekly units are the headline; a daily variant is kept only as a
clearly-labelled secondary figure (computed in consistent daily units,
never mixed).

Reproduction discipline: every NON-DSR number is checked against the
existing committed JSON to 1e-6 before anything is written. A mismatch
stops the script with a clear report rather than silently overwriting
-- this script is a reproducibility check first, a generator second.

Usage:
    python scripts/reselection_phase4_stats.py
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from atlas import backtest, data, metrics

RESULTS_DIR = data.PROJECT_ROOT / "reports" / "results"
DAILY_RETURNS_PATH = RESULTS_DIR / "reselection_phase4_daily_returns.pkl"
PHASE2_GRID_PATH = RESULTS_DIR / "reselection_phase2_grid.csv"
OUT_PATH = RESULTS_DIR / "reselection_phase4_statistics.json"

WINDOWS = {
    "OOS_2022_2026": "2022-01-01",
    "full_2014_2026": "2014-01-01",
}
LABELS = {
    "a_current": "current", "b_phase2": "phase2", "c_permanent": "permanent",
}

N_TRIALS_PREREG = 144
N_PRIOR_OOS_LOOKS = 26
# 170, as originally reported (Phase 2's pre-registered grid + the
# cited prior OOS looks).
N_TRIALS_170 = N_TRIALS_PREREG + N_PRIOR_OOS_LOOKS
# +1: this task's own post-hoc keep/adopt decision, itself an
# OOS-informed choice (docs/RESELECTION.md Step 4a).
N_TRIALS_171 = N_TRIALS_170 + 1
N_TRIALS_HEADLINE = N_TRIALS_171

_NON_DSR_METRIC_KEYS = (
    "ann_return", "ann_vol", "excess_sharpe", "max_drawdown", "years",
    "n_days", "sharpe_se",
)
_PAIRED_KEYS = ("mean", "se", "tstat", "pvalue", "n", "maxlags")
_RTOL = 0.0
_ATOL = 1e-6


def _window_metrics(daily_returns: pd.Series, rf_series: pd.Series) -> dict:
    aligned_rf = rf_series.reindex(daily_returns.index)
    span_days = (daily_returns.index.max() - daily_returns.index.min()).days
    years = span_days / 365.25
    excess_sharpe = metrics.sharpe(daily_returns, rf=aligned_rf)
    return {
        "ann_return": metrics.ann_return(daily_returns),
        "ann_vol": metrics.ann_vol(daily_returns),
        "excess_sharpe": excess_sharpe,
        "max_drawdown": metrics.max_drawdown(daily_returns),
        "years": years,
        "n_days": len(daily_returns),
        "sharpe_se": metrics.sharpe_standard_error(excess_sharpe, years),
    }


def _weekly_returns(daily_returns: pd.Series) -> pd.Series:
    weeks = sorted(backtest.week_end_dates(daily_returns.index))
    nav = (1.0 + daily_returns).cumprod()
    nav_at_weeks = nav.reindex(weeks)
    weekly = nav_at_weeks.pct_change()
    weekly.iloc[0] = nav_at_weeks.iloc[0] - 1.0
    return weekly


def _check_reproduction(old: dict, new: dict) -> list[str]:
    """Returns a list of human-readable mismatch descriptions (empty
    if every non-DSR field matches to `_ATOL`)."""
    mismatches = []
    for window in WINDOWS:
        old_w = old.get("windows", {}).get(window, {})
        new_w = new["windows"][window]
        for cfg_key in LABELS:
            for metric_key in _NON_DSR_METRIC_KEYS:
                old_v = old_w.get(cfg_key, {}).get(metric_key)
                new_v = new_w[cfg_key][metric_key]
                if old_v is None:
                    continue
                if not np.isclose(old_v, new_v, rtol=_RTOL, atol=_ATOL):
                    mismatches.append(
                        f"{window}.{cfg_key}.{metric_key}: old={old_v!r} "
                        f"new={new_v!r}"
                    )
        for pair_key in ("paired_b_vs_c", "paired_a_vs_b"):
            old_p = old_w.get(pair_key, {})
            new_p = new_w[pair_key]
            for k in _PAIRED_KEYS:
                old_v = old_p.get(k)
                new_v = new_p[k]
                if old_v is None:
                    continue
                if not np.isclose(old_v, new_v, rtol=_RTOL, atol=_ATOL):
                    mismatches.append(
                        f"{window}.{pair_key}.{k}: old={old_v!r} "
                        f"new={new_v!r}"
                    )
    return mismatches


def _load_cached_dtb3(fred_series: str) -> pd.DataFrame:
    """Load the most recently cached DTB3 parquet under data/raw/
    directly -- never `data.download_macro` (which would attempt a
    live network fetch if no cache file happens to be dated today).
    This task recomputes statistics from already-committed/cached
    data only; it must not depend on, or risk, a fresh data pull.
    """
    candidates = sorted(
        data.RAW_DIR.glob(f"macro_{fred_series}_*_latest_*.parquet")
    )
    if not candidates:
        raise FileNotFoundError(
            f"No cached macro_{fred_series}_*.parquet found under "
            f"{data.RAW_DIR} -- this script must not fall back to a "
            "live download."
        )
    return pd.read_parquet(candidates[-1])


def main() -> int:
    cfg = data.load_config()
    daily = pd.read_pickle(DAILY_RETURNS_PATH)

    rf_raw = _load_cached_dtb3(cfg["risk_free"]["fred_series"])
    rf_series = (
        rf_raw[cfg["risk_free"]["fred_series"]] / 100.0
    ).reindex(daily["current"].index).ffill()

    phase2_grid = pd.read_csv(PHASE2_GRID_PATH)
    sharpe_std_across_trials = float(
        phase2_grid["excess_sharpe_2014_2021"].std(ddof=1)
    )

    old_json = {}
    if OUT_PATH.exists():
        with open(OUT_PATH, encoding="utf-8") as f:
            old_json = json.load(f)

    report: dict = {"windows": {}}
    for window_name, start in WINDOWS.items():
        report["windows"][window_name] = {}
        for cfg_key, label in LABELS.items():
            report["windows"][window_name][cfg_key] = _window_metrics(
                daily[label].loc[start:], rf_series
            )

        weekly_by_label = {
            k: _weekly_returns(daily[LABELS[k]].loc[start:]) for k in LABELS
        }
        common_idx = weekly_by_label["a_current"].index
        for k in weekly_by_label:
            common_idx = common_idx.intersection(weekly_by_label[k].index)

        diff_b_minus_c = (
            weekly_by_label["b_phase2"].loc[common_idx]
            - weekly_by_label["c_permanent"].loc[common_idx]
        )
        diff_a_minus_b = (
            weekly_by_label["a_current"].loc[common_idx]
            - weekly_by_label["b_phase2"].loc[common_idx]
        )
        report["windows"][window_name]["paired_b_vs_c"] = (
            metrics.newey_west_mean_test(diff_b_minus_c)
        )
        report["windows"][window_name]["paired_a_vs_b"] = (
            metrics.newey_west_mean_test(diff_a_minus_b)
        )

    # --- Reproduction check: STOP before writing anything if any
    # non-DSR number has drifted from what's already committed. ---
    mismatches = _check_reproduction(old_json, report)
    if mismatches:
        print("REPRODUCTION CHECK FAILED -- refusing to write. Mismatches:")
        for m in mismatches:
            print(" ", m)
        return 1
    print(
        f"Reproduction check passed: all {len(_NON_DSR_METRIC_KEYS) * 3 * 2}"
        " per-config metrics and all paired-test fields match the "
        "existing committed JSON to 1e-6."
    )

    # --- Corrected DSR/PSR: weekly units are the headline; daily is a
    # clearly-labelled, consistently-computed secondary figure. ---
    for window_name in WINDOWS:
        m = report["windows"][window_name]["b_phase2"]
        n_weeks = report["windows"][window_name]["paired_b_vs_c"]["n"]
        m["n_weeks"] = n_weeks

        for n_trials, suffix in ((N_TRIALS_170, "170"), (N_TRIALS_171, "171")):
            m[f"deflated_sharpe_ratio_weekly_n{suffix}"] = (
                metrics.deflated_sharpe_ratio(
                    m["excess_sharpe"], n_trials=n_trials, n_obs=n_weeks,
                    periods_per_year=52,
                    sharpe_std_across_trials=sharpe_std_across_trials,
                )
            )
            m[f"deflated_sharpe_ratio_daily_n{suffix}_secondary"] = (
                metrics.deflated_sharpe_ratio(
                    m["excess_sharpe"], n_trials=n_trials, n_obs=m["n_days"],
                    periods_per_year=252,
                    sharpe_std_across_trials=sharpe_std_across_trials,
                )
            )
        m["probabilistic_sharpe_ratio_weekly_no_trial_correction"] = (
            metrics.deflated_sharpe_ratio(
                m["excess_sharpe"], n_trials=1, n_obs=n_weeks,
                periods_per_year=52,
            )
        )

        old_b = old_json.get("windows", {}).get(window_name, {}).get(
            "b_phase2", {}
        )
        m["superseded_dsr_units_bug"] = {
            "deflated_sharpe_ratio_daily_n_obs": old_b.get(
                "deflated_sharpe_ratio_daily_n_obs"
            ),
            "deflated_sharpe_ratio_weekly_n_obs": old_b.get(
                "deflated_sharpe_ratio_weekly_n_obs"
            ),
            "erratum": (
                "These values mixed an ANNUALIZED Sharpe ratio with a "
                "PER-OBSERVATION n_obs (a units bug, not a formula "
                "bug) -- superseded by the periods_per_year-aware "
                "values above. See docs/RESELECTION.md Step 1 erratum."
            ),
        }

    report["n_trials_prereg"] = N_TRIALS_PREREG
    report["n_prior_oos_looks"] = N_PRIOR_OOS_LOOKS
    report["n_trials_total"] = N_TRIALS_170
    report["n_trials_total_171"] = N_TRIALS_171
    report["n_trials_headline"] = N_TRIALS_HEADLINE
    report["sharpe_std_across_trials"] = sharpe_std_across_trials

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"Wrote corrected statistics to {OUT_PATH}")

    for window_name in WINDOWS:
        m = report["windows"][window_name]["b_phase2"]
        dsr_170 = round(m["deflated_sharpe_ratio_weekly_n170"], 4)
        dsr_171 = round(m["deflated_sharpe_ratio_weekly_n171"], 4)
        print(window_name, "DSR weekly N170:", dsr_170, "N171:", dsr_171)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
