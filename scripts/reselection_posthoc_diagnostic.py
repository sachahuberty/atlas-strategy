"""Reselection task Step 3 -- post-hoc IS diagnostic (external review
correction).

Two tables, both derived from already-committed/cached data only --
no new strategy backtest, no OOS access:

1. `reselection_posthoc_is_diagnostic.csv`: for each of the 9
   one-setting-at-a-time challengers vs. the current config, the
   per-fold delta (challenger - current) read straight from the
   already-committed `reselection_phase2_fold_sharpe.csv` -- mean/
   median delta and win/loss/tie counts over all 12 IS folds, and the
   same split into the 2010-2013 cold-start years vs. 2014-2021.

2. `reselection_posthoc_cold_start.csv`: for each IS year, how many of
   that year's weekly decisions had the HMM posture forced to neutral
   by insufficient trailing history, and how many had a covariance
   window shorter than `optimization.lookback_days` -- computed by
   calling the same cheap, pure history-check functions the live
   pipeline itself calls, on IS-only price/return data freshly loaded
   from the local cache (no backtest, no OOS dates).

Usage:
    python scripts/reselection_posthoc_diagnostic.py
"""

from __future__ import annotations

import ast

import pandas as pd

from atlas import backtest, data, reselection, universe

RESULTS_DIR = data.PROJECT_ROOT / "reports" / "results"
FOLD_SHARPE_PATH = RESULTS_DIR / "reselection_phase2_fold_sharpe.csv"

CURRENT_KEY = ("V3_off__V5_off__riskoff_legacy", 2, "2pct_0.5pct")
CHALLENGERS = {
    "V3": ("V3_on__V5_off__riskoff_legacy", 2, "2pct_0.5pct"),
    "V5": ("V3_off__V5_on__riskoff_legacy", 2, "2pct_0.5pct"),
    "risk_off_bonds_cash": (
        "V3_off__V5_off__riskoff_bonds_cash", 2, "2pct_0.5pct",
    ),
    "risk_off_bonds_gold": (
        "V3_off__V5_off__riskoff_bonds_gold", 2, "2pct_0.5pct",
    ),
    "gate_A_1": ("V3_off__V5_off__riskoff_legacy", 1, "2pct_0.5pct"),
    "gate_A_3": ("V3_off__V5_off__riskoff_legacy", 3, "2pct_0.5pct"),
    "gate_A_5": ("V3_off__V5_off__riskoff_legacy", 5, "2pct_0.5pct"),
    "turnover_1pct": ("V3_off__V5_off__riskoff_legacy", 2, "1pct_0.5pct"),
    "turnover_4pct": ("V3_off__V5_off__riskoff_legacy", 2, "4pct_0.5pct"),
}

# From the task's expected-value table, for the reproduction check
# below -- not enforced, only reported if different.
_EXPECTED = {
    "V5": {"mean_delta_all": 0.049, "wins_all": 7, "losses_all": 4,
           "ties_all": 1, "wins_2014_2021": 4, "mean_delta_2014_2021": -0.056},
    "turnover_1pct": {"mean_delta_all": 0.020, "wins_all": 8,
                       "losses_all": 4, "ties_all": 0,
                       "wins_2014_2021": 5, "mean_delta_2014_2021": 0.016},
    "V3": {"mean_delta_all": 0.138, "wins_all": 6, "losses_all": 6,
           "ties_all": 0, "wins_2014_2021": 5, "mean_delta_2014_2021": 0.175},
    "risk_off_bonds_cash": {"ties_all": 5},
    "risk_off_bonds_gold": {"ties_all": 5},
}


def _load_fold_sharpe() -> pd.DataFrame:
    df = pd.read_csv(FOLD_SHARPE_PATH, index_col=0)
    df.columns = [ast.literal_eval(c) for c in df.columns]
    return df


def main() -> int:
    fold_sharpe = _load_fold_sharpe()
    current = fold_sharpe[CURRENT_KEY]

    rows = {}
    for name, key in CHALLENGERS.items():
        rows[name] = reselection.fold_delta_diagnostic(current, fold_sharpe[key])
    table = pd.DataFrame(rows).T
    table.index.name = "challenger"
    table.to_csv(RESULTS_DIR / "reselection_posthoc_is_diagnostic.csv")
    print(table.round(4).to_string())

    print("\nReproduction check against the task's expected values:")
    any_mismatch = False
    for name, expected in _EXPECTED.items():
        for key, expected_val in expected.items():
            actual_val = rows[name][key]
            close = (
                abs(actual_val - expected_val) < 0.01
                if isinstance(expected_val, float)
                else actual_val == expected_val
            )
            flag = "OK" if close else "MISMATCH"
            if not close:
                any_mismatch = True
            print(f"  {name}.{key}: expected={expected_val} actual={actual_val} [{flag}]")
    if any_mismatch:
        print("\nAt least one expected value did NOT match -- see docs/RESELECTION.md for how this is reported.")
    else:
        print("\nAll expected values matched.")

    # --- Cold-start diagnostic: IS-only, no backtest. ---
    cfg = data.load_config()
    as_of_universe = pd.Timestamp(cfg["general"]["is_end_date"])
    universe_df = universe.load_universe(as_of_universe)
    tickers = universe_df.index.tolist()

    prices = data.download_prices(tickers, start=cfg["general"]["start_date"])
    prices = data.align_calendars(prices)
    returns = data.daily_returns(prices)
    is_returns = returns.loc[:as_of_universe]

    dates = sorted(backtest.week_end_dates(is_returns.index))
    cold_start = reselection.cold_start_diagnostic(
        dates, is_returns, cfg["regimes"]["market_ticker"], cfg
    )
    cold_start.to_csv(RESULTS_DIR / "reselection_posthoc_cold_start.csv")
    print("\nCold-start diagnostic (IS years, no OOS access):")
    print(cold_start.to_string())

    return 1 if any_mismatch else 0


if __name__ == "__main__":
    raise SystemExit(main())
