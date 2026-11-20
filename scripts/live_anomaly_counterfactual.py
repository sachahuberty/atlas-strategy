"""Counterfactual: would the 2026-10-23 anchor model have fired on each
live week so far (external review, Part A4)?

Read-only. Scores the committed anchor model (`live/models/anomaly/`,
`fitted_at = 2026-10-23`, Part A3) against each live week's actual
market window, and reports recon_error/threshold/fired next to what
the live run ACTUALLY did that week (read from the already-committed
`live/weekly/<date>/diagnostics.json` files, or computed fresh via a
dry run for a week that has not been decided yet). This is live data
(dates >= 2026-10-23, after the live start), not the 2022-2026
development set, and nothing here is a new backtest: it only re-scores
already-observed weeks against a model that should have been in use
all along.

Never rewrites `live/history.csv`, `live/weekly/*`, or
`live/state.json` -- the record of what actually happened stays as it
happened. Writes a read-only report to
`reports/results/live_anomaly_counterfactual.csv`.

Usage:
    python scripts/live_anomaly_counterfactual.py
"""

from __future__ import annotations

import json

import pandas as pd

from atlas import anomaly, anomaly_store, backtest, data, live

LIVE_START = pd.Timestamp("2026-10-23")
RESULTS_PATH = (
    data.PROJECT_ROOT
    / "reports"
    / "results"
    / "live_anomaly_counterfactual.csv"
)


def _actual_diagnostics(as_of: pd.Timestamp) -> dict | None:
    """What the live run actually recorded for `as_of`, from the
    already-committed diagnostics file -- never from a fresh run."""
    path = (
        data.PROJECT_ROOT
        / "live"
        / "weekly"
        / as_of.date().isoformat()
        / "diagnostics.json"
    )
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f).get("anomaly")


def main() -> int:
    cfg = data.load_config()
    market = live.fetch_data(cfg)
    store = anomaly_store.AnomalyModelStore()
    anchor = store.load(cfg)
    if anchor.reason is not None:
        print(f"Could not load the anchor model: {anchor.reason}")
        return 1
    print(f"Anchor model fitted_at = {anchor.fitted_at.date()}")

    week_ends = sorted(backtest.week_end_dates(market.returns.index))
    live_weeks = [d for d in week_ends if d >= LIVE_START]

    rows = []
    for as_of in live_weeks:
        window = market.returns.loc[:as_of]
        features = anomaly.build_features(window)
        recent = features.tail(anchor.result.seq_len)
        errors = anomaly.reconstruction_error(anchor.result, recent)
        counterfactual_error = float(errors[-1]) if len(errors) > 0 else None
        counterfactual_fired = (
            counterfactual_error is not None
            and counterfactual_error > anchor.result.threshold
        )

        actual = _actual_diagnostics(as_of)
        actual_is_dry_run = False
        if actual is None:
            # Not decided yet as a real week (no committed diagnostics
            # file): a read-only dry run (no state/history/weekly
            # mutation) to see what live WOULD do, for completeness.
            dry = live.run_once(as_of=as_of, cfg=cfg, dry_run=True)
            if dry.status == live.STATUS_OK:
                actual = dry.diagnostics.get("anomaly")
                actual_is_dry_run = True

        rows.append(
            {
                "as_of": as_of.date().isoformat(),
                "actual_fitted_at": (actual or {}).get("fitted_at"),
                "actual_recon_error": (actual or {}).get("recon_error"),
                "actual_threshold": (actual or {}).get("threshold"),
                "actual_fired": (actual or {}).get("fired"),
                "actual_is_dry_run": actual_is_dry_run,
                "counterfactual_recon_error": counterfactual_error,
                "counterfactual_threshold": anchor.result.threshold,
                "counterfactual_fired": counterfactual_fired,
            }
        )

    table = pd.DataFrame(rows).set_index("as_of")
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(RESULTS_PATH)
    print(table.to_string())
    print(f"\nSaved to {RESULTS_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
