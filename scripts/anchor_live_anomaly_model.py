"""One-off: anchor the live anomaly autoencoder at the live start date
(external review, Part A3).

Problem this corrects: `strategy.with_anomaly_override`'s refit cache
lived only in-process, so every weekly GitHub Actions run (a fresh
process) refit the autoencoder from scratch instead of following
`anomaly.refit_frequency_days` the way the backtest does (see
`src/atlas/anomaly_store.py`'s module docstring and Part A2's fix).
Weeks 2026-10-23 through 2026-11-13 all show `fitted_at == as_of` in
their diagnostics -- four refits where the backtest-equivalent
schedule would have done exactly one.

This script reconstructs what that ONE fit should have been: trained
on real market data truncated at 2026-10-23 (the live start date),
with the configured random seed, exactly as `live.build_strategy`'s
first-ever call would have produced it. It is fit TWICE to check
determinism (same seed, same data -> should be bit-identical weights),
and its threshold is compared against the real 2026-10-23 run's own
diagnostics (`live/weekly/2026-10-23/diagnostics.json`) -- TensorFlow
on CPU is not guaranteed bit-reproducible across processes, so this
reports the difference rather than forcing a match.

The result is committed to `live/models/anomaly/` via
`anomaly_store.AnomalyModelStore`, with `fitted_at = 2026-10-23`. The
next refit is then due on or after 2027-02-26
(2026-10-23 + `anomaly.refit_frequency_days` days), matching the
backtest-equivalent schedule from the live start.

Deliberately does NOT touch `live/state.json`, `live/history.csv`, or
`live/weekly/*`: the record of what actually happened stays as it
happened (external review Part A4's rule). `state.json`'s
`anomaly_fitted_at` field self-corrects the next time a real weekly
run executes with the override active, as a normal side effect of
`live._decide`'s existing diagnostics -- it does not need, and must
not get, a direct edit here.

Usage:
    python scripts/anchor_live_anomaly_model.py
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from atlas import anomaly, anomaly_store, data, live

LIVE_START = pd.Timestamp("2026-10-23")
REFERENCE_DIAGNOSTICS = (
    data.PROJECT_ROOT / "live" / "weekly" / "2026-10-23" / "diagnostics.json"
)


def _reference_threshold() -> float | None:
    if not REFERENCE_DIAGNOSTICS.exists():
        return None
    with open(REFERENCE_DIAGNOSTICS, encoding="utf-8") as f:
        diag = json.load(f)
    return diag.get("anomaly", {}).get("threshold")


def main() -> int:
    cfg = data.load_config()
    market = live.fetch_data(cfg)
    window = market.returns.loc[:LIVE_START]
    features = anomaly.build_features(window)

    if not anomaly.has_enough_history(features, cfg):
        print("Not enough history through", LIVE_START.date(), "-- aborting.")
        return 1

    fit_a = anomaly.fit_autoencoder(features, cfg)
    fit_b = anomaly.fit_autoencoder(features, cfg)

    weights_a = fit_a.model.get_weights()
    weights_b = fit_b.model.get_weights()
    weights_identical = all(
        np.array_equal(wa, wb) for wa, wb in zip(weights_a, weights_b)
    )
    threshold_identical = fit_a.threshold == fit_b.threshold
    print(f"Determinism check (two independent fits, same seed/data):")
    print(f"  weights identical:   {weights_identical}")
    print(f"  threshold identical: {threshold_identical} "
          f"({fit_a.threshold!r} vs {fit_b.threshold!r})")
    if not (weights_identical and threshold_identical):
        print(
            "NOT deterministic across repeated fits -- stopping before "
            "committing an anchor model. Report this, do not force it."
        )
        return 1

    ref_threshold = _reference_threshold()
    print(f"\nReconstructed threshold:        {fit_a.threshold!r}")
    print(f"Live 2026-10-23 run's threshold: {ref_threshold!r}")
    if ref_threshold is not None:
        diff = fit_a.threshold - ref_threshold
        print(f"Difference:                      {diff!r}")
    else:
        print("(reference diagnostics file not found; nothing to compare)")

    store = anomaly_store.AnomalyModelStore()
    store.save(fit_a, LIVE_START, cfg)
    print(f"\nAnchor model committed to {store.store_dir}")
    print(f"fitted_at = {LIVE_START.date()}")
    refit_due = LIVE_START + pd.Timedelta(
        days=cfg["anomaly"]["refit_frequency_days"]
    )
    print(f"Next refit due on or after {refit_due.date()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
