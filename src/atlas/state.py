"""Persisted portfolio state + drift reconciliation for live automation.

A weekly scheduled job starts cold every Friday with no in-process
memory: the backtester carries the current portfolio weights in a loop
variable, but a live runner cannot. This module is the missing memory.
It persists the post-execution portfolio state to `live/state.json` and
provides `drift_weights` to roll last week's stored weights forward
through the returns that have happened since, using the SAME arithmetic
`backtest.run` uses so the live path never silently disagrees with the
validated backtest.

This module is pure state + arithmetic. It does not download data, call
the strategy, generate trades, or render output; those are later stages.

Public API:
    empty_state() -> dict
    load_state(path=None) -> dict
    save_state(state, path=None) -> Path        # atomic
    validate_state(state) -> None
    weights_series(state) -> pd.Series
    drift_weights(weights, returns, from_date, to_date) -> pd.Series
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LIVE_DIR = PROJECT_ROOT / "live"
STATE_PATH = LIVE_DIR / "state.json"

SCHEMA_VERSION = 1
_WEIGHT_SUM_TOL = 1e-6


def empty_state() -> dict:
    """A valid, uninitialized state (no portfolio funded yet).

    Every persisted file starts here. `initialized` stays false until
    the first funding run writes real weights, mirroring how
    `backtest.run` holds zero weights until its first rebalance.
    """
    return {
        "schema_version": SCHEMA_VERSION,
        "initialized": False,
        "weights": {},
        "weights_as_of": None,
        "last_decision_date": None,
        "last_execution_date": None,
        "run_count": 0,
        "last_run_utc": None,
        "last_status": None,
    }


def validate_state(state: dict) -> None:
    """Raise if `state` is not a coherent portfolio state.

    Checks: known `schema_version`; and, once `initialized`, that
    `weights_as_of` is present, no weight is negative, and the weights
    sum to 1 within `_WEIGHT_SUM_TOL`. An uninitialized state is only
    required to carry the right schema version (its weights are empty
    by construction).
    """
    version = state.get("schema_version")
    if version != SCHEMA_VERSION:
        raise ValueError(
            f"Unknown schema_version {version!r}; this code speaks "
            f"version {SCHEMA_VERSION}."
        )

    if not state.get("initialized", False):
        return

    if state.get("weights_as_of") is None:
        raise ValueError(
            "Initialized state must carry weights_as_of (the date these "
            "weights were live)."
        )

    weights = state.get("weights", {})
    for ticker, weight in weights.items():
        # NaN/inf must be rejected explicitly: NaN defeats every
        # comparison-based guard below (nan < 0 is False, and any
        # inequality against the sum is False too), so a NaN weight
        # would otherwise sail through and be written to disk.
        if not math.isfinite(weight):
            raise ValueError(
                f"Non-finite weight {weight!r} for {ticker!r}; weights "
                "must be real numbers."
            )
        if weight < 0:
            raise ValueError(
                f"Negative weight {weight!r} for {ticker!r}; the strategy "
                "is long-only."
            )

    total = sum(weights.values())
    if not math.isfinite(total):
        raise ValueError(
            f"Weights sum to a non-finite value ({total!r}); weights must "
            "be real numbers."
        )
    if abs(total - 1.0) > _WEIGHT_SUM_TOL:
        raise ValueError(
            f"Initialized weights must sum to 1 (got {total!r}, tolerance "
            f"{_WEIGHT_SUM_TOL})."
        )


def load_state(path: str | Path | None = None) -> dict:
    """Load persisted portfolio state, or a fresh empty state if absent.

    A missing file is normal on the very first run and returns
    `empty_state()` without raising. A file that exists but is corrupt
    (malformed JSON or a schema that fails validation) DOES raise:
    silently resetting to empty would erase the portfolio's memory,
    which is strictly worse than failing loudly and letting a human
    intervene.
    """
    path = Path(path) if path is not None else STATE_PATH
    if not path.exists():
        return empty_state()

    try:
        with open(path, "r", encoding="utf-8") as f:
            state = json.load(f)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"State file {path} is corrupt (invalid JSON): {exc}. Refusing "
            "to silently reset -- the file is the portfolio's memory."
        ) from exc

    validate_state(state)
    return state


def save_state(state: dict, path: str | Path | None = None) -> Path:
    """Atomically persist `state` to `path`, returning the path written.

    WHY ATOMICITY MATTERS: this file IS the portfolio's memory. If a
    crash, power loss, or disk-full error interrupts a plain in-place
    write, the file is left half-written and unparseable -- and on the
    next run `load_state` would (correctly) refuse to reset, halting the
    portfolio, or worse a naive reader would see garbage. To make a
    partial write impossible, we write the full contents to a temp file
    in the SAME directory (so `os.replace` is a same-filesystem rename,
    which is atomic on POSIX and Windows) and then rename it onto the
    target. The target therefore only ever contains a complete, valid
    prior version or a complete, valid new version -- never a truncated
    one. The temp file goes in the same directory rather than a system
    temp dir precisely so the rename cannot cross a filesystem boundary
    and degrade into a non-atomic copy.

    The state is validated before anything is written, so an invalid
    state can never reach disk.
    """
    validate_state(state)

    path = Path(path) if path is not None else STATE_PATH
    path.parent.mkdir(parents=True, exist_ok=True)

    tmp_path = path.with_suffix(path.suffix + ".tmp")
    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2, sort_keys=False)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)
    except BaseException:
        # A failed write must leave the target's prior version untouched
        # (os.replace is atomic, so it never half-applies) AND leave no
        # orphan temp file to confuse the next run's directory scan.
        tmp_path.unlink(missing_ok=True)
        raise
    return path


def weights_series(state: dict) -> pd.Series:
    """`state["weights"]` as a float `pd.Series` for arithmetic.

    An uninitialized (or otherwise empty) state yields an empty Series.
    """
    return pd.Series(state.get("weights", {}), dtype=float)


def drift_weights(
    weights: pd.Series,
    returns: pd.DataFrame,
    from_date,
    to_date,
) -> pd.Series:
    """Roll `weights` forward through realized returns after `from_date`.

    `weights` are the portfolio's weights as they stood at the CLOSE of
    `from_date` (i.e. what `backtest.run` records in its weights frame
    for that date -- already post-execution, post-drift, summing to 1).
    This applies each trading day strictly AFTER `from_date` up to and
    including `to_date`, reproducing `backtest.run`'s per-day drift
    (see backtest.py lines 153-155) EXACTLY:

        gross = current_weights * (1.0 + day_return)
        total = gross.sum()
        current_weights = gross / total if total > 1e-12 else gross

    Renormalizing every day is the fully-invested convention: weights
    always sum to 1. The `1e-12` guard leaves weights untouched on a day
    where the portfolio's total growth factor collapses to zero (a
    degenerate case that cannot arise from ordinary returns).

    This function deliberately knows nothing about trades. It reproduces
    the backtester's behaviour only over spans with no rebalance
    execution; the live runner is responsible for applying trades
    (start of day, before that day's return) at the appropriate
    boundaries, exactly as `backtest.run` does.

    Args:
        weights: weights at the close of `from_date`, indexed by ticker.
        returns: daily simple returns, one column per ticker.
        from_date: the date `weights` were live (exclusive lower bound).
        to_date: the date to drift up to (inclusive).

    Returns:
        Weights drifted to the close of `to_date`, summing to 1.

    Raises:
        ValueError: if `to_date < from_date`, or if any ticker in
            `weights` is absent from `returns.columns` (a silent
            `fillna(0)` there would fabricate a frozen position rather
            than surface the missing data).
    """
    from_date = pd.Timestamp(from_date)
    to_date = pd.Timestamp(to_date)

    if to_date < from_date:
        raise ValueError(
            f"to_date {to_date.date()} is before from_date "
            f"{from_date.date()}; cannot drift backwards in time."
        )

    missing = [t for t in weights.index if t not in returns.columns]
    if missing:
        raise ValueError(
            f"Tickers {missing} in weights are absent from returns.columns "
            f"{list(returns.columns)}; refusing to fabricate a frozen "
            "position via a silent fill."
        )

    if from_date == to_date:
        return weights.copy()

    # Strictly-after semantics on whatever dates actually exist in the
    # frame: `from_date` itself need not be present (the weights are the
    # close-of-from_date state, already reflecting from_date's return),
    # and any drift day between the two that is missing simply isn't
    # applied -- we drift through the trading days that occurred.
    span = returns.loc[
        (returns.index > from_date) & (returns.index <= to_date)
    ]

    # A NaN return defeats the renormalization guard: `gross` picks up
    # the NaN, that asset's weight vanishes, and the remaining weights
    # renormalize to absorb it -- silently fabricating a position exit.
    # Refuse it, matching this module's stance on missing tickers: a
    # hole in the data must surface, not be smoothed over.
    relevant = span[list(weights.index)]
    if relevant.isna().to_numpy().any():
        bad_dates = relevant.index[relevant.isna().any(axis=1)]
        raise ValueError(
            f"NaN return(s) in the drift span for dates "
            f"{[d.date() for d in bad_dates]}; refusing to renormalize a "
            "position away via a silent NaN."
        )

    current = weights.copy()
    for date in span.index:
        day_return = span.loc[date].reindex(current.index)
        gross = current * (1.0 + day_return)
        total = gross.sum()
        current = gross / total if total > 1e-12 else gross

    return current
