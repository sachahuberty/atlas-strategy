"""State persistence + drift reconciliation tests (live stage 1).

Fast and offline: no network, no TensorFlow, no yfinance. The drift
parity test is the load-bearing one -- it asserts `drift_weights`
reproduces `backtest.run`'s recorded weights row for row, which is the
guarantee that the live path and the validated backtest agree.
"""

import json

import numpy as np
import pandas as pd
import pytest

from atlas import backtest, state


# --------------------------------------------------------------------- #
# empty_state / validate_state
# --------------------------------------------------------------------- #
def test_empty_state_is_valid_and_uninitialized():
    s = state.empty_state()
    assert s["initialized"] is False
    assert s["weights"] == {}
    assert s["schema_version"] == state.SCHEMA_VERSION
    state.validate_state(s)  # must not raise


def test_validate_rejects_weights_summing_to_point_nine():
    s = state.empty_state()
    s["initialized"] = True
    s["weights_as_of"] = "2026-06-05"
    s["weights"] = {"A": 0.5, "B": 0.4}  # sums to 0.9
    with pytest.raises(ValueError, match="sum to 1"):
        state.validate_state(s)


def test_validate_rejects_negative_weight():
    s = state.empty_state()
    s["initialized"] = True
    s["weights_as_of"] = "2026-06-05"
    s["weights"] = {"A": 1.1, "B": -0.1}  # sums to 1 but B < 0
    with pytest.raises(ValueError, match="[Nn]egative"):
        state.validate_state(s)


def test_validate_rejects_unknown_schema_version():
    s = state.empty_state()
    s["schema_version"] = 999
    with pytest.raises(ValueError, match="schema_version"):
        state.validate_state(s)


def test_validate_rejects_initialized_without_weights_as_of():
    s = state.empty_state()
    s["initialized"] = True
    s["weights"] = {"A": 0.6, "B": 0.4}
    s["weights_as_of"] = None
    with pytest.raises(ValueError, match="weights_as_of"):
        state.validate_state(s)


def test_validate_rejects_nan_weight():
    # NaN defeats comparison guards (nan < 0 is False), so it must be
    # caught by an explicit finiteness check, not slip through to disk.
    s = state.empty_state()
    s["initialized"] = True
    s["weights_as_of"] = "2026-06-05"
    s["weights"] = {"SPY": float("nan"), "AGG": 0.5}
    with pytest.raises(ValueError, match="[Nn]on-finite"):
        state.validate_state(s)


def test_validate_rejects_nan_sum():
    # Weights individually caught first, but guard the sum too: an inf
    # and a -inf could each be finite-checked away yet sum to NaN in
    # other arrangements. Here two NaNs make the sum non-finite.
    s = state.empty_state()
    s["initialized"] = True
    s["weights_as_of"] = "2026-06-05"
    s["weights"] = {"A": float("inf"), "B": float("-inf")}
    with pytest.raises(ValueError, match="[Nn]on-finite"):
        state.validate_state(s)


def test_validate_accepts_initialized_state_within_tolerance():
    s = state.empty_state()
    s["initialized"] = True
    s["weights_as_of"] = "2026-06-05"
    # Sum off by less than 1e-6 must still pass.
    s["weights"] = {"A": 0.6, "B": 0.4 + 5e-7}
    state.validate_state(s)  # must not raise


# --------------------------------------------------------------------- #
# load_state
# --------------------------------------------------------------------- #
def test_load_missing_file_returns_empty_state(tmp_path):
    missing = tmp_path / "does_not_exist.json"
    loaded = state.load_state(missing)
    assert loaded == state.empty_state()
    assert not missing.exists()  # loading must not create it


def test_load_corrupt_json_raises_not_silently_resets(tmp_path):
    corrupt = tmp_path / "state.json"
    corrupt.write_text("{ this is not: valid json ,,,", encoding="utf-8")
    with pytest.raises(ValueError, match="corrupt"):
        state.load_state(corrupt)


def test_load_valid_but_invalid_schema_raises(tmp_path):
    # Well-formed JSON that fails validation must also refuse to load,
    # not silently reset.
    bad = tmp_path / "state.json"
    bad.write_text(json.dumps({"schema_version": 42}), encoding="utf-8")
    with pytest.raises(ValueError, match="schema_version"):
        state.load_state(bad)


# --------------------------------------------------------------------- #
# save_state round-trip + atomicity
# --------------------------------------------------------------------- #
def test_save_load_round_trips_exactly(tmp_path):
    path = tmp_path / "state.json"
    s = state.empty_state()
    s["initialized"] = True
    s["weights"] = {"ACWI": 0.25, "AGG": 0.25, "GLD": 0.25, "BSV": 0.25}
    s["weights_as_of"] = "2026-06-05"
    s["last_decision_date"] = "2026-06-05"
    s["last_execution_date"] = "2026-06-08"
    s["run_count"] = 3
    s["last_run_utc"] = "2026-06-05T21:00:00+00:00"
    s["last_status"] = "ok"

    returned = state.save_state(s, path)
    assert returned == path

    loaded = state.load_state(path)
    assert loaded == s
    # Date fields survive as strings, not coerced to Timestamps.
    assert isinstance(loaded["weights_as_of"], str)
    assert isinstance(loaded["last_run_utc"], str)
    assert isinstance(loaded["run_count"], int)


def test_save_creates_live_directory(tmp_path):
    nested = tmp_path / "live" / "state.json"
    state.save_state(state.empty_state(), nested)
    assert nested.exists()


def test_save_leaves_no_temp_file_behind(tmp_path):
    path = tmp_path / "state.json"
    state.save_state(state.empty_state(), path)
    leftovers = list(tmp_path.glob("*.tmp"))
    assert leftovers == []


def test_save_is_atomic_failure_mid_write_preserves_original(
    tmp_path, monkeypatch
):
    path = tmp_path / "state.json"

    # Write a first, valid version.
    good = state.empty_state()
    good["initialized"] = True
    good["weights"] = {"A": 0.6, "B": 0.4}
    good["weights_as_of"] = "2026-06-05"
    state.save_state(good, path)
    original_bytes = path.read_bytes()

    # Simulate a crash at the final rename while writing a new version.
    def boom(src, dst):
        raise OSError("simulated crash during os.replace")

    monkeypatch.setattr(state.os, "replace", boom)

    newer = dict(good)
    newer["weights"] = {"A": 0.7, "B": 0.3}
    with pytest.raises(OSError, match="simulated crash"):
        state.save_state(newer, path)

    # The original file is byte-for-byte intact and still parseable.
    assert path.read_bytes() == original_bytes
    assert state.load_state(path) == good
    # No half-written temp file survives to confuse the next run.
    assert list(tmp_path.glob("*.tmp")) == []


# --------------------------------------------------------------------- #
# weights_series
# --------------------------------------------------------------------- #
def test_weights_series_from_state():
    s = state.empty_state()
    s["weights"] = {"A": 0.6, "B": 0.4}
    ser = state.weights_series(s)
    assert isinstance(ser, pd.Series)
    assert ser["A"] == pytest.approx(0.6)
    assert ser.dtype == float


def test_weights_series_empty_state_is_empty():
    ser = state.weights_series(state.empty_state())
    assert len(ser) == 0


# --------------------------------------------------------------------- #
# drift_weights
# --------------------------------------------------------------------- #
def test_drift_zero_returns_leaves_weights_unchanged():
    dates = pd.bdate_range("2026-06-01", periods=5)
    returns = pd.DataFrame(0.0, index=dates, columns=["A", "B"])
    weights = pd.Series({"A": 0.6, "B": 0.4})
    drifted = state.drift_weights(weights, returns, dates[0], dates[-1])
    pd.testing.assert_series_equal(drifted, weights, check_names=False)


def test_drift_same_from_and_to_returns_unchanged():
    dates = pd.bdate_range("2026-06-01", periods=5)
    rng = np.random.default_rng(0)
    returns = pd.DataFrame(
        rng.normal(0, 0.01, (5, 2)), index=dates, columns=["A", "B"]
    )
    weights = pd.Series({"A": 0.6, "B": 0.4})
    drifted = state.drift_weights(weights, returns, dates[2], dates[2])
    pd.testing.assert_series_equal(drifted, weights, check_names=False)


def test_drift_hand_computed_two_asset_two_day():
    # from_date is day0; drift through day1 and day2 (strictly after).
    dates = pd.to_datetime(
        ["2026-06-01", "2026-06-02", "2026-06-03"]
    )
    returns = pd.DataFrame(
        {"A": [0.00, 0.10, 0.00], "B": [0.00, 0.00, 0.20]}, index=dates
    )
    weights = pd.Series({"A": 0.5, "B": 0.5})

    # Day1: gross = (0.5*1.10, 0.5*1.00) = (0.55, 0.50), total 1.05
    #   -> (0.55/1.05, 0.50/1.05) = (0.523809..., 0.476190...)
    # Day2: gross = (0.523809*1.00, 0.476190*1.20)
    #             = (0.523809..., 0.571428...), total 1.095238...
    #   -> (0.478260..., 0.521739...)
    drifted = state.drift_weights(weights, returns, dates[0], dates[2])
    assert drifted["A"] == pytest.approx(0.5238095238 * 1.0 / 1.0952380952)
    assert drifted["B"] == pytest.approx(0.4761904762 * 1.2 / 1.0952380952)
    assert drifted.sum() == pytest.approx(1.0)


def test_drift_output_always_sums_to_one():
    dates = pd.bdate_range("2026-06-01", periods=30)
    rng = np.random.default_rng(5)
    returns = pd.DataFrame(
        rng.normal(0.0005, 0.02, (30, 3)),
        index=dates,
        columns=["A", "B", "C"],
    )
    weights = pd.Series({"A": 0.2, "B": 0.5, "C": 0.3})
    drifted = state.drift_weights(weights, returns, dates[0], dates[-1])
    assert drifted.sum() == pytest.approx(1.0)


def test_drift_raises_on_to_before_from():
    dates = pd.bdate_range("2026-06-01", periods=5)
    returns = pd.DataFrame(0.0, index=dates, columns=["A", "B"])
    weights = pd.Series({"A": 0.6, "B": 0.4})
    with pytest.raises(ValueError, match="before from_date"):
        state.drift_weights(weights, returns, dates[3], dates[1])


def test_drift_raises_on_missing_ticker():
    dates = pd.bdate_range("2026-06-01", periods=5)
    returns = pd.DataFrame(0.0, index=dates, columns=["A", "B"])
    weights = pd.Series({"A": 0.6, "GLD": 0.4})  # GLD not in returns
    with pytest.raises(ValueError, match="GLD"):
        state.drift_weights(weights, returns, dates[0], dates[-1])


def test_drift_raises_on_nan_return():
    # A missing (NaN) return in the drift span would otherwise make the
    # asset's weight vanish and the rest renormalize to absorb it,
    # silently fabricating a position exit. It must raise instead.
    dates = pd.bdate_range("2026-06-01", periods=5)
    returns = pd.DataFrame(0.01, index=dates, columns=["A", "B"])
    returns.loc[dates[2], "A"] = float("nan")
    weights = pd.Series({"A": 0.6, "B": 0.4})
    with pytest.raises(ValueError, match="NaN"):
        state.drift_weights(weights, returns, dates[0], dates[-1])


def test_drift_from_date_absent_uses_strictly_after_present_dates():
    # from_date falls on a weekend not present in the frame; drift must
    # apply every trading day strictly after it up to to_date.
    dates = pd.to_datetime(["2026-06-01", "2026-06-02", "2026-06-03"])
    returns = pd.DataFrame(
        {"A": [0.05, 0.10, 0.00], "B": [0.05, 0.00, 0.00]}, index=dates
    )
    weights = pd.Series({"A": 0.5, "B": 0.5})
    # from_date = 2026-05-31 (Sunday, absent). Strictly-after means all
    # three present dates are applied.
    from_absent = pd.Timestamp("2026-05-31")
    drifted = state.drift_weights(weights, returns, from_absent, dates[-1])

    # Equivalent to drifting through every row of the frame.
    manual = weights.copy()
    for d in dates:
        gross = manual * (1.0 + returns.loc[d])
        manual = gross / gross.sum()
    pd.testing.assert_series_equal(drifted, manual, check_names=False)


# --------------------------------------------------------------------- #
# THE drift parity test: live and backtest must agree row for row
# --------------------------------------------------------------------- #
def test_drift_parity_reproduces_backtest_recorded_weights():
    dates = pd.bdate_range("2026-01-05", periods=40)
    rng = np.random.default_rng(42)
    tickers = ["A", "B", "C"]
    returns = pd.DataFrame(
        rng.normal(0.0004, 0.012, (40, 3)), index=dates, columns=tickers
    )

    fixed_target = pd.Series({"A": 0.5, "B": 0.3, "C": 0.2})

    def constant_strategy(as_of, window):
        return fixed_target

    cfg = {
        "rebalance": {
            "transaction_cost_bps": 10,
            "max_weekly_turnover": 1.0,
            # A band of 1.0 makes every rebalance after the first funding
            # a no-op (drift alone can never reach 100% one-way turnover),
            # so the only weight changes after funding are pure drift --
            # exactly the span drift_weights models.
            "no_trade_band": 1.0,
        }
    }
    result = backtest.run(constant_strategy, returns, cfg)

    # The first rebalance funds from cash; its trade executes the next
    # trading day, which is the first day weights are non-zero.
    first_rebalance = result.turnover.index[0]
    funded_pos = returns.index.get_loc(first_rebalance) + 1
    funded_day = returns.index[funded_pos]

    # No trade executes after funding, so backtest's recorded weights are
    # pure drift from the funded-day close. drift_weights must reproduce
    # every subsequent recorded row exactly.
    anchor = result.weights.loc[funded_day]
    later_days = returns.index[funded_pos + 1:]
    assert len(later_days) > 0

    for day in later_days:
        expected = result.weights.loc[day]
        got = state.drift_weights(anchor, returns, funded_day, day)
        pd.testing.assert_series_equal(
            got.reindex(expected.index),
            expected,
            check_names=False,
            check_dtype=False,
        )
