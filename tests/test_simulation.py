"""Tests for simulation.py: GBM Monte Carlo forward risk (stage 10)."""

import numpy as np
import pandas as pd
import pytest

from atlas import simulation


def test_business_days_length_and_starts_after_start():
    start = pd.Timestamp("2026-01-01")
    dates = simulation.business_days(start, 50)

    assert len(dates) == 50
    assert dates[0] > start
    assert (dates.weekday < 5).all()


def test_gbm_paths_shape_and_start_column_is_one():
    dates = simulation.business_days(pd.Timestamp("2026-01-01"), 30)
    paths = simulation.gbm_paths(0.08, 0.15, dates, n_paths=200, seed=42)

    assert paths.shape == (200, 31)
    assert (paths[:, 0] == 1.0).all()


def test_gbm_paths_deterministic_with_same_seed():
    dates = simulation.business_days(pd.Timestamp("2026-01-01"), 30)
    a = simulation.gbm_paths(0.08, 0.15, dates, n_paths=50, seed=7)
    b = simulation.gbm_paths(0.08, 0.15, dates, n_paths=50, seed=7)

    assert np.array_equal(a, b)


def test_gbm_paths_different_seeds_differ():
    dates = simulation.business_days(pd.Timestamp("2026-01-01"), 30)
    a = simulation.gbm_paths(0.08, 0.15, dates, n_paths=50, seed=1)
    b = simulation.gbm_paths(0.08, 0.15, dates, n_paths=50, seed=2)

    assert not np.array_equal(a, b)


def test_gbm_paths_zero_vol_matches_deterministic_drift():
    # With sigma=0 every path is identical and equals the exact
    # continuous-compounding drift trajectory -- a known-answer check.
    dates = simulation.business_days(pd.Timestamp("2026-01-01"), 10)
    paths = simulation.gbm_paths(0.10, 0.0, dates, n_paths=5, seed=1)

    dt = 1.0 / simulation.TRADING_DAYS_PER_YEAR
    expected = np.exp(0.10 * dt * np.arange(11))
    for row in paths:
        assert row == pytest.approx(expected)


def test_scenario_summary_matches_manual_quantiles():
    # Construct a paths array with fully known terminal values at
    # day 252 (1 year) so every summary stat is checkable by hand.
    n_days = 252
    terminal_values = np.array(
        [0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7]
    )
    paths = np.ones((len(terminal_values), n_days + 1))
    paths[:, -1] = terminal_values

    summary = simulation.scenario_summary(paths, horizons_years=[1])
    row = summary.loc[1]

    assert row["mean_growth"] == pytest.approx(terminal_values.mean())
    assert row["median_growth"] == pytest.approx(np.median(terminal_values))
    assert row["prob_positive"] == pytest.approx(0.7)  # 7 of 10 above 1.0


def test_recovery_analysis_flat_path_recovers_immediately():
    paths = np.ones((3, 20))
    result = simulation.recovery_analysis(paths, peak=1.0)

    assert result["recovered"].all()
    assert (result["recovery_days"] == 0).all()


def test_recovery_analysis_censored_when_never_reaches_peak():
    paths = np.full((2, 20), 0.9)
    result = simulation.recovery_analysis(paths, peak=1.0)

    assert not result["recovered"].any()
    assert result["recovery_days"].isna().all()


def test_recovery_analysis_recovery_day_is_exact():
    path = np.concatenate([np.full(5, 0.8), np.full(5, 1.2)])
    paths = path.reshape(1, -1)

    result = simulation.recovery_analysis(paths, peak=1.0)

    assert result.loc[0, "recovered"]
    assert result.loc[0, "recovery_days"] == 5
