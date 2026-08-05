"""Tests for momentum.py: V5 dual momentum signal (stage 11,
ANALYSIS_V2.md action 1)."""

import numpy as np
import pandas as pd
import pytest

from atlas import momentum

CFG = {
    "momentum": {
        "lookback_days": 60,
        "skip_days": 5,
        "trend_lookback_days": 60,
        "top_fraction": 0.5,
        "max_view_magnitude": 0.03,
    }
}


def _growth_series(
    n: int, total_return: float, seed: int | None = None
) -> pd.Series:
    """A smooth exponential price path from 100 to 100*(1+total_return)
    over `n` rows, with a tiny deterministic noise wiggle so it isn't
    perfectly monotonic (closer to a real price series)."""
    dates = pd.bdate_range("2020-01-01", periods=n)
    growth = (1.0 + total_return) ** (np.arange(n) / (n - 1))
    values = 100.0 * growth
    if seed is not None:
        rng = np.random.default_rng(seed)
        values = values * (1.0 + rng.normal(0, 0.0005, n))
    return pd.Series(values, index=dates)


def _drift_walk(n=800, drift=0.001, sigma=0.005, seed=1) -> pd.Series:
    rng = np.random.default_rng(seed)
    log_price = np.cumsum(rng.normal(drift, sigma, n))
    dates = pd.bdate_range("2020-01-01", periods=n)
    return pd.Series(100 * np.exp(log_price), index=dates)


def _no_drift_walk(n=800, sigma=0.01, seed=2) -> pd.Series:
    rng = np.random.default_rng(seed)
    log_price = np.cumsum(rng.normal(0.0, sigma, n))
    dates = pd.bdate_range("2020-01-01", periods=n)
    return pd.Series(100 * np.exp(log_price), index=dates)


def test_has_enough_history():
    short = pd.DataFrame({"A": np.zeros(10)})
    long_ = pd.DataFrame({"A": np.zeros(200)})
    assert not momentum.has_enough_history(short, CFG)
    assert momentum.has_enough_history(long_, CFG)


def test_cross_sectional_score_ranks_within_bucket_not_globally():
    n = 70
    prices = pd.DataFrame(
        {
            "A": _growth_series(n, 0.30, seed=1),
            "B": _growth_series(n, 0.10, seed=2),
            "C": _growth_series(n, -0.05, seed=3),
            "D": _growth_series(n, 0.50, seed=4),
            "E": _growth_series(n, 0.01, seed=5),
        }
    )
    class_bucket = pd.Series(
        {"A": "equity", "B": "equity", "C": "equity",
         "D": "fixed_income", "E": "fixed_income"}
    )
    rank = momentum.cross_sectional_score(prices, class_bucket, CFG)

    # Within the 3-member equity bucket: strict order, best = 1.0.
    assert rank["A"] == pytest.approx(1.0)
    assert rank["B"] == pytest.approx(2 / 3)
    assert rank["C"] == pytest.approx(1 / 3)
    # Within the 2-member fixed_income bucket: D beats E, D = 1.0
    # despite its 50% growth being far larger than equity's best (A's
    # 30%) -- rank is bucket-local, not a global comparison.
    assert rank["D"] == pytest.approx(1.0)
    assert rank["E"] == pytest.approx(0.5)


def test_cross_sectional_score_nan_with_too_little_history():
    prices = pd.DataFrame({"A": np.linspace(100, 110, 10)})
    class_bucket = pd.Series({"A": "equity"})
    rank = momentum.cross_sectional_score(prices, class_bucket, CFG)
    assert rank.isna().all()


def test_absolute_trend_flags_uptrend_and_downtrend_buckets():
    n = 61
    prices = pd.DataFrame(
        {
            "A": _growth_series(n, 0.10, seed=1),
            "B": _growth_series(n, 0.08, seed=2),
            "C": _growth_series(n, -0.05, seed=3),
            "D": _growth_series(n, -0.03, seed=4),
        }
    )
    class_bucket = pd.Series(
        {"A": "equity", "B": "equity",
         "C": "fixed_income", "D": "fixed_income"}
    )
    rf_series = pd.Series(0.04, index=prices.index)
    trending = momentum.absolute_trend(prices, class_bucket, rf_series, CFG)
    assert trending["equity"]
    assert not trending["fixed_income"]


def test_absolute_trend_defaults_to_zero_rf_without_rf_series():
    n = 61
    prices = pd.DataFrame(
        {"A": _growth_series(n, 0.02, seed=1),
         "B": _growth_series(n, -0.01, seed=2)}
    )
    class_bucket = pd.Series({"A": "equity", "B": "fixed_income"})
    trending = momentum.absolute_trend(prices, class_bucket, None, CFG)
    assert trending["equity"]
    assert not trending["fixed_income"]


def test_absolute_trend_empty_with_too_little_history():
    prices = pd.DataFrame({"A": np.linspace(100, 110, 10)})
    class_bucket = pd.Series({"A": "equity"})
    trending = momentum.absolute_trend(prices, class_bucket, None, CFG)
    assert trending.empty


def _dual_momentum_scenario():
    n = 70
    prices = pd.DataFrame(
        {
            "A": _growth_series(n, 0.20, seed=1),   # equity, top
            "B": _growth_series(n, 0.02, seed=2),   # equity, bottom
            "C": _growth_series(n, -0.10, seed=3),  # fixed_income, declining
            "D": _growth_series(n, -0.15, seed=4),  # fixed_income, declining
        }
    )
    class_bucket = pd.Series(
        {"A": "equity", "B": "equity",
         "C": "fixed_income", "D": "fixed_income"}
    )
    rf_series = pd.Series(0.04, index=prices.index)
    return prices, class_bucket, rf_series


def test_momentum_view_positive_for_top_half_of_a_trending_bucket():
    prices, class_bucket, rf_series = _dual_momentum_scenario()
    view = momentum.momentum_view(prices, class_bucket, rf_series, CFG)
    assert view["A"] > 0.0
    assert view["A"] <= CFG["momentum"]["max_view_magnitude"] + 1e-9


def test_momentum_view_zero_for_bottom_half_of_a_trending_bucket():
    prices, class_bucket, rf_series = _dual_momentum_scenario()
    view = momentum.momentum_view(prices, class_bucket, rf_series, CFG)
    assert view["B"] == pytest.approx(0.0)


def test_momentum_view_flat_negative_for_a_non_trending_bucket():
    # Absolute leg overrides relative rank entirely: every member of a
    # non-trending bucket gets the same flat negative view, regardless
    # of C vs D's own within-bucket rank ordering.
    prices, class_bucket, rf_series = _dual_momentum_scenario()
    view = momentum.momentum_view(prices, class_bucket, rf_series, CFG)
    assert view["C"] == pytest.approx(-CFG["momentum"]["max_view_magnitude"])
    assert view["D"] == pytest.approx(-CFG["momentum"]["max_view_magnitude"])


def test_momentum_view_zero_with_too_little_history():
    prices = pd.DataFrame({"A": np.linspace(100, 110, 10)})
    class_bucket = pd.Series({"A": "equity"})
    view = momentum.momentum_view(prices, class_bucket, None, CFG)
    assert view["A"] == 0.0


def test_view_magnitude_same_order_as_other_view_families():
    # Same discipline as meanreversion's stage-11 unit-mismatch fix
    # test: a genuine signal's view magnitude should land within one
    # order of magnitude of the other (annual) view families, not be
    # off by 100x from a units bug.
    other_view_magnitudes = [0.03, 0.02]
    prices, class_bucket, rf_series = _dual_momentum_scenario()
    view = momentum.momentum_view(prices, class_bucket, rf_series, CFG)
    magnitude = abs(view["A"])
    assert magnitude > 0
    for other in other_view_magnitudes:
        assert other / 10 <= magnitude <= other * 10, (
            f"view magnitude {magnitude} is not within one order of "
            f"magnitude of {other}"
        )


def test_event_study_hit_rate_higher_for_a_trending_series():
    trend = _drift_walk()
    flat = _no_drift_walk()
    prices = pd.DataFrame({"T": trend, "F": flat})
    result = momentum.event_study(prices, CFG, horizon_days=10)

    assert result.loc["T", "n_events"] > 0
    assert result.loc["F", "n_events"] > 0
    assert result.loc["T", "hit_rate"] > result.loc["F", "hit_rate"]
