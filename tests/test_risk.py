"""Tests for risk.py: historical/sensitivity/reverse stress (stage 10)."""

import pandas as pd
import pytest

from atlas import risk


def _returns_fixture():
    dates = pd.bdate_range("2020-01-01", periods=10)
    data = {
        "A": [0.01, -0.02, 0.03, -0.10, 0.01, 0.02, -0.01, 0.00, 0.01, -0.03],
        "B": [0.00, 0.01, -0.01, -0.05, 0.02, -0.02, 0.01, 0.01, -0.02, 0.00],
    }
    return pd.DataFrame(data, index=dates)


def test_historical_stress_matches_manual_buyhold_calc():
    returns = _returns_fixture()
    weights = pd.Series({"A": 0.6, "B": 0.4})
    window = returns.iloc[2:5]
    scenarios = {
        "test": (str(window.index[0].date()), str(window.index[-1].date()))
    }

    result = risk.historical_stress(weights, returns, scenarios)
    row = result.loc["test"]

    port_returns = window.mul(weights, axis=1).sum(axis=1)
    expected_cum = (1.0 + port_returns).prod() - 1.0

    assert row["n_days"] == 3
    assert row["cumulative_return"] == pytest.approx(expected_cum)
    assert row["worst_day"] == pytest.approx(port_returns.min())


def test_historical_stress_multiple_scenarios_are_independent():
    returns = _returns_fixture()
    weights = pd.Series({"A": 1.0, "B": 0.0})
    scenarios = {
        "early": ("2020-01-01", "2020-01-03"),
        "late": ("2020-01-10", "2020-01-14"),
    }

    result = risk.historical_stress(weights, returns, scenarios)

    assert set(result.index) == {"early", "late"}
    assert result.loc["early", "n_days"] != result.loc["late", "n_days"] or (
        result.loc["early", "cumulative_return"]
        != result.loc["late", "cumulative_return"]
    )


def test_sensitivity_stress_portfolio_impact_is_weight_times_shock():
    weights = pd.Series({"A": 0.5, "B": 0.3, "C": 0.2})
    class_bucket = pd.Series(
        {"A": "equity", "B": "fixed_income", "C": "cash"}
    )
    shocks = {"equity": [-0.1, -0.2], "fixed_income": [-0.05], "cash": [0.0]}

    result = risk.sensitivity_stress(weights, class_bucket, shocks)

    assert len(result) == 4  # 2 + 1 + 1 shock magnitudes
    is_equity = result["bucket"] == "equity"
    is_neg20 = result["shock"] == -0.2
    equity_row = result[is_equity & is_neg20]
    assert equity_row["bucket_weight"].iloc[0] == pytest.approx(0.5)
    assert equity_row["portfolio_impact"].iloc[0] == pytest.approx(-0.1)


def test_sensitivity_stress_multi_ticker_bucket_sums_weights():
    weights = pd.Series({"A": 0.3, "B": 0.3, "C": 0.4})
    class_bucket = pd.Series({"A": "equity", "B": "equity", "C": "cash"})
    shocks = {"equity": [-0.1]}

    result = risk.sensitivity_stress(weights, class_bucket, shocks)

    assert result["bucket_weight"].iloc[0] == pytest.approx(0.6)
    assert result["portfolio_impact"].iloc[0] == pytest.approx(-0.06)


def test_reverse_stress_worst_day_matches_manual_argmin():
    returns = _returns_fixture()
    weights = pd.Series({"A": 0.6, "B": 0.4})

    result = risk.reverse_stress(weights, returns, window_days=3)

    port_returns = returns.mul(weights, axis=1).sum(axis=1)
    assert result["worst_day"] == port_returns.idxmin()
    assert result["worst_day_return"] == pytest.approx(port_returns.min())


def test_reverse_stress_contributors_sum_to_worst_day_return():
    returns = _returns_fixture()
    weights = pd.Series({"A": 0.6, "B": 0.4})

    result = risk.reverse_stress(weights, returns, window_days=3)

    assert result["worst_day_contributors"].sum() == pytest.approx(
        result["worst_day_return"]
    )


def test_reverse_stress_worst_window_is_the_actual_minimum():
    returns = _returns_fixture()
    weights = pd.Series({"A": 0.6, "B": 0.4})
    window_days = 3

    result = risk.reverse_stress(weights, returns, window_days=window_days)

    port_returns = returns.mul(weights, axis=1).sum(axis=1)
    rolling_cum = (
        (1.0 + port_returns)
        .rolling(window_days)
        .apply(lambda x: x.prod() - 1.0, raw=True)
    )
    assert result["worst_window_end"] == rolling_cum.idxmin()
    assert result["worst_window_return"] == pytest.approx(rolling_cum.min())
