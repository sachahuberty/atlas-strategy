"""Known-answer tests for metrics.py (stage 1)."""

import numpy as np
import pandas as pd
import pytest

from atlas.metrics import (
    ann_return,
    cagr,
    calmar,
    hit_rate,
    max_drawdown,
    portfolio_return,
    portfolio_vol,
    sharpe,
    sortino,
    turnover,
    utility,
)


def test_ann_return_constant_daily_return():
    daily = 0.0004
    returns = pd.Series([daily] * 252)
    expected = (1 + daily) ** 252 - 1
    assert ann_return(returns) == pytest.approx(expected)


def test_sharpe_of_constant_return_series_is_zero():
    # Zero volatility is a degenerate case: sharpe() must not divide by zero.
    returns = pd.Series([0.001] * 20)
    assert sharpe(returns) == 0.0


def test_sharpe_known_values():
    returns = pd.Series([0.02, -0.01, 0.03, 0.00, -0.02])
    expected = (returns.mean() / returns.std(ddof=1)) * np.sqrt(252)
    assert sharpe(returns) == pytest.approx(expected)


def test_sortino_ignores_upside_deviation():
    # Only downside deviation should enter the denominator.
    returns = pd.Series([0.05, 0.05, 0.05, -0.01, -0.01])
    downside = returns[returns < 0]
    expected_denom = np.sqrt((downside ** 2).mean())
    expected = (returns.mean() / expected_denom) * np.sqrt(252)
    assert sortino(returns) == pytest.approx(expected)


def test_max_drawdown_known_path():
    # Prices: 100 -> 110 -> 88 -> 99, i.e. a known -20% drawdown from the peak.
    prices = pd.Series([100, 110, 88, 99])
    returns = prices.pct_change().dropna()
    assert max_drawdown(returns) == pytest.approx(-0.20)


def test_calmar_uses_max_drawdown():
    returns = pd.Series([0.01, -0.05, 0.02, 0.01])
    expected = ann_return(returns) / abs(max_drawdown(returns))
    assert calmar(returns) == pytest.approx(expected)


def test_cagr_matches_ann_return():
    returns = pd.Series([0.001, 0.002, -0.001, 0.003])
    assert cagr(returns) == pytest.approx(ann_return(returns))


def test_hit_rate_known_series():
    returns = pd.Series([0.01, -0.01, 0.02, 0.0, 0.03])
    # Three of five returns are strictly positive (0.0 does not count).
    assert hit_rate(returns) == pytest.approx(3 / 5)


def test_portfolio_return_matches_manual_weighted_sum():
    returns = pd.DataFrame({"A": [0.01, 0.02], "B": [-0.01, 0.00]})
    weights = pd.Series({"A": 0.6, "B": 0.4})
    expected = returns["A"] * 0.6 + returns["B"] * 0.4
    actual = portfolio_return(weights, returns)
    pd.testing.assert_series_equal(actual, expected, check_names=False)


def test_portfolio_vol_matches_quadratic_form():
    cov = pd.DataFrame(
        {"A": [0.04, 0.01], "B": [0.01, 0.09]}, index=["A", "B"]
    )
    weights = pd.Series({"A": 0.5, "B": 0.5})
    w = weights.to_numpy()
    expected = np.sqrt(w @ cov.to_numpy() @ w)
    assert portfolio_vol(weights, cov) == pytest.approx(expected)


def test_turnover_known_weight_change():
    w_t = pd.Series({"A": 0.5, "B": 0.5})
    w_prev = pd.Series({"A": 0.3, "B": 0.7})
    # |0.5-0.3| + |0.5-0.7| = 0.4, halved (one-way turnover) = 0.2
    assert turnover(w_t, w_prev) == pytest.approx(0.2)


def test_turnover_handles_new_and_dropped_assets():
    w_t = pd.Series({"A": 1.0})
    w_prev = pd.Series({"B": 1.0})
    # Full portfolio replacement: |1-0| + |0-1| = 2, halved = 1.0
    assert turnover(w_t, w_prev) == pytest.approx(1.0)


def test_utility_ordering_across_risk_aversion():
    low_aversion = utility(exp_ret=0.08, vol=0.15, risk_aversion=2)
    high_aversion = utility(exp_ret=0.08, vol=0.15, risk_aversion=10)
    assert low_aversion > high_aversion


def test_utility_known_value():
    result = utility(exp_ret=0.10, vol=0.20, risk_aversion=5)
    assert result == pytest.approx(0.10 - 0.5 * 5 * 0.20 ** 2)
