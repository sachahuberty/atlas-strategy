"""Known-answer tests for metrics.py (stage 1)."""

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from atlas.metrics import (
    ann_return,
    cagr,
    calmar,
    deflated_sharpe_ratio,
    hit_rate,
    max_drawdown,
    newey_west_mean_test,
    portfolio_return,
    portfolio_vol,
    sharpe,
    sharpe_standard_error,
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


# --------------------------------------------------------------------- #
# sharpe_standard_error (reselection task Phase 4): deliberately the
# sqrt((1+SR^2/2)/years) formula, not README's simpler 0.5/sqrt(years).
# --------------------------------------------------------------------- #
def test_sharpe_standard_error_known_value():
    sr, years = 0.6, 4.5
    expected = np.sqrt((1 + sr ** 2 / 2) / years)
    assert sharpe_standard_error(sr, years) == pytest.approx(expected)


def test_sharpe_standard_error_grows_with_sharpe_magnitude():
    # Holding years fixed, a larger |SR| means a larger SE under this
    # formula (unlike the simpler 0.5/sqrt(years) convention, which is
    # flat in SR).
    se_low = sharpe_standard_error(0.1, 5.0)
    se_high = sharpe_standard_error(1.5, 5.0)
    assert se_high > se_low


def test_sharpe_standard_error_shrinks_with_more_years():
    se_short = sharpe_standard_error(0.6, 2.0)
    se_long = sharpe_standard_error(0.6, 10.0)
    assert se_long < se_short


# --------------------------------------------------------------------- #
# newey_west_mean_test
# --------------------------------------------------------------------- #
def test_newey_west_mean_test_zero_mean_series_not_significant():
    # A series mirrored around zero (x, -x) has an EXACTLY zero sample
    # mean by construction, not merely "close to zero by luck of the
    # seed" -- avoids a flaky draw that happens to reject a true null
    # at the ~5% rate any valid test should.
    rng = np.random.default_rng(1)
    x = rng.normal(0.0, 0.01, 150)
    series = pd.Series(np.concatenate([x, -x]))
    result = newey_west_mean_test(series)
    assert result["mean"] == pytest.approx(0.0, abs=1e-12)
    assert result["pvalue"] > 0.9  # essentially certain not to reject
    assert result["n"] == 300


def test_newey_west_mean_test_detects_a_real_mean_shift():
    rng = np.random.default_rng(2)
    series = pd.Series(rng.normal(0.01, 0.005, 300))  # clear positive drift
    result = newey_west_mean_test(series)
    assert result["mean"] == pytest.approx(series.mean())
    assert result["pvalue"] < 0.01
    assert result["tstat"] > 0


def test_newey_west_mean_test_drops_nan():
    series = pd.Series([0.01, np.nan, 0.01, 0.01, np.nan, 0.01] * 20)
    result = newey_west_mean_test(series)
    assert result["n"] == 80  # 120 - 40 NaNs


def test_newey_west_mean_test_se_larger_than_iid_under_autocorrelation():
    # A strongly autocorrelated series: the HAC SE must exceed the
    # naive i.i.d. SE (std/sqrt(n)), since HAC accounts for the extra
    # uncertainty autocorrelation introduces.
    n = 300
    rng = np.random.default_rng(3)
    noise = rng.normal(0.0, 0.01, n)
    series = pd.Series(noise).rolling(5).mean().dropna() + 0.002
    result = newey_west_mean_test(series, maxlags=10)
    iid_se = series.std(ddof=1) / np.sqrt(len(series))
    assert result["se"] > iid_se


# --------------------------------------------------------------------- #
# deflated_sharpe_ratio
# --------------------------------------------------------------------- #
def test_deflated_sharpe_ratio_without_trial_correction_is_probabilistic_sr():
    # sharpe_std_across_trials omitted -> SR_0 = 0, so this reduces to
    # the ordinary Probabilistic Sharpe Ratio (normal returns: skew=0,
    # kurtosis=3 collapses the variance term to 1/(n_obs-1)).
    sr, n_obs = 0.8, 500
    dsr = deflated_sharpe_ratio(sr, n_trials=1, n_obs=n_obs)
    sigma_sr = np.sqrt(1.0 / (n_obs - 1))
    expected = norm.cdf(sr / sigma_sr)
    assert dsr == pytest.approx(expected)


def test_deflated_sharpe_ratio_decreases_with_more_trials():
    sr, n_obs = 0.8, 500
    dsr_few = deflated_sharpe_ratio(
        sr, n_trials=5, n_obs=n_obs, sharpe_std_across_trials=0.2
    )
    dsr_many = deflated_sharpe_ratio(
        sr, n_trials=200, n_obs=n_obs, sharpe_std_across_trials=0.2
    )
    assert dsr_many < dsr_few


def test_deflated_sharpe_ratio_is_a_probability():
    for n_trials in (1, 10, 144, 171):
        dsr = deflated_sharpe_ratio(
            0.5, n_trials=n_trials, n_obs=300,
            sharpe_std_across_trials=0.15,
        )
        assert 0.0 <= dsr <= 1.0


def test_deflated_sharpe_ratio_higher_sharpe_gives_higher_dsr():
    low = deflated_sharpe_ratio(
        0.2, n_trials=144, n_obs=500, sharpe_std_across_trials=0.2
    )
    high = deflated_sharpe_ratio(
        0.9, n_trials=144, n_obs=500, sharpe_std_across_trials=0.2
    )
    assert high > low


def test_deflated_sharpe_ratio_degenerate_zero_variance():
    # sigma_sr_hat collapses to ~0 when n_obs is at its floor; must not
    # raise a ZeroDivisionError-equivalent, and must return a boundary
    # probability consistent with sharpe_ratio vs sr_0.
    result = deflated_sharpe_ratio(0.1, n_trials=1, n_obs=1)
    assert result in (0.0, 1.0)
