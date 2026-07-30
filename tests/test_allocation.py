"""Constraint tests for allocation.py (stages 2-3, 8)."""

import numpy as np
import pandas as pd
import pytest

from atlas.allocation import (
    apply_defensive_tilt,
    black_litterman,
    covariance_matrix,
    equilibrium_returns,
    gmv,
    hrp,
    max_sharpe,
    mean_returns,
    permanent,
    risk_parity,
    sixty_forty,
    tracking_error_min,
    utility_select,
)

CFG = {"constraints": {"per_asset_cap": 0.6}}


@pytest.fixture
def synthetic_returns():
    rng = np.random.default_rng(42)
    tickers = ["A", "B", "C", "D"]
    data = rng.normal(0.0004, 0.01, size=(500, 4))
    return pd.DataFrame(data, columns=tickers)


def _assert_valid_weights(w: pd.Series, cap: float = 0.6):
    assert w.sum() == pytest.approx(1.0, abs=1e-6)
    assert (w >= -1e-9).all()
    assert (w <= cap + 1e-6).all()


def test_max_sharpe_respects_constraints(synthetic_returns):
    mu = mean_returns(synthetic_returns)
    cov = covariance_matrix(synthetic_returns)
    _assert_valid_weights(max_sharpe(mu, cov, CFG))


def test_max_sharpe_cash_degeneracy_without_exclude():
    # A near-zero-vol, near-zero-return "cash" asset gets a large
    # weight purely from the rf=0 Sharpe objective's degeneracy, not
    # because of any real expected-return edge (stage 11 finding).
    mu = pd.Series({"A": 0.05, "B": 0.08, "CASH": 0.0001})
    cov = pd.DataFrame(
        [[0.04, 0.01, 0.0], [0.01, 0.09, 0.0], [0.0, 0.0, 1e-5]],
        index=["A", "B", "CASH"],
        columns=["A", "B", "CASH"],
    )
    w = max_sharpe(mu, cov, CFG)
    assert w["CASH"] > 0.2


def test_max_sharpe_exclude_removes_cash_and_renormalizes():
    mu = pd.Series({"A": 0.05, "B": 0.08, "CASH": 0.0001})
    cov = pd.DataFrame(
        [[0.04, 0.01, 0.0], [0.01, 0.09, 0.0], [0.0, 0.0, 1e-5]],
        index=["A", "B", "CASH"],
        columns=["A", "B", "CASH"],
    )
    w = max_sharpe(mu, cov, CFG, exclude=["CASH"])
    assert w["CASH"] == 0.0
    _assert_valid_weights(w)


def test_max_sharpe_exclude_none_matches_default_behavior(synthetic_returns):
    mu = mean_returns(synthetic_returns)
    cov = covariance_matrix(synthetic_returns)
    w_default = max_sharpe(mu, cov, CFG)
    w_explicit_none = max_sharpe(mu, cov, CFG, exclude=None)
    pd.testing.assert_series_equal(w_default, w_explicit_none)


def test_gmv_respects_constraints(synthetic_returns):
    cov = covariance_matrix(synthetic_returns)
    _assert_valid_weights(gmv(cov, CFG))


def test_gmv_variance_beats_equal_weight(synthetic_returns):
    cov = covariance_matrix(synthetic_returns)
    n = len(cov)
    equal_weight = pd.Series(1.0 / n, index=cov.index)
    equal_var = equal_weight @ cov.values @ equal_weight

    w = gmv(cov, CFG)
    gmv_var = w.to_numpy() @ cov.to_numpy() @ w.to_numpy()

    assert gmv_var <= equal_var + 1e-9


def test_risk_parity_respects_constraints(synthetic_returns):
    cov = covariance_matrix(synthetic_returns)
    _assert_valid_weights(risk_parity(cov, CFG))


def test_risk_parity_contributions_are_approximately_equal(synthetic_returns):
    cov = covariance_matrix(synthetic_returns)
    w = risk_parity(cov, CFG)
    risk_contrib = w.to_numpy() * (cov.to_numpy() @ w.to_numpy())
    assert risk_contrib.std() < 1e-3


def test_hrp_respects_constraints(synthetic_returns):
    _assert_valid_weights(hrp(synthetic_returns, CFG))


def test_hrp_equal_weight_for_iid_equal_variance_assets():
    # Uncorrelated assets with identical variance: HRP should reduce
    # to (approximately) equal weight, since no cluster dominates.
    rng = np.random.default_rng(42)
    tickers = ["A", "B", "C", "D"]
    data = rng.normal(0.0, 0.01, size=(1000, 4))
    returns = pd.DataFrame(data, columns=tickers)
    w = hrp(returns, CFG)
    assert w.to_numpy() == pytest.approx([0.25] * 4, abs=0.05)


def test_hrp_caps_a_dominant_low_variance_asset():
    # One near-riskless asset (tiny variance) should get capped, not
    # allowed to swallow the whole portfolio (the BIL/T-bill failure
    # mode vanilla HRP is prone to).
    rng = np.random.default_rng(5)
    tickers = ["RISKY_A", "RISKY_B", "RISKY_C", "CASH_LIKE"]
    data = np.column_stack([
        rng.normal(0.0, 0.015, 500),
        rng.normal(0.0, 0.016, 500),
        rng.normal(0.0, 0.017, 500),
        rng.normal(0.0, 0.0002, 500),
    ])
    returns = pd.DataFrame(data, columns=tickers)
    w = hrp(returns, CFG)
    _assert_valid_weights(w)
    assert w["CASH_LIKE"] <= CFG["constraints"]["per_asset_cap"] + 1e-6


def test_tracking_error_min_recovers_feasible_benchmark(synthetic_returns):
    cov = covariance_matrix(synthetic_returns)
    benchmark = pd.Series(0.25, index=cov.index)
    w = tracking_error_min(cov, benchmark, CFG)
    pd.testing.assert_series_equal(w, benchmark, check_exact=False, atol=1e-4)


def test_permanent_splits_25_each():
    class_bucket = pd.Series(
        {
            "SPY": "equity",
            "AGG": "fixed_income",
            "GLD": "commodity",
            "BIL": "cash",
        }
    )
    w = permanent(class_bucket)
    assert w.sum() == pytest.approx(1.0)
    assert w.to_numpy() == pytest.approx([0.25, 0.25, 0.25, 0.25])


def test_permanent_splits_equally_within_bucket():
    class_bucket = pd.Series(
        {"SPY": "equity", "QQQ": "equity", "AGG": "fixed_income",
         "GLD": "commodity", "BIL": "cash"}
    )
    w = permanent(class_bucket)
    assert w["SPY"] == pytest.approx(0.125)
    assert w["QQQ"] == pytest.approx(0.125)
    assert w["AGG"] == pytest.approx(0.25)


def test_sixty_forty_split():
    class_bucket = pd.Series({"SPY": "equity", "AGG": "fixed_income"})
    w = sixty_forty(class_bucket)
    assert w["SPY"] == pytest.approx(0.6)
    assert w["AGG"] == pytest.approx(0.4)


def test_utility_select_picks_higher_utility_candidate():
    mu = pd.Series({"A": 0.10, "B": 0.10})
    cov = pd.DataFrame(
        {"A": [0.04, 0.0], "B": [0.0, 0.01]}, index=["A", "B"]
    )
    candidates = {
        "risky": pd.Series({"A": 1.0, "B": 0.0}),
        "safe": pd.Series({"A": 0.0, "B": 1.0}),
    }
    # Same expected return, lower vol -> "safe" must win on utility.
    name, weights = utility_select(candidates, mu, cov, risk_aversion=5)
    assert name == "safe"
    pd.testing.assert_series_equal(weights, candidates["safe"])


def test_apply_defensive_tilt_shifts_weight_into_target_bucket():
    class_bucket = pd.Series(
        {"A": "equity", "B": "equity", "C": "fixed_income"}
    )
    weights = pd.Series({"A": 0.5, "B": 0.3, "C": 0.2})
    cfg = {"constraints": {"per_asset_cap": 1.0}}

    tilted = apply_defensive_tilt(
        weights, class_bucket, {"fixed_income": 0.10}, cfg
    )

    assert tilted.sum() == pytest.approx(1.0)
    assert tilted["C"] == pytest.approx(0.30, abs=1e-6)
    # Funded pro rata out of equity: the A:B ratio must be preserved.
    assert tilted["A"] / tilted["B"] == pytest.approx(0.5 / 0.3)


def test_apply_defensive_tilt_respects_cap():
    class_bucket = pd.Series({"A": "equity", "B": "fixed_income"})
    weights = pd.Series({"A": 0.85, "B": 0.15})
    cfg = {"constraints": {"per_asset_cap": 0.6}}

    tilted = apply_defensive_tilt(
        weights, class_bucket, {"fixed_income": 0.30}, cfg
    )

    assert tilted.sum() == pytest.approx(1.0)
    assert (tilted <= 0.6 + 1e-6).all()


# --- stage 8: Black-Litterman fusion ----------------------------------

BL_TICKERS = ["A", "B"]
BL_COV = pd.DataFrame(
    {"A": [0.04, 0.01], "B": [0.01, 0.09]}, index=BL_TICKERS
)


def test_equilibrium_returns_known_formula():
    market_weights = pd.Series({"A": 0.5, "B": 0.5})
    delta = 2.5
    prior = equilibrium_returns(market_weights, BL_COV, delta)

    expected = delta * (BL_COV.to_numpy() @ market_weights.to_numpy())
    np.testing.assert_allclose(prior.reindex(BL_TICKERS).to_numpy(), expected)


def test_black_litterman_no_views_returns_prior_unchanged():
    prior = pd.Series({"A": 0.06, "B": 0.12})
    posterior = black_litterman(prior, BL_COV, None, None, None, tau=0.05)
    pd.testing.assert_series_equal(posterior, prior)


def test_black_litterman_high_confidence_view_pins_posterior_to_view():
    prior = pd.Series({"A": 0.06, "B": 0.12})
    P = np.array([[1.0, 0.0]])
    Q = np.array([0.20])
    Omega = np.array([[1e-8]])  # near-zero uncertainty

    posterior = black_litterman(prior, BL_COV, P, Q, Omega, tau=0.05)

    assert posterior["A"] == pytest.approx(0.20, abs=1e-3)
    # Correlated asset B must also shift (BL propagates views through
    # the covariance structure, not just onto the viewed asset).
    assert posterior["B"] > prior["B"]


def test_black_litterman_matches_hand_computed_two_asset_example():
    prior = pd.Series({"A": 0.06, "B": 0.12})
    tau = 0.05
    P = np.array([[1.0, 0.0]])
    Q = np.array([0.10])
    Omega = np.array([[0.02]])

    posterior = black_litterman(prior, BL_COV, P, Q, Omega, tau)

    tau_sigma = tau * BL_COV.to_numpy()
    tau_sigma_inv = np.linalg.inv(tau_sigma)
    omega_inv = np.linalg.inv(Omega)
    precision = tau_sigma_inv + P.T @ omega_inv @ P
    expected = np.linalg.inv(precision) @ (
        tau_sigma_inv @ prior.to_numpy() + P.T @ omega_inv @ Q
    )
    np.testing.assert_allclose(posterior.to_numpy(), expected)
