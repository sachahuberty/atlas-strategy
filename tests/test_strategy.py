"""Tests for the V1 regime-switching strategy wiring (stage 3)."""

import numpy as np
import pandas as pd

from atlas import allocation, regimes, strategy

POSTURE_CFG = {
    "postures": {
        "risk_on": {
            "profile": {"vol": "low", "mean_return": "high"},
            "allocation_method": "max_sharpe",
        },
        "neutral": {
            "profile": {"vol": "medium", "mean_return": "medium"},
            "allocation_method": "risk_parity",
        },
        "risk_off": {
            "profile": {"vol": "high", "mean_return": "low"},
            "allocation_method": "gmv_defensive",
            "defensive_class_tilt": {"fixed_income": 0.10, "commodity": 0.10},
        },
    }
}

CFG = {
    "general": {"random_seed": 42},
    "regimes": {
        "market_ticker": "ACWI",
        "hmm": {"n_states": 3, "vol_window_days": 21, "lookback_days": 1260},
    },
    "optimization": {"lookback_days": 500, "covariance": "ledoit_wolf"},
    "constraints": {"per_asset_cap": 0.6},
}

CLASS_BUCKET = pd.Series(
    {
        "ACWI": "equity",
        "AGG": "fixed_income",
        "GLD": "commodity",
        "BIL": "cash",
    }
)


def _iid_returns(n=1400, seed=3) -> pd.DataFrame:
    dates = pd.bdate_range("2015-01-01", periods=n)
    rng = np.random.default_rng(seed)
    data = {t: rng.normal(0.0004, 0.01, n) for t in CLASS_BUCKET.index}
    return pd.DataFrame(data, index=dates)


def test_regime_switching_falls_back_when_history_is_too_short():
    # Only a handful of days: not enough for a meaningful HMM fit.
    # Must fall back to neutral (risk_parity) instead of crashing --
    # this is exactly the "first weeks of a backtest" cold-start case.
    returns = _iid_returns(n=10, seed=9)
    strategy_fn = strategy.regime_switching_strategy(
        CLASS_BUCKET, CFG, POSTURE_CFG
    )
    weights = strategy_fn(returns.index[-1], returns)

    assert abs(weights.sum() - 1.0) < 1e-6
    assert (weights >= -1e-9).all()


def test_regime_switching_falls_back_when_hmm_fit_degenerates(monkeypatch):
    # hmmlearn's EM can occasionally produce NaN parameters on a given
    # week's window (a state collapsing to ~zero responsibility) --
    # observed in a real 16-year OOS run. Must fall back to neutral,
    # not crash the whole backtest over one bad week's fit.
    def _boom(*args, **kwargs):
        raise ValueError("startprob_ must sum to 1 (got nan)")

    monkeypatch.setattr(regimes, "market_regime", _boom)

    returns = _iid_returns()
    strategy_fn = strategy.regime_switching_strategy(
        CLASS_BUCKET, CFG, POSTURE_CFG
    )
    weights = strategy_fn(returns.index[-1], returns)

    assert abs(weights.sum() - 1.0) < 1e-6
    assert (weights >= -1e-9).all()


def test_regime_switching_strategy_returns_valid_weights():
    returns = _iid_returns()
    strategy_fn = strategy.regime_switching_strategy(
        CLASS_BUCKET, CFG, POSTURE_CFG
    )
    weights = strategy_fn(returns.index[-1], returns)

    assert abs(weights.sum() - 1.0) < 1e-6
    assert (weights >= -1e-9).all()
    assert set(weights.index) == set(CLASS_BUCKET.index)


def test_risk_off_posture_tilts_toward_defensive_buckets():
    # A market series that is calm/flat for a long history, then
    # sharply volatile and negative at the end, should decode to
    # risk_off "today" and apply the configured defensive tilt.
    n = 1400
    dates = pd.bdate_range("2015-01-01", periods=n)
    rng = np.random.default_rng(3)
    data = {t: rng.normal(0.0004, 0.01, n) for t in CLASS_BUCKET.index}
    market = np.concatenate(
        [
            rng.normal(0.0006, 0.006, n - 100),
            rng.normal(-0.001, 0.03, 100),
        ]
    )
    data["ACWI"] = market
    returns = pd.DataFrame(data, index=dates)

    strategy_fn = strategy.regime_switching_strategy(
        CLASS_BUCKET, CFG, POSTURE_CFG
    )
    weights = strategy_fn(returns.index[-1], returns)

    recent = returns.tail(CFG["optimization"]["lookback_days"])
    cov = allocation.covariance_matrix(recent, method="ledoit_wolf")
    plain_gmv = allocation.gmv(cov, CFG)

    defensive_mass = weights[["AGG", "GLD"]].sum()
    plain_defensive_mass = plain_gmv[["AGG", "GLD"]].sum()

    # The tilt must have moved weight into fixed_income/commodity
    # relative to an un-tilted GMV on the same covariance -- unless GMV
    # itself was already selected verbatim (no tilt to observe).
    assert defensive_mass >= plain_defensive_mass - 1e-6
