"""Tests for the V1 regime-switching strategy (stage 3) and the
anomaly risk-override wrapper (stage 4)."""

import numpy as np
import pandas as pd

from atlas import allocation, anomaly, regimes, strategy

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


# --- stage 4: anomaly risk-override wrapper -------------------------

ANOMALY_CFG = {
    "optimization": {"lookback_days": 50, "covariance": "ledoit_wolf"},
    "constraints": {"per_asset_cap": 1.0},
    "anomaly": {
        "sequence_length": 5,
        "refit_frequency_days": 63,
        "blend_to_gmv": 0.5,
    },
}


class _DummyAEResult:
    def __init__(self, seq_len=5, threshold=1.0):
        self.seq_len = seq_len
        self.threshold = threshold


def _base_fn(as_of, window):
    return pd.Series({"A": 0.5, "B": 0.3, "C": 0.2})


def _synthetic_returns(n=200, seed=5):
    dates = pd.bdate_range("2020-01-01", periods=n)
    rng = np.random.default_rng(seed)
    data = {t: rng.normal(0.0004, 0.01, n) for t in ["A", "B", "C"]}
    return pd.DataFrame(data, index=dates)


def test_anomaly_override_passes_through_when_not_flagged(monkeypatch):
    monkeypatch.setattr(anomaly, "has_enough_history", lambda *a, **k: True)
    monkeypatch.setattr(
        anomaly, "fit_autoencoder", lambda *a, **k: _DummyAEResult()
    )
    monkeypatch.setattr(
        anomaly, "reconstruction_error", lambda *a, **k: np.array([0.1])
    )
    monkeypatch.setattr(
        anomaly, "is_anomalous", lambda errors, threshold: np.array([False])
    )

    returns = _synthetic_returns()
    wrapped = strategy.with_anomaly_override(_base_fn, ANOMALY_CFG)
    weights = wrapped(returns.index[-1], returns)

    pd.testing.assert_series_equal(weights, _base_fn(None, None))


def test_anomaly_override_blends_toward_gmv_when_flagged(monkeypatch):
    monkeypatch.setattr(anomaly, "has_enough_history", lambda *a, **k: True)
    monkeypatch.setattr(
        anomaly, "fit_autoencoder", lambda *a, **k: _DummyAEResult()
    )
    monkeypatch.setattr(
        anomaly, "reconstruction_error", lambda *a, **k: np.array([5.0])
    )
    monkeypatch.setattr(
        anomaly, "is_anomalous", lambda errors, threshold: np.array([True])
    )

    returns = _synthetic_returns()
    wrapped = strategy.with_anomaly_override(_base_fn, ANOMALY_CFG)
    weights = wrapped(returns.index[-1], returns)

    recent = returns.tail(ANOMALY_CFG["optimization"]["lookback_days"])
    cov = allocation.covariance_matrix(recent, method="ledoit_wolf")
    gmv_weights = allocation.gmv(cov, ANOMALY_CFG)
    base_weights = _base_fn(None, None).reindex(gmv_weights.index).fillna(0.0)
    expected = 0.5 * base_weights + 0.5 * gmv_weights

    pd.testing.assert_series_equal(
        weights.sort_index(), expected.sort_index()
    )


def test_anomaly_override_refits_only_after_frequency_elapses(monkeypatch):
    call_count = {"n": 0}

    def _fake_fit(features, cfg):
        call_count["n"] += 1
        return _DummyAEResult()

    monkeypatch.setattr(anomaly, "has_enough_history", lambda *a, **k: True)
    monkeypatch.setattr(anomaly, "fit_autoencoder", _fake_fit)
    monkeypatch.setattr(
        anomaly, "reconstruction_error", lambda *a, **k: np.array([0.1])
    )
    monkeypatch.setattr(
        anomaly, "is_anomalous", lambda errors, threshold: np.array([False])
    )

    returns = _synthetic_returns()
    wrapped = strategy.with_anomaly_override(_base_fn, ANOMALY_CFG)

    day0 = returns.index[100]
    day_soon = returns.index[110]  # well within refit_frequency_days=63
    day_later = day0 + pd.Timedelta(days=100)  # past the refit window

    wrapped(day0, returns)
    wrapped(day_soon, returns)
    assert call_count["n"] == 1

    wrapped(day_later, returns)
    assert call_count["n"] == 2


def test_anomaly_override_degenerate_fit_falls_back_to_base(monkeypatch):
    def _boom(features, cfg):
        raise ValueError("training diverged")

    monkeypatch.setattr(anomaly, "has_enough_history", lambda *a, **k: True)
    monkeypatch.setattr(anomaly, "fit_autoencoder", _boom)

    returns = _synthetic_returns()
    wrapped = strategy.with_anomaly_override(_base_fn, ANOMALY_CFG)
    weights = wrapped(returns.index[-1], returns)

    pd.testing.assert_series_equal(weights, _base_fn(None, None))


def test_anomaly_override_passes_through_with_too_little_history():
    # Real (non-monkeypatched) has_enough_history gate: far below the
    # anomaly module's minimum-observations floor.
    returns = _synthetic_returns(n=10)
    wrapped = strategy.with_anomaly_override(_base_fn, ANOMALY_CFG)
    weights = wrapped(returns.index[-1], returns)

    pd.testing.assert_series_equal(weights, _base_fn(None, None))
