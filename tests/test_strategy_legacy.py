"""Tests for the superseded stage 3/5/6 strategy wrappers, moved to
strategy_legacy.py alongside the code they test (S11 Step 5,
ANALYSIS_V2.md action 6): regime_switching_strategy (stage 3),
with_meanreversion_tilt (stage 5), with_technical_view +
technical_phase_flags (stage 6). Stage 4 (anomaly override) and stage
7 (sentiment) remain live and stay in strategy.py / test_strategy.py.
"""

import numpy as np
import pandas as pd

from atlas import (
    allocation,
    meanreversion,
    regimes,
    strategy_legacy,
    technicals,
)

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
    "constraints": {"per_asset_cap": 0.6, "per_class_cap": 0.6},
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


def _base_fn(as_of, window):
    return pd.Series({"A": 0.5, "B": 0.3, "C": 0.2})


def _synthetic_returns(n=200, seed=5):
    dates = pd.bdate_range("2020-01-01", periods=n)
    rng = np.random.default_rng(seed)
    data = {t: rng.normal(0.0004, 0.01, n) for t in ["A", "B", "C"]}
    return pd.DataFrame(data, index=dates)


# --- stage 3: regime-switching (posture-switch) strategy --------------


def test_regime_switching_falls_back_when_history_is_too_short():
    # Only a handful of days: not enough for a meaningful HMM fit.
    # Must fall back to neutral (risk_parity) instead of crashing --
    # this is exactly the "first weeks of a backtest" cold-start case.
    returns = _iid_returns(n=10, seed=9)
    strategy_fn = strategy_legacy.regime_switching_strategy(
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
    strategy_fn = strategy_legacy.regime_switching_strategy(
        CLASS_BUCKET, CFG, POSTURE_CFG
    )
    weights = strategy_fn(returns.index[-1], returns)

    assert abs(weights.sum() - 1.0) < 1e-6
    assert (weights >= -1e-9).all()


def test_regime_switching_strategy_returns_valid_weights():
    returns = _iid_returns()
    strategy_fn = strategy_legacy.regime_switching_strategy(
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

    strategy_fn = strategy_legacy.regime_switching_strategy(
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


# --- stage 5: mean-reversion (V2) tilt wrapper -----------------------

MEANREV_CFG = {
    "optimization": {"lookback_days": 50, "covariance": "ledoit_wolf"},
    "constraints": {"per_asset_cap": 1.0},
    "meanreversion": {
        "lookback_days": 40,
        "entry_z": 2.0,
        "exit_z": 0.5,
        "max_half_life_days": 60,
        "adf_lookback_days": 100,
        "adf_pvalue_threshold": 0.05,
        "vol_filter": False,
        "vol_filter_percentile": 75,
        "max_tilt_pp": 0.05,
    },
}


def test_meanreversion_tilt_passes_through_when_view_is_zero(monkeypatch):
    monkeypatch.setattr(
        meanreversion, "has_enough_history", lambda *a, **k: True
    )
    monkeypatch.setattr(
        meanreversion,
        "mean_reversion_view",
        lambda *a, **k: pd.Series({"A": 0.0, "B": 0.0, "C": 0.0}),
    )

    returns = _synthetic_returns()
    wrapped = strategy_legacy.with_meanreversion_tilt(_base_fn, MEANREV_CFG)
    weights = wrapped(returns.index[-1], returns)

    pd.testing.assert_series_equal(weights, _base_fn(None, None))


def test_meanreversion_tilt_shifts_weight_toward_positive_view(monkeypatch):
    monkeypatch.setattr(
        meanreversion, "has_enough_history", lambda *a, **k: True
    )
    monkeypatch.setattr(
        meanreversion,
        "mean_reversion_view",
        lambda *a, **k: pd.Series({"A": 1.0, "B": -1.0, "C": 0.0}),
    )

    returns = _synthetic_returns()
    max_tilt = MEANREV_CFG["meanreversion"]["max_tilt_pp"]
    wrapped = strategy_legacy.with_meanreversion_tilt(_base_fn, MEANREV_CFG)
    weights = wrapped(returns.index[-1], returns)

    base = _base_fn(None, None)
    expected = pd.Series(
        {
            "A": base["A"] + max_tilt,
            "B": base["B"] - max_tilt,
            "C": base["C"],
        }
    )
    expected = expected / expected.sum()
    pd.testing.assert_series_equal(
        weights.sort_index(), expected.sort_index()
    )


def test_meanreversion_tilt_respects_cap(monkeypatch):
    monkeypatch.setattr(
        meanreversion, "has_enough_history", lambda *a, **k: True
    )
    monkeypatch.setattr(
        meanreversion,
        "mean_reversion_view",
        lambda *a, **k: pd.Series({"A": 1.0, "B": -1.0, "C": 0.0}),
    )

    returns = _synthetic_returns()
    tight_cfg = {
        **MEANREV_CFG,
        "constraints": {"per_asset_cap": 0.52},
    }
    wrapped = strategy_legacy.with_meanreversion_tilt(_base_fn, tight_cfg)
    weights = wrapped(returns.index[-1], returns)

    assert (weights <= 0.52 + 1e-9).all()
    assert abs(weights.sum() - 1.0) < 1e-6


def test_meanreversion_tilt_degenerate_fit_falls_back_to_base(monkeypatch):
    monkeypatch.setattr(
        meanreversion, "has_enough_history", lambda *a, **k: True
    )

    def _boom(*args, **kwargs):
        raise ValueError("ADF blew up")

    monkeypatch.setattr(meanreversion, "mean_reversion_view", _boom)

    returns = _synthetic_returns()
    wrapped = strategy_legacy.with_meanreversion_tilt(_base_fn, MEANREV_CFG)
    weights = wrapped(returns.index[-1], returns)

    pd.testing.assert_series_equal(weights, _base_fn(None, None))


def test_meanreversion_tilt_passes_through_with_too_little_history():
    # Real (non-monkeypatched) has_enough_history gate.
    returns = _synthetic_returns(n=10)
    wrapped = strategy_legacy.with_meanreversion_tilt(_base_fn, MEANREV_CFG)
    weights = wrapped(returns.index[-1], returns)

    pd.testing.assert_series_equal(weights, _base_fn(None, None))


# --- stage 6: technical (V3) tilt + phase flags ----------------------

TECH_CFG = {
    "optimization": {"lookback_days": 50, "covariance": "ledoit_wolf"},
    "constraints": {"per_asset_cap": 1.0},
    "technicals": {
        "pivot_order": 5,
        "sr_cluster_k": 2,
        "zone_width_bps": 100,
        "max_view_magnitude": 0.02,
        "min_history_days": 60,
        "phase_fraction": 0.5,
    },
}


def test_technical_view_passes_through_when_view_is_zero(monkeypatch):
    monkeypatch.setattr(
        technicals, "has_enough_history", lambda *a, **k: True
    )
    monkeypatch.setattr(
        technicals,
        "technical_view",
        lambda *a, **k: pd.Series({"A": 0.0, "B": 0.0, "C": 0.0}),
    )

    returns = _synthetic_returns()
    wrapped = strategy_legacy.with_technical_view(_base_fn, TECH_CFG)
    weights = wrapped(returns.index[-1], returns)

    pd.testing.assert_series_equal(weights, _base_fn(None, None))


def test_technical_view_shifts_weight_toward_positive_view(monkeypatch):
    monkeypatch.setattr(
        technicals, "has_enough_history", lambda *a, **k: True
    )
    monkeypatch.setattr(
        technicals,
        "technical_view",
        lambda *a, **k: pd.Series({"A": 1.0, "B": -1.0, "C": 0.0}),
    )

    returns = _synthetic_returns()
    max_magnitude = TECH_CFG["technicals"]["max_view_magnitude"]
    wrapped = strategy_legacy.with_technical_view(_base_fn, TECH_CFG)
    weights = wrapped(returns.index[-1], returns)

    base = _base_fn(None, None)
    expected = pd.Series(
        {
            "A": base["A"] + max_magnitude,
            "B": base["B"] - max_magnitude,
            "C": base["C"],
        }
    )
    expected = expected / expected.sum()
    pd.testing.assert_series_equal(
        weights.sort_index(), expected.sort_index()
    )


def test_technical_view_degenerate_fit_falls_back_to_base(monkeypatch):
    monkeypatch.setattr(
        technicals, "has_enough_history", lambda *a, **k: True
    )

    def _boom(*args, **kwargs):
        raise ValueError("KMeans blew up")

    monkeypatch.setattr(technicals, "technical_view", _boom)

    returns = _synthetic_returns()
    wrapped = strategy_legacy.with_technical_view(_base_fn, TECH_CFG)
    weights = wrapped(returns.index[-1], returns)

    pd.testing.assert_series_equal(weights, _base_fn(None, None))


def test_technical_view_passes_through_with_too_little_history():
    returns = _synthetic_returns(n=10)
    wrapped = strategy_legacy.with_technical_view(_base_fn, TECH_CFG)
    weights = wrapped(returns.index[-1], returns)

    pd.testing.assert_series_equal(weights, _base_fn(None, None))


def test_technical_phase_flags_returns_resistance_tickers(monkeypatch):
    diag = pd.DataFrame(
        {"role": ["resistance", "support", None]},
        index=["A", "B", "C"],
    )
    monkeypatch.setattr(
        technicals, "has_enough_history", lambda *a, **k: True
    )
    monkeypatch.setattr(technicals, "technical_signal", lambda *a, **k: diag)

    returns = _synthetic_returns()
    phase_flags_fn = strategy_legacy.technical_phase_flags(TECH_CFG)
    flags = phase_flags_fn(returns.index[-1], returns)

    assert flags == {"A"}


def test_technical_phase_flags_empty_with_too_little_history():
    returns = _synthetic_returns(n=10)
    phase_flags_fn = strategy_legacy.technical_phase_flags(TECH_CFG)
    flags = phase_flags_fn(returns.index[-1], returns)

    assert flags == set()


def test_technical_phase_flags_empty_on_degenerate_fit(monkeypatch):
    monkeypatch.setattr(
        technicals, "has_enough_history", lambda *a, **k: True
    )

    def _boom(*args, **kwargs):
        raise ValueError("KMeans blew up")

    monkeypatch.setattr(technicals, "technical_signal", _boom)

    returns = _synthetic_returns()
    phase_flags_fn = strategy_legacy.technical_phase_flags(TECH_CFG)
    flags = phase_flags_fn(returns.index[-1], returns)

    assert flags == set()
