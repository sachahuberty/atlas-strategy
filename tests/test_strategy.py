"""Tests for the V1 regime-switching strategy (stage 3), the anomaly
risk-override wrapper (stage 4), the mean-reversion tilt (stage 5),
the technical tilt + phase flags (stage 6), the sentiment tilt
(stage 7), and the Black-Litterman fusion strategy (stage 8).
"""

import copy

import numpy as np
import pandas as pd
import pytest

from atlas import (
    allocation,
    anomaly,
    meanreversion,
    regimes,
    strategy,
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
    wrapped = strategy.with_meanreversion_tilt(_base_fn, MEANREV_CFG)
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
    wrapped = strategy.with_meanreversion_tilt(_base_fn, MEANREV_CFG)
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
    wrapped = strategy.with_meanreversion_tilt(_base_fn, tight_cfg)
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
    wrapped = strategy.with_meanreversion_tilt(_base_fn, MEANREV_CFG)
    weights = wrapped(returns.index[-1], returns)

    pd.testing.assert_series_equal(weights, _base_fn(None, None))


def test_meanreversion_tilt_passes_through_with_too_little_history():
    # Real (non-monkeypatched) has_enough_history gate.
    returns = _synthetic_returns(n=10)
    wrapped = strategy.with_meanreversion_tilt(_base_fn, MEANREV_CFG)
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
    wrapped = strategy.with_technical_view(_base_fn, TECH_CFG)
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
    wrapped = strategy.with_technical_view(_base_fn, TECH_CFG)
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
    wrapped = strategy.with_technical_view(_base_fn, TECH_CFG)
    weights = wrapped(returns.index[-1], returns)

    pd.testing.assert_series_equal(weights, _base_fn(None, None))


def test_technical_view_passes_through_with_too_little_history():
    returns = _synthetic_returns(n=10)
    wrapped = strategy.with_technical_view(_base_fn, TECH_CFG)
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
    phase_flags_fn = strategy.technical_phase_flags(TECH_CFG)
    flags = phase_flags_fn(returns.index[-1], returns)

    assert flags == {"A"}


def test_technical_phase_flags_empty_with_too_little_history():
    returns = _synthetic_returns(n=10)
    phase_flags_fn = strategy.technical_phase_flags(TECH_CFG)
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
    phase_flags_fn = strategy.technical_phase_flags(TECH_CFG)
    flags = phase_flags_fn(returns.index[-1], returns)

    assert flags == set()


# --- stage 7: sentiment (V4) tilt -------------------------------------

SENTIMENT_CFG = {
    "constraints": {"per_asset_cap": 1.0},
    "sentiment": {"max_view_magnitude": 0.01},
}

SENTIMENT_CLASS_BUCKET = pd.Series(
    {"A": "equity", "B": "fixed_income", "C": "cash"}
)


def test_sentiment_view_passes_through_when_scores_are_empty():
    empty_scores = pd.Series(dtype=float)
    wrapped = strategy.with_sentiment_view(
        _base_fn, SENTIMENT_CFG, SENTIMENT_CLASS_BUCKET, empty_scores
    )
    returns = _synthetic_returns()
    weights = wrapped(returns.index[-1], returns)

    pd.testing.assert_series_equal(weights, _base_fn(None, None))


def test_sentiment_view_shifts_weight_by_bucket_score():
    bucket_scores = pd.Series({"equity": 0.8, "fixed_income": -0.6})
    wrapped = strategy.with_sentiment_view(
        _base_fn, SENTIMENT_CFG, SENTIMENT_CLASS_BUCKET, bucket_scores
    )
    returns = _synthetic_returns()
    weights = wrapped(returns.index[-1], returns)

    base = _base_fn(None, None)
    max_magnitude = SENTIMENT_CFG["sentiment"]["max_view_magnitude"]
    expected = pd.Series(
        {
            "A": base["A"] + 0.8 * max_magnitude,
            "B": base["B"] - 0.6 * max_magnitude,
            "C": base["C"],
        }
    )
    expected = expected / expected.sum()
    pd.testing.assert_series_equal(
        weights.sort_index(), expected.sort_index()
    )


def test_sentiment_view_respects_cap():
    # Realistic magnitude (VADER's full +/-1.0 range, but the small
    # max_view_magnitude sentiment_view is actually configured with,
    # matching V2/V3's tilt scale) -- a mild, easily-resolvable cap
    # breach, not the pathological case where an extreme tilt clips an
    # asset to exactly zero before capping even runs (that starves it
    # of any room in cap_and_renormalize's pro-rata redistribution,
    # which is a separate, known limitation, not what this test is
    # checking).
    bucket_scores = pd.Series({"equity": 1.0, "fixed_income": -1.0})
    tight_cfg = {
        "constraints": {"per_asset_cap": 0.52},
        "sentiment": {"max_view_magnitude": 0.05},
    }
    wrapped = strategy.with_sentiment_view(
        _base_fn, tight_cfg, SENTIMENT_CLASS_BUCKET, bucket_scores
    )
    returns = _synthetic_returns()
    weights = wrapped(returns.index[-1], returns)

    assert (weights <= 0.52 + 1e-9).all()
    assert abs(weights.sum() - 1.0) < 1e-6


# --- stage 8: Black-Litterman fusion strategy -------------------------

BL_CFG = {
    **CFG,
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
    "technicals": {
        "pivot_order": 5,
        "sr_cluster_k": 3,
        "zone_width_bps": 50,
        "max_view_magnitude": 0.02,
        "min_history_days": 60,
        "phase_fraction": 0.5,
    },
    "modules": {
        "regime_view": True,
        "meanreversion_view": True,
        "technical_view": True,
        "sentiment_view": True,
    },
    "black_litterman": {
        "tau": 0.05,
        "delta": 2.5,
        "regime_view_magnitude": 0.03,
        "risk_aversion_for_utility_gate": 5,
        "gate": "utility",
        "vol_floor_multiplier": 1.5,
        "view_confidence": {
            "regime": 1.0,
            "meanreversion": 0.8,
            "technical": 0.5,
            "sentiment": 0.3,
        },
    },
}


def test_black_litterman_strategy_returns_valid_weights():
    returns = _iid_returns()
    strategy_fn = strategy.black_litterman_strategy(
        CLASS_BUCKET, BL_CFG, POSTURE_CFG
    )
    weights = strategy_fn(returns.index[-1], returns)

    assert abs(weights.sum() - 1.0) < 1e-6
    assert (weights >= -1e-9).all()
    assert set(weights.index) == set(CLASS_BUCKET.index)


def test_black_litterman_strategy_falls_back_with_short_history():
    # Cold-start case: not enough data for a meaningful HMM read, so
    # posture defaults to neutral (no regime view) instead of crashing.
    returns = _iid_returns(n=10, seed=9)
    strategy_fn = strategy.black_litterman_strategy(
        CLASS_BUCKET, BL_CFG, POSTURE_CFG
    )
    weights = strategy_fn(returns.index[-1], returns)

    assert abs(weights.sum() - 1.0) < 1e-6
    assert (weights >= -1e-9).all()


def test_black_litterman_strategy_falls_back_when_hmm_fit_degenerates(
    monkeypatch,
):
    def _boom(*args, **kwargs):
        raise ValueError("startprob_ must sum to 1 (got nan)")

    monkeypatch.setattr(regimes, "market_regime", _boom)

    returns = _iid_returns()
    strategy_fn = strategy.black_litterman_strategy(
        CLASS_BUCKET, BL_CFG, POSTURE_CFG
    )
    weights = strategy_fn(returns.index[-1], returns)

    assert abs(weights.sum() - 1.0) < 1e-6
    assert (weights >= -1e-9).all()


def test_black_litterman_strategy_uses_permanent_as_market_weights(
    monkeypatch,
):
    captured = {}
    original = allocation.equilibrium_returns

    def spy(market_weights, cov, delta):
        captured["market_weights"] = market_weights.copy()
        captured["delta"] = delta
        return original(market_weights, cov, delta)

    monkeypatch.setattr(allocation, "equilibrium_returns", spy)

    returns = _iid_returns()
    strategy_fn = strategy.black_litterman_strategy(
        CLASS_BUCKET, BL_CFG, POSTURE_CFG
    )
    strategy_fn(returns.index[-1], returns)

    expected = allocation.permanent(CLASS_BUCKET)
    pd.testing.assert_series_equal(
        captured["market_weights"].sort_index(), expected.sort_index()
    )
    assert captured["delta"] == BL_CFG["black_litterman"]["delta"]


def test_black_litterman_strategy_offers_three_candidates_to_utility_select(
    monkeypatch,
):
    captured = {}
    original = allocation.utility_select

    def spy(candidates, mu, cov, risk_aversion, **kwargs):
        captured["keys"] = set(candidates.keys())
        return original(candidates, mu, cov, risk_aversion, **kwargs)

    monkeypatch.setattr(allocation, "utility_select", spy)

    returns = _iid_returns()
    strategy_fn = strategy.black_litterman_strategy(
        CLASS_BUCKET, BL_CFG, POSTURE_CFG
    )
    strategy_fn(returns.index[-1], returns)

    assert captured["keys"] == {"black_litterman", "gmv", "risk_parity"}


def test_black_litterman_strategy_overrides_cash_prior_with_rf(monkeypatch):
    captured = {}
    original = allocation.black_litterman

    def spy(prior, cov, P, Q, Omega, tau):
        captured["prior"] = prior.copy()
        return original(prior, cov, P, Q, Omega, tau)

    monkeypatch.setattr(allocation, "black_litterman", spy)

    returns = _iid_returns()
    rf_series = pd.Series(0.04, index=returns.index)
    strategy_fn = strategy.black_litterman_strategy(
        CLASS_BUCKET, BL_CFG, POSTURE_CFG, rf_series=rf_series
    )
    strategy_fn(returns.index[-1], returns)

    assert captured["prior"]["BIL"] == pytest.approx(0.04)


def test_black_litterman_strategy_defaults_rf_to_zero_without_rf_series(
    monkeypatch,
):
    captured = {}
    original = allocation.black_litterman

    def spy(prior, cov, P, Q, Omega, tau):
        captured["prior"] = prior.copy()
        return original(prior, cov, P, Q, Omega, tau)

    monkeypatch.setattr(allocation, "black_litterman", spy)

    returns = _iid_returns()
    strategy_fn = strategy.black_litterman_strategy(
        CLASS_BUCKET, BL_CFG, POSTURE_CFG
    )
    strategy_fn(returns.index[-1], returns)

    assert captured["prior"]["BIL"] == 0.0


def test_black_litterman_strategy_rf_lookup_ignores_future_values(
    monkeypatch,
):
    # A no-lookahead canary for the rf lookup: rf_series contains a
    # future rate change well after the decision date, but the
    # decision must only see what was prevailing at (or before) as_of.
    captured = []
    original = allocation.black_litterman

    def spy(prior, cov, P, Q, Omega, tau):
        captured.append(prior.copy())
        return original(prior, cov, P, Q, Omega, tau)

    monkeypatch.setattr(allocation, "black_litterman", spy)

    returns = _iid_returns()
    change_date = returns.index[900]
    rf_series = pd.Series(0.01, index=returns.index)
    rf_series.loc[change_date:] = 0.05

    strategy_fn = strategy.black_litterman_strategy(
        CLASS_BUCKET, BL_CFG, POSTURE_CFG, rf_series=rf_series
    )
    earlier_date = returns.index[700]
    strategy_fn(earlier_date, returns.loc[:earlier_date])

    assert captured[-1]["BIL"] == pytest.approx(0.01)


def test_black_litterman_strategy_rf_defaults_zero_before_series_start(
    monkeypatch,
):
    captured = {}
    original = allocation.black_litterman

    def spy(prior, cov, P, Q, Omega, tau):
        captured["prior"] = prior.copy()
        return original(prior, cov, P, Q, Omega, tau)

    monkeypatch.setattr(allocation, "black_litterman", spy)

    returns = _iid_returns()
    rf_series = pd.Series(0.03, index=returns.index[500:])

    strategy_fn = strategy.black_litterman_strategy(
        CLASS_BUCKET, BL_CFG, POSTURE_CFG, rf_series=rf_series
    )
    early_date = returns.index[100]
    strategy_fn(early_date, returns.loc[:early_date])

    assert captured["prior"]["BIL"] == 0.0


def test_black_litterman_strategy_passes_rf_to_utility_select(monkeypatch):
    captured = {}
    original = allocation.utility_select

    def spy(candidates, mu, cov, risk_aversion, **kwargs):
        captured["rf"] = kwargs.get("rf")
        return original(candidates, mu, cov, risk_aversion, **kwargs)

    monkeypatch.setattr(allocation, "utility_select", spy)

    returns = _iid_returns()
    rf_series = pd.Series(0.045, index=returns.index)
    strategy_fn = strategy.black_litterman_strategy(
        CLASS_BUCKET, BL_CFG, POSTURE_CFG, rf_series=rf_series
    )
    strategy_fn(returns.index[-1], returns)

    assert captured["rf"] == pytest.approx(0.045)


def test_black_litterman_strategy_gate_off_bypasses_utility_select(
    monkeypatch,
):
    called = {"utility_select": False}
    original = allocation.utility_select

    def spy(*args, **kwargs):
        called["utility_select"] = True
        return original(*args, **kwargs)

    monkeypatch.setattr(allocation, "utility_select", spy)

    cfg = copy.deepcopy(BL_CFG)
    cfg["black_litterman"]["gate"] = "off"

    returns = _iid_returns()
    strategy_fn = strategy.black_litterman_strategy(
        CLASS_BUCKET, cfg, POSTURE_CFG
    )
    weights = strategy_fn(returns.index[-1], returns)

    assert not called["utility_select"]
    assert weights.sum() == pytest.approx(1.0)


def test_black_litterman_strategy_gate_vol_floor(monkeypatch):
    # Distinct, easily-identifiable sentinel weight vectors so the
    # gate's choice can be read straight off the returned weights,
    # rather than needing to reason about the real BL/GMV solution.
    bl_sentinel = pd.Series({"ACWI": 0.9, "AGG": 0.0, "GLD": 0.0, "BIL": 0.1})
    gmv_sentinel = pd.Series({"ACWI": 0.1, "AGG": 0.3, "GLD": 0.3, "BIL": 0.3})
    monkeypatch.setattr(
        allocation, "max_sharpe", lambda *a, **k: bl_sentinel.copy()
    )
    monkeypatch.setattr(allocation, "gmv", lambda *a, **k: gmv_sentinel.copy())

    returns = _iid_returns()
    lookback = BL_CFG["optimization"]["lookback_days"]
    recent = returns.tail(lookback)
    cov = allocation.covariance_matrix(
        recent, method=BL_CFG["optimization"]["covariance"]
    )
    bl_vol = allocation.portfolio_vol(bl_sentinel, cov)
    gmv_vol = allocation.portfolio_vol(gmv_sentinel, cov)
    assert bl_vol > gmv_vol  # sanity check the sentinels differ as intended

    cfg = copy.deepcopy(BL_CFG)
    cfg["black_litterman"]["gate"] = "vol_floor"

    # Generous multiplier: BL's vol clears the floor -> BL selected.
    cfg["black_litterman"]["vol_floor_multiplier"] = bl_vol / gmv_vol + 1.0
    strategy_fn = strategy.black_litterman_strategy(
        CLASS_BUCKET, cfg, POSTURE_CFG
    )
    w = strategy_fn(returns.index[-1], returns)
    pd.testing.assert_series_equal(
        w.sort_index(), bl_sentinel.sort_index(), check_names=False
    )

    # Strict multiplier: BL's vol exceeds the floor -> falls back to GMV.
    cfg["black_litterman"]["vol_floor_multiplier"] = bl_vol / gmv_vol - 0.1
    strategy_fn = strategy.black_litterman_strategy(
        CLASS_BUCKET, cfg, POSTURE_CFG
    )
    w = strategy_fn(returns.index[-1], returns)
    pd.testing.assert_series_equal(
        w.sort_index(), gmv_sentinel.sort_index(), check_names=False
    )


def test_black_litterman_strategy_unknown_gate_raises():
    cfg = copy.deepcopy(BL_CFG)
    cfg["black_litterman"]["gate"] = "not_a_real_gate"

    returns = _iid_returns()
    strategy_fn = strategy.black_litterman_strategy(
        CLASS_BUCKET, cfg, POSTURE_CFG
    )
    with pytest.raises(ValueError, match="not_a_real_gate"):
        strategy_fn(returns.index[-1], returns)
