"""Tests for the anomaly risk-override wrapper (stage 4), the
sentiment tilt (stage 7), and the Black-Litterman fusion strategy
(stage 8/11). Stage 3/5/6 (regime-switching, mean-reversion tilt,
technical tilt + phase flags) moved to test_strategy_legacy.py
alongside strategy_legacy.py (S11 Step 5, ANALYSIS_V2.md action 6).
"""

import copy

import numpy as np
import pandas as pd
import pytest

from atlas import (
    allocation,
    anomaly,
    meanreversion,
    metrics,
    momentum,
    regimes,
    strategy,
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
    "momentum": {
        "lookback_days": 60,
        "skip_days": 5,
        "trend_lookback_days": 60,
        "top_fraction": 0.5,
        "max_view_magnitude": 0.03,
    },
    "modules": {
        "regime_view": True,
        "meanreversion_view": True,
        "technical_view": True,
        "sentiment_view": True,
        "momentum_view": True,
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
            "momentum": 0.8,
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


def test_black_litterman_strategy_hmm_degeneracy_surfaced_in_diagnostics(
    monkeypatch,
):
    # ANALYSIS_V4 Sec 2.3a: a degenerate HMM fit must not fail silently
    # -- the diagnostics sink carries an explicit reason, not just an
    # inferable absence of "current_state".
    def _boom(*args, **kwargs):
        raise ValueError("startprob_ must sum to 1 (got nan)")

    monkeypatch.setattr(regimes, "market_regime", _boom)

    diag = {}
    returns = _iid_returns()
    strategy_fn = strategy.black_litterman_strategy(
        CLASS_BUCKET, BL_CFG, POSTURE_CFG, record=diag.update
    )
    strategy_fn(returns.index[-1], returns)

    assert diag["regime"]["posture"] == "neutral"
    assert "current_state" not in diag["regime"]
    assert "hmm_fit_degenerate" in diag["regime"]["fallback_reason"]
    assert "startprob_" in diag["regime"]["fallback_reason"]


def test_black_litterman_strategy_thin_history_fallback_reason():
    diag = {}
    returns = _iid_returns(n=10, seed=9)
    strategy_fn = strategy.black_litterman_strategy(
        CLASS_BUCKET, BL_CFG, POSTURE_CFG, record=diag.update
    )
    strategy_fn(returns.index[-1], returns)

    assert diag["regime"]["posture"] == "neutral"
    assert diag["regime"]["fallback_reason"] == "insufficient_history"


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


def test_black_litterman_strategy_self_calibrating_delta(monkeypatch):
    captured = {}
    original = allocation.equilibrium_returns

    def spy(market_weights, cov, delta):
        captured["delta"] = delta
        return original(market_weights, cov, delta)

    monkeypatch.setattr(allocation, "equilibrium_returns", spy)

    cfg = copy.deepcopy(BL_CFG)
    cfg["black_litterman"]["delta_mode"] = "self_calibrating"

    returns = _iid_returns()
    rf_series = pd.Series(0.03, index=returns.index)
    strategy_fn = strategy.black_litterman_strategy(
        CLASS_BUCKET, cfg, POSTURE_CFG, rf_series=rf_series
    )
    strategy_fn(returns.index[-1], returns)

    lookback = cfg["optimization"]["lookback_days"]
    recent = returns.tail(lookback)
    market_weights = allocation.permanent(CLASS_BUCKET)
    aligned = market_weights.reindex(recent.columns).fillna(0.0)
    market_returns = metrics.portfolio_return(aligned, recent)
    expected_delta = allocation.implied_risk_aversion(
        market_returns, rf=0.03, fallback=cfg["black_litterman"]["delta"]
    )

    assert captured["delta"] == pytest.approx(expected_delta)
    # Sanity: this fixture's synthetic data genuinely differs from the
    # fixed constant, so this isn't accidentally testing the fallback.
    assert captured["delta"] != cfg["black_litterman"]["delta"]


def test_black_litterman_strategy_delta_mode_defaults_to_fixed(monkeypatch):
    # No delta_mode key at all (matches every pre-Step-6 config/test
    # fixture) must behave exactly as "fixed", not raise or silently
    # self-calibrate.
    captured = {}
    original = allocation.equilibrium_returns

    def spy(market_weights, cov, delta):
        captured["delta"] = delta
        return original(market_weights, cov, delta)

    monkeypatch.setattr(allocation, "equilibrium_returns", spy)

    cfg = copy.deepcopy(BL_CFG)
    assert "delta_mode" not in cfg["black_litterman"]

    returns = _iid_returns()
    strategy_fn = strategy.black_litterman_strategy(
        CLASS_BUCKET, cfg, POSTURE_CFG
    )
    strategy_fn(returns.index[-1], returns)

    assert captured["delta"] == cfg["black_litterman"]["delta"]


def test_black_litterman_strategy_unknown_delta_mode_raises():
    cfg = copy.deepcopy(BL_CFG)
    cfg["black_litterman"]["delta_mode"] = "not_a_real_mode"

    returns = _iid_returns()
    strategy_fn = strategy.black_litterman_strategy(
        CLASS_BUCKET, cfg, POSTURE_CFG
    )
    with pytest.raises(ValueError, match="not_a_real_mode"):
        strategy_fn(returns.index[-1], returns)


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


def test_black_litterman_strategy_excess_prior_leaves_cash_unoverridden(
    monkeypatch,
):
    # Default convention (ANALYSIS_V4 Sec 2.1 fix): Pi is already an
    # excess return, so cash's near-zero equilibrium Pi is correct AS
    # IS and must NOT be overridden by rf -- regardless of rf's value.
    captured = {}
    original = allocation.black_litterman

    def spy(prior, cov, P, Q, Omega, tau):
        captured["prior"] = prior.copy()
        return original(prior, cov, P, Q, Omega, tau)

    monkeypatch.setattr(allocation, "black_litterman", spy)

    returns = _iid_returns()
    rf_series = pd.Series(0.04, index=returns.index)
    assert "excess_return_prior" not in BL_CFG["black_litterman"]  # default
    strategy_fn = strategy.black_litterman_strategy(
        CLASS_BUCKET, BL_CFG, POSTURE_CFG, rf_series=rf_series
    )
    strategy_fn(returns.index[-1], returns)

    # Not overridden to rf=0.04; left at the natural (near-zero, but not
    # exactly 0 or rf) reverse-optimization value for a near-zero-vol
    # asset against this fixture's actual (noisy) covariance.
    assert captured["prior"]["BIL"] != pytest.approx(0.04)
    assert captured["prior"]["BIL"] != 0.0
    assert abs(captured["prior"]["BIL"]) < 0.03


def test_black_litterman_strategy_legacy_overrides_cash_prior_with_rf(
    monkeypatch,
):
    # excess_return_prior: false reproduces the deployed-through-2026-10
    # behavior for the Phase-2 side-by-side ablation (ANALYSIS_V4 Sec
    # 2.1/6 Tier 1 action 5).
    captured = {}
    original = allocation.black_litterman

    def spy(prior, cov, P, Q, Omega, tau):
        captured["prior"] = prior.copy()
        return original(prior, cov, P, Q, Omega, tau)

    monkeypatch.setattr(allocation, "black_litterman", spy)

    cfg = copy.deepcopy(BL_CFG)
    cfg["black_litterman"]["excess_return_prior"] = False
    returns = _iid_returns()
    rf_series = pd.Series(0.04, index=returns.index)
    strategy_fn = strategy.black_litterman_strategy(
        CLASS_BUCKET, cfg, POSTURE_CFG, rf_series=rf_series
    )
    strategy_fn(returns.index[-1], returns)

    assert captured["prior"]["BIL"] == pytest.approx(0.04)


def test_black_litterman_strategy_rf_defaults_to_zero_without_rf_series():
    # rf itself (not the cash-prior convention) defaults to 0 absent an
    # rf_series -- observed via the diagnostics sink, which is agnostic
    # to excess_return_prior.
    diag = {}
    strategy_fn = strategy.black_litterman_strategy(
        CLASS_BUCKET, BL_CFG, POSTURE_CFG, record=diag.update
    )
    returns = _iid_returns()
    strategy_fn(returns.index[-1], returns)

    assert diag["rf"] == 0.0


# --- ANALYSIS_V4 Sec 2.1/6 Tier 1 action 5: the regression test -------
# With no views at all, the corrected forward step must return the
# anchor up to the effect of the 20% per-asset cap (cash 25% -> 20%,
# the excess redistributed to the other buckets), and every risky
# asset's PURE prior (no views, no cap) must have positive expected
# excess return. `gate: off` isolates the max_sharpe forward step
# itself from the utility gate's separate choice among candidate
# books (a low-vol bucket can win the gate as GMV/risk_parity
# regardless of this fix, which is not what this test is about).
# Asset classes carry different volatilities (like the real universe:
# bonds low, equities/commodities high) -- with the three non-cash
# buckets at identical vol, as in an earlier draft of this fixture,
# the legacy double-subtraction shifts every risky asset's net return
# down by the SAME constant and happens to reproduce the anchor
# anyway; the bug's real signature only shows up when it can push a
# LOW-vol (low-Pi) bucket below cash while a HIGH-vol (high-Pi)
# bucket stays above it.

_MULTI_ASSET_CLASS_BUCKET = pd.Series(
    {
        "EQ1": "equity", "EQ2": "equity", "EQ3": "equity",
        "FI1": "fixed_income", "FI2": "fixed_income", "FI3": "fixed_income",
        "CO1": "commodity", "CO2": "commodity", "CO3": "commodity",
        "CASH": "cash",
    }
)
_MULTI_ASSET_ANN_VOL = {
    "equity": 0.20, "fixed_income": 0.03, "commodity": 0.25, "cash": 0.003,
}


def _no_view_cfg():
    cfg = copy.deepcopy(BL_CFG)
    cfg["constraints"] = {"per_asset_cap": 0.20, "per_class_cap": 0.60}
    cfg["modules"] = {
        "regime_view": False,
        "meanreversion_view": False,
        "technical_view": False,
        "sentiment_view": False,
        "momentum_view": False,
    }
    cfg["black_litterman"]["gate"] = "off"
    return cfg


def _multi_vol_returns(tickers, n=1400, seed=3) -> pd.DataFrame:
    """Independent (zero cross-covariance) returns per ticker, each at
    its own asset class's annualized vol -- isolates Pi's dependence on
    each asset's OWN variance (Pi_i ~ delta * var_i * w_mkt_i when
    cross-covariance is ~0), rather than letting correlation structure
    muddy the no-views reproduction."""
    dates = pd.bdate_range("2015-01-01", periods=n)
    rng = np.random.default_rng(seed)
    data = {}
    for t in tickers:
        daily_vol = _MULTI_ASSET_ANN_VOL[_MULTI_ASSET_CLASS_BUCKET[t]]
        daily_vol /= np.sqrt(252)
        data[t] = rng.normal(0.0004, daily_vol, n)
    return pd.DataFrame(data, index=dates)


def test_black_litterman_strategy_no_views_returns_capped_anchor():
    cfg = _no_view_cfg()
    returns = _multi_vol_returns(_MULTI_ASSET_CLASS_BUCKET.index)
    rf_series = pd.Series(0.04, index=returns.index)

    diag = {}
    strategy_fn = strategy.black_litterman_strategy(
        _MULTI_ASSET_CLASS_BUCKET, cfg, POSTURE_CFG, market_ticker="EQ1",
        rf_series=rf_series, record=diag.update,
    )
    weights = strategy_fn(returns.index[-1], returns)

    assert not diag["views_fired"]  # genuinely no views
    assert weights.sum() == pytest.approx(1.0)

    by_bucket = weights.groupby(_MULTI_ASSET_CLASS_BUCKET).sum()
    # Cash capped at 20% (anchor was 25%); the excess 5% redistributed
    # across the other three buckets, not collapsed into one or two of
    # them -- the deployed bug's signature (see the legacy test below)
    # is 20/0/60/20 (commodity only, fixed_income starved to 0%).
    assert by_bucket["cash"] == pytest.approx(0.20, abs=1e-3)
    for bucket in ("equity", "fixed_income", "commodity"):
        assert 0.24 < by_bucket[bucket] < 0.30, (bucket, by_bucket[bucket])

    # Every risky asset's PURE prior (no views, no cap) has positive
    # expected excess return under the corrected convention.
    prior = pd.Series(diag["equilibrium_prior"])
    risky = prior[_MULTI_ASSET_CLASS_BUCKET.reindex(prior.index) != "cash"]
    assert (risky > 0).all()


def test_black_litterman_strategy_legacy_no_views_matches_deployed_bug():
    # The same no-views setup under excess_return_prior: false (the
    # legacy/deployed convention) reproduces the bug ANALYSIS_V4 Sec
    # 2.1 measured: the low-vol (low-Pi) bucket is starved to 0% while
    # a high-vol (high-Pi) bucket piles up to its combined per-asset
    # cap -- kept reproducible for the Phase-2 side-by-side ablation.
    cfg = _no_view_cfg()
    cfg["black_litterman"]["excess_return_prior"] = False
    returns = _multi_vol_returns(_MULTI_ASSET_CLASS_BUCKET.index)
    rf_series = pd.Series(0.04, index=returns.index)

    strategy_fn = strategy.black_litterman_strategy(
        _MULTI_ASSET_CLASS_BUCKET, cfg, POSTURE_CFG, market_ticker="EQ1",
        rf_series=rf_series,
    )
    weights = strategy_fn(returns.index[-1], returns)
    by_bucket = weights.groupby(_MULTI_ASSET_CLASS_BUCKET).sum()

    assert by_bucket["fixed_income"] == pytest.approx(0.0, abs=1e-6)
    assert by_bucket["commodity"] > 0.45  # piled up near 3 x its 20% cap
    # Not the corrected (~26/28/26/20) split.
    assert not (
        0.24 < by_bucket["equity"] < 0.30
        and 0.24 < by_bucket["fixed_income"] < 0.30
        and 0.24 < by_bucket["commodity"] < 0.30
    )


def test_black_litterman_strategy_passes_meanrev_eligible_through(
    monkeypatch,
):
    captured = {}
    original = meanreversion.mean_reversion_view

    def spy(prices, cfg, eligible=None):
        captured["eligible"] = eligible
        return original(prices, cfg, eligible=eligible)

    monkeypatch.setattr(meanreversion, "mean_reversion_view", spy)

    returns = _iid_returns()
    eligible = pd.Index(["ACWI", "GLD"])
    strategy_fn = strategy.black_litterman_strategy(
        CLASS_BUCKET, BL_CFG, POSTURE_CFG, meanrev_eligible=eligible
    )
    strategy_fn(returns.index[-1], returns)

    assert captured["eligible"] is eligible


def test_black_litterman_strategy_defaults_meanrev_eligible_to_none(
    monkeypatch,
):
    captured = {}
    original = meanreversion.mean_reversion_view

    def spy(prices, cfg, eligible=None):
        captured["eligible"] = eligible
        return original(prices, cfg, eligible=eligible)

    monkeypatch.setattr(meanreversion, "mean_reversion_view", spy)

    returns = _iid_returns()
    strategy_fn = strategy.black_litterman_strategy(
        CLASS_BUCKET, BL_CFG, POSTURE_CFG
    )
    strategy_fn(returns.index[-1], returns)

    assert captured["eligible"] is None


def test_black_litterman_strategy_calls_momentum_view(monkeypatch):
    captured = {}
    original = momentum.momentum_view

    def spy(prices, class_bucket_arg, rf_series_arg, cfg):
        captured["class_bucket"] = class_bucket_arg
        captured["rf_series"] = rf_series_arg
        return original(prices, class_bucket_arg, rf_series_arg, cfg)

    monkeypatch.setattr(momentum, "momentum_view", spy)

    returns = _iid_returns()
    rf_series = pd.Series(0.04, index=returns.index)
    strategy_fn = strategy.black_litterman_strategy(
        CLASS_BUCKET, BL_CFG, POSTURE_CFG, rf_series=rf_series
    )
    strategy_fn(returns.index[-1], returns)

    assert "class_bucket" in captured
    pd.testing.assert_series_equal(
        captured["class_bucket"].sort_index(), CLASS_BUCKET.sort_index()
    )
    assert captured["rf_series"] is rf_series


def test_black_litterman_strategy_rf_lookup_ignores_future_values():
    # A no-lookahead canary for the rf lookup: rf_series contains a
    # future rate change well after the decision date, but the
    # decision must only see what was prevailing at (or before) as_of.
    # Observed via the diagnostics sink's "rf" key, which reflects the
    # looked-up prevailing rate regardless of excess_return_prior.
    returns = _iid_returns()
    change_date = returns.index[900]
    rf_series = pd.Series(0.01, index=returns.index)
    rf_series.loc[change_date:] = 0.05

    diag = {}
    strategy_fn = strategy.black_litterman_strategy(
        CLASS_BUCKET, BL_CFG, POSTURE_CFG, rf_series=rf_series,
        record=diag.update,
    )
    earlier_date = returns.index[700]
    strategy_fn(earlier_date, returns.loc[:earlier_date])

    assert diag["rf"] == pytest.approx(0.01)


def test_black_litterman_strategy_rf_defaults_zero_before_series_start():
    returns = _iid_returns()
    rf_series = pd.Series(0.03, index=returns.index[500:])

    diag = {}
    strategy_fn = strategy.black_litterman_strategy(
        CLASS_BUCKET, BL_CFG, POSTURE_CFG, rf_series=rf_series,
        record=diag.update,
    )
    early_date = returns.index[100]
    strategy_fn(early_date, returns.loc[:early_date])

    assert diag["rf"] == 0.0


def test_black_litterman_strategy_excess_prior_passes_zero_to_utility_select(
    monkeypatch,
):
    # Default convention: mu_bl is already excess, so utility_select
    # must receive opt_rf=0.0 regardless of the prevailing rf -- else
    # it would double-subtract (ANALYSIS_V4 Sec 2.1, the root cause).
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

    assert captured["rf"] == 0.0


def test_black_litterman_strategy_legacy_passes_rf_to_utility_select(
    monkeypatch,
):
    captured = {}
    original = allocation.utility_select

    def spy(candidates, mu, cov, risk_aversion, **kwargs):
        captured["rf"] = kwargs.get("rf")
        return original(candidates, mu, cov, risk_aversion, **kwargs)

    monkeypatch.setattr(allocation, "utility_select", spy)

    cfg = copy.deepcopy(BL_CFG)
    cfg["black_litterman"]["excess_return_prior"] = False
    returns = _iid_returns()
    rf_series = pd.Series(0.045, index=returns.index)
    strategy_fn = strategy.black_litterman_strategy(
        CLASS_BUCKET, cfg, POSTURE_CFG, rf_series=rf_series
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


BUCKET_CLASS_BUCKET = pd.concat(
    [CLASS_BUCKET, pd.Series({"VNQ": "real_estate"})]
)


def _iid_returns_with_real_estate(n=1400, seed=3) -> pd.DataFrame:
    dates = pd.bdate_range("2015-01-01", periods=n)
    rng = np.random.default_rng(seed)
    data = {t: rng.normal(0.0004, 0.01, n) for t in BUCKET_CLASS_BUCKET.index}
    return pd.DataFrame(data, index=dates)


def test_bucket_black_litterman_strategy_returns_valid_weights():
    returns = _iid_returns_with_real_estate()
    strategy_fn = strategy.bucket_black_litterman_strategy(
        BUCKET_CLASS_BUCKET, BL_CFG, POSTURE_CFG
    )
    weights = strategy_fn(returns.index[-1], returns)

    assert abs(weights.sum() - 1.0) < 1e-6
    assert (weights >= -1e-9).all()
    assert set(weights.index) == set(BUCKET_CLASS_BUCKET.index)


def test_bucket_black_litterman_strategy_excludes_zero_weight_bucket():
    # real_estate has zero equilibrium weight (allocation.permanent
    # only references equity/fixed_income/commodity/cash), so it must
    # be excluded from the bucket-level universe entirely, not just
    # given a small weight.
    returns = _iid_returns_with_real_estate()
    strategy_fn = strategy.bucket_black_litterman_strategy(
        BUCKET_CLASS_BUCKET, BL_CFG, POSTURE_CFG
    )
    weights = strategy_fn(returns.index[-1], returns)

    assert weights["VNQ"] == 0.0


def test_bucket_black_litterman_strategy_equal_weights_within_bucket():
    # ACWI is the only equity ticker in this fixture, so this doesn't
    # exercise the equal-split directly; instead check fixed_income,
    # which (per CLASS_BUCKET) has no additional members either --
    # use a fixture where a bucket genuinely has 2+ members.
    class_bucket = pd.concat(
        [CLASS_BUCKET, pd.Series({"IEF": "fixed_income"})]
    )
    dates = pd.bdate_range("2015-01-01", periods=1400)
    rng = np.random.default_rng(9)
    data = {t: rng.normal(0.0004, 0.01, 1400) for t in class_bucket.index}
    returns = pd.DataFrame(data, index=dates)

    strategy_fn = strategy.bucket_black_litterman_strategy(
        class_bucket, BL_CFG, POSTURE_CFG
    )
    weights = strategy_fn(returns.index[-1], returns)

    assert weights["AGG"] == pytest.approx(weights["IEF"])


def test_bucket_black_litterman_strategy_respects_per_class_cap():
    cfg = copy.deepcopy(BL_CFG)
    cfg["constraints"]["per_class_cap"] = 0.6
    returns = _iid_returns_with_real_estate()
    strategy_fn = strategy.bucket_black_litterman_strategy(
        BUCKET_CLASS_BUCKET, cfg, POSTURE_CFG
    )
    weights = strategy_fn(returns.index[-1], returns)

    bucket_totals = weights.groupby(BUCKET_CLASS_BUCKET).sum()
    assert (bucket_totals <= 0.6 + 1e-6).all()


@pytest.mark.parametrize("gate", ["off", "vol_floor", "utility"])
def test_bucket_black_litterman_strategy_gate_modes(gate):
    cfg = copy.deepcopy(BL_CFG)
    cfg["black_litterman"]["gate"] = gate
    returns = _iid_returns_with_real_estate()
    strategy_fn = strategy.bucket_black_litterman_strategy(
        BUCKET_CLASS_BUCKET, cfg, POSTURE_CFG
    )
    weights = strategy_fn(returns.index[-1], returns)

    assert abs(weights.sum() - 1.0) < 1e-6
