"""IS-only re-selection harness tests (reselection task, external
review). Proves the efficient caching path in `reselection.py`
reproduces the real (uncached) pipeline exactly, that gate re-selection
on cached candidates matches `strategy._select_gated_candidate`
itself, and that the lookahead canary refuses any frame touching OOS.

The equivalence tests run with `modules.anomaly_override: False` (both
sides) -- anomaly's own correctness is covered by `test_anomaly.py`
and `test_strategy.py`; what's under test here is the cache/variant/
gate/turnover plumbing, not autoencoder determinism across separate
fit calls.
"""

import copy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from atlas import backtest, data, reselection, strategy


def _small_cfg():
    cfg = copy.deepcopy(data.load_config())
    cfg["regimes"]["market_ticker"] = "EQ1"
    cfg["regimes"]["hmm"]["n_states"] = 2
    cfg["regimes"]["hmm"]["lookback_days"] = 120
    cfg["regimes"]["hmm"]["vol_window_days"] = 5
    cfg["optimization"]["lookback_days"] = 60
    cfg["meanreversion"]["lookback_days"] = 20
    cfg["meanreversion"]["adf_lookback_days"] = 40
    cfg["technicals"]["min_history_days"] = 30
    cfg["momentum"]["lookback_days"] = 60
    cfg["momentum"]["trend_lookback_days"] = 60
    cfg["momentum"]["skip_days"] = 5
    cfg["modules"]["anomaly_override"] = False
    cfg["modules"]["technical_view"] = True
    cfg["modules"]["momentum_view"] = True
    cfg["general"]["is_end_date"] = "2021-12-31"
    cfg["general"]["oos_start_date"] = "2022-01-01"
    return cfg


def _market(seed=7, n=260):
    rng = np.random.default_rng(seed)
    tickers = ["EQ1", "EQ2", "FI1", "FI2", "CO1", "CO2", "CA1"]
    buckets = {
        "EQ1": "equity", "EQ2": "equity",
        "FI1": "fixed_income", "FI2": "fixed_income",
        "CO1": "commodity", "CO2": "commodity",
        "CA1": "cash",
    }
    vol = {
        "equity": 0.012, "fixed_income": 0.004,
        "commodity": 0.014, "cash": 0.0005,
    }
    mu = {
        "equity": 0.0004, "fixed_income": 0.0001,
        "commodity": 0.0003, "cash": 8e-5,
    }
    dates = pd.bdate_range("2020-06-01", periods=n)
    cols = {}
    for t in tickers:
        b = buckets[t]
        r = rng.normal(mu[b], vol[b], len(dates))
        cols[t] = 100 * np.cumprod(1 + r)
    prices = data.align_calendars(pd.DataFrame(cols, index=dates))
    returns = data.daily_returns(prices)
    class_bucket = pd.Series(buckets)
    rf_series = pd.Series(0.03, index=returns.index)
    return returns, class_bucket, rf_series


def _posture_cfg():
    return data.load_config(
        data.PROJECT_ROOT / "config" / "regime_posture.yaml"
    )


# --------------------------------------------------------------------- #
# ViewVariant / variant_cfg
# --------------------------------------------------------------------- #
def test_all_view_variants_has_12_combinations():
    assert len(reselection.ALL_VIEW_VARIANTS) == 12
    keys = {v.key for v in reselection.ALL_VIEW_VARIANTS}
    assert len(keys) == 12  # every key distinct


def test_variant_cfg_never_mutates_base_cfg():
    base = _small_cfg()
    before = copy.deepcopy(base)
    reselection.variant_cfg(base, reselection.ALL_VIEW_VARIANTS[0])
    assert base == before


def test_variant_cfg_sets_module_flags_and_risk_off_target():
    base = _small_cfg()
    variant = reselection.ViewVariant(
        v3=True, v5=False, risk_off_target="bonds_gold"
    )
    cfg = reselection.variant_cfg(base, variant)
    assert cfg["modules"]["technical_view"] is True
    assert cfg["modules"]["momentum_view"] is False
    assert cfg["black_litterman"]["regime_targets"]["risk_off"] == (
        "bonds_gold"
    )


# --------------------------------------------------------------------- #
# Lookahead canary
# --------------------------------------------------------------------- #
def test_assert_no_oos_access_raises_on_oos_touching_frame():
    cfg = _small_cfg()
    dates = pd.bdate_range("2021-12-01", "2022-01-15")
    frame = pd.DataFrame(0.0, index=dates, columns=["A"])
    with pytest.raises(ValueError, match="oos_start_date"):
        reselection.assert_no_oos_access(frame, cfg)


def test_assert_no_oos_access_allows_pure_is_frame():
    cfg = _small_cfg()
    dates = pd.bdate_range("2021-01-01", "2021-12-31")
    frame = pd.DataFrame(0.0, index=dates, columns=["A"])
    reselection.assert_no_oos_access(frame, cfg)  # must not raise


def test_build_shared_caches_refuses_oos_touching_returns():
    # oos_start_date set to the market fixture's OWN midpoint, so the
    # full returns frame deterministically straddles it regardless of
    # the fixture's exact calendar span.
    cfg = _small_cfg()
    returns, class_bucket, rf_series = _market()
    midpoint = returns.index[len(returns) // 2]
    cfg["general"]["oos_start_date"] = midpoint.date().isoformat()
    dates = sorted(backtest.week_end_dates(returns.index))[-3:]
    with pytest.raises(ValueError, match="oos_start_date"):
        reselection.build_shared_caches(
            dates, returns, class_bucket, cfg, _posture_cfg(),
            "EQ1", rf_series, None,
        )


def test_run_turnover_variant_refuses_oos_touching_returns():
    cfg = _small_cfg()
    returns, _, _ = _market()
    midpoint = returns.index[len(returns) // 2]
    cfg["general"]["oos_start_date"] = midpoint.date().isoformat()
    with pytest.raises(ValueError, match="oos_start_date"):
        reselection.run_turnover_variant(
            {}, returns.columns, returns, cfg
        )


# --------------------------------------------------------------------- #
# with_delta (Phase 3 sensitivity): recomputes `prior` cheaply, without
# re-running any of compute_shared_entry's expensive parts.
# --------------------------------------------------------------------- #
def test_with_delta_reproduces_prior_unchanged_at_the_same_delta():
    cfg = _small_cfg()
    returns, class_bucket, rf_series = _market()
    is_returns = returns.loc[:"2021-12-31"]
    as_of = max(backtest.week_end_dates(is_returns.index))
    window = is_returns.loc[:as_of]

    shared = reselection.compute_shared_entry(
        as_of, window, class_bucket, cfg, _posture_cfg(), "EQ1",
        rf_series, None, {"result": None, "fitted_at": None},
    )
    same_delta = reselection.with_delta(
        shared, class_bucket, shared["delta"]
    )
    pd.testing.assert_series_equal(same_delta["prior"], shared["prior"])


def test_with_delta_matches_manual_equilibrium_returns():
    cfg = _small_cfg()
    returns, class_bucket, rf_series = _market()
    is_returns = returns.loc[:"2021-12-31"]
    as_of = max(backtest.week_end_dates(is_returns.index))
    window = is_returns.loc[:as_of]

    shared = reselection.compute_shared_entry(
        as_of, window, class_bucket, cfg, _posture_cfg(), "EQ1",
        rf_series, None, {"result": None, "fitted_at": None},
    )
    new_delta = 4.0
    recomputed = reselection.with_delta(shared, class_bucket, new_delta)

    from atlas import allocation

    market_weights = allocation.permanent(class_bucket)
    expected = allocation.equilibrium_returns(
        market_weights.reindex(shared["tickers"]).fillna(0.0),
        shared["cov"],
        new_delta,
    )
    pd.testing.assert_series_equal(recomputed["prior"], expected)
    # Everything else (cov, anomaly flag, raw view series) is reused
    # unchanged -- only prior/delta differ.
    pd.testing.assert_frame_equal(recomputed["cov"], shared["cov"])
    assert recomputed["delta"] == new_delta


# --------------------------------------------------------------------- #
# Equivalence: cached gate re-selection matches
# strategy._select_gated_candidate directly.
# --------------------------------------------------------------------- #
def test_gate_select_matches_select_gated_candidate_directly():
    cfg = _small_cfg()
    returns, class_bucket, rf_series = _market()
    is_returns = returns.loc[:"2021-12-31"]
    as_of = max(backtest.week_end_dates(is_returns.index))
    window = is_returns.loc[:as_of]

    anomaly_cache = {"result": None, "fitted_at": None}
    shared = reselection.compute_shared_entry(
        as_of, window, class_bucket, cfg, _posture_cfg(), "EQ1",
        rf_series, None, anomaly_cache,
    )
    variant = reselection.ALL_VIEW_VARIANTS[0]
    v_cfg = reselection.variant_cfg(cfg, variant)
    variant_entry = reselection.compute_variant_entry(
        shared, class_bucket, v_cfg, shared["tickers"]
    )

    for risk_aversion in (1, 2, 3, 5):
        _, cached_weights = reselection.gate_select(
            variant_entry, v_cfg["black_litterman"], risk_aversion
        )
        direct_weights = strategy._select_gated_candidate(
            variant_entry["candidates"],
            variant_entry["mu_bl"],
            variant_entry["cov"],
            {
                **v_cfg["black_litterman"],
                "gate": "utility",
                "risk_aversion_for_utility_gate": risk_aversion,
            },
            variant_entry["rf"],
        )
        pd.testing.assert_series_equal(
            cached_weights.sort_index(),
            direct_weights.sort_index(),
            check_names=False,
        )


# --------------------------------------------------------------------- #
# Equivalence: the full cached path (shared cache -> variant cache ->
# gate -> turnover re-run) reproduces an UNCACHED real backtest.run
# exactly, on a short IS-only window.
# --------------------------------------------------------------------- #
def test_cached_path_matches_uncached_backtest_run_exactly():
    cfg = _small_cfg()
    returns, class_bucket, rf_series = _market()
    is_returns = returns.loc[:"2021-12-31"]
    tickers = is_returns.columns
    risk_aversion = 2

    variant = reselection.ViewVariant(
        v3=True, v5=True, risk_off_target="legacy"
    )
    v_cfg = reselection.variant_cfg(cfg, variant)
    v_cfg["black_litterman"]["risk_aversion_for_utility_gate"] = risk_aversion

    # Reference: the real, uncached pipeline.
    reference_fn = strategy.black_litterman_strategy(
        class_bucket, v_cfg, _posture_cfg(), rf_series=rf_series,
    )
    reference_result = backtest.run(reference_fn, is_returns, v_cfg)

    # Cached path.
    dates = sorted(backtest.week_end_dates(is_returns.index))
    shared_caches = reselection.build_shared_caches(
        dates, is_returns, class_bucket, cfg, _posture_cfg(), "EQ1",
        rf_series, None,
    )
    variant_caches = reselection.build_variant_caches(
        shared_caches, class_bucket, cfg, variant
    )
    targets = reselection.target_weight_series(
        variant_caches, v_cfg["black_litterman"], risk_aversion
    )
    cached_result = reselection.run_turnover_variant(
        targets, tickers, is_returns, v_cfg
    )

    pd.testing.assert_series_equal(
        cached_result.daily_returns, reference_result.daily_returns
    )
    pd.testing.assert_frame_equal(
        cached_result.weights, reference_result.weights
    )
    assert cached_result.metrics["sharpe"] == pytest.approx(
        reference_result.metrics["sharpe"]
    )


def test_turnover_variants_reuse_same_targets_different_execution():
    cfg = _small_cfg()
    returns, class_bucket, rf_series = _market()
    is_returns = returns.loc[:"2021-12-31"]
    tickers = is_returns.columns

    variant = reselection.ALL_VIEW_VARIANTS[0]
    v_cfg = reselection.variant_cfg(cfg, variant)
    dates = sorted(backtest.week_end_dates(is_returns.index))
    shared_caches = reselection.build_shared_caches(
        dates, is_returns, class_bucket, cfg, _posture_cfg(), "EQ1",
        rf_series, None,
    )
    variant_caches = reselection.build_variant_caches(
        shared_caches, class_bucket, cfg, variant
    )
    targets = reselection.target_weight_series(
        variant_caches, v_cfg["black_litterman"], 2
    )

    tight_cfg = copy.deepcopy(v_cfg)
    tight_cfg["rebalance"]["max_weekly_turnover"] = 0.01
    tight_cfg["rebalance"]["no_trade_band"] = 0.005
    loose_cfg = copy.deepcopy(v_cfg)
    loose_cfg["rebalance"]["max_weekly_turnover"] = 0.04
    loose_cfg["rebalance"]["no_trade_band"] = 0.005

    tight_result = reselection.run_turnover_variant(
        targets, tickers, is_returns, tight_cfg
    )
    loose_result = reselection.run_turnover_variant(
        targets, tickers, is_returns, loose_cfg
    )
    # Same target weights, different execution: the tighter cap must
    # bind more often, producing lower average realized turnover.
    assert (
        tight_result.metrics["avg_weekly_turnover"]
        <= loose_result.metrics["avg_weekly_turnover"]
    )


# --------------------------------------------------------------------- #
# Checkpointing: resuming from a partial cache must not recompute
# already-cached dates, and must produce the identical final cache.
# --------------------------------------------------------------------- #
def test_build_shared_caches_resumes_from_checkpoint(tmp_path):
    cfg = _small_cfg()
    returns, class_bucket, rf_series = _market()
    is_returns = returns.loc[:"2021-12-31"]
    dates = sorted(backtest.week_end_dates(is_returns.index))[-6:]
    checkpoint = tmp_path / "shared.pkl"

    partial = reselection.build_shared_caches(
        dates[:3], is_returns, class_bucket, cfg, _posture_cfg(), "EQ1",
        rf_series, None, checkpoint_path=checkpoint,
    )
    assert set(partial.keys()) == set(dates[:3])

    full = reselection.build_shared_caches(
        dates, is_returns, class_bucket, cfg, _posture_cfg(), "EQ1",
        rf_series, None, checkpoint_path=checkpoint,
    )
    assert set(full.keys()) == set(dates)
    # The first 3 dates' cached entries are untouched by the resume.
    for d in dates[:3]:
        pd.testing.assert_series_equal(full[d]["prior"], partial[d]["prior"])


# --------------------------------------------------------------------- #
# save_checkpoint's OneDrive-lock retry (this project lives in a
# OneDrive-synced folder, CLAUDE.md's own documented environment
# caveat; a checkpoint saved after every decision date is exactly the
# write pattern that can collide with a transient sync lock).
# --------------------------------------------------------------------- #
def test_save_checkpoint_retries_through_transient_permission_error(
    tmp_path, monkeypatch
):
    path = tmp_path / "cache.pkl"
    calls = {"n": 0}
    real_replace = Path.replace

    def flaky_replace(self, target):
        calls["n"] += 1
        if calls["n"] < 3:
            raise PermissionError("simulated OneDrive lock")
        return real_replace(self, target)

    monkeypatch.setattr(Path, "replace", flaky_replace)
    monkeypatch.setattr(reselection.time, "sleep", lambda _s: None)

    reselection.save_checkpoint(
        {"a": 1}, path, retries=5, retry_delay_seconds=0.0
    )
    assert calls["n"] == 3
    assert reselection.load_checkpoint(path) == {"a": 1}


def test_save_checkpoint_raises_after_exhausting_retries(
    tmp_path, monkeypatch
):
    path = tmp_path / "cache.pkl"

    def always_locked(self, target):
        raise PermissionError("simulated OneDrive lock")

    monkeypatch.setattr(Path, "replace", always_locked)
    monkeypatch.setattr(reselection.time, "sleep", lambda _s: None)

    with pytest.raises(PermissionError):
        reselection.save_checkpoint(
            {"a": 1}, path, retries=3, retry_delay_seconds=0.0
        )
    # No orphan .tmp left behind to confuse the next run's resume scan.
    assert not path.with_suffix(path.suffix + ".tmp").exists()
