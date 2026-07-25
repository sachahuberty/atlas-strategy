"""Weekly decision pipeline (glues L0-L5). See PROJECT_STRUCTURE.md
section 5.

Stage 3: V1 regime view. The HMM posture (from regimes.py) selects
which classical allocation book (from allocation.py) to run each
Friday -- "posture switching," the project's first full strategy
version.

Stage 4: an anomaly risk override wrapper (L4, S8). Both stages build
on the same contract -- a `backtest.run`-compatible
`strategy_fn(as_of, window) -> pd.Series` -- so later stages
(mean-reversion, technicals, sentiment, Black-Litterman) can keep
layering in without changing it.

Public API (stage 3):
    regime_switching_strategy(class_bucket, cfg, posture_cfg,
                               market_ticker) -> strategy_fn

Public API (stage 4):
    with_anomaly_override(strategy_fn, cfg) -> strategy_fn
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd
import tensorflow as tf

from . import allocation, anomaly, regimes

StrategyFn = Callable[[pd.Timestamp, pd.DataFrame], pd.Series]


def regime_switching_strategy(
    class_bucket: pd.Series,
    cfg: dict,
    posture_cfg: dict,
    market_ticker: str | None = None,
) -> StrategyFn:
    """Build a `backtest.run`-compatible strategy_fn implementing V1:
    each Friday, detect the HMM posture from trailing market data and
    run that posture's configured allocation book (S4/S7)."""
    lookback = cfg["optimization"]["lookback_days"]
    cov_method = cfg["optimization"]["covariance"]
    market_ticker = market_ticker or cfg["regimes"]["market_ticker"]

    def strategy_fn(as_of: pd.Timestamp, window: pd.DataFrame) -> pd.Series:
        market_returns = window[market_ticker]
        if not regimes.has_enough_history(market_returns, cfg):
            # Not enough trailing data yet for a meaningful HMM read
            # (e.g. the first few weeks of a backtest): default to
            # neutral rather than force a fit on almost no data.
            posture = "neutral"
        else:
            try:
                regime_result = regimes.market_regime(
                    market_returns, cfg, posture_cfg
                )
                posture = regime_result["current_posture"]
            except (ValueError, np.linalg.LinAlgError):
                # HMM EM can occasionally degenerate on a given week's
                # window (a state collapses to ~zero responsibility,
                # producing NaN parameters) -- a known hmmlearn failure
                # mode on live/evolving data, not a logic bug. Fall
                # back to neutral rather than crash the whole backtest
                # over one bad week's fit.
                posture = "neutral"
        posture_spec = posture_cfg["postures"][posture]
        method = posture_spec["allocation_method"]

        recent = window.tail(lookback)
        cov = allocation.covariance_matrix(recent, method=cov_method)

        if method == "max_sharpe":
            mu = allocation.mean_returns(recent)
            return allocation.max_sharpe(mu, cov, cfg)
        if method == "risk_parity":
            return allocation.risk_parity(cov, cfg)
        if method == "gmv_defensive":
            weights = allocation.gmv(cov, cfg)
            tilt = posture_spec["defensive_class_tilt"]
            return allocation.apply_defensive_tilt(
                weights, class_bucket, tilt, cfg
            )
        raise ValueError(f"Unknown allocation_method: {method}")

    return strategy_fn


def with_anomaly_override(strategy_fn: StrategyFn, cfg: dict) -> StrategyFn:
    """Wrap any strategy_fn with the anomaly risk override (L4, S8):
    if the autoencoder flags the current window as anomalous, blend
    its weights toward a defensive plain-GMV book by
    `anomaly.blend_to_gmv` (1.0 = fully defensive).

    The autoencoder is refit every `anomaly.refit_frequency_days` and
    cached in between -- a lightweight stand-in for the fold-based
    walk-forward refit stage 9 will formalize with real per-fold
    persistence in models/.
    """
    acfg = cfg["optimization"]
    cov_method = acfg["covariance"]
    lookback = acfg["lookback_days"]
    blend = cfg["anomaly"]["blend_to_gmv"]
    refit_every = cfg["anomaly"]["refit_frequency_days"]

    cache: dict = {"result": None, "fitted_at": None}

    def strategy_fn_with_override(
        as_of: pd.Timestamp, window: pd.DataFrame
    ) -> pd.Series:
        base_weights = strategy_fn(as_of, window)

        features = anomaly.build_features(window)
        if not anomaly.has_enough_history(features, cfg):
            return base_weights

        stale = (
            cache["fitted_at"] is None
            or (as_of - cache["fitted_at"]).days >= refit_every
        )
        if stale:
            try:
                cache["result"] = anomaly.fit_autoencoder(features, cfg)
                cache["fitted_at"] = as_of
            except (ValueError, tf.errors.InvalidArgumentError):
                # Same spirit as regimes.py's HMM guard: a degenerate
                # fit on one week's window shouldn't crash the whole
                # backtest. Keep the last good model (if any) and
                # retry on the next refit date.
                if cache["result"] is None:
                    return base_weights

        result = cache["result"]
        recent_features = features.tail(result.seq_len)
        errors = anomaly.reconstruction_error(result, recent_features)
        flagged = len(errors) > 0 and anomaly.is_anomalous(
            errors, result.threshold
        )[-1]
        if not flagged:
            return base_weights

        recent = window.tail(lookback)
        cov = allocation.covariance_matrix(recent, method=cov_method)
        gmv_weights = allocation.gmv(cov, cfg)
        aligned_base = base_weights.reindex(gmv_weights.index).fillna(0.0)
        return (1 - blend) * aligned_base + blend * gmv_weights

    return strategy_fn_with_override
