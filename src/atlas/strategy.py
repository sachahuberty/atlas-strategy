"""Weekly decision pipeline (glues L0-L5). See PROJECT_STRUCTURE.md
section 5.

Stage 3: V1 regime view. The HMM posture (from regimes.py) selects
which classical allocation book (from allocation.py) to run each
Friday -- "posture switching," the project's first full strategy
version.

Stage 4: an anomaly risk override wrapper (L4, S8).

Stage 5: a mean-reversion (V2) tilt wrapper (S10).

Stage 6: a technical (V3) tilt wrapper, plus a separate
`phase_flags_fn` builder for backtest.run's execution-timing hook
(S12) -- not a strategy_fn wrapper itself, since phasing needs the
backtester's own current-vs-target trade, not just a weight vector.

Stage 7: a sentiment (V4) tilt wrapper (S14). Unlike every other
wrapper here, it takes a fixed `bucket_scores` snapshot rather than
recomputing from `window` -- free news sources have no historical
archive (same limitation as stage 6's options positioning), so
sentiment is LIVE-ONLY: meant for a forward/live loop that re-scrapes
and passes in a fresh snapshot each real week, never for the
historical OOS backtest (applying one fixed live snapshot across
history would be a lookahead violation).

All strategy_fn stages build on the same contract -- a
`backtest.run`-compatible `strategy_fn(as_of, window) -> pd.Series`.

Stage 8: `black_litterman_strategy` replaces stages 3-6's naive
posture-switch + additive-tilt chain with the real fusion
PROJECT_STRUCTURE.md 5.1 always intended: V1 (regime) becomes an
asset-class-level view instead of a book switch; V2 (mean-reversion)
and V3 (technical) become per-asset views instead of post-hoc tilts;
Black-Litterman fuses all of them with an equilibrium prior into one
posterior mu_BL; a single max_sharpe(mu_BL, Sigma) book is compared by
utility against GMV and Risk Parity (defensive/fallback), and whichever
wins is returned. V4 (sentiment) is omitted from the backtest (still
live-only, see sentiment.py) but the pipeline accepts it for live use.
The stage 3-7 wrappers remain in the codebase, tested and functional,
as the historical record of the naive-combination era they were built
for -- not deleted, just superseded as the "current" strategy.

Public API (stage 3):
    regime_switching_strategy(class_bucket, cfg, posture_cfg,
                               market_ticker) -> strategy_fn

Public API (stage 4):
    with_anomaly_override(strategy_fn, cfg) -> strategy_fn

Public API (stage 5):
    with_meanreversion_tilt(strategy_fn, cfg) -> strategy_fn

Public API (stage 6):
    with_technical_view(strategy_fn, cfg) -> strategy_fn
    technical_phase_flags(cfg) -> phase_flags_fn

Public API (stage 7):
    with_sentiment_view(strategy_fn, cfg, class_bucket,
                         bucket_scores) -> strategy_fn

Public API (stage 8):
    black_litterman_strategy(class_bucket, cfg, posture_cfg,
                              market_ticker) -> strategy_fn
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd
import tensorflow as tf

from . import (
    allocation,
    anomaly,
    meanreversion,
    regimes,
    sentiment,
    technicals,
    views,
)

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


def with_meanreversion_tilt(strategy_fn: StrategyFn, cfg: dict) -> StrategyFn:
    """Wrap any strategy_fn with the V2 mean-reversion tilt (S10):
    nudge weights toward assets with a strong OU-implied expected
    reversion drift, scaled to a bounded max per-asset tilt (S10).

    Until Black-Litterman (stage 8) properly fuses views into one
    posterior, this combines naively as an additive tilt on top of
    whatever book V1 selected (PROJECT_STRUCTURE.md 5.1): the raw view
    is rescaled to [-1, 1] by its own cross-sectional max magnitude,
    then multiplied by `meanreversion.max_tilt_pp` before being added
    to the base weights and re-capped.
    """
    mcfg = cfg["meanreversion"]
    max_tilt = mcfg["max_tilt_pp"]
    cap = cfg["constraints"]["per_asset_cap"]

    def strategy_fn_with_tilt(
        as_of: pd.Timestamp, window: pd.DataFrame
    ) -> pd.Series:
        base_weights = strategy_fn(as_of, window)

        # Returns are all backtest.run passes strategy_fn; reconstruct
        # a price-relative series (exact for log-price purposes, since
        # detrending/z-scoring/OU regression are all invariant to the
        # arbitrary constant scale factor of a $1 starting level).
        prices = (1.0 + window).cumprod()
        if not meanreversion.has_enough_history(prices, cfg):
            return base_weights

        try:
            view = meanreversion.mean_reversion_view(prices, cfg)
        except ValueError:
            # Same defensive spirit as the HMM/anomaly guards: an
            # unexpected numerical failure in one week's ADF/OU fit
            # shouldn't crash the whole backtest.
            return base_weights

        view = view.reindex(base_weights.index).fillna(0.0)
        if (view == 0).all():
            return base_weights

        scale = view.abs().max()
        tilt = (view / scale) * max_tilt if scale > 1e-12 else view

        tilted = (base_weights + tilt).clip(lower=0.0)
        return allocation.cap_and_renormalize(tilted, cap)

    return strategy_fn_with_tilt


def with_technical_view(strategy_fn: StrategyFn, cfg: dict) -> StrategyFn:
    """Wrap any strategy_fn with the V3 technical tilt (S12): nudge
    weights toward assets sitting near a support zone (expect a
    bounce) and away from resistance (expect rejection), bounded by
    `technicals.max_view_magnitude`. Same naive additive-tilt
    combination as V2 until Black-Litterman (stage 8).

    Options-derived signals (OI notional, call/put walls, gamma proxy)
    are live-only (yfinance has no chain history) and are not part of
    this backtestable view -- see technicals.py's module docstring.
    """
    cap = cfg["constraints"]["per_asset_cap"]

    def strategy_fn_with_view(
        as_of: pd.Timestamp, window: pd.DataFrame
    ) -> pd.Series:
        base_weights = strategy_fn(as_of, window)

        prices = (1.0 + window).cumprod()
        if not technicals.has_enough_history(prices, cfg):
            return base_weights

        try:
            view = technicals.technical_view(prices, cfg)
        except ValueError:
            return base_weights

        view = view.reindex(base_weights.index).fillna(0.0)
        if (view == 0).all():
            return base_weights

        scale = view.abs().max()
        max_magnitude = cfg["technicals"]["max_view_magnitude"]
        tilt = (view / scale) * max_magnitude if scale > 1e-12 else view

        tilted = (base_weights + tilt).clip(lower=0.0)
        return allocation.cap_and_renormalize(tilted, cap)

    return strategy_fn_with_view


def technical_phase_flags(
    cfg: dict,
) -> Callable[[pd.Timestamp, pd.DataFrame], set]:
    """Build a `backtest.run`-compatible phase_flags_fn (S12 execution
    timing): flags any asset currently sitting near a resistance zone,
    so backtest.run phases that rebalance's trade in rather than
    executing it in one shot."""

    def phase_flags_fn(as_of: pd.Timestamp, window: pd.DataFrame) -> set:
        prices = (1.0 + window).cumprod()
        if not technicals.has_enough_history(prices, cfg):
            return set()
        try:
            diag = technicals.technical_signal(prices, cfg)
        except ValueError:
            return set()
        return set(diag[diag["role"] == "resistance"].index)

    return phase_flags_fn


def with_sentiment_view(
    strategy_fn: StrategyFn,
    cfg: dict,
    class_bucket: pd.Series,
    bucket_scores: pd.Series,
) -> StrategyFn:
    """Wrap any strategy_fn with the V4 sentiment tilt (S14, live use
    only -- see module docstring). `sentiment.sentiment_view` already
    bounds each asset's tilt to `sentiment.max_view_magnitude` (VADER's
    compound score is itself in [-1, 1]), so unlike V2/V3 there is no
    further cross-sectional rescaling here -- the view is already
    safely bounded by construction.
    """
    cap = cfg["constraints"]["per_asset_cap"]
    view = sentiment.sentiment_view(bucket_scores, class_bucket, cfg)

    def strategy_fn_with_sentiment(
        as_of: pd.Timestamp, window: pd.DataFrame
    ) -> pd.Series:
        base_weights = strategy_fn(as_of, window)
        aligned_view = view.reindex(base_weights.index).fillna(0.0)
        if (aligned_view == 0).all():
            return base_weights

        tilted = (base_weights + aligned_view).clip(lower=0.0)
        return allocation.cap_and_renormalize(tilted, cap)

    return strategy_fn_with_sentiment


def black_litterman_strategy(
    class_bucket: pd.Series,
    cfg: dict,
    posture_cfg: dict,
    market_ticker: str | None = None,
) -> StrategyFn:
    """Build a `backtest.run`-compatible strategy_fn implementing the
    full Black-Litterman fusion pipeline (S4, stage 8):

    1. Detect the HMM posture (same guard/fallback pattern as
       regime_switching_strategy).
    2. Equilibrium prior Pi from `permanent()`'s asset-class weights
       (notebook 08: "equilibrium returns from asset-class weights").
    3. V1 (regime posture -> asset-class view), V2 (mean-reversion),
       V3 (technical) each contribute views; V4 (sentiment) is
       omitted here -- still live-only (see sentiment.py).
    4. Fuse into mu_BL via Black-Litterman.
    5. max_sharpe(mu_BL, Sigma) is the primary book; GMV and Risk
       Parity are defensive/fallback books; utility_select (using
       mu_BL for all three) picks whichever wins.
    """
    lookback = cfg["optimization"]["lookback_days"]
    cov_method = cfg["optimization"]["covariance"]
    market_ticker = market_ticker or cfg["regimes"]["market_ticker"]
    bcfg = cfg["black_litterman"]
    risk_aversion = bcfg["risk_aversion_for_utility_gate"]
    market_weights = allocation.permanent(class_bucket)

    def strategy_fn(as_of: pd.Timestamp, window: pd.DataFrame) -> pd.Series:
        market_returns = window[market_ticker]
        if not regimes.has_enough_history(market_returns, cfg):
            posture = "neutral"
        else:
            try:
                regime_result = regimes.market_regime(
                    market_returns, cfg, posture_cfg
                )
                posture = regime_result["current_posture"]
            except (ValueError, np.linalg.LinAlgError):
                posture = "neutral"

        recent = window.tail(lookback)
        cov = allocation.covariance_matrix(recent, method=cov_method)
        tickers = recent.columns

        prior = allocation.equilibrium_returns(
            market_weights.reindex(tickers).fillna(0.0), cov, bcfg["delta"]
        )

        prices = (1.0 + window).cumprod()
        if meanreversion.has_enough_history(prices, cfg):
            try:
                meanrev_view = meanreversion.mean_reversion_view(prices, cfg)
            except ValueError:
                meanrev_view = pd.Series(0.0, index=tickers)
        else:
            meanrev_view = pd.Series(0.0, index=tickers)

        if technicals.has_enough_history(prices, cfg):
            try:
                tech_view = technicals.technical_view(prices, cfg)
            except ValueError:
                tech_view = pd.Series(0.0, index=tickers)
        else:
            tech_view = pd.Series(0.0, index=tickers)

        view_sets = [
            views.regime_view(posture, class_bucket, prior, cfg),
            views.meanreversion_views(meanrev_view, prior, cfg),
            views.technical_views(tech_view, prior, cfg),
        ]
        P, Q, Omega, _ = views.assemble(view_sets, tickers, cov, cfg)
        mu_bl = allocation.black_litterman(
            prior, cov, P, Q, Omega, bcfg["tau"]
        )

        candidates = {
            "black_litterman": allocation.max_sharpe(mu_bl, cov, cfg),
            "gmv": allocation.gmv(cov, cfg),
            "risk_parity": allocation.risk_parity(cov, cfg),
        }
        _, selected = allocation.utility_select(
            candidates, mu_bl, cov, risk_aversion
        )
        return selected

    return strategy_fn
