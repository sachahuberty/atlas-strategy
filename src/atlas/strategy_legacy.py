"""Superseded strategy_fn wrappers: stages 3, 5, 6 (S11 Step 5,
ANALYSIS_V2.md action 6).

`black_litterman_strategy` (stage 8, in strategy.py) replaced the
naive posture-switch + additive-tilt chain these three built:
`regime_switching_strategy` (V1 as a book switch), `with_
meanreversion_tilt` (V2 as a post-hoc tilt), `with_technical_view` +
`technical_phase_flags` (V3 as a post-hoc tilt, plus its execution-
timing hook). None of them are called by the current pipeline or any
canonical notebook run. They are kept, not deleted -- a defensible
choice this project has made throughout (REVIEW.md) -- as the tested,
functional historical record of the naive-combination era
PROJECT_STRUCTURE.md 5.1 describes, physically separated from
strategy.py's live path (`black_litterman_strategy`, `bucket_
black_litterman_strategy`) purely for readability now that the live
module has grown large enough that the current path was hard to find.

Stage 4 (`with_anomaly_override`) and stage 7 (`with_sentiment_view`)
are NOT here: both are still live -- the former is composed with
`black_litterman_strategy` in every canonical OOS run (see notebook
09 Part D), the latter is still the documented live/forward entry
point for V4 (see sentiment.py, notebook 07). Only wrappers with zero
current callers moved.

Public API (stage 3):
    regime_switching_strategy(class_bucket, cfg, posture_cfg,
                               market_ticker) -> strategy_fn

Public API (stage 5):
    with_meanreversion_tilt(strategy_fn, cfg) -> strategy_fn

Public API (stage 6):
    with_technical_view(strategy_fn, cfg) -> strategy_fn
    technical_phase_flags(cfg) -> phase_flags_fn
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

from . import allocation, meanreversion, regimes, technicals
from .strategy import StrategyFn


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
