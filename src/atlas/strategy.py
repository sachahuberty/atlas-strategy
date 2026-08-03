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
                              market_ticker, rf_series,
                              meanrev_eligible) -> strategy_fn

Public API (stage 11, Tier 2): bucket_black_litterman_strategy runs
the same Black-Litterman/gate machinery on the four asset-class
bucket portfolios instead of 22 individual assets (DIAGNOSTIC.md's
most robust finding: with ~22 assets that are really 4-5 uncorrelated
bets, per-asset covariance/return estimation is mostly noise). V1 is
the only active view -- it is already asset-class-level, and the
ablation's only positively-contributing module; V2/V3 have no natural
bucket-level analog. Kept alongside, not instead of, the asset-level
path so both can be compared on the same OOS window.
    bucket_black_litterman_strategy(class_bucket, cfg, posture_cfg,
                                     market_ticker, rf_series) -> strategy_fn
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd
import tensorflow as tf

from . import (
    allocation,
    anomaly,
    buckets,
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


def _select_gated_candidate(
    candidates: dict[str, pd.Series],
    mu_bl: pd.Series,
    cov: pd.DataFrame,
    bcfg: dict,
    rf: float,
) -> pd.Series:
    """Shared gate-selection logic (S11 Tier 2) between the asset-
    level and bucket-level Black-Litterman strategies: `bcfg["gate"]`
    picks "utility" (mean-variance utility_select), "off" (always the
    max_sharpe book), or "vol_floor" (max_sharpe unless its vol
    exceeds `vol_floor_multiplier` x GMV's vol). See
    black_litterman_strategy's docstring for the full rationale."""
    gate = bcfg["gate"]
    if gate == "off":
        return candidates["black_litterman"]
    if gate == "vol_floor":
        bl_vol = allocation.portfolio_vol(candidates["black_litterman"], cov)
        gmv_vol = allocation.portfolio_vol(candidates["gmv"], cov)
        if bl_vol <= gmv_vol * bcfg["vol_floor_multiplier"]:
            return candidates["black_litterman"]
        return candidates["gmv"]
    if gate == "utility":
        risk_aversion = bcfg["risk_aversion_for_utility_gate"]
        _, selected = allocation.utility_select(
            candidates, mu_bl, cov, risk_aversion, rf=rf
        )
        return selected
    raise ValueError(f"Unknown black_litterman.gate: {gate!r}")


def black_litterman_strategy(
    class_bucket: pd.Series,
    cfg: dict,
    posture_cfg: dict,
    market_ticker: str | None = None,
    rf_series: pd.Series | None = None,
    meanrev_eligible: pd.Index | None = None,
) -> StrategyFn:
    """Build a `backtest.run`-compatible strategy_fn implementing the
    full Black-Litterman fusion pipeline (S4, stage 8):

    1. Detect the HMM posture (same guard/fallback pattern as
       regime_switching_strategy).
    2. Equilibrium prior Pi from `permanent()`'s asset-class weights
       (notebook 08: "equilibrium returns from asset-class weights"),
       with the `cash` bucket's entry overridden by the prevailing
       risk-free rate (see `rf_series` below) rather than Pi's near-
       zero implied return for a zero-covariance asset.
    3. V1 (regime posture -> asset-class view), V2 (mean-reversion),
       V3 (technical) each contribute views; V4 (sentiment) is
       omitted here -- still live-only (see sentiment.py).
    4. Fuse into mu_BL via Black-Litterman.
    5. max_sharpe(mu_BL, Sigma, rf) is the primary book; GMV and Risk
       Parity are defensive/fallback books; `cfg["black_litterman"]
       ["gate"]` picks whichever wins:
       - "utility" (default): mean-variance utility_select (using
         mu_BL and rf for all three candidates).
       - "off": always the max_sharpe book, no gating at all.
       - "vol_floor": the max_sharpe book unless its volatility
         exceeds `vol_floor_multiplier` x GMV's volatility, in which
         case GMV. A simple risk-budget gate instead of a quadratic
         utility comparison, which structurally favors minimum
         variance regardless of signal quality (DIAGNOSTIC.md Sec 3;
         stage 11's gate ablation, notebook 11, measured the "off"
         and A-sensitivity arms before this option was added).

    `rf_series`, if given, is an annual risk-free rate indexed by
    date (e.g. FRED DTB3/100); the prevailing rate as of each `as_of`
    (via `.asof`, so never a future value) is used as `rf` throughout
    this week's decision. Defaults to a constant 0.0 (the stage 1-10
    total-return convention) if omitted.

    `meanrev_eligible`, if given (stage 11 Tier 3), restricts V2's
    view to tickers in that index -- see `meanreversion.
    hit_rate_eligible_tickers`. Intended to be computed once from an
    IS-only price slice and frozen for the OOS run, not recomputed
    weekly. Defaults to None (every ticker eligible, the pre-Tier-3
    behavior).

    Stage-11 note: an earlier version of this function excluded the
    `cash` bucket from max_sharpe's universe entirely (rf=0 makes a
    near-zero-vol cash proxy degenerate in a Sharpe objective,
    regardless of its actual expected return -- see
    allocation.max_sharpe's docstring for that degeneracy, which is
    still real). That exclusion was reverted: a gate census
    (DIAGNOSTIC.md Sec 0/2.3) showed cash was not merely riding the
    Sharpe degeneracy, it was the *variance anchor* that let the
    max_sharpe book clear the utility gate at all -- with cash
    allowed, the BL book won the gate 239/239 OOS weeks; with cash
    excluded, 0/239 (the pipeline silently became plain GMV every
    week). Removing the asset was the wrong remedy; pricing it
    properly via `rf_series` fixes the degeneracy at its source
    without deleting the variance anchor the gate depends on.
    """
    lookback = cfg["optimization"]["lookback_days"]
    cov_method = cfg["optimization"]["covariance"]
    market_ticker = market_ticker or cfg["regimes"]["market_ticker"]
    bcfg = cfg["black_litterman"]
    market_weights = allocation.permanent(class_bucket)
    cash_tickers = class_bucket[class_bucket == "cash"].index

    def strategy_fn(as_of: pd.Timestamp, window: pd.DataFrame) -> pd.Series:
        rf = 0.0
        if rf_series is not None:
            # .asof never looks past as_of; NaN before rf_series' own
            # start (e.g. a cold-start week) falls back to 0.0 rather
            # than propagating a NaN through mu_bl/utility.
            looked_up = rf_series.asof(as_of)
            if pd.notna(looked_up):
                rf = float(looked_up)
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
        # Pi is ~0 for cash by construction (near-zero covariance with
        # everything), regardless of the prevailing rate -- overriding
        # it with the actual risk-free rate is what fixes the cash
        # degeneracy at its source (see docstring).
        priced_cash = cash_tickers.intersection(tickers)
        if len(priced_cash) > 0:
            prior.loc[priced_cash] = rf

        prices = (1.0 + window).cumprod()
        if meanreversion.has_enough_history(prices, cfg):
            try:
                meanrev_view = meanreversion.mean_reversion_view(
                    prices, cfg, eligible=meanrev_eligible
                )
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
            "black_litterman": allocation.max_sharpe(mu_bl, cov, cfg, rf=rf),
            "gmv": allocation.gmv(cov, cfg),
            "risk_parity": allocation.risk_parity(cov, cfg),
        }
        return _select_gated_candidate(candidates, mu_bl, cov, bcfg, rf)

    return strategy_fn


def bucket_black_litterman_strategy(
    class_bucket: pd.Series,
    cfg: dict,
    posture_cfg: dict,
    market_ticker: str | None = None,
    rf_series: pd.Series | None = None,
) -> StrategyFn:
    """Black-Litterman fusion at the asset-CLASS-BUCKET level (S11
    Tier 2, DIAGNOSTIC.md action 5) -- the same machinery as
    `black_litterman_strategy`, run on the four bucket portfolios
    (equity, fixed_income, commodity, cash: the same four
    `allocation.permanent` uses) instead of 22 individual assets.

    Rationale: with ~22 assets that are really 4-5 uncorrelated bets,
    estimating a 22x22 covariance and 22 expected returns is mostly
    estimating noise; collapsing to bucket-level estimation makes both
    problems tractable (DIAGNOSTIC.md's most robust finding, measured
    independently of this implementation at Sharpe 1.279 OOS / 0.885
    IS / 1.036 full-period for a bucket-level book, vs. Permanent's
    1.067 / 0.476 / 0.690 -- the only change that beat Permanent in
    both windows).

    Only V1 (regime posture) contributes a view: it is already asset-
    class-level (so it needs no broadcasting/aggregation to fit this
    universe) and is the only view with a positive marginal OOS
    contribution in stage 11's ablation. V2/V3 are per-asset signals
    with no natural bucket-level analog and are not applied here.

    A bucket with zero equilibrium weight (`real_estate`, since
    `allocation.permanent`/`sixty_forty` only reference four buckets)
    is excluded from the bucket-level optimization entirely, matching
    that existing asymmetry rather than silently fixing it here (see
    DIAGNOSTIC.md Sec 5.2 item 4). The resulting bucket weights are
    split equally across each bucket's member tickers
    (`buckets.expand_bucket_weights`) before being returned, so this
    strategy_fn is a drop-in, backtest.run-compatible replacement for
    the asset-level path -- both can be run side by side on the same
    OOS window for comparison.

    Uses `constraints.per_class_cap` (not `per_asset_cap`) as the cap
    on each bucket's weight -- a bucket is not a single concentrated
    position the way an individual asset is, so the tighter per-asset
    cap doesn't apply at this level of abstraction.
    """
    lookback = cfg["optimization"]["lookback_days"]
    cov_method = cfg["optimization"]["covariance"]
    market_ticker = market_ticker or cfg["regimes"]["market_ticker"]
    bcfg = cfg["black_litterman"]

    bucket_cfg = dict(cfg)
    bucket_cfg["constraints"] = dict(cfg["constraints"])
    bucket_cfg["constraints"]["per_asset_cap"] = cfg["constraints"][
        "per_class_cap"
    ]

    market_weights_by_asset = allocation.permanent(class_bucket)
    market_weights_bucket = market_weights_by_asset.groupby(class_bucket).sum()
    active_buckets = market_weights_bucket[market_weights_bucket > 0]
    active_buckets = active_buckets.index.tolist()
    market_weights_bucket = market_weights_bucket.loc[active_buckets]
    # V1's regime_view expects a class_bucket-shaped Series mapping
    # each "asset" to its class; at this level each bucket IS its own
    # class, so an identity mapping lets the existing, unmodified
    # views.regime_view apply unchanged.
    bucket_identity = pd.Series(active_buckets, index=active_buckets)

    def strategy_fn(as_of: pd.Timestamp, window: pd.DataFrame) -> pd.Series:
        rf = 0.0
        if rf_series is not None:
            looked_up = rf_series.asof(as_of)
            if pd.notna(looked_up):
                rf = float(looked_up)

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

        bucket_rets = buckets.bucket_returns(
            window, class_bucket, active_buckets
        )
        recent = bucket_rets.tail(lookback)
        cov = allocation.covariance_matrix(recent, method=cov_method)

        prior = allocation.equilibrium_returns(
            market_weights_bucket, cov, bcfg["delta"]
        )
        if "cash" in prior.index:
            prior.loc["cash"] = rf

        view_set = views.regime_view(posture, bucket_identity, prior, cfg)
        P, Q, Omega, _ = views.assemble(
            [view_set], pd.Index(active_buckets), cov, cfg
        )
        mu_bl = allocation.black_litterman(
            prior, cov, P, Q, Omega, bcfg["tau"]
        )

        candidates = {
            "black_litterman": allocation.max_sharpe(
                mu_bl, cov, bucket_cfg, rf=rf
            ),
            "gmv": allocation.gmv(cov, bucket_cfg),
            "risk_parity": allocation.risk_parity(cov, bucket_cfg),
        }
        selected_bucket = _select_gated_candidate(
            candidates, mu_bl, cov, bcfg, rf
        )
        return buckets.expand_bucket_weights(selected_bucket, class_bucket)

    return strategy_fn
