"""Weekly decision pipeline (glues L0-L5). See PROJECT_STRUCTURE.md
section 5.

Stage 4: an anomaly risk override wrapper (L4, S8) -- still live,
composed with `black_litterman_strategy` in every canonical OOS run.

Stage 7: a sentiment (V4) tilt wrapper (S14) -- still the documented
live/forward entry point for V4. Unlike every other wrapper here, it
takes a fixed `bucket_scores` snapshot rather than recomputing from
`window` -- free news sources have no historical archive, so sentiment
is LIVE-ONLY: meant for a forward/live loop that re-scrapes and passes
in a fresh snapshot each real week, never for the historical OOS
backtest (applying one fixed live snapshot across history would be a
lookahead violation).

All strategy_fn stages build on the same contract -- a
`backtest.run`-compatible `strategy_fn(as_of, window) -> pd.Series`.

Stage 8: `black_litterman_strategy` replaces stages 3, 5, 6's naive
posture-switch + additive-tilt chain (now in strategy_legacy.py, S11
Step 5) with the real fusion PROJECT_STRUCTURE.md 5.1 always intended:
V1 (regime) becomes an asset-class-level view instead of a book
switch; V2 (mean-reversion) and V3 (technical) become per-asset views
instead of post-hoc tilts; Black-Litterman fuses all of them with an
equilibrium prior into one posterior mu_BL; a single max_sharpe(mu_BL,
Sigma) book is compared by utility against GMV and Risk Parity
(defensive/fallback), and whichever wins is returned. V4 (sentiment)
is omitted from the backtest (still live-only) but the pipeline
accepts it for live use.

Stage 11 (ANALYSIS_V2.md action 1): V5 dual momentum (momentum.py)
joins V1-V3 as another per-asset view in the same fusion, the one
course-covered factor family (S4) this project hadn't implemented.

Stage 11 Step 5 (ANALYSIS_V2.md action 6): stages 3, 5, 6's wrappers
(`regime_switching_strategy`, `with_meanreversion_tilt`, `with_
technical_view`, `technical_phase_flags`) moved to strategy_legacy.py
-- tested and functional, kept as the naive-combination era's
historical record, not deleted, just physically separated now that
this module has grown large enough that the live path was hard to
find. Stages 4 and 7 stayed here: both are still live, not superseded.

Public API (stage 4):
    with_anomaly_override(strategy_fn, cfg) -> strategy_fn

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
    metrics,
    momentum,
    regimes,
    sentiment,
    technicals,
    views,
)

StrategyFn = Callable[[pd.Timestamp, pd.DataFrame], pd.Series]


def with_anomaly_override(
    strategy_fn: StrategyFn,
    cfg: dict,
    record: Callable[[dict], None] | None = None,
) -> StrategyFn:
    """Wrap any strategy_fn with the anomaly risk override (L4, S8):
    if the autoencoder flags the current window as anomalous, blend
    its weights toward a defensive plain-GMV book by
    `anomaly.blend_to_gmv` (1.0 = fully defensive).

    The autoencoder is refit every `anomaly.refit_frequency_days` and
    cached in between -- a lightweight stand-in for the fold-based
    walk-forward refit stage 9 will formalize with real per-fold
    persistence in models/.

    `record`, if given, is called once per invocation with a JSON-
    serializable dict describing the override decision (reconstruction
    error vs threshold, whether it fired, the blend applied) for the
    live runner's diagnostics sink. Default None leaves every existing
    caller (backtest, ablation) untouched.
    """
    acfg = cfg["optimization"]
    cov_method = acfg["covariance"]
    lookback = acfg["lookback_days"]
    blend = cfg["anomaly"]["blend_to_gmv"]
    refit_every = cfg["anomaly"]["refit_frequency_days"]

    cache: dict = {"result": None, "fitted_at": None}

    def _emit(diag: dict) -> None:
        if record is not None:
            record(diag)

    def strategy_fn_with_override(
        as_of: pd.Timestamp, window: pd.DataFrame
    ) -> pd.Series:
        base_weights = strategy_fn(as_of, window)

        features = anomaly.build_features(window)
        if not anomaly.has_enough_history(features, cfg):
            _emit({"active": False, "reason": "insufficient_history"})
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
                    _emit({"active": False, "reason": "fit_failed"})
                    return base_weights

        result = cache["result"]
        recent_features = features.tail(result.seq_len)
        errors = anomaly.reconstruction_error(result, recent_features)
        flagged = len(errors) > 0 and anomaly.is_anomalous(
            errors, result.threshold
        )[-1]
        latest_error = float(errors[-1]) if len(errors) > 0 else None
        if not flagged:
            _emit(
                {
                    "active": True,
                    "fired": False,
                    "recon_error": latest_error,
                    "threshold": float(result.threshold),
                    "fitted_at": cache["fitted_at"].isoformat(),
                }
            )
            return base_weights

        recent = window.tail(lookback)
        cov = allocation.covariance_matrix(recent, method=cov_method)
        gmv_weights = allocation.gmv(cov, cfg)
        aligned_base = base_weights.reindex(gmv_weights.index).fillna(0.0)
        _emit(
            {
                "active": True,
                "fired": True,
                "recon_error": latest_error,
                "threshold": float(result.threshold),
                "blend_to_gmv": float(blend),
                "fitted_at": cache["fitted_at"].isoformat(),
            }
        )
        return (1 - blend) * aligned_base + blend * gmv_weights

    return strategy_fn_with_override


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


def _resolve_delta(
    market_weights: pd.Series, recent: pd.DataFrame, rf: float, bcfg: dict
) -> float:
    """Resolve this week's Black-Litterman delta (S11 Step 6,
    ANALYSIS_V3.md Sec 4/6.2): a fixed constant (`bcfg["delta"]`, the
    default) or, if `bcfg.get("delta_mode") == "self_calibrating"`,
    `allocation.implied_risk_aversion` computed from the trailing
    window (`recent`, the same one already used for `cov`) held at
    `market_weights` -- see that function's docstring for the
    rationale (a fixed equity-calibrated delta understates the
    equilibrium risk premium for a lower-volatility multi-asset
    "market" proxy once real rates are priced in). Missing
    `delta_mode` defaults to "fixed" (the pre-Step-6 behavior)."""
    mode = bcfg.get("delta_mode", "fixed")
    if mode == "fixed":
        return bcfg["delta"]
    if mode == "self_calibrating":
        aligned = market_weights.reindex(recent.columns).fillna(0.0)
        market_returns = metrics.portfolio_return(aligned, recent)
        return allocation.implied_risk_aversion(
            market_returns, rf=rf, fallback=bcfg["delta"]
        )
    raise ValueError(f"Unknown black_litterman.delta_mode: {mode!r}")


def _bl_diagnostics(
    as_of: pd.Timestamp,
    rf: float,
    opt_rf: float,
    delta: float,
    posture: str,
    regime_result: dict | None,
    regime_fallback_reason: str | None,
    prior: pd.Series,
    mu_bl: pd.Series,
    mu_bl_total: pd.Series,
    view_sets: list,
    cov: pd.DataFrame,
    candidates: dict[str, pd.Series],
    selected: pd.Series,
    bcfg: dict,
    excess_return_prior: bool,
) -> dict:
    """Assemble a JSON-serializable snapshot of one weekly Black-
    Litterman decision for the live runner's diagnostics sink (S8
    explainability). A pure, read-only reduction of values already
    computed inside `strategy_fn`: it never influences the selection,
    and is only built when a `record` callback is supplied, so ordinary
    backtest/ablation callers pay nothing for it.

    `rf` is the prevailing risk-free rate (display/provenance only,
    e.g. the report's "rf" line); `opt_rf` is the rate actually passed
    to `max_sharpe`/`utility_select` this week -- 0.0 under the fixed
    `excess_return_prior` convention (ANALYSIS_V4 Sec 2.1: `mu_bl` is
    already an excess return, so subtracting `rf` again would double-
    count it), or `rf` under the legacy (pre-fix) convention kept for
    the Phase-2 side-by-side ablation. `mu_bl_total` = `mu_bl + rf`,
    reported alongside `mu_bl` so both excess and total expected
    returns are visible regardless of which convention produced them.
    """
    tickers = prior.index
    prior_arr = prior.to_numpy()

    views_fired = []
    for view_set in view_sets:
        for i in range(len(view_set)):
            p = np.asarray(view_set.p_rows[i], dtype=float)
            support = {
                t: float(p[j]) for j, t in enumerate(tickers) if p[j] != 0.0
            }
            base = float(p @ prior_arr)
            views_fired.append(
                {
                    "label": view_set.labels[i],
                    "family": view_set.labels[i].split("_")[0],
                    "q": float(view_set.q_rows[i]),
                    "tilt": float(view_set.q_rows[i]) - base,
                    "confidence": float(view_set.confidence[i]),
                    "support": support,
                }
            )

    gate_aversion = bcfg["risk_aversion_for_utility_gate"]
    books = {}
    for name, w in candidates.items():
        wa = w.reindex(mu_bl.index).fillna(0.0)
        exp_ret = float(wa @ mu_bl)
        vol = metrics.portfolio_vol(wa, cov)
        books[name] = {
            "exp_return": exp_ret,
            "exp_return_total": float(wa @ mu_bl_total.reindex(wa.index)),
            "vol": vol,
            "utility": float(
                metrics.utility(exp_ret, vol, gate_aversion, rf=opt_rf)
            ),
            "weights": {t: float(v) for t, v in wa.items()},
        }

    selected_name = None
    for name, w in candidates.items():
        wa = w.reindex(selected.index).fillna(0.0).to_numpy()
        if np.allclose(wa, selected.to_numpy(), atol=1e-12):
            selected_name = name
            break
    ranked = sorted((b["utility"] for b in books.values()), reverse=True)
    utility_margin = float(ranked[0] - ranked[1]) if len(ranked) > 1 else None

    # 100%-cash reference row (ANALYSIS_V4 Sec 2.1): by construction
    # cash has 0 excess return and 0 vol, so its utility is always
    # exactly 0 -- a reference point for whether the gate's winner
    # actually clears "do nothing", not a candidate the gate picks
    # from. Added after utility_margin so it never competes for the
    # winner/runner-up margin.
    books["cash"] = {
        "exp_return": 0.0,
        "exp_return_total": float(rf),
        "vol": 0.0,
        "utility": 0.0,
        "weights": {},
    }

    regime = {"posture": posture}
    if regime_fallback_reason is not None:
        regime["fallback_reason"] = regime_fallback_reason
    if regime_result is not None:
        states = regime_result["states"]
        current_state = int(states.iloc[-1])
        transition = regime_result["transition_matrix"].loc[current_state]
        regime.update(
            {
                "current_state": current_state,
                "state_to_posture": {
                    int(k): v
                    for k, v in regime_result["state_to_posture"].items()
                },
                "state_profiles": {
                    int(s): {c: float(val) for c, val in row.items()}
                    for s, row in regime_result["state_profiles"]
                    .to_dict("index")
                    .items()
                },
                "transition_row": {
                    int(c): float(v) for c, v in transition.items()
                },
            }
        )

    return {
        "as_of": as_of.isoformat(),
        "rf": float(rf),
        "excess_return_prior": excess_return_prior,
        "delta": float(delta),
        "regime": regime,
        "equilibrium_prior": {t: float(v) for t, v in prior.items()},
        "views_fired": views_fired,
        "mu_bl": {t: float(v) for t, v in mu_bl.items()},
        "mu_bl_total": {t: float(v) for t, v in mu_bl_total.items()},
        "candidate_books": books,
        "gate": {
            "type": bcfg["gate"],
            "risk_aversion": gate_aversion,
            "selected": selected_name,
            "utility_margin": utility_margin,
        },
    }


def black_litterman_strategy(
    class_bucket: pd.Series,
    cfg: dict,
    posture_cfg: dict,
    market_ticker: str | None = None,
    rf_series: pd.Series | None = None,
    meanrev_eligible: pd.Index | None = None,
    record: Callable[[dict], None] | None = None,
) -> StrategyFn:
    """Build a `backtest.run`-compatible strategy_fn implementing the
    full Black-Litterman fusion pipeline (S4, stage 8):

    1. Detect the HMM posture (same guard/fallback pattern as
       regime_switching_strategy).
    2. Equilibrium prior Pi from `permanent()`'s asset-class weights
       (notebook 08: "equilibrium returns from asset-class weights").
       Pi = delta*Sigma*w_mkt is an EXCESS return (He-Litterman reverse
       optimization: delta is calibrated as (E[Rm]-Rf)/sigma_m^2, so Pi
       already has rf removed). Under the default
       `black_litterman.excess_return_prior: true` (ANALYSIS_V4 Sec
       2.1), it is used as-is, including for `cash` (whose Pi is ~0 by
       construction -- a zero-covariance asset's equilibrium EXCESS
       return over itself is correctly ~0, not the prevailing rate).
       `false` keeps the legacy (deployed-through-2026-10) behavior for
       the Phase-2 side-by-side ablation: `cash`'s prior is overridden
       to the prevailing risk-free rate, treating Pi as if it were a
       total return.
    3. V1 (regime posture -> asset-class view), V2 (mean-reversion),
       V3 (technical), V5 (dual momentum) each contribute views; V4
       (sentiment) is omitted here -- still live-only (see
       sentiment.py). Views add a tilt relative to `prior`'s own scale
       (excess under the default convention), so they need no
       adjustment either way.
    4. Fuse into mu_BL via Black-Litterman -- same scale as `prior`.
    5. max_sharpe(mu_BL, Sigma, opt_rf) is the primary book; GMV and
       Risk Parity are defensive/fallback books; `cfg["black_litterman"]
       ["gate"]` picks whichever wins:
       - "utility" (default): mean-variance utility_select (using
         mu_BL and opt_rf for all three candidates).
       - "off": always the max_sharpe book, no gating at all.
       - "vol_floor": the max_sharpe book unless its volatility
         exceeds `vol_floor_multiplier` x GMV's volatility, in which
         case GMV. A simple risk-budget gate instead of a quadratic
         utility comparison, which structurally favors minimum
         variance regardless of signal quality (DIAGNOSTIC.md Sec 3;
         stage 11's gate ablation, notebook 11, measured the "off"
         and A-sensitivity arms before this option was added).

    `opt_rf` (the rate actually passed to max_sharpe/utility_select) is
    0.0 when `excess_return_prior` is true -- mu_BL is already excess,
    so subtracting the prevailing rate again would double-count it
    (ANALYSIS_V4 Sec 2.1, the root-cause bug: cash's prior overridden
    to rf AND rf subtracted again downstream, pushing 19 of 22 assets'
    implied excess return below cash and producing an unintended
    high-volatility/commodity tilt) -- or `rf` when false (legacy,
    double-subtracting on purpose to reproduce the deployed arm).

    `rf_series`, if given, is an annual risk-free rate indexed by
    date (e.g. FRED DTB3/100); the prevailing rate as of each `as_of`
    (via `.asof`, so never a future value) is used as `rf` throughout
    this week's decision (display/provenance, and the legacy cash
    override/double-subtraction when `excess_return_prior` is false).
    Defaults to a constant 0.0 if omitted.

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
    excess_return_prior = bcfg.get("excess_return_prior", True)
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
        regime_result = None
        regime_fallback_reason = None
        if not regimes.has_enough_history(market_returns, cfg):
            posture = "neutral"
            regime_fallback_reason = "insufficient_history"
        else:
            try:
                regime_result = regimes.market_regime(
                    market_returns, cfg, posture_cfg
                )
                posture = regime_result["current_posture"]
            except (ValueError, np.linalg.LinAlgError) as exc:
                posture = "neutral"
                # ANALYSIS_V4 Sec 2.3a: a degenerate HMM fit (NaN
                # params, ~2% of OOS fits) must not fail silently --
                # surfaced here instead of only being inferable from
                # the absence of "current_state" in diagnostics.
                regime_fallback_reason = f"hmm_fit_degenerate: {exc}"

        recent = window.tail(lookback)
        cov = allocation.covariance_matrix(recent, method=cov_method)
        tickers = recent.columns

        delta = _resolve_delta(market_weights, recent, rf, bcfg)
        prior = allocation.equilibrium_returns(
            market_weights.reindex(tickers).fillna(0.0), cov, delta
        )
        if not excess_return_prior:
            # Legacy (pre-ANALYSIS_V4) behavior, kept only for the
            # Phase-2 side-by-side ablation: treats Pi as a total
            # return by overriding cash's ~0 excess prior with the
            # prevailing rate -- then opt_rf below subtracts rf a
            # second time from every other asset's already-excess Pi.
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

        if momentum.has_enough_history(prices, cfg):
            try:
                mom_view = momentum.momentum_view(
                    prices, class_bucket, rf_series, cfg
                )
            except ValueError:
                mom_view = pd.Series(0.0, index=tickers)
        else:
            mom_view = pd.Series(0.0, index=tickers)

        view_sets = [
            views.regime_view(posture, class_bucket, prior, cfg),
            views.meanreversion_views(meanrev_view, prior, cfg),
            views.technical_views(tech_view, prior, cfg),
            views.momentum_views(mom_view, prior, cfg),
        ]
        P, Q, Omega, _ = views.assemble(view_sets, tickers, cov, cfg)
        mu_bl = allocation.black_litterman(
            prior, cov, P, Q, Omega, bcfg["tau"]
        )
        # 0.0 under the fixed convention: mu_bl is already excess, so
        # max_sharpe/utility_select must not subtract rf a second time
        # (ANALYSIS_V4 Sec 2.1). `rf` under the legacy convention,
        # reproducing the original double subtraction on purpose.
        opt_rf = rf if not excess_return_prior else 0.0
        mu_bl_total = mu_bl + rf

        candidates = {
            "black_litterman": allocation.max_sharpe(
                mu_bl, cov, cfg, rf=opt_rf
            ),
            "gmv": allocation.gmv(cov, cfg),
            "risk_parity": allocation.risk_parity(cov, cfg),
        }
        selected = _select_gated_candidate(
            candidates, mu_bl, cov, bcfg, opt_rf
        )

        if record is not None:
            record(
                _bl_diagnostics(
                    as_of, rf, opt_rf, delta, posture, regime_result,
                    regime_fallback_reason, prior, mu_bl, mu_bl_total,
                    view_sets, cov, candidates, selected, bcfg,
                    excess_return_prior,
                )
            )
        return selected

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

    Respects `black_litterman.excess_return_prior` identically to
    `black_litterman_strategy` (ANALYSIS_V4 Sec 2.1/6 Tier 1 action
    5) -- same prior, same double-subtraction bug, same fix.
    """
    lookback = cfg["optimization"]["lookback_days"]
    cov_method = cfg["optimization"]["covariance"]
    market_ticker = market_ticker or cfg["regimes"]["market_ticker"]
    bcfg = cfg["black_litterman"]
    excess_return_prior = bcfg.get("excess_return_prior", True)

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

        delta = _resolve_delta(market_weights_bucket, recent, rf, bcfg)
        prior = allocation.equilibrium_returns(
            market_weights_bucket, cov, delta
        )
        if not excess_return_prior and "cash" in prior.index:
            prior.loc["cash"] = rf  # legacy convention, see asset-level fn

        view_set = views.regime_view(posture, bucket_identity, prior, cfg)
        P, Q, Omega, _ = views.assemble(
            [view_set], pd.Index(active_buckets), cov, cfg
        )
        mu_bl = allocation.black_litterman(
            prior, cov, P, Q, Omega, bcfg["tau"]
        )
        opt_rf = rf if not excess_return_prior else 0.0

        candidates = {
            "black_litterman": allocation.max_sharpe(
                mu_bl, cov, bucket_cfg, rf=opt_rf
            ),
            "gmv": allocation.gmv(cov, bucket_cfg),
            "risk_parity": allocation.risk_parity(cov, bucket_cfg),
        }
        selected_bucket = _select_gated_candidate(
            candidates, mu_bl, cov, bcfg, opt_rf
        )
        return buckets.expand_bucket_weights(selected_bucket, class_bucket)

    return strategy_fn
