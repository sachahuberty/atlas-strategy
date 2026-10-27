"""IS-only re-selection harness (reselection task, external review).

See `docs/RESELECTION_PREREG.md` Sec 5 for the efficiency design this
module implements: per decision date, everything INDEPENDENT of which
(V3 on/off, V5 on/off, risk-off-target) variant is active -- rf,
posture, covariance, the equilibrium prior, the raw V2/V3/V5 per-asset
view series, the anomaly flag, and the GMV/risk-parity candidate books
-- is computed and cached exactly once (`compute_shared_entry`). Only
the 12 variants that actually change `mu_bl` get their own
Black-Litterman candidate computed (`compute_variant_entry`). Gate
risk-aversion re-selects among a variant's cached candidates with no
re-run (`gate_select`, a thin wrapper around `strategy.
_select_gated_candidate` -- the SAME function the live pipeline calls,
not a reimplementation). Turnover cap/band re-run only `backtest.run`'s
execution layer over an already-decided weekly target-weight lookup
(`run_turnover_variant`), never the signal pipeline.

Caches persist incrementally to `data/processed/reselection_cache/`
(gitignored) via `save_checkpoint`/`load_checkpoint`, so a stalled run
(the known Windows SLSQP/HMM stall, S9) does not lose completed
decision dates -- `build_shared_caches`/`build_variant_caches` resume
from the latest checkpoint rather than recomputing from scratch.

`assert_no_oos_access` is the lookahead canary for this selection step
itself (pre-registration Sec "no-lookahead-within-a-fold rule"):
every public entry point that takes a `returns`/`prices` frame calls it
first, refusing to run at all if that frame extends on or after
`general.oos_start_date`.

Public API:
    ViewVariant                                        # dataclass
    ALL_VIEW_VARIANTS                                  # the 12 combos
    variant_cfg(base_cfg, variant) -> dict
    assert_no_oos_access(frame, cfg) -> None            # lookahead canary
    compute_shared_entry(as_of, window, class_bucket, cfg, posture_cfg,
        market_ticker, rf_series, meanrev_eligible, anomaly_cache) -> dict
    compute_variant_entry(shared_entry, class_bucket, variant_cfg_dict,
        tickers) -> dict
    gate_select(variant_entry, bcfg, risk_aversion) -> (str, pd.Series)
    build_shared_caches(dates, window_fn, ..., checkpoint_path=None) -> dict
    build_variant_caches(shared_caches, class_bucket, base_cfg, variant,
        checkpoint_path=None) -> dict
    target_weight_series(variant_caches, bcfg, risk_aversion) -> dict
    run_turnover_variant(target_weights_by_date, tickers, is_returns,
        cfg) -> backtest.BacktestResult
    save_checkpoint(obj, path) / load_checkpoint(path) -> obj | None
"""

from __future__ import annotations

import copy
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from . import allocation, anomaly, backtest, meanreversion, momentum
from . import regimes, strategy, technicals, views

CACHE_DIR = Path(__file__).resolve().parents[2] / "data" / "processed" / (
    "reselection_cache"
)

_RISK_OFF_VARIANTS = ("legacy", "bonds_cash", "bonds_gold")


@dataclass(frozen=True)
class ViewVariant:
    """One of the 12 (V3 on/off x V5 on/off x risk-off-target)
    combinations whose own Black-Litterman candidate must actually be
    computed -- see module docstring. `risk_off_assets` only matters
    when `risk_off_target == "bonds_gold"` (views._risk_off_targets
    ignores it otherwise)."""

    v3: bool
    v5: bool
    risk_off_target: str
    risk_off_assets: tuple[str, ...] = ("GLD",)

    @property
    def key(self) -> str:
        v3_s = "on" if self.v3 else "off"
        v5_s = "on" if self.v5 else "off"
        return f"V3_{v3_s}__V5_{v5_s}__riskoff_{self.risk_off_target}"


ALL_VIEW_VARIANTS: tuple[ViewVariant, ...] = tuple(
    ViewVariant(v3=v3, v5=v5, risk_off_target=risk_off)
    for v3 in (False, True)
    for v5 in (False, True)
    for risk_off in _RISK_OFF_VARIANTS
)
assert len(ALL_VIEW_VARIANTS) == 12


def variant_cfg(base_cfg: dict, variant: ViewVariant) -> dict:
    """Deep-copy `base_cfg` (never mutated) and overlay this variant's
    module flags + risk-off target -- the SAME in-memory-override
    pattern notebook 09/11 already use for grid search (`copy.
    deepcopy(cfg)` + override specific keys), never writing to
    config.yaml itself."""
    cfg = copy.deepcopy(base_cfg)
    cfg["modules"]["technical_view"] = variant.v3
    cfg["modules"]["momentum_view"] = variant.v5
    cfg["black_litterman"]["regime_targets"] = {
        "risk_off": variant.risk_off_target,
        "risk_off_assets": list(variant.risk_off_assets),
    }
    return cfg


def assert_no_oos_access(frame: pd.DataFrame | pd.Series, cfg: dict) -> None:
    """Lookahead canary for the selection step itself (pre-registration
    Sec 1): refuse to run at all if `frame`'s index reaches on or after
    `general.oos_start_date`. Every public entry point below that takes
    a returns/prices frame calls this first."""
    oos_start = pd.Timestamp(cfg["general"]["oos_start_date"])
    if len(frame.index) and frame.index.max() >= oos_start:
        raise ValueError(
            f"Frame extends to {frame.index.max().date()}, on or after "
            f"oos_start_date {oos_start.date()}; the reselection task "
            "may not read OOS data until Phase 4."
        )


def compute_shared_entry(
    as_of: pd.Timestamp,
    window: pd.DataFrame,
    class_bucket: pd.Series,
    cfg: dict,
    posture_cfg: dict,
    market_ticker: str,
    rf_series: pd.Series | None,
    meanrev_eligible: pd.Index | None,
    anomaly_cache: dict,
) -> dict:
    """Everything independent of the (V3, V5, risk-off-target) variant,
    for one decision date. `anomaly_cache` is a caller-owned mutable
    dict (`{"result": None, "fitted_at": None}`) threaded across calls
    exactly like `strategy.with_anomaly_override`'s own closure cache,
    so the autoencoder refit schedule (`anomaly.refit_frequency_days`)
    is respected across the whole IS walk, not refit every call.
    """
    rf = 0.0
    if rf_series is not None:
        looked_up = rf_series.asof(as_of)
        if pd.notna(looked_up):
            rf = float(looked_up)

    market_returns = window[market_ticker]
    posture = "neutral"
    if regimes.has_enough_history(market_returns, cfg):
        try:
            posture = regimes.market_regime(
                market_returns, cfg, posture_cfg
            )["current_posture"]
        except (ValueError, np.linalg.LinAlgError):
            posture = "neutral"

    lookback = cfg["optimization"]["lookback_days"]
    cov_method = cfg["optimization"]["covariance"]
    recent = window.tail(lookback)
    tickers = recent.columns
    cov = allocation.covariance_matrix(recent, method=cov_method)

    bcfg = cfg["black_litterman"]
    market_weights = allocation.permanent(class_bucket)
    delta = strategy._resolve_delta(market_weights, recent, rf, bcfg)
    prior = allocation.equilibrium_returns(
        market_weights.reindex(tickers).fillna(0.0), cov, delta
    )
    # excess_return_prior stays true throughout this task (the
    # corrected pipeline) -- no cash-prior override.

    prices = (1.0 + window).cumprod()
    meanrev_view = (
        meanreversion.mean_reversion_view(
            prices, cfg, eligible=meanrev_eligible
        )
        if meanreversion.has_enough_history(prices, cfg)
        else pd.Series(0.0, index=tickers)
    )
    tech_view = (
        technicals.technical_view(prices, cfg)
        if technicals.has_enough_history(prices, cfg)
        else pd.Series(0.0, index=tickers)
    )
    mom_view = (
        momentum.momentum_view(prices, class_bucket, rf_series, cfg)
        if momentum.has_enough_history(prices, cfg)
        else pd.Series(0.0, index=tickers)
    )

    gmv_weights = allocation.gmv(cov, cfg)
    risk_parity_weights = allocation.risk_parity(cov, cfg)

    anomaly_flagged = False
    acfg = cfg["anomaly"]
    # Matches live.build_strategy's own gating exactly: with
    # modules.anomaly_override off, the real pipeline never wraps
    # with_anomaly_override at all, so this cache must not silently
    # compute/apply an override the live decision never makes either.
    if cfg["modules"].get("anomaly_override", False):
        features = anomaly.build_features(window)
    else:
        features = None
    if features is not None and anomaly.has_enough_history(features, cfg):
        refit_every = acfg["refit_frequency_days"]
        stale = (
            anomaly_cache.get("fitted_at") is None
            or (as_of - anomaly_cache["fitted_at"]).days >= refit_every
        )
        if stale:
            try:
                anomaly_cache["result"] = anomaly.fit_autoencoder(
                    features, cfg
                )
                anomaly_cache["fitted_at"] = as_of
            except Exception:  # noqa: BLE001 -- same guard strategy.py uses
                pass
        result = anomaly_cache.get("result")
        if result is not None:
            recent_features = features.tail(result.seq_len)
            errors = anomaly.reconstruction_error(result, recent_features)
            anomaly_flagged = len(errors) > 0 and bool(
                anomaly.is_anomalous(errors, result.threshold)[-1]
            )

    return {
        "as_of": as_of,
        "rf": rf,
        "posture": posture,
        "cov": cov,
        "delta": delta,
        "prior": prior,
        "meanrev_view": meanrev_view,
        "tech_view": tech_view,
        "mom_view": mom_view,
        "gmv_weights": gmv_weights,
        "risk_parity_weights": risk_parity_weights,
        "anomaly_flagged": anomaly_flagged,
        "tickers": tickers,
    }


def compute_variant_entry(
    shared_entry: dict,
    class_bucket: pd.Series,
    variant_cfg_dict: dict,
    tickers: pd.Index,
) -> dict:
    """The one part of the pipeline that actually depends on the
    (V3, V5, risk-off-target) variant: assemble this variant's views,
    fuse into `mu_bl`, compute its Black-Litterman candidate, and apply
    the (variant-independent) anomaly blend using the shared GMV book.
    """
    bcfg = variant_cfg_dict["black_litterman"]
    prior = shared_entry["prior"]
    cov = shared_entry["cov"]

    view_sets = [
        views.regime_view(
            shared_entry["posture"], class_bucket, prior, variant_cfg_dict
        ),
        views.meanreversion_views(
            shared_entry["meanrev_view"], prior, variant_cfg_dict
        ),
        views.technical_views(
            shared_entry["tech_view"], prior, variant_cfg_dict
        ),
        views.momentum_views(
            shared_entry["mom_view"], prior, variant_cfg_dict
        ),
    ]
    P, Q, Omega, _ = views.assemble(view_sets, tickers, cov, variant_cfg_dict)
    mu_bl = allocation.black_litterman(prior, cov, P, Q, Omega, bcfg["tau"])

    bl_weights = allocation.max_sharpe(mu_bl, cov, variant_cfg_dict, rf=0.0)
    if shared_entry["anomaly_flagged"]:
        blend = variant_cfg_dict["anomaly"]["blend_to_gmv"]
        aligned_gmv = shared_entry["gmv_weights"].reindex(
            bl_weights.index
        ).fillna(0.0)
        bl_weights = (1 - blend) * bl_weights + blend * aligned_gmv

    candidates = {
        "black_litterman": bl_weights,
        "gmv": shared_entry["gmv_weights"],
        "risk_parity": shared_entry["risk_parity_weights"],
    }
    return {"mu_bl": mu_bl, "cov": cov, "candidates": candidates, "rf": 0.0}


def gate_select(
    variant_entry: dict, bcfg: dict, risk_aversion: float
) -> tuple[str, pd.Series]:
    """Re-select the gate winner among a variant's CACHED candidates,
    no re-run. The returned weights come from `strategy.
    _select_gated_candidate` -- the exact function the live pipeline
    calls, not a reimplementation, forced to `gate="utility"` with
    `risk_aversion` overridden (Phase 2 only varies `A`, never the gate
    type itself). The winner's NAME is read from `allocation.
    utility_select` directly (the same call `_select_gated_candidate`'s
    own "utility" branch already makes internally) rather than
    re-matching weights by value, which would be fragile if two
    candidates ever tied exactly.
    """
    gate_bcfg = {
        **bcfg,
        "gate": "utility",
        "risk_aversion_for_utility_gate": risk_aversion,
    }
    selected = strategy._select_gated_candidate(
        variant_entry["candidates"],
        variant_entry["mu_bl"],
        variant_entry["cov"],
        gate_bcfg,
        variant_entry["rf"],
    )
    selected_name, _ = allocation.utility_select(
        variant_entry["candidates"],
        variant_entry["mu_bl"],
        variant_entry["cov"],
        risk_aversion,
        rf=variant_entry["rf"],
    )
    return selected_name, selected


def target_weight_series(
    variant_caches: dict[pd.Timestamp, dict],
    bcfg: dict,
    risk_aversion: float,
) -> dict[pd.Timestamp, pd.Series]:
    """Gate-select every cached decision date's candidates at a given
    risk aversion -- no pipeline re-run, see `gate_select`."""
    return {
        as_of: gate_select(entry, bcfg, risk_aversion)[1]
        for as_of, entry in variant_caches.items()
    }


def run_turnover_variant(
    target_weights_by_date: dict[pd.Timestamp, pd.Series],
    tickers: pd.Index,
    is_returns: pd.DataFrame,
    cfg: dict,
) -> backtest.BacktestResult:
    """Re-run ONLY `backtest.run`'s execution layer (drift, no-trade
    band, turnover cap, costs) over an already-decided weekly target-
    weight lookup -- no signal recomputation. `cfg["rebalance"]` is the
    only thing that may differ between calls on the same
    `target_weights_by_date`."""
    assert_no_oos_access(is_returns, cfg)

    def lookup_strategy(
        as_of: pd.Timestamp, window: pd.DataFrame
    ) -> pd.Series:
        return target_weights_by_date[as_of].reindex(tickers).fillna(0.0)

    return backtest.run(lookup_strategy, is_returns, cfg)


def save_checkpoint(obj, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "wb") as f:
        pickle.dump(obj, f)
    tmp.replace(path)


def load_checkpoint(path: str | Path):
    path = Path(path)
    if not path.exists():
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


def build_shared_caches(
    dates: list[pd.Timestamp],
    returns: pd.DataFrame,
    class_bucket: pd.Series,
    cfg: dict,
    posture_cfg: dict,
    market_ticker: str,
    rf_series: pd.Series | None,
    meanrev_eligible: pd.Index | None,
    checkpoint_path: str | Path | None = None,
) -> dict[pd.Timestamp, dict]:
    """Build (or resume) the shared, variant-independent cache across
    every decision date in `dates`, checkpointing after each date so a
    stalled run (the known Windows SLSQP/HMM stall) does not lose
    completed work."""
    assert_no_oos_access(returns, cfg)

    cache: dict[pd.Timestamp, dict] = {}
    anomaly_cache: dict = {"result": None, "fitted_at": None}
    if checkpoint_path is not None:
        loaded = load_checkpoint(checkpoint_path)
        if loaded is not None:
            cache, anomaly_cache = loaded

    for as_of in dates:
        if as_of in cache:
            continue
        window = returns.loc[:as_of]
        cache[as_of] = compute_shared_entry(
            as_of, window, class_bucket, cfg, posture_cfg, market_ticker,
            rf_series, meanrev_eligible, anomaly_cache,
        )
        if checkpoint_path is not None:
            save_checkpoint((cache, anomaly_cache), checkpoint_path)
    return cache


def build_variant_caches(
    shared_caches: dict[pd.Timestamp, dict],
    class_bucket: pd.Series,
    base_cfg: dict,
    variant: ViewVariant,
    checkpoint_path: str | Path | None = None,
) -> dict[pd.Timestamp, dict]:
    """Build (or resume) one view-variant's cache from the already-
    built shared cache -- no signal pipeline re-run, just view
    assembly + Black-Litterman fusion per date."""
    cache: dict[pd.Timestamp, dict] = {}
    if checkpoint_path is not None:
        loaded = load_checkpoint(checkpoint_path)
        if loaded is not None:
            cache = loaded

    v_cfg = variant_cfg(base_cfg, variant)
    for as_of, shared_entry in shared_caches.items():
        if as_of in cache:
            continue
        cache[as_of] = compute_variant_entry(
            shared_entry, class_bucket, v_cfg, shared_entry["tickers"]
        )
        if checkpoint_path is not None:
            save_checkpoint(cache, checkpoint_path)
    return cache
