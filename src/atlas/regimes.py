"""Regime detection (L1): HMM primary switch + K-Means macro context.

Stage 3. The HMM is refit on a trailing window every time it's called
(the "signal clock" style used by allocation.py's classical books in
stage 2) -- cheap enough weekly per config, so no walk-forward fold
persistence is needed yet; that lands in stage 9. Regimes are always
identified by PROFILE (relative vol/return, or growth/inflation for
the macro side), never by raw cluster/state index, since that index
is arbitrary and can flip on every re-fit.

Macro K-Means is structural context only in this stage (component 3):
it is not yet wired into the V1 decision in strategy.py. It becomes a
confidence input alongside the anomaly/GMM agreement score in stage 4.

Stage 11 Tier 3: `n_states` (HMM regime count) was a fixed, reasoned-
but-untuned default (3) through stage 10. `hmm_bic_curve` fits a
candidate HMM at each `n_states` in a range and scores it by BIC
(hmmlearn's built-in `GaussianHMM.bic`, which already accounts for
`covariance_type="diag"`'s parameter count -- no need to hand-derive
it), so a caller can freeze the argmin on an IS-only window, the same
"grid search IS-only, freeze for OOS" convention as `meanreversion.
lookback_days`/`entry_z` and the turnover cap (notebook 09).

Public API:
    has_enough_history(market_returns, cfg) -> bool
    fit_hmm(market_returns, cfg) -> (model, features_df)
    hmm_bic_curve(market_returns, cfg, n_states_range) -> pd.Series
    decode_states(model, features_df) -> pd.Series
    state_profiles(states, market_returns) -> pd.DataFrame
    map_states_to_postures(profiles, posture_cfg) -> dict[state, name]
    market_regime(market_returns, cfg, posture_cfg) -> dict
    macro_kmeans(macro_df, cfg) -> dict
    macro_elbow_curve(macro_df, cfg, k_range) -> pd.Series
    label_macro_clusters(macro_df, labels, cfg) -> dict[cluster, name]
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# hmmlearn logs a "Model is not converging" warning via `logging` (not
# `warnings`) whenever a fit exhausts n_iter without meeting tolerance
# -- expected and already accounted for by the fixed n_iter cap below,
# not something to act on. Left at WARNING level it floods Jupyter's
# IOPub channel on every weekly refit across a multi-year backtest,
# which measurably slows notebook execution (confirmed: an identical
# grid search ran ~4x faster outside a notebook kernel, stage 9).
logging.getLogger("hmmlearn").setLevel(logging.ERROR)

_PROFILE_RANK = {"low": 0, "medium": 1, "high": 2}


def _hmm_features(market_returns: pd.Series, vol_window: int) -> pd.DataFrame:
    rolling_vol = market_returns.rolling(vol_window).std()
    features = pd.concat([market_returns, rolling_vol], axis=1)
    features.columns = ["return", "rolling_vol"]
    return features.dropna()


def has_enough_history(market_returns: pd.Series, cfg: dict) -> bool:
    """Whether there's enough trailing data for a meaningful HMM fit:
    at least one full rolling-vol window, plus enough rows to tell
    `n_states` regimes apart. Guards the early weeks of a backtest
    (or a live system's first weeks) where a real regime read isn't
    possible yet -- callers should fall back to a neutral default
    rather than force a fit on almost no data."""
    hmm_cfg = cfg["regimes"]["hmm"]
    min_obs = hmm_cfg["vol_window_days"] + 10 * hmm_cfg["n_states"]
    return market_returns.dropna().shape[0] >= min_obs


def fit_hmm(
    market_returns: pd.Series, cfg: dict
) -> tuple[GaussianHMM, pd.DataFrame]:
    """Fit a Gaussian HMM on (return, rolling_vol) for a market proxy
    series, using the trailing `regimes.hmm.lookback_days` window."""
    hmm_cfg = cfg["regimes"]["hmm"]
    window = market_returns.tail(hmm_cfg["lookback_days"])
    features = _hmm_features(window, hmm_cfg["vol_window_days"])
    model = GaussianHMM(
        n_components=hmm_cfg["n_states"],
        # Diagonal, not full, covariance: with only 2 features this is
        # far more numerically stable across weekly refits on live
        # data than full covariance, which can go singular (Cholesky
        # failure) whenever a state's assignments happen to be nearly
        # degenerate.
        covariance_type="diag",
        random_state=cfg["general"]["random_seed"],
        n_iter=100,
    )
    model.fit(features.values)
    return model, features


def hmm_bic_curve(
    market_returns: pd.Series, cfg: dict, n_states_range: range
) -> pd.Series:
    """BIC (lower is better) for a GaussianHMM fit at each candidate
    `n_states` in `n_states_range`, on the same trailing
    `regimes.hmm.lookback_days` window and feature set `fit_hmm` uses
    (stage 11 Tier 3). Intended for an IS-only elbow-style selection,
    mirroring `macro_elbow_curve`: fit once per candidate, pick the
    argmin, freeze that value into `regimes.hmm.n_states` for the OOS
    run -- not refit inside the weekly backtest loop."""
    hmm_cfg = cfg["regimes"]["hmm"]
    window = market_returns.tail(hmm_cfg["lookback_days"])
    features = _hmm_features(window, hmm_cfg["vol_window_days"])
    bics = {}
    for k in n_states_range:
        model = GaussianHMM(
            n_components=k,
            covariance_type="diag",
            random_state=cfg["general"]["random_seed"],
            n_iter=100,
        )
        model.fit(features.values)
        bics[k] = model.bic(features.values)
    return pd.Series(bics)


def decode_states(model: GaussianHMM, features: pd.DataFrame) -> pd.Series:
    """Viterbi-decode the most likely state sequence for `features`."""
    states = model.predict(features.values)
    return pd.Series(states, index=features.index, name="state")


def state_profiles(
    states: pd.Series, market_returns: pd.Series
) -> pd.DataFrame:
    """Annualized mean return and vol of the market series, per state."""
    aligned = market_returns.reindex(states.index)
    grouped = aligned.groupby(states)
    return pd.DataFrame(
        {
            "mean_return": grouped.mean() * 252,
            "vol": grouped.std() * np.sqrt(252),
        }
    )


def map_states_to_postures(
    profiles: pd.DataFrame, posture_cfg: dict
) -> dict:
    """Assign each fitted state to its nearest posture by relative
    vol/return profile -- never by state index. Uses nearest-score
    matching rather than a strict one-to-one assignment, since a
    short or quiet window can leave the Viterbi path never visiting
    every configured state; that must degrade gracefully, not crash a
    week-long backtest loop."""
    postures = posture_cfg["postures"]
    target_scores = {
        name: (
            _PROFILE_RANK[spec["profile"]["vol"]]
            - _PROFILE_RANK[spec["profile"]["mean_return"]]
        )
        for name, spec in postures.items()
    }

    vol_rank = profiles["vol"].rank(method="first") - 1
    return_rank = profiles["mean_return"].rank(method="first") - 1
    empirical_score = vol_rank - return_rank

    return {
        state: min(
            target_scores, key=lambda name: abs(target_scores[name] - score)
        )
        for state, score in empirical_score.items()
    }


def market_regime(
    market_returns: pd.Series, cfg: dict, posture_cfg: dict
) -> dict:
    """End-to-end regime detection: fit HMM, decode states, map to
    postures by profile (S7/S8, primary switch).

    Returns a dict with keys: posture_series, current_posture,
    transition_matrix, state_profiles, state_to_posture, states.
    """
    model, features = fit_hmm(market_returns, cfg)
    states = decode_states(model, features)
    profiles = state_profiles(states, market_returns)
    state_to_posture = map_states_to_postures(profiles, posture_cfg)
    posture_series = states.map(state_to_posture)
    transition_matrix = pd.DataFrame(
        model.transmat_,
        index=range(model.n_components),
        columns=range(model.n_components),
    )
    return {
        "posture_series": posture_series,
        "current_posture": posture_series.iloc[-1],
        "transition_matrix": transition_matrix,
        "state_profiles": profiles,
        "state_to_posture": state_to_posture,
        "states": states,
    }


def _macro_features(macro_df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Resample raw FRED series to month-end and transform to
    stationary features (S8 rule: models only see stationary inputs):
    YoY % change for the growth/inflation level indicators, 12-month
    difference for rate/spread/sentiment series already in
    percentage-point or index-point terms."""
    monthly = macro_df.resample("ME").last().ffill()
    kcfg = cfg["regimes"]["kmeans_macro"]
    pct_change_cols = {kcfg["growth_indicator"], kcfg["inflation_indicator"]}

    features = pd.DataFrame(index=monthly.index)
    for col in monthly.columns:
        if col in pct_change_cols:
            features[col] = monthly[col].pct_change(12)
        else:
            features[col] = monthly[col].diff(12)
    return features.dropna()


def macro_kmeans(macro_df: pd.DataFrame, cfg: dict) -> dict:
    """StandardScaler (train-only fit) + K-Means + 2-D PCA on
    stationary FRED macro features (S8). Structural context only in
    this stage -- see module docstring."""
    kcfg = cfg["regimes"]["kmeans_macro"]
    seed = cfg["general"]["random_seed"]
    features = _macro_features(macro_df, cfg)

    scaler = StandardScaler().fit(features.values)
    scaled = scaler.transform(features.values)

    model = KMeans(n_clusters=kcfg["k"], random_state=seed, n_init=10)
    labels = pd.Series(
        model.fit_predict(scaled), index=features.index, name="cluster"
    )

    pca = PCA(n_components=2, random_state=seed).fit(scaled)
    coords = pd.DataFrame(
        pca.transform(scaled), index=features.index, columns=["pc1", "pc2"]
    )

    return {
        "features": features,
        "labels": labels,
        "pca_coords": coords,
        "explained_variance_ratio": pca.explained_variance_ratio_,
        "model": model,
        "scaler": scaler,
    }


def macro_elbow_curve(
    macro_df: pd.DataFrame, cfg: dict, k_range: range
) -> pd.Series:
    """K-Means inertia across `k_range`, for the elbow chart (S8)."""
    seed = cfg["general"]["random_seed"]
    features = _macro_features(macro_df, cfg)
    scaled = StandardScaler().fit_transform(features.values)
    inertias = {
        k: KMeans(n_clusters=k, random_state=seed, n_init=10)
        .fit(scaled)
        .inertia_
        for k in k_range
    }
    return pd.Series(inertias)


def label_macro_clusters(
    macro_df: pd.DataFrame, labels: pd.Series, cfg: dict
) -> dict:
    """Map macro clusters to the standard four-quadrant business-cycle
    labels (Expansion/Slowdown/Contraction/Recovery), by each cluster's
    OWN mean growth/inflation level (a profile, never a cluster index):

        growth \\ inflation |  low        high
        --------------------+---------------------
        high                | Recovery    Expansion
        low                 | Contraction Slowdown
    """
    kcfg = cfg["regimes"]["kmeans_macro"]
    features = _macro_features(macro_df, cfg)
    growth = features[kcfg["growth_indicator"]].groupby(labels).mean()
    inflation = features[kcfg["inflation_indicator"]].groupby(labels).mean()

    growth_high = growth > growth.median()
    inflation_high = inflation > inflation.median()

    quadrant = {
        (True, False): "Recovery",
        (True, True): "Expansion",
        (False, True): "Slowdown",
        (False, False): "Contraction",
    }
    return {
        cluster: quadrant[(growth_high[cluster], inflation_high[cluster])]
        for cluster in growth.index
    }
