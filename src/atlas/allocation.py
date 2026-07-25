"""Optimizers (L3): classical books now, Black-Litterman fusion later.

Stage 2: max_sharpe, gmv, risk_parity, hrp, tracking_error_min,
efficient_frontier, permanent, sixty_forty, utility_select, plus the
mean/covariance helpers they share.

Stage 3 addition: apply_defensive_tilt, for the regime-conditional
risk-off tilt wired in strategy.py (S7).

Stage 8 (not yet implemented): equilibrium_returns, black_litterman.
Those need the view machinery from views.py and are out of scope
until that stage.

All functions return pd.Series of weights indexed by ticker:
weights >= 0, sum to 1. Constraints (long-only, per-asset cap) come
from config; covariance from Ledoit-Wolf shrinkage by default.

Public API (stage 2):
    mean_returns(returns) -> pd.Series
    covariance_matrix(returns, method) -> pd.DataFrame
    max_sharpe(mu, cov, cfg) -> weights           # SLSQP (S2/S3)
    gmv(cov, cfg) -> weights                      # S5
    risk_parity(cov, cfg) -> weights              # S5 risk contributions
    hrp(returns, cfg) -> weights                  # S4 dendrogram
    tracking_error_min(cov, benchmark, cfg) -> weights   # S2
    efficient_frontier(mu, cov, cfg) -> pd.DataFrame     # S2/S3
    permanent(class_bucket) / sixty_forty(class_bucket) -> weights
    utility_select(candidates, mu, cov, risk_aversion) -> (name, weights)
    cap_and_renormalize(weights, cap) -> weights   # waterfall re-cap, reused
                                                    # by hrp and every tilt

Public API (stage 3):
    apply_defensive_tilt(weights, class_bucket, tilt, cfg) -> weights
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage
from scipy.optimize import minimize
from scipy.spatial.distance import squareform
from sklearn.covariance import LedoitWolf

from .metrics import ann_return, portfolio_vol, utility

TRADING_DAYS_PER_YEAR = 252


def mean_returns(
    returns: pd.DataFrame, periods_per_year: int = TRADING_DAYS_PER_YEAR
) -> pd.Series:
    """Annualized mean return per asset (reuses metrics.ann_return)."""
    return returns.apply(
        lambda col: ann_return(col.dropna(), periods_per_year)
    )


def covariance_matrix(
    returns: pd.DataFrame,
    method: str = "ledoit_wolf",
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> pd.DataFrame:
    """Annualized asset covariance. `method` is 'ledoit_wolf' (shrunk,
    the config default) or 'sample' (plain sample covariance)."""
    if method == "ledoit_wolf":
        cov = LedoitWolf().fit(returns.values).covariance_
    elif method == "sample":
        cov = returns.cov().values
    else:
        raise ValueError(f"Unknown covariance method: {method}")
    cov = cov * periods_per_year
    return pd.DataFrame(cov, index=returns.columns, columns=returns.columns)


def _sum_to_one() -> dict:
    return {"type": "eq", "fun": lambda w: w.sum() - 1.0}


def _renormalize(x: np.ndarray, tickers: pd.Index) -> pd.Series:
    """Clip tiny negative SLSQP residuals and rescale to sum exactly 1."""
    w = pd.Series(np.clip(x, 0.0, None), index=tickers)
    return w / w.sum()


def max_sharpe(
    mu: pd.Series, cov: pd.DataFrame, cfg: dict, rf: float = 0.0
) -> pd.Series:
    """Max-Sharpe weights via SLSQP (S2/S3)."""
    tickers = mu.index
    cov = cov.loc[tickers, tickers].values
    mu_vals = mu.values
    cap = cfg["constraints"]["per_asset_cap"]
    n = len(tickers)

    def neg_sharpe(w: np.ndarray) -> float:
        ret = w @ mu_vals - rf
        vol = np.sqrt(w @ cov @ w)
        return 0.0 if vol < 1e-12 else -ret / vol

    result = minimize(
        neg_sharpe,
        np.full(n, 1.0 / n),
        method="SLSQP",
        bounds=[(0.0, cap)] * n,
        constraints=(_sum_to_one(),),
    )
    return _renormalize(result.x, tickers)


def gmv(cov: pd.DataFrame, cfg: dict) -> pd.Series:
    """Global minimum-variance weights via SLSQP (S5)."""
    tickers = cov.index
    cov_vals = cov.values
    cap = cfg["constraints"]["per_asset_cap"]
    n = len(tickers)

    result = minimize(
        lambda w: w @ cov_vals @ w,
        np.full(n, 1.0 / n),
        method="SLSQP",
        bounds=[(0.0, cap)] * n,
        constraints=(_sum_to_one(),),
    )
    return _renormalize(result.x, tickers)


def risk_parity(cov: pd.DataFrame, cfg: dict) -> pd.Series:
    """Weights equalizing per-asset risk contributions (S5)."""
    tickers = cov.index
    cov_vals = cov.values
    cap = cfg["constraints"]["per_asset_cap"]
    n = len(tickers)

    def objective(w: np.ndarray) -> float:
        port_var = w @ cov_vals @ w
        risk_contrib = w * (cov_vals @ w)
        target = port_var / n
        return float(np.sum((risk_contrib - target) ** 2))

    result = minimize(
        objective,
        np.full(n, 1.0 / n),
        method="SLSQP",
        # strictly positive lower bound: risk contributions need w > 0
        bounds=[(1e-6, cap)] * n,
        constraints=(_sum_to_one(),),
    )
    return _renormalize(result.x, tickers)


def tracking_error_min(
    cov: pd.DataFrame, benchmark_weights: pd.Series, cfg: dict
) -> pd.Series:
    """Weights minimizing tracking-error variance vs a benchmark (S2)."""
    tickers = cov.index
    cov_vals = cov.values
    benchmark = benchmark_weights.reindex(tickers).fillna(0.0).values
    cap = cfg["constraints"]["per_asset_cap"]
    n = len(tickers)

    def tracking_variance(w: np.ndarray) -> float:
        active = w - benchmark
        return float(active @ cov_vals @ active)

    result = minimize(
        tracking_variance,
        benchmark.copy(),
        method="SLSQP",
        bounds=[(0.0, cap)] * n,
        constraints=(_sum_to_one(),),
    )
    return _renormalize(result.x, tickers)


def efficient_frontier(
    mu: pd.Series, cov: pd.DataFrame, cfg: dict, n_points: int = 25
) -> pd.DataFrame:
    """Minimum-variance weights across a grid of target returns spanning
    the achievable range (S2/S3). Returns columns: target_return, ret, vol."""
    tickers = mu.index
    cov_vals = cov.loc[tickers, tickers].values
    mu_vals = mu.values
    cap = cfg["constraints"]["per_asset_cap"]
    n = len(tickers)
    bounds = [(0.0, cap)] * n
    x0 = np.full(n, 1.0 / n)

    rows = []
    for target in np.linspace(mu.min(), mu.max(), n_points):
        constraints = (
            _sum_to_one(),
            {"type": "eq", "fun": lambda w, t=target: w @ mu_vals - t},
        )
        result = minimize(
            lambda w: w @ cov_vals @ w,
            x0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
        )
        if not result.success:
            continue
        w = _renormalize(result.x, tickers).values
        rows.append(
            {
                "target_return": target,
                "ret": float(w @ mu_vals),
                "vol": float(np.sqrt(w @ cov_vals @ w)),
            }
        )
    return pd.DataFrame(rows)


def _corr_to_dist(corr: pd.DataFrame) -> pd.DataFrame:
    return np.sqrt(0.5 * (1.0 - corr))


def _quasi_diagonalize(link: np.ndarray) -> list[int]:
    """Order leaves so the correlation matrix is (quasi-)block-diagonal
    along the dendrogram, by walking linkage merges bottom-up (HRP)."""
    link = link.astype(int)
    num_items = link[-1, 3]
    sort_ix = pd.Series([link[-1, 0], link[-1, 1]])
    while sort_ix.max() >= num_items:
        sort_ix.index = range(0, sort_ix.shape[0] * 2, 2)
        clusters = sort_ix[sort_ix >= num_items]
        i = clusters.index
        j = clusters.values - num_items
        sort_ix[i] = link[j, 0]
        df0 = pd.Series(link[j, 1], index=i + 1)
        sort_ix = pd.concat([sort_ix, df0]).sort_index()
        sort_ix.index = range(sort_ix.shape[0])
    return sort_ix.tolist()


def _cluster_variance(cov: pd.DataFrame, items: list[str]) -> float:
    sub = cov.loc[items, items]
    inv_var = 1.0 / np.diag(sub)
    inv_var /= inv_var.sum()
    return float(inv_var @ sub.values @ inv_var)


def _recursive_bisection(
    cov: pd.DataFrame, ordered_tickers: list[str]
) -> pd.Series:
    weights = pd.Series(1.0, index=ordered_tickers)
    clusters = [ordered_tickers]
    while clusters:
        clusters = [
            c[start:end]
            for c in clusters
            for start, end in ((0, len(c) // 2), (len(c) // 2, len(c)))
            if len(c) > 1
        ]
        for i in range(0, len(clusters), 2):
            left, right = clusters[i], clusters[i + 1]
            var_left = _cluster_variance(cov, left)
            var_right = _cluster_variance(cov, right)
            alpha = 1.0 - var_left / (var_left + var_right)
            weights[left] *= alpha
            weights[right] *= 1.0 - alpha
    return weights


def cap_and_renormalize(weights: pd.Series, cap: float) -> pd.Series:
    """Clip anything above `cap` and redistribute the excess pro rata
    across uncapped assets, iterating to convergence (waterfall cap).

    Vanilla HRP has no box constraints: its recursive bisection can
    concentrate heavily in a low-variance asset (e.g. a T-bill ETF),
    which would violate the same per-asset cap every other book in
    this module respects.

    Known limitation: redistribution is pro rata to each uncapped
    asset's CURRENT weight, so an asset already at exactly zero never
    receives any of the redistributed excess. With few assets and an
    extreme, artificial excess (confirmed while testing a tilt
    wrapper: a large enough tilt clipped one asset to zero, then two
    assets alone couldn't hold the remainder under the cap), this can
    oscillate rather than converge within `len(weights)` iterations.
    Not an issue at the realistic scale this module is actually used
    at (~20 assets, modest tilts), but a real edge case if a caller
    ever passes something more extreme.
    """
    w = weights.astype(float).copy()
    for _ in range(len(w)):
        over = w > cap
        if not over.any():
            break
        excess = (w[over] - cap).sum()
        w[over] = cap
        under = ~over
        room = w[under].sum()
        if room <= 1e-12:
            break
        w[under] += excess * (w[under] / room)
    return w / w.sum()


def hrp(returns: pd.DataFrame, cfg: dict) -> pd.Series:
    """Hierarchical Risk Parity weights (Lopez de Prado, S4): cluster
    assets by correlation distance, order leaves quasi-diagonally,
    allocate top-down by inverse-variance recursive bisection, then
    apply the per-asset cap (S4 dendrogram)."""
    tickers = list(returns.columns)
    cap = cfg["constraints"]["per_asset_cap"]
    if len(tickers) == 1:
        return pd.Series({tickers[0]: 1.0})

    corr = returns.corr()
    cov = returns.cov()
    dist = _corr_to_dist(corr)
    condensed = squareform(dist.values, checks=False)
    link = linkage(condensed, method="single")
    ordered = [tickers[i] for i in _quasi_diagonalize(link)]
    weights = _recursive_bisection(cov, ordered)
    weights = weights.reindex(returns.columns)
    return cap_and_renormalize(weights, cap)


def _bucket_equal_weight(
    class_bucket: pd.Series, bucket: str, total_weight: float
) -> pd.Series:
    members = class_bucket[class_bucket == bucket].index
    if len(members) == 0:
        raise ValueError(
            f"No tickers found for asset-class bucket '{bucket}'"
        )
    return pd.Series(total_weight / len(members), index=members)


def permanent(class_bucket: pd.Series) -> pd.Series:
    """Permanent Portfolio benchmark: 25% each into equity,
    fixed_income, commodity, and cash, split equally within each
    bucket (S4)."""
    parts = [
        _bucket_equal_weight(class_bucket, "equity", 0.25),
        _bucket_equal_weight(class_bucket, "fixed_income", 0.25),
        _bucket_equal_weight(class_bucket, "commodity", 0.25),
        _bucket_equal_weight(class_bucket, "cash", 0.25),
    ]
    weights = pd.concat(parts)
    return weights.reindex(class_bucket.index).fillna(0.0)


def sixty_forty(class_bucket: pd.Series) -> pd.Series:
    """60/40 benchmark: 60% equity, 40% fixed income, split equally
    within each bucket (S1/S5)."""
    parts = [
        _bucket_equal_weight(class_bucket, "equity", 0.60),
        _bucket_equal_weight(class_bucket, "fixed_income", 0.40),
    ]
    weights = pd.concat(parts)
    return weights.reindex(class_bucket.index).fillna(0.0)


def utility_select(
    candidates: dict[str, pd.Series],
    mu: pd.Series,
    cov: pd.DataFrame,
    risk_aversion: float,
) -> tuple[str, pd.Series]:
    """Pick the candidate book with the highest mean-variance utility
    U = E[r] - 0.5*A*sigma^2 (S1/S3)."""
    best_name, best_utility, best_weights = None, -np.inf, None
    for name, weights in candidates.items():
        w = weights.reindex(mu.index).fillna(0.0)
        exp_ret = float(w @ mu)
        vol = portfolio_vol(w, cov)
        u = utility(exp_ret, vol, risk_aversion)
        if u > best_utility:
            best_name, best_utility, best_weights = name, u, w
    return best_name, best_weights


def apply_defensive_tilt(
    weights: pd.Series,
    class_bucket: pd.Series,
    tilt: dict[str, float],
    cfg: dict,
) -> pd.Series:
    """Add an additive risk-off tilt to specific asset-class buckets
    (e.g. + fixed_income, + commodity), funded pro rata out of every
    other bucket, then re-apply the per-asset cap (S7 regime-
    conditional caps)."""
    w = weights.reindex(class_bucket.index).fillna(0.0).astype(float)
    total_tilt = sum(tilt.values())

    for bucket, add in tilt.items():
        members = class_bucket[class_bucket == bucket].index
        if len(members) == 0:
            raise ValueError(f"No tickers found for bucket '{bucket}'")
        current = w[members].sum()
        if current > 1e-12:
            w[members] += add * (w[members] / current)
        else:
            w[members] += add / len(members)

    other_members = class_bucket[~class_bucket.isin(tilt.keys())].index
    other_total = w[other_members].sum()
    if other_total > 1e-12:
        w[other_members] -= total_tilt * (w[other_members] / other_total)

    cap = cfg["constraints"]["per_asset_cap"]
    return cap_and_renormalize(w.clip(lower=0.0), cap)
