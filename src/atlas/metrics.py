"""Metrics engine: single source of truth for every stat (S1, S3, S5, S10).

Stage 1. All functions operate on daily simple-return series/frames
unless noted.

Public API:
    ann_return, ann_vol, sharpe, sortino, calmar, cagr
    max_drawdown, drawdown_series, hit_rate
    portfolio_return(weights, returns), portfolio_vol(weights, cov)
    turnover(weights_t, weights_t_minus_1)
    utility(exp_ret, vol, risk_aversion, rf=0.0)  # U = (E[r]-rf) -
                                                   # 0.5*A*sigma^2 (S1/S3)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 252
_ZERO_VOL_TOL = 1e-12


def ann_return(
    returns: pd.Series, periods_per_year: int = TRADING_DAYS_PER_YEAR
) -> float:
    """Geometric annualized return from a daily simple-return series."""
    growth = (1.0 + returns).prod()
    n_periods = returns.shape[0]
    if n_periods == 0:
        return 0.0
    return growth ** (periods_per_year / n_periods) - 1.0


def ann_vol(
    returns: pd.Series, periods_per_year: int = TRADING_DAYS_PER_YEAR
) -> float:
    """Annualized volatility from a daily simple-return series."""
    return returns.std(ddof=1) * np.sqrt(periods_per_year)


def sharpe(
    returns: pd.Series,
    rf: float | pd.Series = 0.0,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> float:
    """Annualized Sharpe ratio. `rf` is an annual risk-free rate: a
    scalar (the stage 1-10 default, `rf=0.0`, giving a total-return
    Sharpe), or a `pd.Series` of annual rates already aligned to
    `returns.index` (e.g. a daily T-bill yield, for an excess-return
    Sharpe) -- pandas broadcasts either case identically below."""
    excess = returns - rf / periods_per_year
    vol = excess.std(ddof=1)
    # Floating-point noise can leave std() slightly above zero even for a
    # perfectly constant series, so guard with a tolerance, not `== 0`.
    if vol < _ZERO_VOL_TOL:
        return 0.0
    return (excess.mean() / vol) * np.sqrt(periods_per_year)


def sortino(
    returns: pd.Series,
    rf: float = 0.0,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> float:
    """Annualized Sortino ratio using downside deviation below rf."""
    target = rf / periods_per_year
    excess = returns - target
    downside = excess[excess < 0]
    if len(downside) == 0:
        return 0.0
    downside_dev = np.sqrt((downside ** 2).mean())
    if downside_dev < _ZERO_VOL_TOL:
        return 0.0
    return (excess.mean() / downside_dev) * np.sqrt(periods_per_year)


def drawdown_series(returns: pd.Series) -> pd.Series:
    """Running drawdown (negative fraction) from a daily return series."""
    cum = (1.0 + returns).cumprod()
    running_max = cum.cummax()
    return cum / running_max - 1.0


def max_drawdown(returns: pd.Series) -> float:
    """Maximum drawdown (a negative number, e.g. -0.35 for -35%)."""
    dd = drawdown_series(returns)
    if len(dd) == 0:
        return 0.0
    return dd.min()


def calmar(
    returns: pd.Series, periods_per_year: int = TRADING_DAYS_PER_YEAR
) -> float:
    """Calmar ratio: annualized return / |max drawdown|."""
    mdd = max_drawdown(returns)
    if mdd == 0:
        return 0.0
    return ann_return(returns, periods_per_year) / abs(mdd)


def cagr(
    returns: pd.Series, periods_per_year: int = TRADING_DAYS_PER_YEAR
) -> float:
    """Compound annual growth rate. Same as ann_return for simple returns."""
    return ann_return(returns, periods_per_year)


def hit_rate(returns: pd.Series) -> float:
    """Fraction of periods with a strictly positive return."""
    if len(returns) == 0:
        return 0.0
    return (returns > 0).mean()


def portfolio_return(
    weights: pd.Series, returns: pd.DataFrame
) -> pd.Series:
    """Period-by-period return given fixed weights and asset returns."""
    weights = weights.reindex(returns.columns)
    return returns.mul(weights, axis=1).sum(axis=1)


def portfolio_vol(weights: pd.Series, cov: pd.DataFrame) -> float:
    """Portfolio volatility: sqrt(w' Cov w)."""
    weights = weights.reindex(cov.columns).fillna(0.0)
    w = weights.to_numpy()
    return float(np.sqrt(w @ cov.to_numpy() @ w))


def turnover(weights_t: pd.Series, weights_t_minus_1: pd.Series) -> float:
    """One-way turnover: half the L1 distance between weight vectors."""
    idx = weights_t.index.union(weights_t_minus_1.index)
    w_t = weights_t.reindex(idx).fillna(0.0)
    w_prev = weights_t_minus_1.reindex(idx).fillna(0.0)
    return 0.5 * (w_t - w_prev).abs().sum()


def utility(
    exp_ret: float, vol: float, risk_aversion: float, rf: float = 0.0
) -> float:
    """Mean-variance utility on excess return: U = (E[r] - rf) -
    0.5 * A * sigma^2 (S1/S3). `rf` defaults to 0 (total-return
    utility) for backward compatibility; pass the prevailing
    risk-free rate to compare candidates on their excess return."""
    return (exp_ret - rf) - 0.5 * risk_aversion * vol ** 2
