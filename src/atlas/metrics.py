"""Metrics engine: single source of truth for every stat (S1, S3, S5, S10).

Stage 1. All functions operate on daily simple-return series/frames
unless noted.

Reselection task (external review) Phase 4 addition: three inference
tools for the single, pre-registered OOS look -- sharpe_standard_error
(the sqrt((1+SR^2/2)/years) asymptotic formula, not the simpler
0.5/sqrt(years) convention used elsewhere in this project's README),
newey_west_mean_test (a HAC-robust paired significance test on a
return-difference series), and deflated_sharpe_ratio (Bailey & Lopez
de Prado 2014, correcting a reported Sharpe for how many
configurations were compared before it was selected).

Public API:
    ann_return, ann_vol, sharpe, sortino, calmar, cagr
    max_drawdown, drawdown_series, hit_rate
    portfolio_return(weights, returns), portfolio_vol(weights, cov)
    turnover(weights_t, weights_t_minus_1)
    utility(exp_ret, vol, risk_aversion, rf=0.0)  # U = (E[r]-rf) -
                                                   # 0.5*A*sigma^2 (S1/S3)
    sharpe_standard_error(sharpe_ratio, n_years) -> float
    newey_west_mean_test(series, maxlags=None) -> dict
    deflated_sharpe_ratio(sharpe_ratio, n_trials, n_obs, skew=0.0,
        kurtosis=3.0, sharpe_std_across_trials=None) -> float
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import norm

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


def sharpe_standard_error(sharpe_ratio: float, n_years: float) -> float:
    """Asymptotic standard error of an ANNUALIZED Sharpe ratio
    estimated from `n_years` years of data:
    SE = sqrt((1 + SR^2 / 2) / n_years).

    Reselection task (external review) Phase 4: deliberately NOT the
    simpler `0.5 / sqrt(n_years)` convention this project's README
    uses elsewhere for a quick approximate SE -- that convention
    drops the SR-dependent term, understating the true SE whenever
    |SR| is not small. This is the textbook asymptotic formula
    (Lo 2002 / the standard large-sample approximation for the Sharpe
    ratio's sampling distribution under i.i.d., normally distributed
    returns), requested explicitly for this task's statistics.
    """
    return float(np.sqrt((1.0 + sharpe_ratio ** 2 / 2.0) / n_years))


def newey_west_mean_test(
    series: pd.Series, maxlags: int | None = None
) -> dict:
    """HAC (Newey-West) robust test of H0: mean(series) == 0, via an
    intercept-only OLS regression with a heteroskedasticity- and
    autocorrelation-consistent covariance estimator (statsmodels'
    `cov_type="HAC"`).

    Reselection task (external review) Phase 4: the paired test of
    weekly return DIFFERENCES between two strategies. Plain i.i.d.
    standard errors understate the true uncertainty here because
    overlapping signal lookbacks and shared market exposure make
    consecutive weekly differences autocorrelated; HAC corrects for
    that without assuming a specific AR structure.

    `maxlags` defaults to the common `floor(4 * (n/100)^(2/9))` rule
    of thumb (statsmodels/Newey-West's own default heuristic) when
    not given explicitly.

    Returns a dict with `mean`, `se`, `tstat`, `pvalue`, `n`.
    """
    clean = series.dropna()
    n = len(clean)
    if maxlags is None:
        maxlags = max(1, int(np.floor(4 * (n / 100.0) ** (2.0 / 9.0))))
    x = np.ones((n, 1))
    model = sm.OLS(clean.to_numpy(), x).fit(
        cov_type="HAC", cov_kwds={"maxlags": maxlags}
    )
    return {
        "mean": float(model.params[0]),
        "se": float(model.bse[0]),
        "tstat": float(model.tvalues[0]),
        "pvalue": float(model.pvalues[0]),
        "n": n,
        "maxlags": maxlags,
    }


_EULER_MASCHERONI = 0.5772156649015329


def deflated_sharpe_ratio(
    sharpe_ratio: float,
    n_trials: int,
    n_obs: int,
    skew: float = 0.0,
    kurtosis: float = 3.0,
    sharpe_std_across_trials: float | None = None,
) -> float:
    """Deflated Sharpe Ratio (Bailey & Lopez de Prado, 2014, "The
    Deflated Sharpe Ratio: Correcting for Selection Bias, Backtest
    Overfitting, and Non-Normality"): the probability that the true
    Sharpe ratio is positive, after correcting for (a) how many
    independent configurations (`n_trials`) were compared before this
    one was selected/reported, and (b) the non-normality of the
    underlying returns (`skew`, `kurtosis`, of the SAME frequency
    `sharpe_ratio` was estimated from).

    Mechanics: `n_trials` skill-less strategies' estimated Sharpe
    ratios are themselves a random variable with some cross-trial
    standard deviation (`sharpe_std_across_trials` -- pass the
    EMPIRICAL std of Sharpe ratios actually measured across the
    trials, e.g. this task's Phase 2 144-cell grid, rather than an
    assumed value, per Bailey & Lopez de Prado's own recommended
    practice when real trial data exists). Their expected maximum
    (SR_0, what the best of N skill-less trials would show by chance
    alone) is:

        SR_0 = sharpe_std_across_trials * [
            (1 - gamma) * Phi^-1(1 - 1/n_trials)
            + gamma * Phi^-1(1 - 1/(n_trials * e))
        ]

    (gamma = Euler-Mascheroni constant, Phi^-1 = inverse normal CDF).
    The deflated ratio is then the ordinary Probabilistic Sharpe Ratio
    of `sharpe_ratio` against this, SR_0-adjusted, benchmark:

        DSR = Phi( (sharpe_ratio - SR_0) / sigma(SR_hat) )

    where sigma(SR_hat) is `sharpe_ratio`'s own standard error,
    accounting for non-normality:

        sigma(SR_hat) = sqrt(
            (1 - skew*SR + (kurtosis - 1)/4 * SR^2) / (n_obs - 1)
        )

    `n_obs` is the number of return OBSERVATIONS (at the frequency
    `sharpe_ratio` was computed from, e.g. daily) used to estimate
    `sharpe_ratio` -- not years, and not the number of trials.

    If `sharpe_std_across_trials` is omitted, SR_0 is taken as 0.0
    (no correction for selection bias, only for non-normality) -- a
    strictly more generous DSR than the fully-corrected version, so a
    caller omitting it should treat the result as an upper bound, not
    the headline number.
    """
    if sharpe_std_across_trials is None or n_trials <= 1:
        sr_0 = 0.0
    else:
        z_a = norm.ppf(1.0 - 1.0 / n_trials)
        z_b = norm.ppf(1.0 - 1.0 / (n_trials * np.e))
        sr_0 = sharpe_std_across_trials * (
            (1 - _EULER_MASCHERONI) * z_a + _EULER_MASCHERONI * z_b
        )

    if n_obs <= 1:
        # Degenerate: no sample to estimate sigma(SR_hat) from at all.
        return 1.0 if sharpe_ratio > sr_0 else 0.0

    sr_variance_term = (
        1.0
        - skew * sharpe_ratio
        + ((kurtosis - 1.0) / 4.0) * sharpe_ratio ** 2
    )
    sigma_sr_hat = np.sqrt(max(sr_variance_term, 0.0) / (n_obs - 1))
    if sigma_sr_hat < 1e-12:
        return 1.0 if sharpe_ratio > sr_0 else 0.0
    return float(norm.cdf((sharpe_ratio - sr_0) / sigma_sr_hat))
