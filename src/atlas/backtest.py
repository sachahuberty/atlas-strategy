"""Weekly backtesting engine (L5): the referee (S5, S10, S4).

Stage 2 (this file): `run` and `compare`. Decision at each week's last
available trading day ("Friday close") uses only data through that
day; the resulting trade is executed at the START of the next trading
day (position lagging, S10) and costs are charged on that execution
day. Weights drift with asset returns between rebalances. A no-trade
band skips negligible trades; a turnover cap scales down large ones
(S4) -- except the very first rebalance, which funds the portfolio
from cash and is exempt from the cap (there is no "prior week" trade
to bound).

Stage 6 addition: an optional `phase_flags_fn` for execution timing
(S12). When it flags any asset in a given rebalance, the WHOLE trade
that week is scaled by `technicals.phase_fraction`, not just the
flagged asset's own component -- scaling one leg alone would break the
zero-sum trade invariant (weights must sum to 1: if one asset's entry
is held back, another's exit must be held back correspondingly). The
rest catches up whenever the strategy re-targets it next. Applied
before the no-trade band and turnover cap, so a phased-down trade can
still be skipped/capped like any other. Backward compatible: omitting
it reproduces stage 2-5 behavior exactly.

Stage 9 addition: walk_forward, memoize_strategy, grid_search.
walk_forward slices an existing run's daily returns into folds and
reports per-fold metrics plus a cross-fold stability score (S10: "OOS
performance = average over folds, not one split") -- it reuses the
SAME continuous, chronological `run` rather than re-fitting a wholly
separate model per fold, since the per-signal refit caches (HMM
weekly, autoencoder every `anomaly.refit_frequency_days`) already
implement the "model clock" walk-forward stage 3-8 established; folds
here are a reporting slice, not a different computation. grid_search
is a generic IS-only tuning utility; memoize_strategy caches a
strategy_fn's output by date, making a rebalance-parameter-only grid
search (e.g. turnover cap, no-trade band) cheap by reusing one
expensive weekly signal computation across every candidate.

Stage 11 (not yet implemented): ablation. Needs the per-module on/off
flag wiring from views.py stage 8 already built, but the actual
marginal-Sharpe comparison table is out of scope until that stage.

Public API (stage 2):
    BacktestResult                          # dataclass: equity, returns,
                                             # weights, turnover, costs,
                                             # metrics
    run(strategy_fn, returns, cfg, phase_flags_fn=None) -> BacktestResult
    compare(results: dict[str, BacktestResult]) -> pd.DataFrame

Public API (stage 9):
    WalkForwardResult                       # dataclass: fold_metrics,
                                             # summary
    walk_forward(daily_returns, cfg, fold=None) -> WalkForwardResult
    memoize_strategy(strategy_fn) -> strategy_fn
    grid_search(param_grid, evaluate_fn) -> pd.DataFrame
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Callable

import pandas as pd

from .metrics import (
    ann_return,
    ann_vol,
    calmar,
    hit_rate,
    max_drawdown,
    sharpe,
    sortino,
)

StrategyFn = Callable[[pd.Timestamp, pd.DataFrame], pd.Series]
PhaseFlagsFn = Callable[[pd.Timestamp, pd.DataFrame], set]


@dataclass
class BacktestResult:
    """Output of a single `run`: daily accounting plus summary metrics."""

    equity_curve: pd.Series
    daily_returns: pd.Series
    weights: pd.DataFrame
    turnover: pd.Series
    costs: pd.Series
    metrics: dict = field(default_factory=dict)


def _week_end_dates(index: pd.DatetimeIndex) -> set[pd.Timestamp]:
    """Last available trading day of each calendar week ("Friday
    close", or the prior trading day if Friday is a holiday)."""
    iso = index.isocalendar()
    grouped = pd.Series(index, index=index).groupby(
        [iso["year"], iso["week"]]
    )
    return set(grouped.max())


def run(
    strategy_fn: StrategyFn,
    returns: pd.DataFrame,
    cfg: dict,
    phase_flags_fn: PhaseFlagsFn | None = None,
) -> BacktestResult:
    """Run the weekly backtest of `strategy_fn` over `returns`.

    `strategy_fn(as_of, returns_window)` must return target weights
    using only `returns_window` (already truncated to `as_of`).

    `phase_flags_fn(as_of, returns_window)`, if given, returns the set
    of tickers that should trigger phased execution this rebalance
    (S12 execution timing): if any is present, the WHOLE trade is
    scaled by `technicals.phase_fraction` before the no-trade band and
    turnover cap apply.
    """
    tickers = returns.columns
    dates = returns.index
    cost_rate = cfg["rebalance"]["transaction_cost_bps"] / 1e4
    turnover_cap = cfg["rebalance"]["max_weekly_turnover"]
    no_trade_band = cfg["rebalance"]["no_trade_band"]
    phase_fraction = (
        cfg["technicals"]["phase_fraction"]
        if phase_flags_fn is not None
        else None
    )
    rebalance_dates = _week_end_dates(dates)

    current_weights = pd.Series(0.0, index=tickers)
    pending_trade = None
    pending_turnover = 0.0
    is_first_rebalance = True

    daily_port_returns = pd.Series(0.0, index=dates)
    weight_rows = {}
    turnover_rows = {}
    cost_rows = {}

    for date in dates:
        day_return = returns.loc[date]

        cost_today = 0.0
        if pending_trade is not None:
            current_weights = current_weights + pending_trade
            cost_today = cost_rate * pending_turnover
            pending_trade = None

        port_return = float((current_weights * day_return).sum()) - cost_today
        daily_port_returns.loc[date] = port_return
        cost_rows[date] = cost_today

        gross = current_weights * (1.0 + day_return)
        total = gross.sum()
        current_weights = gross / total if total > 1e-12 else gross
        weight_rows[date] = current_weights.copy()

        if date in rebalance_dates:
            window = returns.loc[:date]
            target = strategy_fn(date, window).reindex(tickers).fillna(0.0)
            trade = target - current_weights

            if phase_flags_fn is not None and not is_first_rebalance:
                # Funding from cash always goes straight to target;
                # phasing only applies to steady-state rebalances.
                # Scaling only the flagged asset's own trade component
                # would break the zero-sum trade invariant (weights
                # must sum to 1): if B's entry is held back, A's exit
                # must be held back correspondingly. So when any
                # flagged asset is involved, the WHOLE trade this
                # rebalance is scaled by phase_fraction -- the rest
                # catches up next time the strategy re-targets it.
                flagged = phase_flags_fn(date, window)
                if flagged and tickers.isin(flagged).any():
                    trade = trade * phase_fraction

            one_way = trade.abs().sum() / 2.0

            if is_first_rebalance:
                # Funding the portfolio from cash is not "turnover" in
                # the steady-state sense; only later rebalances are
                # capped.
                pending_trade = trade
                pending_turnover = one_way
                is_first_rebalance = False
            elif one_way < no_trade_band:
                pending_trade = None
                pending_turnover = 0.0
            else:
                if one_way > turnover_cap:
                    trade = trade * (turnover_cap / one_way)
                    one_way = turnover_cap
                pending_trade = trade
                pending_turnover = one_way

            turnover_rows[date] = pending_turnover

    equity_curve = (1.0 + daily_port_returns).cumprod()
    weights_df = pd.DataFrame(weight_rows).T
    turnover_series = pd.Series(turnover_rows, dtype=float).sort_index()
    cost_series = pd.Series(cost_rows, dtype=float).sort_index()

    result_metrics = {
        "ann_return": ann_return(daily_port_returns),
        "ann_vol": ann_vol(daily_port_returns),
        "sharpe": sharpe(daily_port_returns),
        "sortino": sortino(daily_port_returns),
        "calmar": calmar(daily_port_returns),
        "max_drawdown": max_drawdown(daily_port_returns),
        "hit_rate": hit_rate(daily_port_returns),
        "avg_weekly_turnover": (
            turnover_series.mean() if len(turnover_series) else 0.0
        ),
        "total_cost_drag": cost_series.sum(),
    }

    return BacktestResult(
        equity_curve=equity_curve,
        daily_returns=daily_port_returns,
        weights=weights_df,
        turnover=turnover_series,
        costs=cost_series,
        metrics=result_metrics,
    )


def compare(results: dict[str, BacktestResult]) -> pd.DataFrame:
    """Strategy-vs-benchmark comparison table, one row per result."""
    return pd.DataFrame(
        {name: result.metrics for name, result in results.items()}
    ).T


@dataclass
class WalkForwardResult:
    """Per-fold metrics for an already-computed daily-returns series.

    `stability_score` in `summary` is `mean(fold sharpe) - penalty *
    std(fold sharpe)` (S10 grid-search criterion): it rewards a
    strategy that holds up across folds over one with a higher average
    driven by a single lucky fold.
    """

    fold_metrics: pd.DataFrame
    summary: dict


_FOLD_PERIOD_CODE = {"quarterly": "Q", "annual": "Y"}


def walk_forward(
    daily_returns: pd.Series, cfg: dict, fold: str | None = None
) -> WalkForwardResult:
    """Slice `daily_returns` into folds and report metrics per fold.

    `fold` is `"quarterly"` or `"annual"`; defaults to
    `cfg["backtest"]["walk_forward_fold"]`. This slices the output of
    one continuous, chronological `run` into reporting periods -- it
    does not re-fit any model per fold. The per-signal refit caches
    already in `strategy.py` (HMM refit weekly, autoencoder every
    `anomaly.refit_frequency_days`) are the actual "model clock"; folds
    here only change how the resulting daily returns are aggregated
    for reporting.
    """
    code = _FOLD_PERIOD_CODE[fold or cfg["backtest"]["walk_forward_fold"]]
    periods = daily_returns.index.to_period(code)

    rows = {}
    for period, fold_returns in daily_returns.groupby(periods):
        rows[str(period)] = {
            "ann_return": ann_return(fold_returns),
            "ann_vol": ann_vol(fold_returns),
            "sharpe": sharpe(fold_returns),
            "sortino": sortino(fold_returns),
            "max_drawdown": max_drawdown(fold_returns),
            "hit_rate": hit_rate(fold_returns),
            "n_days": len(fold_returns),
        }
    fold_metrics = pd.DataFrame(rows).T

    penalty = cfg["grid_search"]["stability_penalty"]
    sharpe_mean = fold_metrics["sharpe"].mean()
    sharpe_std = (
        fold_metrics["sharpe"].std(ddof=1) if len(fold_metrics) > 1 else 0.0
    )
    summary = {
        "sharpe_mean": sharpe_mean,
        "sharpe_std": sharpe_std,
        "stability_score": sharpe_mean - penalty * sharpe_std,
    }
    return WalkForwardResult(fold_metrics=fold_metrics, summary=summary)


def memoize_strategy(strategy_fn: StrategyFn) -> StrategyFn:
    """Cache `strategy_fn`'s target weights by `as_of` date.

    Target weights are a deterministic function of `(as_of, window)`,
    and `window = returns.loc[:as_of]` is itself determined by `as_of`
    for a fixed `returns` DataFrame, so caching on `as_of` alone is
    safe. This lets a grid search over rebalance-only parameters
    (turnover cap, no-trade band) reuse one expensive weekly signal
    computation across every candidate combination instead of
    recomputing it from scratch each time.
    """
    cache: dict[pd.Timestamp, pd.Series] = {}

    def memoized(as_of: pd.Timestamp, window: pd.DataFrame) -> pd.Series:
        if as_of not in cache:
            cache[as_of] = strategy_fn(as_of, window)
        return cache[as_of].copy()

    return memoized


def grid_search(
    param_grid: dict[str, list],
    evaluate_fn: Callable[[dict], dict],
) -> pd.DataFrame:
    """Evaluate `evaluate_fn` over the Cartesian product of `param_grid`.

    `evaluate_fn(params)` must return a metrics dict for that
    combination. In-sample-only tuning (S10): select on
    `stability_score` from `walk_forward`, not on peak single-fold
    performance.
    """
    keys = list(param_grid.keys())
    rows = []
    for combo in itertools.product(*param_grid.values()):
        params = dict(zip(keys, combo))
        rows.append({**params, **evaluate_fn(params)})
    return pd.DataFrame(rows)
