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

Stage 9 (not yet implemented): walk_forward, ablation. Those need the
per-fold model refits from later stages and are out of scope here.

Public API (stage 2):
    BacktestResult                          # dataclass: equity, returns,
                                             # weights, turnover, costs,
                                             # metrics
    run(strategy_fn, returns, cfg) -> BacktestResult
    compare(results: dict[str, BacktestResult]) -> pd.DataFrame
"""

from __future__ import annotations

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
    strategy_fn: StrategyFn, returns: pd.DataFrame, cfg: dict
) -> BacktestResult:
    """Run the weekly backtest of `strategy_fn` over `returns`.

    `strategy_fn(as_of, returns_window)` must return target weights
    using only `returns_window` (already truncated to `as_of`).
    """
    tickers = returns.columns
    dates = returns.index
    cost_rate = cfg["rebalance"]["transaction_cost_bps"] / 1e4
    turnover_cap = cfg["rebalance"]["max_weekly_turnover"]
    no_trade_band = cfg["rebalance"]["no_trade_band"]
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
