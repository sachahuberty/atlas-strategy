"""Weekly backtesting engine (L5): the referee (S5, S10, S4).

Stage 2.
Rules: decision Friday close -> execution next trading day (position lagging);
costs = bps * turnover; max weekly turnover cap + no-trade band; weights drift
intra-week; IS/OOS split; quarterly walk-forward with per-fold model refits.

Planned public API:
    run(strategy_fn, returns, cfg) -> BacktestResult   # dataclass: equity, weights,
                                                       # turnover, costs, metrics
    walk_forward(strategy_factory, data, cfg) -> WFResult
    compare(results: dict) -> pd.DataFrame             # strategy vs benchmark table
    ablation(cfg, modules) -> pd.DataFrame             # marginal OOS Sharpe per module
"""
