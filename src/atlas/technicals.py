"""Technical levels & options positioning (L2 view V3 + execution timing, S12).

Stage 6.
Price part (backtestable): pivot highs/lows -> K-Means support/resistance zones ->
level-proximity signal + event-study sanity check.
Options part (LIVE ONLY - yfinance has no chain history): OI notional, call/put
walls, gamma proxy -> positioning risk score.

Planned public API:
    pivots(ohlc, order) -> pd.DataFrame
    sr_zones(pivot_prices, k, width_bps) -> list[Zone]
    level_proximity(price, zones) -> float
    event_study(prices, zones) -> pd.DataFrame
    option_walls(chain) -> WallResult                # live only
    gamma_proxy(chain, spot) -> float                # live only
"""
