"""Accounting and no-lookahead tests for backtest.py (stage 2)."""

# TODO stage 2:
# - costs = bps * turnover exactly
# - turnover cap never exceeded
# - LOOKAHEAD CANARY: shifting prices by one day changes decisions accordingly
# - equity curve of zero-cost buy-and-hold matches cumulative returns
