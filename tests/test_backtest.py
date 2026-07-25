"""Accounting and no-lookahead tests for backtest.py (stage 2)."""

import numpy as np
import pandas as pd
import pytest

from atlas import backtest


def test_costs_equal_bps_times_turnover():
    dates = pd.bdate_range("2022-01-03", periods=20)
    tickers = ["A", "B"]
    returns = pd.DataFrame(0.0, index=dates, columns=tickers)
    target = pd.Series({"A": 0.6, "B": 0.4})

    def constant_strategy(as_of, window):
        return target

    bps = 10
    cfg = {
        "rebalance": {
            "transaction_cost_bps": bps,
            "max_weekly_turnover": 1.0,
            "no_trade_band": 0.0,
        }
    }
    result = backtest.run(constant_strategy, returns, cfg)

    # Zero returns -> weights never drift, so only the initial funding
    # trade (turnover 0.5) ever executes.
    assert result.turnover.iloc[0] == pytest.approx(0.5)
    assert (result.turnover.iloc[1:] == 0).all()

    expected_cost = (bps / 1e4) * 0.5
    assert result.costs.sum() == pytest.approx(expected_cost)
    assert result.costs.max() == pytest.approx(expected_cost)


def test_turnover_cap_never_exceeded_after_funding():
    dates = pd.bdate_range("2022-01-03", periods=60)
    tickers = ["A", "B"]
    rng = np.random.default_rng(11)
    returns = pd.DataFrame(
        rng.normal(0.0, 0.01, size=(60, 2)), index=dates, columns=tickers
    )

    def flip_flop_strategy(as_of, window):
        # Alternate hard between two extreme books each week, trying
        # to force turnover above the cap.
        week = as_of.isocalendar().week
        if week % 2 == 0:
            return pd.Series({"A": 1.0, "B": 0.0})
        return pd.Series({"A": 0.0, "B": 1.0})

    cap = 0.02
    cfg = {
        "rebalance": {
            "transaction_cost_bps": 0,
            "max_weekly_turnover": cap,
            "no_trade_band": 0.0,
        }
    }
    result = backtest.run(flip_flop_strategy, returns, cfg)

    # The first rebalance funds from cash and is exempt from the cap;
    # every later rebalance must respect it.
    assert (result.turnover.iloc[1:] <= cap + 1e-9).all()


def test_lookahead_canary_shifted_returns_change_decisions():
    dates = pd.bdate_range("2022-01-03", periods=40)
    tickers = ["A", "B"]
    rng = np.random.default_rng(7)
    returns = pd.DataFrame(
        rng.normal(0.0, 0.01, size=(40, 2)), index=dates, columns=tickers
    )

    def momentum_strategy(as_of, window):
        # Reacts to the most recent available day only -- a decision
        # that must change if that day's data is shifted away.
        last_row = window.iloc[-1]
        if last_row["A"] > last_row["B"]:
            return pd.Series({"A": 1.0, "B": 0.0})
        return pd.Series({"A": 0.0, "B": 1.0})

    cfg = {
        "rebalance": {
            "transaction_cost_bps": 0,
            "max_weekly_turnover": 1.0,
            "no_trade_band": 0.0,
        }
    }

    result_original = backtest.run(momentum_strategy, returns, cfg)
    shifted_returns = returns.shift(1).fillna(0.0)
    result_shifted = backtest.run(momentum_strategy, shifted_returns, cfg)

    assert not result_original.weights.equals(result_shifted.weights)


def test_phase_flags_fn_spreads_entry_across_multiple_rebalances():
    # Zero returns -> no drift, so weight changes come only from the
    # phased-trade sequence itself, making the convergence math exact:
    # each week the flagged asset closes half the remaining gap to
    # target (B_n = 1 - 0.5**n after n post-switch rebalances).
    dates = pd.bdate_range("2022-01-03", periods=60)
    tickers = ["A", "B"]
    returns = pd.DataFrame(0.0, index=dates, columns=tickers)
    switch_week = dates[20].isocalendar().week

    def switching_strategy(as_of, window):
        if as_of.isocalendar().week < switch_week:
            return pd.Series({"A": 1.0, "B": 0.0})
        return pd.Series({"A": 0.0, "B": 1.0})

    def phase_flags_fn(as_of, window):
        return {"B"}

    cfg = {
        "rebalance": {
            "transaction_cost_bps": 0,
            "max_weekly_turnover": 1.0,
            "no_trade_band": 0.0,
        },
        "technicals": {"phase_fraction": 0.5},
    }
    result = backtest.run(
        switching_strategy, returns, cfg, phase_flags_fn=phase_flags_fn
    )

    rebalance_dates = sorted(result.turnover.index)
    post_switch = [
        d for d in rebalance_dates if d.isocalendar().week >= switch_week
    ][:3]
    # Execution lags one trading day behind each rebalance decision.
    execution_days = [
        returns.index[returns.index.get_loc(d) + 1] for d in post_switch
    ]

    expected = [0.5, 0.75, 0.875]
    for day, exp in zip(execution_days, expected):
        assert result.weights.loc[day, "B"] == pytest.approx(exp)


def test_phase_flags_fn_is_backward_compatible_when_omitted():
    dates = pd.bdate_range("2022-01-03", periods=20)
    tickers = ["A", "B"]
    returns = pd.DataFrame(0.0, index=dates, columns=tickers)
    target = pd.Series({"A": 0.6, "B": 0.4})

    def constant_strategy(as_of, window):
        return target

    cfg = {
        "rebalance": {
            "transaction_cost_bps": 0,
            "max_weekly_turnover": 1.0,
            "no_trade_band": 0.0,
        }
    }
    with_none = backtest.run(
        constant_strategy, returns, cfg, phase_flags_fn=None
    )
    without_arg = backtest.run(constant_strategy, returns, cfg)
    pd.testing.assert_frame_equal(with_none.weights, without_arg.weights)


def test_buy_and_hold_matches_independently_computed_growth():
    dates = pd.bdate_range("2022-01-03", periods=40)
    tickers = ["A", "B"]
    rng = np.random.default_rng(3)
    returns = pd.DataFrame(
        rng.normal(0.0003, 0.01, size=(40, 2)), index=dates, columns=tickers
    )
    initial_weights = pd.Series({"A": 0.6, "B": 0.4})

    def constant_strategy(as_of, window):
        return initial_weights

    cfg = {
        "rebalance": {
            "transaction_cost_bps": 0,
            "max_weekly_turnover": 1.0,
            # A band of 1.0 makes every rebalance after the first a
            # no-op, since drift alone can never reach 100% turnover.
            "no_trade_band": 1.0,
        }
    }
    result = backtest.run(constant_strategy, returns, cfg)

    assert (result.turnover.iloc[1:] == 0).all()

    first_funded_day = result.turnover.index[0]
    execution_pos = returns.index.get_loc(first_funded_day) + 1
    execution_day = returns.index[execution_pos]

    # No position is held before funding: equity curve stays flat.
    assert (result.equity_curve.loc[:first_funded_day] == 1.0).all()

    # Independent path: per-asset growth of $1, split by the initial
    # weights and never rebalanced (true buy-and-hold), should match
    # the backtester's drift-based equity curve exactly.
    asset_growth = (1.0 + returns.loc[execution_day:]).cumprod()
    manual_bh = asset_growth.mul(initial_weights, axis=1).sum(axis=1)

    pd.testing.assert_series_equal(
        result.equity_curve.loc[execution_day:],
        manual_bh,
        check_names=False,
    )
