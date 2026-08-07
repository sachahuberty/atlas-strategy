"""Tests for plotting.py: the shared chart library (S11 Step 5,
ANALYSIS_V2.md action 5)."""

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest
from matplotlib.figure import Figure

from atlas import plotting


def _returns(n=100, seed=1) -> pd.Series:
    dates = pd.bdate_range("2022-01-01", periods=n)
    rng = np.random.default_rng(seed)
    return pd.Series(rng.normal(0.0004, 0.01, n), index=dates)


def test_equity_curves_returns_a_figure_with_one_line_per_series():
    returns = {"A": _returns(seed=1), "B": _returns(seed=2)}
    fig = plotting.equity_curves(returns)

    assert isinstance(fig, Figure)
    ax = fig.axes[0]
    assert len(ax.get_lines()) == 2
    labels = {line.get_label() for line in ax.get_lines()}
    assert labels == {"A", "B"}


def test_equity_curves_respects_oos_start_slicing():
    dates = pd.bdate_range("2022-01-01", periods=100)
    r = pd.Series(0.001, index=dates)
    oos_start = dates[50]

    fig = plotting.equity_curves({"A": r}, oos_start=oos_start)
    ax = fig.axes[0]
    line = ax.get_lines()[0]
    x_data = line.get_xdata()

    assert len(x_data) == 50  # only the last 50 dates plotted
    # Rebased at oos_start: first plotted value is (1 + r)^1, not a
    # multi-year compounded growth from the very start of the series.
    y_data = line.get_ydata()
    assert y_data[0] == pytest.approx(1.001, abs=1e-9)


def test_equity_curves_highlight_gets_a_thicker_line():
    returns = {"A": _returns(seed=1), "B": _returns(seed=2)}
    fig = plotting.equity_curves(returns, highlight="A")
    ax = fig.axes[0]
    widths = {
        line.get_label(): line.get_linewidth() for line in ax.get_lines()
    }

    assert widths["A"] > widths["B"]


def test_drawdown_curves_returns_a_figure_with_non_positive_values():
    returns = {"A": _returns(seed=3)}
    fig = plotting.drawdown_curves(returns)
    ax = fig.axes[0]

    assert isinstance(fig, Figure)
    line = ax.get_lines()[0]
    assert (line.get_ydata() <= 1e-9).all()


def test_weight_evolution_returns_a_stacked_area_figure():
    dates = pd.bdate_range("2022-01-01", periods=20)
    weights = pd.DataFrame(
        {"SPY": np.full(20, 0.6), "AGG": np.full(20, 0.4)}, index=dates
    )
    fig = plotting.weight_evolution(weights)

    assert isinstance(fig, Figure)
    ax = fig.axes[0]
    # Area charts still register one Line2D-backed collection per
    # column boundary; a simpler, robust check is that the axes has
    # some drawn content and a legend with both asset names.
    legend_labels = {t.get_text() for t in ax.get_legend().get_texts()}
    assert legend_labels == {"SPY", "AGG"}


def test_marginal_contribution_bars_colors_by_sign():
    marginal = pd.Series({"V1": 0.08, "V3": -0.26, "V2": -0.01})
    fig = plotting.marginal_contribution_bars(marginal, title="t", xlabel="x")

    assert isinstance(fig, Figure)
    ax = fig.axes[0]
    assert ax.get_title() == "t"
    assert ax.get_xlabel() == "x"
    bars = ax.patches
    assert len(bars) == 3
    colors = [bar.get_facecolor() for bar in bars]
    # seagreen and firebrick are distinct RGBA tuples; the positive
    # entry (V1) must differ in color from the two negative entries.
    positive_idx = list(marginal.index).index("V1")
    negative_idx = list(marginal.index).index("V3")
    assert colors[positive_idx] != colors[negative_idx]
