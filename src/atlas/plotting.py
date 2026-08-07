"""Standard chart library (S11 Step 5, ANALYSIS_V2.md action 5): the
four chart patterns hand-rolled repeatedly across notebooks 09 and
11 (equity curves, drawdowns, weight evolution, marginal-contribution
bar charts), collected here so notebooks call one tested function
instead of re-copying ~15 lines of matplotlib boilerplate each time.

Every function takes plain pandas objects (return series, a weight
DataFrame, a marginal-contribution Series) rather than a `backtest.
BacktestResult`, so it has no dependency on backtest.py and works
identically for a real strategy's returns or a synthetic series (e.g.
notebook 09's risk-matched levered-benchmark line). Every function
returns a `Figure` rather than calling `plt.show()` -- the caller
decides whether to display it (`plt.show()` in a notebook) and/or
save it (`fig.savefig(...)`, e.g. to `reports/figures/`).

Not every chart hand-rolled across notebooks 02-11 is covered here
(e.g. the HMM-posture-colored equity curve in notebook 09 or the
correlation heatmap in notebook 06 are one-off enough not to be worth
generalizing yet) -- this covers the four patterns that repeat
verbatim across the OOS backtest (notebook 09) and the ablation study
(notebook 11).

Public API:
    equity_curves(returns, oos_start, highlight, title, ylabel,
                  figsize) -> Figure
    drawdown_curves(returns, oos_start, highlight, title, ylabel,
                     figsize) -> Figure
    weight_evolution(weights, title, ylabel, figsize) -> Figure
    marginal_contribution_bars(marginal, title, xlabel, figsize) -> Figure
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.figure import Figure

from . import metrics


def equity_curves(
    returns: dict[str, pd.Series],
    oos_start: pd.Timestamp | None = None,
    highlight: str | None = None,
    title: str = "Equity curves",
    ylabel: str = "Growth of $1",
    figsize: tuple[float, float] = (11, 6),
) -> Figure:
    """Growth-of-$1 curves for each named daily-return series in
    `returns`, optionally sliced from `oos_start` onward and rebased
    there. `highlight`, if given, draws that one series with a
    thicker line so it stands out among several benchmarks."""
    fig, ax = plt.subplots(figsize=figsize)
    for name, r in returns.items():
        series = r.loc[oos_start:] if oos_start is not None else r
        curve = (1.0 + series).cumprod()
        linewidth = 2 if name == highlight else 1
        curve.plot(ax=ax, label=name, linewidth=linewidth)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def drawdown_curves(
    returns: dict[str, pd.Series],
    oos_start: pd.Timestamp | None = None,
    highlight: str | None = None,
    title: str = "Drawdowns",
    ylabel: str = "Drawdown",
    figsize: tuple[float, float] = (11, 4),
) -> Figure:
    """Running drawdown for each named daily-return series in
    `returns` (via `metrics.drawdown_series`), same slicing/highlight
    conventions as `equity_curves`."""
    fig, ax = plt.subplots(figsize=figsize)
    for name, r in returns.items():
        series = r.loc[oos_start:] if oos_start is not None else r
        dd = metrics.drawdown_series(series)
        linewidth = 2 if name == highlight else 1
        dd.plot(ax=ax, label=name, linewidth=linewidth)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def weight_evolution(
    weights: pd.DataFrame,
    title: str = "Weight evolution",
    ylabel: str = "Weight",
    figsize: tuple[float, float] = (11, 5),
) -> Figure:
    """Stacked-area chart of a strategy's per-asset weights over
    time."""
    fig, ax = plt.subplots(figsize=figsize)
    weights.plot.area(ax=ax, title=title)
    ax.set_ylabel(ylabel)
    ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0), fontsize=8)
    fig.tight_layout()
    return fig


def marginal_contribution_bars(
    marginal: pd.Series,
    title: str | None = None,
    xlabel: str | None = None,
    figsize: tuple[float, float] = (9, 5),
) -> Figure:
    """Horizontal bar chart of a marginal-contribution Series (e.g.
    notebook 11's ablation results): green for positive, red for
    negative, with a zero reference line -- the same chart used for
    both the module and gate ablations."""
    fig, ax = plt.subplots(figsize=figsize)
    colors = ["seagreen" if v > 0 else "firebrick" for v in marginal]
    ax.barh(marginal.index, marginal.to_numpy(), color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if title is not None:
        ax.set_title(title)
    fig.tight_layout()
    return fig
