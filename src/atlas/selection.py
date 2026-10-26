"""IS-only fold/selection primitives (reselection task, external review).

Extracted verbatim (behavior-preserving) from where they already lived:
the annual-fold partitioning and the `mean - penalty * std` stability
score both appear independently in notebook 09's meanreversion/momentum
grid-search cells (`fold_periods = is_prices.index.to_period("Y")
.unique()`, `mean_hr - penalty * std_hr`, `mean_spread - penalty *
std_spread`) and a third time inside `backtest.walk_forward`'s own
summary (`sharpe_mean - penalty * sharpe_std`). This module gives that
one formula a single, tested home so every IS-only grid search --
existing or new -- reuses the exact same fold scheme and stability
definition, per CLAUDE.md's "don't duplicate logic" rule and the
reselection task's "reuse exactly, do not invent a new fold scheme."

`select_candidate_or_keep_current` implements the pre-registered tie
rule (docs/RESELECTION_PREREG.md): a challenger replaces the current
config only if it beats it on stability score AND wins a strict
majority of the folds compared; ties -- on either criterion -- keep the
current (simpler) config.

Public API:
    annual_fold_periods(index) -> pd.PeriodIndex
    stability_score(fold_values, penalty) -> float
    SelectionDecision                       # dataclass
    select_candidate_or_keep_current(current_fold_values,
        candidate_fold_values, penalty) -> SelectionDecision
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


def annual_fold_periods(index: pd.DatetimeIndex) -> pd.PeriodIndex:
    """The exact fold scheme notebook 09's IS-only grid searches use:
    one fold per calendar year present in `index`
    (`index.to_period("Y").unique()`), matching
    `grid_search.is_validation_fold: annual` in config.yaml. Not a new
    fold definition -- this is the pre-existing one, just given a
    tested, reusable home.
    """
    return index.to_period("Y").unique()


def stability_score(fold_values: pd.Series, penalty: float) -> float:
    """`mean(fold_values) - penalty * std(fold_values, ddof=1)`: the
    same formula notebook 09's `evaluate_meanreversion`
    (`mean_hr - penalty * std_hr`), `evaluate_momentum`
    (`mean_spread - penalty * std_spread`), and
    `backtest.walk_forward`'s summary (`sharpe_mean - penalty *
    sharpe_std`) each already compute independently. NaN values are
    dropped before aggregating (a fold with no qualifying events
    contributes nothing, not a NaN-poisoned score); `std` is 0.0 with
    fewer than two valid folds, matching every one of those three
    existing call sites' own edge-case handling.
    """
    valid = pd.Series(fold_values).dropna()
    if len(valid) == 0:
        return float("nan")
    mean = float(valid.mean())
    std = float(valid.std(ddof=1)) if len(valid) > 1 else 0.0
    return mean - penalty * std


@dataclass
class SelectionDecision:
    """One pre-registered tie-rule verdict (docs/RESELECTION_PREREG.md):
    a challenger is adopted only if `beats_on_stability` AND
    `wins_strict_majority` are both true; otherwise the current config
    is kept. `reason` is a human-readable one-line explanation for the
    report, not re-derived logic."""

    adopt_candidate: bool
    current_stability: float
    candidate_stability: float
    beats_on_stability: bool
    fold_win_count: int
    n_folds: int
    wins_strict_majority: bool
    reason: str


def select_candidate_or_keep_current(
    current_fold_values: pd.Series,
    candidate_fold_values: pd.Series,
    penalty: float,
) -> SelectionDecision:
    """Pre-registered tie rule: keep the CURRENT config value unless a
    candidate (a) beats it on `stability_score` AND (b) wins a STRICT
    majority of the folds the two share (a per-fold tie counts for
    neither side, so it cannot count toward the candidate's majority).
    Current/simpler wins every tie on either criterion -- this function
    never adopts a challenger that only matches, not beats, the
    incumbent.

    Folds are compared only where both series have a non-NaN value
    (a fold with no qualifying events for one side carries no
    information about which config is better there).
    """
    current = pd.Series(current_fold_values).dropna()
    candidate = pd.Series(candidate_fold_values).dropna()
    common = current.index.intersection(candidate.index)
    current_common = current.loc[common]
    candidate_common = candidate.loc[common]

    current_stability = stability_score(current_fold_values, penalty)
    candidate_stability = stability_score(candidate_fold_values, penalty)
    beats_on_stability = (
        pd.notna(candidate_stability)
        and pd.notna(current_stability)
        and candidate_stability > current_stability
    )

    n_folds = len(common)
    fold_win_count = int((candidate_common > current_common).sum())
    wins_strict_majority = n_folds > 0 and fold_win_count > n_folds / 2

    adopt = beats_on_stability and wins_strict_majority
    if adopt:
        reason = (
            "candidate adopted: beats current on stability_score "
            f"({candidate_stability:.4f} > {current_stability:.4f}) and "
            f"wins a strict majority of folds ({fold_win_count}/{n_folds})"
        )
    elif not beats_on_stability:
        reason = (
            "current kept: candidate does not beat current on "
            f"stability_score ({candidate_stability:.4f} <= "
            f"{current_stability:.4f})"
        )
    else:
        reason = (
            "current kept: candidate beats current on stability_score "
            f"but does not win a strict majority of folds "
            f"({fold_win_count}/{n_folds})"
        )

    return SelectionDecision(
        adopt_candidate=adopt,
        current_stability=current_stability,
        candidate_stability=candidate_stability,
        beats_on_stability=beats_on_stability,
        fold_win_count=fold_win_count,
        n_folds=n_folds,
        wins_strict_majority=wins_strict_majority,
        reason=reason,
    )
