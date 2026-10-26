"""IS-only fold/selection primitives (reselection task, external
review). Proves `selection.py` reproduces the fold scheme and
stability-score formula already used independently in notebook 09's
grid searches and in `backtest.walk_forward`, rather than inventing a
new one."""

import numpy as np
import pandas as pd
import pytest

from atlas import backtest, selection


def test_annual_fold_periods_matches_to_period_year_unique():
    dates = pd.bdate_range("2014-01-01", "2021-12-31")
    expected = dates.to_period("Y").unique()
    result = selection.annual_fold_periods(dates)
    assert list(result) == list(expected)
    assert len(result) == 8  # 2014..2021 inclusive


def test_stability_score_matches_manual_mean_minus_penalty_std():
    values = pd.Series([0.5, 0.6, 0.4, 0.55])
    penalty = 1.5
    expected = values.mean() - penalty * values.std(ddof=1)
    assert selection.stability_score(values, penalty) == pytest.approx(
        expected
    )


def test_stability_score_zero_std_with_one_fold():
    # Matches every one of the three existing call sites' own edge case
    # (notebook 09's evaluate_meanreversion/evaluate_momentum,
    # backtest.walk_forward's summary): std is 0.0, not NaN, with a
    # single valid fold.
    assert selection.stability_score(pd.Series([0.7]), 1.0) == pytest.approx(
        0.7
    )


def test_stability_score_drops_nan_folds():
    with_nan = pd.Series([0.5, np.nan, 0.5, 0.5])
    without_nan = pd.Series([0.5, 0.5, 0.5])
    assert selection.stability_score(
        with_nan, 1.0
    ) == pytest.approx(selection.stability_score(without_nan, 1.0))


def test_stability_score_matches_walk_forward_summary_exactly():
    # Equivalence test (not just "looks similar"): recomputing
    # backtest.walk_forward's own per-fold Sharpe series through
    # selection.stability_score must reproduce its summary exactly --
    # proving this module is the SAME formula, not a reimplementation.
    dates = pd.bdate_range("2020-01-01", periods=500)
    rng = np.random.default_rng(5)
    daily_returns = pd.Series(
        rng.normal(0.0003, 0.01, len(dates)), index=dates
    )
    penalty = 1.3
    cfg = {
        "backtest": {"walk_forward_fold": "quarterly"},
        "grid_search": {"stability_penalty": penalty},
    }
    wf = backtest.walk_forward(daily_returns, cfg)

    recomputed = selection.stability_score(wf.fold_metrics["sharpe"], penalty)
    assert recomputed == pytest.approx(wf.summary["stability_score"])


# --------------------------------------------------------------------- #
# select_candidate_or_keep_current: the pre-registered tie rule
# --------------------------------------------------------------------- #
def test_candidate_adopted_when_it_beats_stability_and_wins_majority():
    # Candidate is uniformly 0.10 higher every fold -- same cross-fold
    # variance as current, strictly higher mean -- so it wins all three
    # folds AND beats current on stability score.
    current = pd.Series({"2019": 0.30, "2020": 0.35, "2021": 0.20})
    candidate = pd.Series({"2019": 0.40, "2020": 0.45, "2021": 0.30})
    decision = selection.select_candidate_or_keep_current(
        current, candidate, penalty=1.0
    )
    assert decision.fold_win_count == 3
    assert decision.n_folds == 3
    assert decision.wins_strict_majority is True
    assert decision.adopt_candidate is True


def test_current_kept_when_candidate_wins_majority_but_not_stability():
    # Wins 2 of 3 folds narrowly but its variance is high enough that
    # the aggregate stability score still loses to current.
    current = pd.Series({"2019": 0.50, "2020": 0.50, "2021": 0.50})
    candidate = pd.Series({"2019": 0.51, "2020": 0.51, "2021": -2.0})
    decision = selection.select_candidate_or_keep_current(
        current, candidate, penalty=1.0
    )
    assert decision.fold_win_count == 2
    assert decision.wins_strict_majority is True
    assert decision.beats_on_stability is False
    assert decision.adopt_candidate is False


def test_current_kept_when_candidate_beats_stability_but_not_majority():
    # Candidate wins only 1 of 3 folds (not a strict majority) even
    # though one large win drags its aggregate stability score above
    # current's.
    current = pd.Series({"2019": 0.10, "2020": 0.10, "2021": 0.10})
    candidate = pd.Series({"2019": 0.05, "2020": 0.05, "2021": 5.0})
    decision = selection.select_candidate_or_keep_current(
        current, candidate, penalty=0.0
    )
    assert decision.fold_win_count == 1
    assert decision.wins_strict_majority is False
    assert decision.beats_on_stability is True
    assert decision.adopt_candidate is False


def test_exact_tie_keeps_current():
    current = pd.Series({"2019": 0.30, "2020": 0.30})
    candidate = pd.Series({"2019": 0.30, "2020": 0.30})
    decision = selection.select_candidate_or_keep_current(
        current, candidate, penalty=1.0
    )
    assert decision.fold_win_count == 0  # no fold is strictly greater
    assert decision.adopt_candidate is False


def test_per_fold_tie_does_not_count_as_a_win_for_either_side():
    current = pd.Series({"2019": 0.30, "2020": 0.20, "2021": 0.20})
    candidate = pd.Series({"2019": 0.30, "2020": 0.40, "2021": 0.10})
    decision = selection.select_candidate_or_keep_current(
        current, candidate, penalty=0.0
    )
    # 2019 ties (no win either way), 2020 candidate wins, 2021 current
    # wins -- 1 win out of 3 folds, not a strict majority.
    assert decision.n_folds == 3
    assert decision.fold_win_count == 1
    assert decision.wins_strict_majority is False
    assert decision.adopt_candidate is False


def test_only_compares_folds_both_sides_have():
    current = pd.Series({"2019": 0.30, "2020": 0.40, "2021": 0.50})
    candidate = pd.Series({"2019": 0.50, "2020": 0.60})  # 2021 missing
    decision = selection.select_candidate_or_keep_current(
        current, candidate, penalty=0.0
    )
    assert decision.n_folds == 2
    assert decision.fold_win_count == 2
    # Stability score itself still uses each series' OWN full set of
    # valid folds (consistent with stability_score's own contract).
    assert decision.current_stability == pytest.approx(current.mean())
    assert decision.candidate_stability == pytest.approx(candidate.mean())
