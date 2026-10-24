"""Live weekly runner tests (live stage 2).

Two load-bearing parity tests prove the live path cannot silently drift
from the validated backtest: `test_replay_parity_matches_backtest_target`
(live target == backtest's computed target for the same date) and
`test_week_over_week_drift_reconciliation` (reconciling last week's
stored post-execution weights across a real multi-day gap reproduces the
backtest's `current` this week -- the anti-one-day-divergence guard). The
rest cover the sizing/reconcile/persist mechanics and failure semantics.

All tests are offline: the real-strategy parity tests run the genuine
HMM/Black-Litterman pipeline on synthetic returns with the anomaly
override (TensorFlow) disabled; every other test mocks the data and
strategy seams so it stays fast and hermetic.
"""

import copy
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from atlas import backtest, data, live, state as state_mod, strategy


# --------------------------------------------------------------------- #
# Fixtures / builders
# --------------------------------------------------------------------- #
BUCKETS_4 = {
    "EQ": "equity",
    "FI": "fixed_income",
    "CO": "commodity",
    "CA": "cash",
}


def _small_cfg():
    """Real config with lookbacks shrunk so the genuine strategy runs on
    a short synthetic history, and the anomaly override (TF) off."""
    cfg = copy.deepcopy(data.load_config())
    cfg["regimes"]["market_ticker"] = "EQ1"
    cfg["regimes"]["hmm"]["n_states"] = 2
    cfg["regimes"]["hmm"]["lookback_days"] = 120
    cfg["regimes"]["hmm"]["vol_window_days"] = 5
    cfg["optimization"]["lookback_days"] = 120
    cfg["meanreversion"]["lookback_days"] = 20
    cfg["meanreversion"]["adf_lookback_days"] = 40
    cfg["general"]["is_end_date"] = "2021-12-31"
    cfg["modules"]["anomaly_override"] = False
    return cfg


def _real_market(cfg, seed=7, n=200):
    """Synthetic multi-asset market covering all four buckets, long
    enough for the real HMM/BL pipeline to run at the final week-end."""
    rng = np.random.default_rng(seed)
    tickers = ["EQ1", "EQ2", "FI1", "FI2", "CO1", "CO2", "CA1"]
    buckets = {
        "EQ1": "equity",
        "EQ2": "equity",
        "FI1": "fixed_income",
        "FI2": "fixed_income",
        "CO1": "commodity",
        "CO2": "commodity",
        "CA1": "cash",
    }
    dates = pd.bdate_range("2021-06-01", periods=n)
    vol = {
        "equity": 0.012,
        "fixed_income": 0.004,
        "commodity": 0.014,
        "cash": 0.0005,
    }
    mu = {
        "equity": 0.0004,
        "fixed_income": 0.0001,
        "commodity": 0.0003,
        "cash": 8e-5,
    }
    cols = {}
    for t in tickers:
        b = buckets[t]
        r = rng.normal(mu[b], vol[b], len(dates))
        cols[t] = 100 * np.cumprod(1 + r)
    prices = data.align_calendars(pd.DataFrame(cols, index=dates))
    returns = data.daily_returns(prices)
    return live.MarketData(
        returns=returns,
        prices=prices,
        rf_series=pd.Series(0.04, index=returns.index),
        rf_true_last_date=returns.index.max(),
        class_bucket=pd.Series(buckets),
        tickers=tickers,
        last_price_date=prices.index.max(),
    )


def _toy_market(seed=1, n=40):
    """Tiny 4-ticker (one per bucket) market for the fast, mocked
    tests -- no real strategy is run against it."""
    rng = np.random.default_rng(seed)
    tickers = ["EQ", "FI", "CO", "CA"]
    dates = pd.bdate_range("2022-01-03", periods=n)
    cols = {
        t: 100 * np.cumprod(1 + rng.normal(0.0002, 0.01, n)) for t in tickers
    }
    prices = data.align_calendars(pd.DataFrame(cols, index=dates))
    returns = data.daily_returns(prices)
    return live.MarketData(
        returns=returns,
        prices=prices,
        rf_series=pd.Series(0.03, index=returns.index),
        rf_true_last_date=returns.index.max(),
        class_bucket=pd.Series(BUCKETS_4),
        tickers=tickers,
        last_price_date=prices.index.max(),
    )


def _fixed_strategy_builder(target):
    """A drop-in for live.build_strategy that ignores data and returns a
    constant target -- lets the sizing/persist tests drive an exact
    wanted-turnover without running the real pipeline."""

    def build(*args, **kwargs):
        def strat(as_of, window):
            return target

        return strat

    return build


def _no_meanrev(prices, cfg):
    return pd.Index([])


def _mock_fixed_decision(monkeypatch, market, target):
    """Patch the data + strategy seams so run_once uses `market` and a
    constant `target`, without running the real pipeline or TensorFlow."""
    monkeypatch.setattr(live, "fetch_data", lambda _cfg: market)
    monkeypatch.setattr(
        live, "build_strategy", _fixed_strategy_builder(target)
    )
    monkeypatch.setattr(live, "compute_meanrev_eligible", _no_meanrev)


def _seed_state(path, weights, weights_as_of, last_decision="2000-01-01"):
    st = state_mod.empty_state()
    st["initialized"] = True
    st["weights"] = {t: float(v) for t, v in weights.items()}
    st["weights_as_of"] = pd.Timestamp(weights_as_of).isoformat()
    st["last_decision_date"] = last_decision
    st["run_count"] = 1
    state_mod.save_state(st, path)
    return path


def _toy_target():
    return pd.Series({"EQ": 0.4, "FI": 0.2, "CO": 0.2, "CA": 0.2})


# --------------------------------------------------------------------- #
# size_trade: the shared sizing primitive (unit level)
# --------------------------------------------------------------------- #
def test_size_trade_funding_is_cap_exempt():
    trade = pd.Series({"A": 1.0, "B": 0.0})  # from cash, one-way 0.5
    sized = backtest.size_trade(
        trade, is_funding=True, no_trade_band=0.005, turnover_cap=0.02
    )
    assert sized.executed_one_way == pytest.approx(0.5)
    assert not sized.cap_binding and not sized.skipped
    pd.testing.assert_series_equal(sized.executed_trade, trade)


def test_size_trade_band_skips():
    trade = pd.Series({"A": 0.001, "B": -0.001})  # one-way 0.001 < band
    sized = backtest.size_trade(
        trade, is_funding=False, no_trade_band=0.005, turnover_cap=0.02
    )
    assert sized.skipped and sized.executed_one_way == 0.0
    assert (sized.executed_trade == 0.0).all()


def test_size_trade_cap_scales_to_exactly_cap():
    trade = pd.Series({"A": 0.5, "B": -0.5})  # one-way 0.5 > cap
    sized = backtest.size_trade(
        trade, is_funding=False, no_trade_band=0.005, turnover_cap=0.02
    )
    assert sized.cap_binding
    assert sized.executed_one_way == pytest.approx(0.02)
    assert sized.executed_trade.abs().sum() / 2 == pytest.approx(0.02)


# --------------------------------------------------------------------- #
# resolve_as_of
# --------------------------------------------------------------------- #
def test_resolve_as_of_excludes_in_progress_week():
    calendar = pd.bdate_range("2026-05-01", "2026-06-10")
    # Reference is a Wednesday; the resolved decision date must be the
    # prior completed week's last trading day (a Friday), not this
    # week's partial data.
    reference = pd.Timestamp("2026-06-10")  # Wednesday
    resolved = live.resolve_as_of(calendar, reference=reference)
    assert resolved == pd.Timestamp("2026-06-05")  # prior Friday
    assert resolved.isocalendar().week != reference.isocalendar().week


@pytest.mark.parametrize(
    "reference",
    [pd.Timestamp("2026-06-13"), pd.Timestamp("2026-06-14")],  # Sat, Sun
)
def test_resolve_as_of_weekend_reference_decides_for_friday_just_closed(
    reference,
):
    # Friday 2026-06-12 is the last trading day of the week that just
    # closed; a weekend run must decide for it, not for the week before
    # (ANALYSIS_V4 Sec 2.2 -- the pre-fix code excluded it unconditionally
    # and was a full week stale on every weekend run).
    calendar = pd.bdate_range("2026-05-01", "2026-06-12")
    resolved = live.resolve_as_of(calendar, reference=reference)
    assert resolved == pd.Timestamp("2026-06-12")


@pytest.mark.parametrize(
    "reference",
    [pd.Timestamp(d) for d in
     ["2026-06-15", "2026-06-16", "2026-06-17", "2026-06-18", "2026-06-19"]],
)
def test_resolve_as_of_weekday_reference_decides_for_previous_completed_week(
    reference,
):
    # Mon-Fri of the following week: the in-progress week is excluded,
    # so every weekday resolves to the same prior Friday.
    calendar = pd.bdate_range("2026-05-01", "2026-06-19")
    resolved = live.resolve_as_of(calendar, reference=reference)
    assert resolved == pd.Timestamp("2026-06-12")


def test_resolve_as_of_holiday_week_resolves_to_thursday():
    # Good Friday 2026-04-03 is a market holiday: it is simply absent
    # from the calendar (as it would be from real downloaded prices).
    # Queried the following week (a fully-elapsed past week, immune to
    # the weekend staleness guard), the week-end must be Thursday.
    calendar = pd.bdate_range("2026-03-01", "2026-04-10").drop(
        pd.Timestamp("2026-04-03")
    )
    reference = pd.Timestamp("2026-04-06")  # Monday after
    resolved = live.resolve_as_of(calendar, reference=reference)
    assert resolved == pd.Timestamp("2026-04-02")  # Thursday


def test_resolve_as_of_saturday_with_friday_missing_holds():
    # Friday 2026-06-12 has not posted: the current week's data stops
    # on Thursday. A Saturday run must HOLD rather than silently decide
    # using Thursday's close as if it were the week's final data.
    calendar = pd.bdate_range("2026-05-01", "2026-06-11")  # thru Thursday
    reference = pd.Timestamp("2026-06-13")  # Saturday
    with pytest.raises(live.StaleDataError):
        live.resolve_as_of(calendar, reference=reference)


# --------------------------------------------------------------------- #
# THE replay parity tests (real strategy, offline)
# --------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def bt_reference():
    """Run the genuine backtest ONCE over the synthetic history and share
    its recorded targets + weights across the parity tests. A recorder
    wraps the exact same strategy construction `live.build_strategy`
    uses, so `recorded[d]` is the target `backtest.run` computed at each
    rebalance `d`."""
    cfg = _small_cfg()
    market = _real_market(cfg)
    posture_cfg = live._load_posture_cfg(cfg)
    meanrev = live.compute_meanrev_eligible(market.prices, cfg)

    bl_fn = strategy.black_litterman_strategy(
        market.class_bucket,
        cfg,
        posture_cfg,
        rf_series=market.rf_series,
        meanrev_eligible=meanrev,
    )
    recorded = {}

    def recorder(d, w):
        out = bl_fn(d, w)
        recorded[d] = out
        return out

    bt = backtest.run(recorder, market.returns, cfg)
    return SimpleNamespace(
        cfg=cfg,
        market=market,
        bt=bt,
        recorded=recorded,
        week_ends=sorted(backtest.week_end_dates(market.returns.index)),
    )


def test_replay_parity_matches_backtest_target(
    tmp_path, monkeypatch, bt_reference
):
    cfg, market, bt, recorded = (
        bt_reference.cfg,
        bt_reference.market,
        bt_reference.bt,
        bt_reference.recorded,
    )
    as_of = bt_reference.week_ends[-1]
    ref_target = recorded[as_of].reindex(market.returns.columns).fillna(0.0)

    # Seed live state with the weights the backtest held at `as_of`,
    # anchored at `as_of` so drift is the identity and the TARGET is
    # isolated. (The week-over-week test below exercises real drift.)
    _seed_state(
        tmp_path / "state.json", bt.weights.loc[as_of].to_dict(), as_of
    )
    monkeypatch.setattr(live, "fetch_data", lambda _cfg: market)

    res = live.run_once(
        as_of=as_of,
        cfg=cfg,
        dry_run=True,
        state_path=tmp_path / "state.json",
    )

    assert res.status == live.STATUS_OK
    # PARITY: live target == backtest's computed target, row for row.
    pd.testing.assert_series_equal(
        res.target_weights.reindex(ref_target.index),
        ref_target,
        atol=1e-8,
        check_names=False,
    )
    # The shared size_trade helper agrees on this week's turnover.
    assert res.one_way_executed == pytest.approx(
        float(bt.turnover.loc[as_of]), abs=1e-10
    )

    # Diagnostics captured the real pipeline and serialize to disk.
    bl = res.diagnostics["black_litterman"]
    assert bl["gate"]["selected"] in {"black_litterman", "gmv", "risk_parity"}
    assert "posture" in bl["regime"]
    weekly_dir = live.write_run_outputs(res, base_dir=tmp_path / "out")
    with open(weekly_dir / "diagnostics.json", encoding="utf-8") as f:
        json.load(f)  # must be valid JSON (no numpy keys leaking through)


def test_week_over_week_drift_reconciliation(
    tmp_path, monkeypatch, bt_reference
):
    # The anti-one-day-divergence guard: seed state as of the PREVIOUS
    # week and let run_once drift the stored weights across the real
    # multi-day gap to this week, then assert the reconstructed `current`
    # equals the backtest's recorded weights going into this week. Unlike
    # every other live test, weights_as_of != as_of here, so
    # drift_weights does genuine multi-day work (not its from==to no-op).
    cfg, market, bt, recorded = (
        bt_reference.cfg,
        bt_reference.market,
        bt_reference.bt,
        bt_reference.recorded,
    )
    cols = market.returns.columns
    # The last two FULL consecutive week-ends (the synthetic calendar can
    # end mid-week, leaving week_ends[-1] a 1-day stub); these two are
    # Fridays a full trading week apart.
    w_prev, w = bt_reference.week_ends[-3], bt_reference.week_ends[-2]

    # Guard: there really is a multi-day gap to drift across.
    gap = market.returns.loc[
        (market.returns.index > w_prev) & (market.returns.index <= w)
    ]
    assert w_prev != w and len(gap) >= 3

    # Reconstruct the POST-execution weights as of w_prev exactly as the
    # backtest applied them at the next trading day: the pre-trade
    # (close-of-w_prev) weights plus that week's executed trade, using
    # the SAME shared size_trade -- matching live.py's stored convention.
    target_prev = recorded[w_prev].reindex(cols).fillna(0.0)
    current_prev = bt.weights.loc[w_prev]
    sized_prev = backtest.size_trade(
        target_prev - current_prev,
        is_funding=False,
        no_trade_band=cfg["rebalance"]["no_trade_band"],
        turnover_cap=cfg["rebalance"]["max_weekly_turnover"],
    )
    post_exec_prev = current_prev + sized_prev.executed_trade

    _seed_state(tmp_path / "state.json", post_exec_prev.to_dict(), w_prev)
    monkeypatch.setattr(live, "fetch_data", lambda _cfg: market)

    res = live.run_once(
        as_of=w, cfg=cfg, dry_run=True, state_path=tmp_path / "state.json"
    )
    assert res.status == live.STATUS_OK

    # DRIFT PARITY: reconciling last week's post-execution weights across
    # the week reproduces the backtest's `current` at this week exactly.
    pd.testing.assert_series_equal(
        res.current_weights.reindex(bt.weights.columns),
        bt.weights.loc[w],
        atol=1e-10,
        check_names=False,
    )
    # And the resulting trade matches the backtest's sized turnover at w,
    # confirming the whole reconcile -> decide -> size chain lines up.
    assert res.one_way_executed == pytest.approx(
        float(bt.turnover.loc[w]), abs=1e-10
    )


# --------------------------------------------------------------------- #
# Idempotency
# --------------------------------------------------------------------- #
def test_second_run_is_already_run_and_leaves_state_untouched(
    tmp_path, monkeypatch
):
    market = _toy_market()
    as_of = max(backtest.week_end_dates(market.returns.index))
    _mock_fixed_decision(monkeypatch, market, _toy_target())

    cfg = data.load_config()
    sp = tmp_path / "state.json"

    first = live.run_once(as_of=as_of, cfg=cfg, state_path=sp)
    assert first.status == live.STATUS_OK
    after_first = sp.read_bytes()

    second = live.run_once(as_of=as_of, cfg=cfg, state_path=sp)
    assert second.status == live.STATUS_ALREADY_RUN
    assert sp.read_bytes() == after_first  # byte-identical: no re-trade


# --------------------------------------------------------------------- #
# Failure semantics: every non-ok outcome leaves state byte-identical
# --------------------------------------------------------------------- #
def test_stale_data_holds_without_touching_state(tmp_path, monkeypatch):
    market = _toy_market()
    stale_as_of = market.last_price_date + pd.Timedelta(days=7)
    monkeypatch.setattr(live, "fetch_data", lambda _cfg: market)

    sp = _seed_state(
        tmp_path / "state.json", {"CA": 1.0}, market.last_price_date
    )
    before = sp.read_bytes()

    res = live.run_once(
        as_of=stale_as_of, cfg=data.load_config(), state_path=sp
    )
    assert res.status == live.STATUS_HELD
    assert res.exit_code == live.EXIT_HELD
    assert sp.read_bytes() == before


def test_risk_free_unavailable_holds(tmp_path, monkeypatch):
    def raising_fetch(_cfg):
        raise live.RiskFreeUnavailableError("no rf")

    monkeypatch.setattr(live, "fetch_data", raising_fetch)
    sp = _seed_state(tmp_path / "state.json", {"CA": 1.0}, "2022-01-07")
    before = sp.read_bytes()

    res = live.run_once(
        as_of="2022-02-04", cfg=data.load_config(), state_path=sp
    )
    assert res.status == live.STATUS_HELD
    assert sp.read_bytes() == before


def test_unexpected_exception_fails_without_touching_state(
    tmp_path, monkeypatch
):
    market = _toy_market()
    as_of = max(backtest.week_end_dates(market.returns.index))

    def exploding_builder(*args, **kwargs):
        def strat(a, w):
            raise RuntimeError("boom in decide")

        return strat

    monkeypatch.setattr(live, "fetch_data", lambda _cfg: market)
    monkeypatch.setattr(live, "build_strategy", exploding_builder)
    monkeypatch.setattr(live, "compute_meanrev_eligible", _no_meanrev)

    sp = _seed_state(tmp_path / "state.json", {"CA": 1.0}, as_of)
    before = sp.read_bytes()

    res = live.run_once(as_of=as_of, cfg=data.load_config(), state_path=sp)
    assert res.status == live.STATUS_FAILED
    assert res.exit_code == live.EXIT_FAILED
    assert "boom in decide" in res.messages[0]
    assert sp.read_bytes() == before


def test_corrupt_state_fails_and_is_not_reset(tmp_path):
    sp = tmp_path / "state.json"
    sp.write_text("{ not valid json ,,,", encoding="utf-8")
    before = sp.read_bytes()

    res = live.run_once(
        as_of="2022-02-04", cfg=data.load_config(), state_path=sp
    )
    assert res.status == live.STATUS_FAILED
    assert sp.read_bytes() == before  # loud failure, memory preserved


# --------------------------------------------------------------------- #
# Funding path from an uninitialized state
# --------------------------------------------------------------------- #
def test_funding_path_executes_full_and_sums_to_one(tmp_path, monkeypatch):
    market = _toy_market()
    as_of = max(backtest.week_end_dates(market.returns.index))
    target = _toy_target()
    _mock_fixed_decision(monkeypatch, market, target)

    sp = tmp_path / "state.json"  # absent -> uninitialized
    res = live.run_once(as_of=as_of, cfg=data.load_config(), state_path=sp)

    assert res.status == live.STATUS_OK
    assert res.diagnostics["is_funding"] is True
    # Funding is cap-exempt: the full 0.5 one-way funding trade executes
    # despite the 0.02 cap.
    assert res.one_way_executed == pytest.approx(0.5)
    assert not res.cap_binding
    assert res.executed_weights.sum() == pytest.approx(1.0)
    assert (res.executed_weights >= 0).all()
    pd.testing.assert_series_equal(
        res.executed_weights.reindex(target.index),
        target,
        check_names=False,
    )
    # State is now initialized and anchored at the decision date.
    saved = state_mod.load_state(sp)
    assert saved["initialized"] is True
    assert saved["weights_as_of"] == pd.Timestamp(as_of).isoformat()


# --------------------------------------------------------------------- #
# Cap binding and no-trade band, driven through run_once
# --------------------------------------------------------------------- #
def test_cap_binding_scales_trade_to_exactly_the_cap(tmp_path, monkeypatch):
    market = _toy_market()
    as_of = max(backtest.week_end_dates(market.returns.index))
    _mock_fixed_decision(monkeypatch, market, _toy_target())

    cfg = data.load_config()
    cap = cfg["rebalance"]["max_weekly_turnover"]
    # Current all in cash, anchored at as_of (drift = identity): a large
    # desired move, so the cap must bind.
    sp = _seed_state(tmp_path / "state.json", {"CA": 1.0}, as_of)

    res = live.run_once(as_of=as_of, cfg=cfg, state_path=sp)

    assert res.status == live.STATUS_OK
    assert res.cap_binding
    assert res.one_way_executed == pytest.approx(cap)
    assert res.trade.abs().sum() / 2 == pytest.approx(cap)
    assert res.executed_weights.sum() == pytest.approx(1.0)


def test_no_trade_band_skips_and_holds(tmp_path, monkeypatch):
    market = _toy_market()
    as_of = max(backtest.week_end_dates(market.returns.index))
    target = _toy_target()
    _mock_fixed_decision(monkeypatch, market, target)

    # Current already at the target (anchored at as_of), so the desired
    # trade is ~0 and falls inside the no-trade band.
    sp = _seed_state(tmp_path / "state.json", target.to_dict(), as_of)

    res = live.run_once(as_of=as_of, cfg=data.load_config(), state_path=sp)

    assert res.status == live.STATUS_OK
    assert res.one_way_executed == 0.0
    assert res.trade.abs().sum() == pytest.approx(0.0, abs=1e-12)
    pd.testing.assert_series_equal(
        res.executed_weights.reindex(target.index),
        target,
        check_names=False,
    )


# --------------------------------------------------------------------- #
# dry_run never writes
# --------------------------------------------------------------------- #
def test_dry_run_never_writes_state(tmp_path, monkeypatch):
    market = _toy_market()
    as_of = max(backtest.week_end_dates(market.returns.index))
    _mock_fixed_decision(monkeypatch, market, _toy_target())

    sp = tmp_path / "state.json"  # absent
    res = live.run_once(
        as_of=as_of, cfg=data.load_config(), dry_run=True, state_path=sp
    )

    assert res.status == live.STATUS_OK
    assert not sp.exists()  # dry run leaves no state behind
    assert not list(tmp_path.glob("*.tmp"))


# --------------------------------------------------------------------- #
# Rounding (ANALYSIS_V4 Sec 2.7): SLSQP residuals like 4.5e-17 must not
# reach allocation.csv / trades.csv / state.json.
# --------------------------------------------------------------------- #
def test_round_weights_rounds_and_renormalizes():
    noisy = pd.Series({"A": 0.25 + 4.5e-17, "B": 0.25, "C": 0.5 - 1e-16})
    rounded = live._round_weights(noisy)
    assert rounded["A"] == 0.25
    assert rounded.sum() == pytest.approx(1.0, abs=1e-12)


def test_round_weights_handles_all_zero():
    zeros = pd.Series({"A": 0.0, "B": 0.0})
    rounded = live._round_weights(zeros)
    assert (rounded == 0.0).all()


def test_funding_path_writes_rounded_outputs(tmp_path, monkeypatch):
    market = _toy_market()
    as_of = max(backtest.week_end_dates(market.returns.index))
    # EQ/CA carry a near-cancelling SLSQP-style residual; FI/CO are clean.
    noisy_target = pd.Series(
        {"EQ": 0.4 + 4.5e-17, "FI": 0.2, "CO": 0.2, "CA": 0.2 - 4.5e-17}
    )
    _mock_fixed_decision(monkeypatch, market, noisy_target)

    sp = tmp_path / "state.json"
    res = live.run_once(as_of=as_of, cfg=data.load_config(), state_path=sp)
    assert res.status == live.STATUS_OK

    saved = state_mod.load_state(sp)  # validate_state must still pass
    for w in saved["weights"].values():
        assert abs(w - round(w, 6)) < 1e-9

    weekly_dir = live.write_run_outputs(res, base_dir=tmp_path / "out")
    alloc = pd.read_csv(weekly_dir / "allocation.csv", index_col="ticker")
    trades = pd.read_csv(weekly_dir / "trades.csv", index_col="ticker")
    for df, cols in ((alloc, ["weight"]), (trades, ["current", "target"])):
        for col in cols:
            for w in df[col]:
                assert abs(w - round(w, 6)) < 1e-9


# --------------------------------------------------------------------- #
# Risk-free freshness (external-review Phase B1): rf_as_of must be
# FRED's own last valid observation, not the price-calendar-ffilled
# date, and a stale FRED feed must hold rather than silently use rf=0.
# --------------------------------------------------------------------- #
def test_rf_staleness_within_bound_is_ok():
    live._check_rf_staleness(
        pd.Timestamp("2026-06-05"), pd.Timestamp("2026-06-03"), 7
    )  # gap=2 days: must not raise


def test_rf_staleness_exceeds_bound_raises():
    with pytest.raises(live.RiskFreeUnavailableError):
        live._check_rf_staleness(
            pd.Timestamp("2026-06-05"), pd.Timestamp("2026-05-26"), 7
        )  # gap=10 days


def test_rf_stale_by_10_days_holds_without_touching_state(
    tmp_path, monkeypatch
):
    market = _toy_market()
    as_of = max(backtest.week_end_dates(market.returns.index))
    market.rf_true_last_date = as_of - pd.Timedelta(days=10)
    monkeypatch.setattr(live, "fetch_data", lambda _cfg: market)

    sp = _seed_state(
        tmp_path / "state.json", {"CA": 1.0}, as_of - pd.Timedelta(days=7)
    )
    before = sp.read_bytes()

    res = live.run_once(as_of=as_of, cfg=data.load_config(), state_path=sp)
    assert res.status == live.STATUS_HELD
    assert sp.read_bytes() == before


def test_rf_lag_of_2_days_is_ok(tmp_path, monkeypatch):
    market = _toy_market()
    as_of = max(backtest.week_end_dates(market.returns.index))
    market.rf_true_last_date = as_of - pd.Timedelta(days=2)
    _mock_fixed_decision(monkeypatch, market, _toy_target())

    sp = tmp_path / "state.json"
    res = live.run_once(as_of=as_of, cfg=data.load_config(), state_path=sp)
    assert res.status == live.STATUS_OK


def test_diagnostics_rf_as_of_is_true_fred_date_not_price_date(
    tmp_path, monkeypatch
):
    market = _toy_market()
    as_of = max(backtest.week_end_dates(market.returns.index))
    # The exact bug this guards against: the true FRED date trails the
    # price calendar's last close, but is still within tolerance.
    true_rf_date = as_of - pd.Timedelta(days=2)
    market.rf_true_last_date = true_rf_date
    _mock_fixed_decision(monkeypatch, market, _toy_target())

    sp = tmp_path / "state.json"
    res = live.run_once(as_of=as_of, cfg=data.load_config(), state_path=sp)
    assert res.status == live.STATUS_OK
    assert res.diagnostics["data"]["rf_as_of"] == true_rf_date.isoformat()
    assert res.diagnostics["data"]["rf_as_of"] != (
        market.last_price_date.isoformat()
    )


# --------------------------------------------------------------------- #
# Holiday-Friday catch-up (external-review Phase B2): a Saturday run
# that holds because Friday is a market holiday must be decidable by a
# later (Monday) run; an ordinary already-decided week must be a TRUE
# no-op for that later run.
# --------------------------------------------------------------------- #
def test_resolve_as_of_saturday_held_then_monday_decides_thursday():
    # Good Friday: absent from the calendar, exactly as it would be
    # from real downloaded prices.
    calendar = pd.bdate_range("2026-03-01", "2026-04-10").drop(
        pd.Timestamp("2026-04-03")
    )
    saturday = pd.Timestamp("2026-04-04")
    with pytest.raises(live.StaleDataError):
        live.resolve_as_of(calendar, reference=saturday)

    monday = pd.Timestamp("2026-04-06")  # a NEW ISO week vs. the holiday one
    resolved = live.resolve_as_of(calendar, reference=monday)
    assert resolved == pd.Timestamp("2026-04-02")  # Thursday


def test_already_run_is_a_true_noop_no_files_touched(tmp_path, monkeypatch):
    # Saturday decides an ordinary week; Monday's catch-up run resolves
    # to the SAME date (already_run) and must touch nothing -- no
    # history row, no weekly directory -- so the scheduler's
    # `git diff --cached --quiet` finds no changes to commit.
    market = _toy_market()
    as_of = max(backtest.week_end_dates(market.returns.index))
    _mock_fixed_decision(monkeypatch, market, _toy_target())

    cfg = data.load_config()
    sp = tmp_path / "state.json"
    out_dir = tmp_path / "out"

    first = live.run_once(as_of=as_of, cfg=cfg, state_path=sp)
    assert first.status == live.STATUS_OK
    assert live.write_run_outputs(first, base_dir=out_dir) is not None
    state_before = sp.read_bytes()
    history_before = (out_dir / "history.csv").read_bytes()
    weekly_before = sorted((out_dir / "weekly").rglob("*"))

    second = live.run_once(as_of=as_of, cfg=cfg, state_path=sp)
    assert second.status == live.STATUS_ALREADY_RUN
    result_dir = live.write_run_outputs(second, base_dir=out_dir)

    assert result_dir is None
    assert sp.read_bytes() == state_before
    assert (out_dir / "history.csv").read_bytes() == history_before
    assert sorted((out_dir / "weekly").rglob("*")) == weekly_before


# --------------------------------------------------------------------- #
# Net-of-cost live NAV (external-review Phase B3): book_return_net must
# subtract the PRIOR week's executed one-way turnover (the trade that
# executed at the start of this window), using backtest.run's exact
# cost formula (cost_rate * one-way, not two-way).
# --------------------------------------------------------------------- #
def test_book_return_net_subtracts_prior_week_execution_cost(
    tmp_path, monkeypatch
):
    market = _toy_market()
    weeks = sorted(backtest.week_end_dates(market.returns.index))
    prev, as_of = weeks[-2], weeks[-1]

    cfg = data.load_config()
    sp = tmp_path / "state.json"

    target1 = pd.Series({"EQ": 0.4, "FI": 0.3, "CO": 0.2, "CA": 0.1})
    _mock_fixed_decision(monkeypatch, market, target1)
    first = live.run_once(as_of=prev, cfg=cfg, state_path=sp)
    assert first.status == live.STATUS_OK
    prior_one_way = first.one_way_executed
    assert prior_one_way > 0  # a real trade, so the cost is non-zero

    target2 = pd.Series({"EQ": 0.3, "FI": 0.3, "CO": 0.2, "CA": 0.2})
    _mock_fixed_decision(monkeypatch, market, target2)
    second = live.run_once(as_of=as_of, cfg=cfg, state_path=sp)
    assert second.status == live.STATUS_OK

    perf = second.diagnostics["performance"]
    cost_rate = cfg["rebalance"]["transaction_cost_bps"] / 1e4
    expected_net = perf["book_return"] - cost_rate * prior_one_way
    assert perf["book_return_net"] == pytest.approx(expected_net, abs=1e-12)
    assert perf["book_return_net"] != pytest.approx(perf["book_return"])


def test_book_return_net_is_none_on_funding_week(tmp_path, monkeypatch):
    market = _toy_market()
    as_of = max(backtest.week_end_dates(market.returns.index))
    _mock_fixed_decision(monkeypatch, market, _toy_target())

    sp = tmp_path / "state.json"
    res = live.run_once(as_of=as_of, cfg=data.load_config(), state_path=sp)
    assert res.status == live.STATUS_OK
    assert res.diagnostics["performance"]["book_return_net"] is None


def test_history_row_carries_book_return_net_column(tmp_path, monkeypatch):
    market = _toy_market()
    weeks = sorted(backtest.week_end_dates(market.returns.index))
    prev, as_of = weeks[-2], weeks[-1]
    cfg = data.load_config()
    sp = tmp_path / "state.json"
    out_dir = tmp_path / "out"

    _mock_fixed_decision(monkeypatch, market, _toy_target())
    first = live.run_once(as_of=prev, cfg=cfg, state_path=sp)
    live.write_run_outputs(first, base_dir=out_dir)

    second = live.run_once(as_of=as_of, cfg=cfg, state_path=sp)
    live.write_run_outputs(second, base_dir=out_dir)

    hist = pd.read_csv(out_dir / "history.csv")
    assert "book_return_net" in hist.columns
    assert "book_return" in hist.columns
    assert pd.notna(hist["book_return_net"].iloc[-1])
