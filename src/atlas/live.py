"""Live weekly runner (L6): produce one week's allocation from scratch.

A scheduled Friday job invokes `run_once`, which turns the weekly
decision into a single callable: fetch data -> drift last week's stored
weights forward -> decide (the SAME `strategy.black_litterman_strategy`
the backtest calls) -> size the trade (the SAME `backtest.size_trade`)
-> persist -> hand back a fully diagnosed `RunResult`. Rendering that
result to disk (allocation.csv, trades.csv, diagnostics.json, history)
lives in `scripts/run_weekly.py`; HTML/dashboard, GitHub Actions, and
alerting are later stages.

Correctness bar: for any historical `as_of`, the live path must produce
the SAME target weights the backtester produced for that date. It gets
there by reusing the exact decision and sizing functions rather than
re-implementing them (see `tests/test_live.py`'s replay parity test).

State-weight convention (the single most important boundary here)
-----------------------------------------------------------------
`state.json` stores POST-EXECUTION weights anchored at the DECISION
date (the Friday), i.e. `weights_as_of` is the Friday and `weights` are
what the book holds AFTER the trade that executes the next trading day.

Why this is consistent with `state.drift_weights` and the backtest:
`backtest.run` applies a Friday-decided trade at the START of the next
trading day (Monday), before Monday's return, and there is no drift
between Friday's close and Monday's open (no calendar days in the
returns index between them). So the post-execution weights take effect
for every trading day strictly after the Friday. `drift_weights`
anchored at the Friday drifts strictly-after (starting Monday), which
reproduces the backtester's recorded weights exactly. Storing Friday's
date with Monday's weights is therefore not off-by-one: it is precisely
the anchor `drift_weights` expects. Next week's `run_once` drifts these
stored weights from `weights_as_of` to the new Friday to reconstruct
the current book, then diffs against the new target -- identical to how
`backtest.run` computes `trade = target - current_weights` at a
rebalance.

Public API:
    RunResult                                # dataclass: full decision record
    STATUS_OK / STATUS_HELD / STATUS_FAILED / STATUS_ALREADY_RUN
    run_once(as_of=None, cfg=None, dry_run=False, state_path=None) -> RunResult
    resolve_as_of(calendar, reference=None) -> pd.Timestamp
    fetch_data(cfg) -> MarketData
    build_strategy(...) -> strategy_fn
    compute_meanrev_eligible(prices, cfg) -> pd.Index
    exit_code_for(status) -> int
    write_run_outputs(result, base_dir=None) -> Path | None
"""

from __future__ import annotations

import hashlib
import json
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from . import (
    allocation,
    backtest,
    data,
    meanreversion,
    state as state_mod,
    strategy,
    universe,
)

PROJECT_ROOT = data.PROJECT_ROOT
LIVE_DIR = PROJECT_ROOT / "live"
POSTURE_CONFIG_PATH = PROJECT_ROOT / "config" / "regime_posture.yaml"

STATUS_OK = "ok"
STATUS_HELD = "held"
STATUS_FAILED = "failed"
STATUS_ALREADY_RUN = "already_run"

EXIT_OK = 0
EXIT_FAILED = 1
EXIT_HELD = 2


class HeldError(Exception):
    """A recoverable condition that means HOLD the current book: no
    decision, no state mutation. `run_once` turns it into a
    `STATUS_HELD` result, never a crash."""


class StaleDataError(HeldError):
    """Price data has not caught up to `as_of` (weekend run before the
    provider posts, or a provider outage)."""


class RiskFreeUnavailableError(HeldError):
    """The risk-free series could not be fetched. rf is load-bearing
    for the cash bucket's equilibrium prior, so we hold rather than
    silently fall back to rf=0."""


@dataclass
class MarketData:
    """Everything the weekly decision needs from the data layer.

    `rf_true_last_date` (external-review Phase B1) is FRED's own last
    valid DTB3 observation, captured BEFORE `rf_series` is reindexed
    onto the price calendar and forward-filled. The reindex/ffill makes
    `rf_series` itself look fresh through `last_price_date` even when
    the underlying FRED series is stale -- `rf_true_last_date` is what
    the live runner actually checks for staleness and reports in
    diagnostics (see `fetch_data`, `_check_rf_staleness`).
    """

    returns: pd.DataFrame
    prices: pd.DataFrame
    rf_series: pd.Series
    rf_true_last_date: pd.Timestamp
    class_bucket: pd.Series
    tickers: list[str]
    last_price_date: pd.Timestamp


@dataclass
class RunResult:
    """The full record of one weekly run: the decision, the sized
    trade, and the diagnostics needed to explain it after the fact."""

    as_of: pd.Timestamp | None
    status: str
    run_utc: pd.Timestamp | None = None
    target_weights: pd.Series | None = None
    current_weights: pd.Series | None = None
    executed_weights: pd.Series | None = None
    trade: pd.Series | None = None
    one_way_turnover_wanted: float | None = None
    one_way_executed: float | None = None
    cap_binding: bool = False
    diagnostics: dict = field(default_factory=dict)
    messages: list[str] = field(default_factory=list)

    @property
    def exit_code(self) -> int:
        return exit_code_for(self.status)


def exit_code_for(status: str) -> int:
    """Map a run status to a process exit code (PART 3 semantics)."""
    if status in (STATUS_OK, STATUS_ALREADY_RUN):
        return EXIT_OK
    if status == STATUS_HELD:
        return EXIT_HELD
    return EXIT_FAILED


def _load_posture_cfg(cfg: dict) -> dict:
    return data.load_config(POSTURE_CONFIG_PATH)


def resolve_as_of(
    calendar: pd.DatetimeIndex, reference: pd.Timestamp | None = None
) -> pd.Timestamp:
    """The most recent COMPLETED week-end trading day on `calendar`.

    Uses `backtest.week_end_dates` (the same calendar the backtester
    rebalances on, holidays included) rather than hand-rolling a Friday
    date.

    Only a reference that falls ON the in-progress ISO week (Mon-Fri,
    `reference.weekday() < 5`) excludes that week: a job running on
    Wednesday must decide for last week's close, not a half-finished
    week. A Sat/Sun reference's own ISO week has already fully elapsed
    (Friday has closed), so it is NOT excluded -- the prior behaviour
    excluded it unconditionally, which made every weekend run decide a
    full week late (ANALYSIS_V4 Sec 2.2).

    For that Sat/Sun case, if the current week's data stops before
    Friday (`weekday() != 4`) -- the provider hasn't posted Friday's
    close yet, rather than Friday being a genuine market holiday, which
    would already be reflected in `calendar` for any fully-elapsed past
    week -- raise `StaleDataError` instead of silently deciding on the
    partial week. (This is deliberately conservative: a real market
    holiday that lands on the Friday of the CURRENT week and is queried
    on that same Saturday will also hold, re-resolving correctly once
    `reference` moves to Monday; holding is always a safe outcome here,
    never a wrong decision.)
    """
    reference = (reference or pd.Timestamp.today()).normalize()
    ref_year, ref_week, _ = reference.isocalendar()
    exclude_current_week = reference.weekday() < 5

    week_ends = backtest.week_end_dates(calendar)
    candidates = [d for d in week_ends if d <= reference]

    if exclude_current_week:
        candidates = [
            d
            for d in candidates
            if (d.isocalendar().year, d.isocalendar().week)
            != (ref_year, ref_week)
        ]
    else:
        current_week_end = next(
            (
                d
                for d in candidates
                if (d.isocalendar().year, d.isocalendar().week)
                == (ref_year, ref_week)
            ),
            None,
        )
        if current_week_end is not None and current_week_end.weekday() != 4:
            raise StaleDataError(
                "Data for the current week only reaches "
                f"{current_week_end.date()} "
                f"({current_week_end.day_name()}); Friday's close has not "
                "posted yet. Holding this week."
            )

    if not candidates:
        raise HeldError(
            "No completed week-end trading day is available in the data yet."
        )
    return max(candidates)


def _check_rf_staleness(
    as_of: pd.Timestamp,
    rf_true_last_date: pd.Timestamp,
    max_staleness_days: int,
) -> None:
    """Hold rather than decide on a stale risk-free series (ANALYSIS_V4
    external-review Phase B1). `rf_true_last_date` is FRED's own last
    valid observation (see `MarketData`'s docstring) -- DTB3 normally
    lags `as_of` by 1-2 business days; the max historical gap in the
    cached 2010-2026 series is 4 calendar days (holiday weekends), so
    the default 7-day threshold never trips in normal operation and
    only catches a genuine multi-day FRED outage.
    """
    gap_days = (as_of - rf_true_last_date).days
    if gap_days > max_staleness_days:
        raise RiskFreeUnavailableError(
            f"Risk-free series stale: last FRED observation "
            f"{rf_true_last_date.date()} is {gap_days} days before as_of "
            f"{as_of.date()} (max {max_staleness_days}); holding this week."
        )


def compute_meanrev_eligible(prices: pd.DataFrame, cfg: dict) -> pd.Index:
    """V2's eligible-ticker set, computed IS-only exactly as notebook 09
    freezes it: `meanreversion.hit_rate_eligible_tickers` over the price
    history through `general.is_end_date`, using the (already frozen)
    meanreversion params in `cfg`. Deterministic, no model fitting, so
    reproducing it live gives the same set the OOS backtest used."""
    is_end = pd.Timestamp(cfg["general"]["is_end_date"])
    is_prices = prices.loc[:is_end]
    return meanreversion.hit_rate_eligible_tickers(is_prices, cfg)


def fetch_data(cfg: dict) -> MarketData:
    """Load the universe, prices, returns, and the risk-free series the
    weekly decision needs, exactly as notebook 09's frozen OOS run does.

    Raises `RiskFreeUnavailableError` (a `HeldError`) if the risk-free
    series cannot be built: rf is load-bearing for the cash prior, so a
    missing rf is a held week, never a silent fall back to rf=0.
    """
    as_of_universe = pd.Timestamp(cfg["general"]["is_end_date"])
    universe_df = universe.load_universe(as_of_universe)
    tickers = universe_df.index.tolist()
    class_bucket = universe_df["class_bucket"]

    prices = data.download_prices(tickers, start=cfg["general"]["start_date"])
    prices = data.align_calendars(prices)
    returns = data.daily_returns(prices)

    fred_series = cfg["risk_free"]["fred_series"]
    try:
        rf_raw = data.download_macro(
            [fred_series], start=cfg["general"]["start_date"]
        )
        # Captured BEFORE the reindex/ffill below, which otherwise masks
        # staleness behind the last PRICE date (ANALYSIS_V4 external-
        # review Phase B1): a FRED outage would still leave rf_series
        # looking "fresh" through today's close.
        rf_true_last_date = rf_raw[fred_series].dropna().index.max()
        rf_series = (
            (rf_raw[fred_series] / 100.0).reindex(returns.index).ffill()
        )
    except Exception as exc:  # noqa: BLE001 -- any rf failure is a held week
        raise RiskFreeUnavailableError(
            f"Could not fetch risk-free series {fred_series!r}: {exc}"
        ) from exc
    if rf_series.dropna().empty:
        raise RiskFreeUnavailableError(
            f"Risk-free series {fred_series!r} is entirely empty after "
            "alignment; refusing to price the cash prior at rf=0."
        )

    return MarketData(
        returns=returns,
        prices=prices,
        rf_series=rf_series,
        rf_true_last_date=rf_true_last_date,
        class_bucket=class_bucket,
        tickers=tickers,
        last_price_date=prices.index.max(),
    )


def build_strategy(
    cfg: dict,
    posture_cfg: dict,
    class_bucket: pd.Series,
    rf_series: pd.Series,
    meanrev_eligible: pd.Index,
    bl_record=None,
    anomaly_record=None,
) -> strategy.StrategyFn:
    """Construct the weekly decision function exactly as notebook 09's
    frozen OOS run does: `black_litterman_strategy`, wrapped with
    `with_anomaly_override` when `modules.anomaly_override` is on.

    Unlike the notebook (which wraps unconditionally), the anomaly
    override here is gated on the config flag. In production the flag is
    true, so behaviour matches; turning it off (e.g. offline tests that
    must avoid TensorFlow) yields the bare, deterministic BL strategy --
    which is exactly what makes the backtest replay-parity test possible
    without a GPU.
    """
    bl_fn = strategy.black_litterman_strategy(
        class_bucket,
        cfg,
        posture_cfg,
        rf_series=rf_series,
        meanrev_eligible=meanrev_eligible,
        record=bl_record,
    )
    if cfg["modules"].get("anomaly_override", False):
        return strategy.with_anomaly_override(
            bl_fn, cfg, record=anomaly_record
        )
    return bl_fn


def _empty_result(
    status: str, as_of, messages: list[str], run_utc: pd.Timestamp
) -> RunResult:
    as_of_ts = pd.Timestamp(as_of) if as_of is not None else None
    return RunResult(
        as_of=as_of_ts, status=status, run_utc=run_utc, messages=messages
    )


def run_once(
    as_of: str | pd.Timestamp | None = None,
    cfg: dict | None = None,
    dry_run: bool = False,
    state_path: str | Path | None = None,
) -> RunResult:
    """Produce one week's allocation.

    `as_of` defaults to the most recent completed week-end trading day
    (`resolve_as_of`). Passing it explicitly is what makes the replay
    test possible.

    Failure semantics (PART 3), invariant: any non-"ok" outcome leaves
    state.json byte-identical.
      - stale/missing prices or rf unavailable -> STATUS_HELD, no
        mutation, exit 2
      - already run for this as_of -> STATUS_ALREADY_RUN, no mutation,
        exit 0
      - any unexpected exception -> STATUS_FAILED, no mutation, exit 1
        (full traceback in `messages`)
    A model fit failure is NOT a held week: `strategy.py`'s own guards
    fall back to a neutral posture; that fallback is surfaced in
    `messages`, never swallowed.
    """
    cfg = cfg if cfg is not None else data.load_config()
    posture_cfg = _load_posture_cfg(cfg)
    resolved_as_of = pd.Timestamp(as_of) if as_of is not None else None
    # Captured once, at the start, so the report's "run date" reflects
    # when the job actually executed, for every outcome (ok/held/failed),
    # not just runs that reach a successful state-mutating decision.
    run_utc = pd.Timestamp.now(tz="UTC")

    try:
        # Load state first: a corrupt state file must fail loudly (via
        # state.load_state) rather than let us trade on lost memory.
        current_state = state_mod.load_state(state_path)

        market = fetch_data(cfg)

        if resolved_as_of is None:
            resolved_as_of = resolve_as_of(market.returns.index)

        if market.last_price_date < resolved_as_of:
            raise StaleDataError(
                f"Latest available close {market.last_price_date.date()} is "
                f"before as_of {resolved_as_of.date()}; holding this week."
            )
        _check_rf_staleness(
            resolved_as_of,
            market.rf_true_last_date,
            cfg["risk_free"]["max_staleness_days"],
        )

        already = current_state.get("last_decision_date")
        if already == resolved_as_of.isoformat():
            return _empty_result(
                STATUS_ALREADY_RUN,
                resolved_as_of,
                [f"Already decided for {resolved_as_of.date()}; no re-trade."],
                run_utc,
            )

        return _decide(
            current_state, market, cfg, posture_cfg, resolved_as_of,
            dry_run, state_path, run_utc,
        )
    except HeldError as exc:
        return _empty_result(STATUS_HELD, resolved_as_of, [str(exc)], run_utc)
    except Exception:  # noqa: BLE001 -- terminal guard: never mutate on error
        return _empty_result(
            STATUS_FAILED, resolved_as_of, [traceback.format_exc()], run_utc
        )


def _decide(
    current_state: dict,
    market: MarketData,
    cfg: dict,
    posture_cfg: dict,
    as_of: pd.Timestamp,
    dry_run: bool,
    state_path,
    run_utc: pd.Timestamp,
) -> RunResult:
    """Reconcile -> decide -> size -> persist. Returns a STATUS_OK
    result. Raises only on genuinely unexpected errors (handled by the
    caller's terminal guard)."""
    tickers = market.returns.columns
    messages: list[str] = []

    # 1. Reconcile: drift last week's stored weights forward to as_of.
    performance: dict = {
        "book_return": None,
        "book_return_net": None,
        "permanent_return": None,
    }
    if current_state.get("initialized"):
        stored = state_mod.weights_series(current_state)
        anchor = pd.Timestamp(current_state["weights_as_of"])
        current = state_mod.drift_weights(
            stored, market.returns, anchor, as_of
        )
        current = current.reindex(tickers).fillna(0.0)
        is_funding = False
        # Realized weekly return of the held book and of the Permanent
        # benchmark since the last decision (gross of costs) -- pure
        # measurement, recorded for the report's context equity curve;
        # it never feeds the decision.
        cap = cfg["constraints"]["per_asset_cap"]
        perm = allocation.permanent(market.class_bucket, cap=cap)
        gross_book_return = _realized_week_return(
            stored, market.prices, anchor, as_of
        )
        # Net of the cost of the trade that executed at the START of
        # this window (external-review Phase B3): that trade was SIZED
        # at the prior decision (`anchor`), not this one, so its
        # one-way turnover is read back from what THAT run persisted,
        # not from this week's `sizing` (computed below). Same formula
        # backtest.run charges on the execution day (backtest.py:
        # `cost_today = cost_rate * pending_turnover`, subtracted from
        # that one day's simple return before it compounds forward) --
        # here applied directly to the week's cumulative return, which
        # differs from backtest.run's day-then-compound order only by a
        # second-order (cost x subsequent-day-return) term far below
        # this reporting's precision.
        cost_rate = cfg["rebalance"]["transaction_cost_bps"] / 1e4
        prior_one_way = float(current_state.get("last_one_way_executed", 0.0))
        book_return_net = (
            gross_book_return - cost_rate * prior_one_way
            if gross_book_return is not None
            else None
        )
        performance = {
            "book_return": gross_book_return,
            "book_return_net": book_return_net,
            "permanent_return": _realized_week_return(
                perm, market.prices, anchor, as_of
            ),
            "anchor": anchor.isoformat(),
        }
    else:
        current = pd.Series(0.0, index=tickers)
        is_funding = True
        messages.append("Uninitialized state: funding the book from cash.")

    # 2. Decide: call the SAME strategy the backtest calls, once.
    bl_diag: dict = {}
    anomaly_diag: dict = {}
    strat = build_strategy(
        cfg, posture_cfg, market.class_bucket, market.rf_series,
        compute_meanrev_eligible(market.prices, cfg),
        bl_record=bl_diag.update,
        anomaly_record=anomaly_diag.update,
    )
    window = market.returns.loc[:as_of]
    raw_target = strat(as_of, window)
    target = raw_target.reindex(tickers).fillna(0.0)

    # Surface (never swallow) a regime-model fallback: strategy.py drops
    # to a neutral posture on an HMM fit failure or thin history.
    # `fallback_reason` (ANALYSIS_V4 Sec 2.3a) distinguishes the two
    # causes explicitly; its absence (older diagnostics) falls back to
    # the previous, less specific wording via "current_state"'s absence.
    regime = bl_diag.get("regime", {})
    reason = regime.get("fallback_reason")
    if reason is not None:
        messages.append(
            f"Regime model fallback ({reason}); posture defaulted to "
            f"{regime.get('posture')!r}."
        )
    elif "current_state" not in regime:
        messages.append(
            "Regime model produced no state read (thin history or fit "
            f"fallback); posture defaulted to {regime.get('posture')!r}."
        )

    # 3. Size the trade with the SHARED helper (no duplicated logic).
    trade = target - current
    sizing = backtest.size_trade(
        trade,
        is_funding=is_funding,
        no_trade_band=cfg["rebalance"]["no_trade_band"],
        turnover_cap=cfg["rebalance"]["max_weekly_turnover"],
    )
    executed_weights = current + sizing.executed_trade
    if sizing.skipped:
        messages.append(
            "Trade below the no-trade band; holding (weights unchanged)."
        )
    elif sizing.cap_binding:
        messages.append(
            f"Turnover cap binding: wanted {sizing.wanted_one_way:.4f} "
            f"one-way, executed {sizing.executed_one_way:.4f}."
        )

    diagnostics = {
        "as_of": as_of.isoformat(),
        "is_funding": is_funding,
        "rf": bl_diag.get("rf"),
        "black_litterman": bl_diag,
        "anomaly": anomaly_diag or {"active": False, "reason": "disabled"},
        "turnover": {
            "wanted_one_way": sizing.wanted_one_way,
            "executed_one_way": sizing.executed_one_way,
            "cap": cfg["rebalance"]["max_weekly_turnover"],
            "no_trade_band": cfg["rebalance"]["no_trade_band"],
            "cap_binding": sizing.cap_binding,
            "skipped": sizing.skipped,
        },
        # Presentation metadata for the report (rendered, never decided
        # on): asset-class map, the binding per-asset cap, module on/off
        # flags, provenance, and realized returns for the context curve.
        "class_bucket": {
            t: str(market.class_bucket.get(t)) for t in tickers
        },
        "constraints": {"per_asset_cap": cfg["constraints"]["per_asset_cap"]},
        "modules": dict(cfg["modules"]),
        "performance": performance,
        "data": {
            "prices_as_of": market.last_price_date.isoformat(),
            # FRED's own last valid observation (ANALYSIS_V4 external-
            # review Phase B1), NOT the last price date: `rf_series`
            # itself is ffilled onto the price calendar, so reading its
            # own index.max() here would always report freshness
            # through today's close regardless of how stale the
            # underlying FRED series actually is.
            "rf_as_of": market.rf_true_last_date.isoformat(),
        },
        "config_hash": _config_hash(cfg),
    }

    # 4. Persist (skip entirely on dry_run).
    if not dry_run:
        execution_date = _next_trading_day(as_of)
        new_state = dict(current_state)
        new_state["initialized"] = True
        new_state["weights"] = {
            t: float(v) for t, v in _round_weights(executed_weights).items()
        }
        # Anchor at the DECISION date (Friday), holding POST-execution
        # weights -- see the module docstring for why this is the anchor
        # drift_weights expects next week.
        new_state["weights_as_of"] = as_of.isoformat()
        new_state["last_decision_date"] = as_of.isoformat()
        new_state["last_execution_date"] = execution_date.isoformat()
        new_state["run_count"] = int(current_state.get("run_count", 0)) + 1
        new_state["last_run_utc"] = run_utc.isoformat()
        new_state["last_status"] = STATUS_OK
        # This decision's one-way turnover, for NEXT week's net-of-cost
        # book_return (Phase B3): the trade sized here executes at the
        # START of next week's reconciliation window, so that window's
        # cost basis is THIS week's executed turnover, not its own.
        new_state["last_one_way_executed"] = float(sizing.executed_one_way)
        state_mod.save_state(new_state, state_path)

    return RunResult(
        as_of=as_of,
        status=STATUS_OK,
        run_utc=run_utc,
        target_weights=target,
        current_weights=current,
        executed_weights=executed_weights,
        trade=sizing.executed_trade,
        one_way_turnover_wanted=sizing.wanted_one_way,
        one_way_executed=sizing.executed_one_way,
        cap_binding=sizing.cap_binding,
        diagnostics=diagnostics,
        messages=messages,
    )


def _realized_week_return(
    weights: pd.Series, prices: pd.DataFrame, start, end
) -> float | None:
    """Gross realized return of holding `weights` (fully invested,
    drifting) from `start` to `end`, via the price ratio. Returns None
    if either date is absent. Measurement only -- for the report's
    context curve, never part of any decision."""
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    if start not in prices.index or end not in prices.index:
        return None
    ratio = prices.loc[end] / prices.loc[start]
    aligned = weights.reindex(prices.columns).fillna(0.0)
    return float((aligned * ratio).sum()) - 1.0


def _config_hash(cfg: dict) -> str:
    """Short, stable digest of the config that produced a run, for the
    report footer's provenance line."""
    blob = json.dumps(cfg, sort_keys=True, default=str)
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:12]


def _round_weights(weights: pd.Series, dp: int = 6) -> pd.Series:
    """Round a weight vector to `dp` decimals for display/persistence,
    then renormalize so the sum is unaffected.

    SLSQP residuals like `4.5e-17` round to exactly 0, which is the
    point; but rounding every element independently can drift the sum
    by up to `len(weights) * 0.5 * 10**-dp`, which would fail
    `state.validate_state`'s sum-to-1 tolerance for a ~20-asset book.
    Renormalizing after rounding keeps the sum exact without reversing
    the cosmetic fix (the SLSQP noise this targets is many orders of
    magnitude below the renormalization's own effect on each weight).
    """
    rounded = weights.round(dp)
    total = rounded.sum()
    return rounded / total if total > 1e-12 else rounded


def _next_trading_day(as_of: pd.Timestamp) -> pd.Timestamp:
    """The expected execution day: the next business day after `as_of`.

    Informational metadata only (`last_execution_date`); the drift
    anchor is `weights_as_of` = the decision date, not this. Uses a
    plain business-day offset, so a holiday-shifted real execution day
    can differ by a day without affecting any reconciliation.
    """
    return (as_of + pd.offsets.BDay(1)).normalize()


def write_run_outputs(
    result: RunResult, base_dir: str | Path | None = None
) -> Path | None:
    """Write a STATUS_OK run's deliverables and append a history row.

    Creates `live/weekly/<as_of>/` with allocation.csv (executed
    weights), trades.csv (current/target/executed), and
    diagnostics.json, and appends one summary row to `live/history.csv`.
    A non-ok result writes only the history row (no weekly directory,
    since there are no weights) -- EXCEPT `STATUS_ALREADY_RUN`, which is
    a TRUE no-op (external-review Phase B2): that outcome means this
    `as_of` was already decided and committed by an earlier run (e.g.
    Saturday's cron), so a later run for the same date (e.g. the Monday
    holiday-catch-up cron) must touch NOTHING -- no history row, no
    weekly directory -- so the scheduler's `git diff --cached --quiet`
    finds no changes to commit. Returns the weekly directory, or None.
    """
    base = Path(base_dir) if base_dir is not None else LIVE_DIR
    base.mkdir(parents=True, exist_ok=True)

    if result.status == STATUS_ALREADY_RUN:
        return None

    as_of_label = (
        result.as_of.date().isoformat() if result.as_of else "unknown"
    )
    _append_history(result, base, as_of_label)

    if result.status != STATUS_OK:
        return None

    weekly_dir = base / "weekly" / as_of_label
    weekly_dir.mkdir(parents=True, exist_ok=True)

    alloc = _round_weights(result.executed_weights).rename("weight")
    alloc.index.name = "ticker"
    alloc.to_csv(weekly_dir / "allocation.csv")

    # current/target are weight vectors (round + renormalize); trade is
    # a delta with no sum-to-1 constraint, so it is only rounded.
    trades = pd.DataFrame(
        {
            "current": _round_weights(result.current_weights),
            "target": _round_weights(result.target_weights),
            "trade": result.trade.round(6),
        }
    )
    trades.index.name = "ticker"
    trades.to_csv(weekly_dir / "trades.csv")

    with open(weekly_dir / "diagnostics.json", "w", encoding="utf-8") as f:
        json.dump(result.diagnostics, f, indent=2, sort_keys=False)

    return weekly_dir


def _append_history(result: RunResult, base: Path, as_of_label: str) -> None:
    perf = result.diagnostics.get("performance", {})
    row = {
        "as_of": as_of_label,
        "status": result.status,
        "run_utc": (
            result.run_utc.isoformat()
            if result.run_utc is not None
            else datetime.now(timezone.utc).isoformat()
        ),
        "one_way_wanted": result.one_way_turnover_wanted,
        "one_way_executed": result.one_way_executed,
        "cap_binding": result.cap_binding,
        # Realized weekly returns accumulate into the report's context
        # equity curve (None on funding / non-ok weeks). `book_return`
        # is gross; `book_return_net` (Phase B3) subtracts the prior
        # week's execution cost using backtest.run's own cost formula --
        # kept alongside, not replacing, the gross column.
        "book_return": perf.get("book_return"),
        "book_return_net": perf.get("book_return_net"),
        "permanent_return": perf.get("permanent_return"),
        "messages": " | ".join(result.messages),
    }
    history_path = base / "history.csv"
    header = not history_path.exists()
    pd.DataFrame([row]).to_csv(
        history_path, mode="a", header=header, index=False
    )
