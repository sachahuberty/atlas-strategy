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

import json
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from . import (
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
    """Everything the weekly decision needs from the data layer."""

    returns: pd.DataFrame
    prices: pd.DataFrame
    rf_series: pd.Series
    class_bucket: pd.Series
    tickers: list[str]
    last_price_date: pd.Timestamp


@dataclass
class RunResult:
    """The full record of one weekly run: the decision, the sized
    trade, and the diagnostics needed to explain it after the fact."""

    as_of: pd.Timestamp | None
    status: str
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
    date. The current, still-in-progress ISO week is excluded: a job
    running on Wednesday must decide for last week's close, not for a
    half-finished week.
    """
    reference = (reference or pd.Timestamp.today()).normalize()
    ref_year, ref_week, _ = reference.isocalendar()
    candidates = [
        d
        for d in backtest.week_end_dates(calendar)
        if (d.isocalendar().year, d.isocalendar().week) != (ref_year, ref_week)
        and d <= reference
    ]
    if not candidates:
        raise HeldError(
            "No completed week-end trading day is available in the data yet."
        )
    return max(candidates)


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


def _empty_result(status: str, as_of, messages: list[str]) -> RunResult:
    as_of_ts = pd.Timestamp(as_of) if as_of is not None else None
    return RunResult(as_of=as_of_ts, status=status, messages=messages)


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

        already = current_state.get("last_decision_date")
        if already == resolved_as_of.isoformat():
            return _empty_result(
                STATUS_ALREADY_RUN,
                resolved_as_of,
                [f"Already decided for {resolved_as_of.date()}; no re-trade."],
            )

        return _decide(
            current_state, market, cfg, posture_cfg, resolved_as_of,
            dry_run, state_path,
        )
    except HeldError as exc:
        return _empty_result(STATUS_HELD, resolved_as_of, [str(exc)])
    except Exception:  # noqa: BLE001 -- terminal guard: never mutate on error
        return _empty_result(
            STATUS_FAILED, resolved_as_of, [traceback.format_exc()]
        )


def _decide(
    current_state: dict,
    market: MarketData,
    cfg: dict,
    posture_cfg: dict,
    as_of: pd.Timestamp,
    dry_run: bool,
    state_path,
) -> RunResult:
    """Reconcile -> decide -> size -> persist. Returns a STATUS_OK
    result. Raises only on genuinely unexpected errors (handled by the
    caller's terminal guard)."""
    tickers = market.returns.columns
    messages: list[str] = []

    # 1. Reconcile: drift last week's stored weights forward to as_of.
    if current_state.get("initialized"):
        stored = state_mod.weights_series(current_state)
        anchor = pd.Timestamp(current_state["weights_as_of"])
        current = state_mod.drift_weights(
            stored, market.returns, anchor, as_of
        )
        current = current.reindex(tickers).fillna(0.0)
        is_funding = False
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
    regime = bl_diag.get("regime", {})
    if "current_state" not in regime:
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
    }

    # 4. Persist (skip entirely on dry_run).
    if not dry_run:
        execution_date = _next_trading_day(as_of)
        new_state = dict(current_state)
        new_state["initialized"] = True
        new_state["weights"] = {
            t: float(v) for t, v in executed_weights.items()
        }
        # Anchor at the DECISION date (Friday), holding POST-execution
        # weights -- see the module docstring for why this is the anchor
        # drift_weights expects next week.
        new_state["weights_as_of"] = as_of.isoformat()
        new_state["last_decision_date"] = as_of.isoformat()
        new_state["last_execution_date"] = execution_date.isoformat()
        new_state["run_count"] = int(current_state.get("run_count", 0)) + 1
        new_state["last_run_utc"] = datetime.now(timezone.utc).isoformat()
        new_state["last_status"] = STATUS_OK
        state_mod.save_state(new_state, state_path)

    return RunResult(
        as_of=as_of,
        status=STATUS_OK,
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
    since there are no weights). Returns the weekly directory, or None.
    """
    base = Path(base_dir) if base_dir is not None else LIVE_DIR
    base.mkdir(parents=True, exist_ok=True)

    as_of_label = (
        result.as_of.date().isoformat() if result.as_of else "unknown"
    )
    _append_history(result, base, as_of_label)

    if result.status != STATUS_OK:
        return None

    weekly_dir = base / "weekly" / as_of_label
    weekly_dir.mkdir(parents=True, exist_ok=True)

    alloc = result.executed_weights.rename("weight")
    alloc.index.name = "ticker"
    alloc.to_csv(weekly_dir / "allocation.csv")

    trades = pd.DataFrame(
        {
            "current": result.current_weights,
            "target": result.target_weights,
            "trade": result.trade,
        }
    )
    trades.index.name = "ticker"
    trades.to_csv(weekly_dir / "trades.csv")

    with open(weekly_dir / "diagnostics.json", "w", encoding="utf-8") as f:
        json.dump(result.diagnostics, f, indent=2, sort_keys=False)

    return weekly_dir


def _append_history(result: RunResult, base: Path, as_of_label: str) -> None:
    row = {
        "as_of": as_of_label,
        "status": result.status,
        "run_utc": datetime.now(timezone.utc).isoformat(),
        "one_way_wanted": result.one_way_turnover_wanted,
        "one_way_executed": result.one_way_executed,
        "cap_binding": result.cap_binding,
        "messages": " | ".join(result.messages),
    }
    history_path = base / "history.csv"
    header = not history_path.exists()
    pd.DataFrame([row]).to_csv(
        history_path, mode="a", header=header, index=False
    )
