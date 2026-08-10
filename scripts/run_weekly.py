"""Thin CLI around `atlas.live.run_once` for the weekly scheduled job.

Prints a readable summary (posture, selected book, top holdings, the
trade list, turnover vs cap) and exits with the PART-3 status code:
0 ok / already-run, 1 failed, 2 held. On a non-dry run it writes
live/weekly/<as_of>/{allocation,trades,diagnostics}, appends a row to
live/history.csv (via `atlas.live.write_run_outputs`), renders the HTML
report, and drops a stable copy at live/report_latest.html for the CI
artifact. It also prints a machine-readable `ATLAS_STATUS=<status>` line
the scheduler greps to tell traded / held / failed / already-run apart.

Examples:
    python scripts/run_weekly.py
    python scripts/run_weekly.py --as-of 2026-06-05 --dry-run --verbose
"""

from __future__ import annotations

import argparse
import shutil
import sys

import pandas as pd

from atlas import live, report


def _format_weights(weights: pd.Series, top: int = 8) -> str:
    held = weights[weights > 1e-6].sort_values(ascending=False)
    if held.empty:
        return "    (all cash / empty)"
    lines = [f"    {t:<8} {w:7.2%}" for t, w in held.head(top).items()]
    return "\n".join(lines)


def _format_trades(result: live.RunResult) -> str:
    trade = result.trade
    moved = trade[trade.abs() > 1e-6].sort_values()
    if moved.empty:
        return "    (no trade this week)"
    lines = [f"    {t:<8} {v:+7.2%}" for t, v in moved.items()]
    return "\n".join(lines)


def _print_summary(result: live.RunResult, verbose: bool) -> None:
    as_of = result.as_of.date() if result.as_of is not None else "unknown"
    print(f"ATLAS weekly run | as_of {as_of} | status: {result.status}")

    for msg in result.messages:
        first_line = msg.splitlines()[0] if msg else msg
        print(f"  - {first_line}")

    if result.status != live.STATUS_OK:
        if verbose and result.status == live.STATUS_FAILED and result.messages:
            print(result.messages[0])
        return

    bl = result.diagnostics.get("black_litterman", {})
    posture = bl.get("regime", {}).get("posture")
    gate = bl.get("gate", {})
    turnover = result.diagnostics.get("turnover", {})

    print(f"  posture: {posture}")
    print(
        f"  selected book: {gate.get('selected')} "
        f"(gate={gate.get('type')}, margin={gate.get('utility_margin')})"
    )
    print(
        f"  turnover: wanted {turnover.get('wanted_one_way'):.4f} / "
        f"executed {turnover.get('executed_one_way'):.4f} / "
        f"cap {turnover.get('cap')}"
        + ("  [CAP BINDING]" if result.cap_binding else "")
    )
    print("  target holdings:")
    print(_format_weights(result.target_weights))
    print("  trades (executed):")
    print(_format_trades(result))

    if verbose:
        views = bl.get("views_fired", [])
        print(f"  views fired: {len(views)}")
        for v in views:
            print(
                f"    {v['label']:<20} q={v['q']:+.4f} "
                f"tilt={v['tilt']:+.4f} conf={v['confidence']}"
            )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="ATLAS weekly live runner.")
    parser.add_argument(
        "--as-of",
        type=str,
        default=None,
        help="Decision date YYYY-MM-DD (default: most recent completed "
        "week-end trading day).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute and print the decision without writing any state "
        "or output files.",
    )
    parser.add_argument(
        "--state-path",
        type=str,
        default=None,
        help="Override the state.json path (default: live/state.json).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print the views that fired and the full failure traceback.",
    )
    args = parser.parse_args(argv)

    as_of = pd.Timestamp(args.as_of) if args.as_of else None
    result = live.run_once(
        as_of=as_of, dry_run=args.dry_run, state_path=args.state_path
    )

    _print_summary(result, verbose=args.verbose)

    if not args.dry_run:
        weekly_dir = live.write_run_outputs(result)
        if weekly_dir is not None:
            print(f"  wrote: {weekly_dir}")
        report_path = _render_report(result)
        print(f"  report: {report_path}")

    # Machine-readable status line for the scheduler (distinguishes
    # already_run from ok, which share exit code 0).
    print(f"ATLAS_STATUS={result.status}")
    return result.exit_code


def _load_history() -> pd.DataFrame | None:
    history_path = live.LIVE_DIR / "history.csv"
    if not history_path.exists():
        return None
    try:
        return pd.read_csv(history_path)
    except Exception:  # noqa: BLE001 - a bad history must not block the run
        return None


def _render_report(result: live.RunResult):
    """Render the HTML report (degraded runs included) into the dated
    weekly directory, and copy it to a stable path the CI artifact step
    always finds regardless of as_of or outcome."""
    history = _load_history()
    report_path = report.render_weekly_report(result, history=history)
    latest = live.LIVE_DIR / "report_latest.html"
    latest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(report_path, latest)
    return report_path


if __name__ == "__main__":
    sys.exit(main())
