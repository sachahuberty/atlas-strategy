"""Weekly HTML report (L6): render one run's diagnostics into a single
self-contained research note a finance professional can read cold.

`render_weekly_report(run_result, history=None, out_path=None)` produces
ONE standalone .html file: no external CSS/JS, no CDN, no network. The
one chart (context equity curve) is embedded as inline SVG. The file
opens correctly from disk, over email, and years later.

This module RENDERS; it never decides. Every number comes from the
`RunResult` and its diagnostics dict (built by `live._decide`); nothing
is recomputed here except display statistics of an already-recorded
series (a curve's drawdown/vol). It degrades gracefully: a held/failed
run still yields a report saying so and showing the last known book, and
a missing diagnostics key renders "not available" rather than raising --
this is the artifact you read WHEN something went wrong, so it must not
be the thing that also breaks.

To avoid importing the (TensorFlow-heavy) decision stack, this module
does NOT import `live`; it treats the run result as a plain object with
the documented attributes and compares `status` against string literals
that mirror `live.STATUS_*`.

Public API:
    render_weekly_report(run_result, history=None, out_path=None) -> Path
"""

from __future__ import annotations

import base64
import io
import os
import subprocess
from datetime import datetime, timezone
from html import escape
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib

matplotlib.use("Agg")  # headless: no display, deterministic SVG output
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from . import plotting  # noqa: E402
from . import state as state_mod  # noqa: E402

if TYPE_CHECKING:  # pragma: no cover - typing only, no runtime TF import
    from .live import RunResult

# Mirror of live.STATUS_* (imported by literal to keep this module free
# of the decision stack; see module docstring).
_STATUS_OK = "ok"
_STATUS_HELD = "held"
_STATUS_FAILED = "failed"
_STATUS_ALREADY_RUN = "already_run"

_MIN_CONTEXT_WEEKS = 8

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LIVE_DIR = PROJECT_ROOT / "live"

# Display names for the fixed ETF universe (presentation only; a ticker
# not listed simply shows as itself).
_NAMES = {
    "ACWI": "iShares MSCI ACWI",
    "EFA": "iShares MSCI EAFE",
    "IWM": "iShares Russell 2000",
    "QQQ": "Invesco QQQ Trust",
    "SPY": "SPDR S&P 500",
    "VEA": "Vanguard FTSE Developed Mkts",
    "VTI": "Vanguard Total Stock Market",
    "VWO": "Vanguard FTSE Emerging Mkts",
    "AGG": "iShares Core US Aggregate Bond",
    "BND": "Vanguard Total Bond Market",
    "BSV": "Vanguard Short-Term Bond",
    "HYG": "iShares iBoxx HY Corp Bond",
    "IEF": "iShares 7-10Y Treasury",
    "LQD": "iShares iBoxx IG Corp Bond",
    "SHY": "iShares 1-3Y Treasury",
    "TIP": "iShares TIPS Bond",
    "TLT": "iShares 20+Y Treasury",
    "DBC": "Invesco DB Commodity",
    "GLD": "SPDR Gold Shares",
    "SLV": "iShares Silver Trust",
    "USO": "United States Oil Fund",
    "BIL": "SPDR 1-3 Month T-Bill",
    "VNQ": "Vanguard Real Estate",
}

_BUCKET_LABEL = {
    "equity": "Equity",
    "fixed_income": "Fixed income",
    "commodity": "Commodity",
    "cash": "Cash",
    "real_estate": "Real estate",
}

_POSTURE = {
    "risk_on": ("Risk-on", "#1a7f4b"),
    "neutral": ("Neutral", "#5b6470"),
    "risk_off": ("Risk-off", "#b3261e"),
}

_STATUS_STYLE = {
    _STATUS_OK: ("TRADED", "#1a7f4b"),
    _STATUS_HELD: ("HELD", "#b8860b"),
    _STATUS_FAILED: ("FAILED", "#b3261e"),
    _STATUS_ALREADY_RUN: ("ALREADY RUN", "#5b6470"),
}

# One-line reasons for the disabled signal families (paraphrasing the
# config comments; presentation text only).
_DISABLED_REASON = {
    "technical_view": "V3 technical levels: disabled, measured -0.265 "
    "OOS Sharpe (largest negative in the ablation).",
    "sentiment_view": "V4 sentiment: live-only, no historical archive to "
    "backtest against.",
    "momentum_view": "V5 dual momentum: disabled, measured -0.236 OOS "
    "Sharpe once fused through Black-Litterman.",
}
_VIEW_MODULE = {
    "regime_view": "V1 regime",
    "meanreversion_view": "V2 mean-reversion",
    "technical_view": "V3 technical",
    "sentiment_view": "V4 sentiment",
    "momentum_view": "V5 momentum",
}


# --------------------------------------------------------------------- #
# Small formatting helpers
# --------------------------------------------------------------------- #
def _pct(x, dp: int = 1) -> str:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return "n/a"
    return f"{x * 100:.{dp}f}%"


def _signed_pct(x, dp: int = 2) -> str:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return "n/a"
    return f"{x * 100:+.{dp}f}%"


def _num(x, dp: int = 4) -> str:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return "n/a"
    return f"{x:.{dp}f}"


def _name_of(ticker: str) -> str:
    return _NAMES.get(ticker, ticker)


def _card(title: str, body: str, note: str | None = None) -> str:
    head = f"<h2>{escape(title)}</h2>"
    if note:
        head += f"<p class='note'>{escape(note)}</p>"
    return f"<section class='card'>{head}{body}</section>"


def _series_from(obj) -> pd.Series | None:
    if obj is None:
        return None
    if isinstance(obj, pd.Series):
        return obj
    try:
        return pd.Series(obj, dtype=float)
    except Exception:  # noqa: BLE001
        return None


# --------------------------------------------------------------------- #
# Sections
# --------------------------------------------------------------------- #
def _header(run) -> str:
    as_of = (
        run.as_of.date().isoformat() if run.as_of is not None else "unknown"
    )
    status = run.status
    label, color = _STATUS_STYLE.get(status, (status.upper(), "#5b6470"))
    diag = run.diagnostics or {}
    posture = diag.get("black_litterman", {}).get("regime", {}).get("posture")
    plabel, pcolor = _POSTURE.get(posture, ("Posture n/a", "#5b6470"))

    messages = ""
    if run.messages:
        items = "".join(
            f"<li>{escape(str(m).splitlines()[0])}</li>" for m in run.messages
        )
        messages = f"<ul class='messages'>{items}</ul>"

    return f"""
    <header>
      <div class="titlerow">
        <div>
          <div class="eyebrow">ATLAS Systematic Allocation</div>
          <h1>Weekly Signal Report</h1>
        </div>
        <div class="asof">
          <div class="asof-date">{escape(as_of)}</div>
          <div class="badges">
            <span class="badge" style="--c:{color}">{escape(label)}</span>
            <span class="badge" style="--c:{pcolor}">{escape(plabel)}</span>
          </div>
        </div>
      </div>
      <div class="disclaimer">
        SIGNAL REPORT - NOT A TRADE INSTRUCTION. Model-driven weekly
        target allocation for review. No discretionary overrides.
      </div>
      {messages}
    </header>
    """


def _last_known_weights(run) -> pd.Series | None:
    """Best-effort last committed book, for degraded (held/failed) runs
    that produced no new weights."""
    try:
        st = state_mod.load_state()
        if st.get("initialized"):
            return state_mod.weights_series(st)
    except Exception:  # noqa: BLE001 - report must never break on this
        return None
    return None


def _book_section(run) -> str:
    diag = run.diagnostics or {}
    buckets = diag.get("class_bucket", {})
    cap = diag.get("constraints", {}).get("per_asset_cap")

    target = _series_from(run.target_weights)
    current = _series_from(run.current_weights)

    if target is None:
        fallback = _last_known_weights(run)
        if fallback is None:
            return _card(
                "The Book",
                "<p class='muted'>No allocation available for this run "
                "and no prior book on file.</p>",
            )
        note = (
            "No new decision this week; showing the last committed "
            "allocation, which stands."
        )
        table = _weights_table(fallback, None, buckets, cap)
        return _card("The Book", table, note)

    body = _weights_table(target, current, buckets, cap)
    body += _bucket_summary(target, buckets)
    return _card("The Book", body)


def _weights_table(target, current, buckets, cap) -> str:
    rows = []
    held = target[target > 1e-6].sort_values(ascending=False)
    at_cap = cap is not None
    for ticker, w in held.items():
        cur = float(current.get(ticker, 0.0)) if current is not None else None
        delta = (w - cur) if cur is not None else None
        capped = at_cap and w >= cap - 1e-6
        cls = " class='capped'" if capped else ""
        bucket = _BUCKET_LABEL.get(str(buckets.get(ticker)), "-")
        cur_cell = _pct(cur) if current is not None else "n/a"
        delta_cell = _signed_pct(delta) if delta is not None else "n/a"
        flag = " <span class='cap-flag'>cap</span>" if capped else ""
        rows.append(
            f"<tr{cls}><td class='tk'>{escape(ticker)}{flag}</td>"
            f"<td class='nm'>{escape(_name_of(ticker))}</td>"
            f"<td>{escape(bucket)}</td>"
            f"<td class='num'>{_pct(w)}</td>"
            f"<td class='num'>{cur_cell}</td>"
            f"<td class='num'>{delta_cell}</td></tr>"
        )
    note = ""
    if at_cap:
        note = (
            f"<p class='note'>Positions marked <b>cap</b> sit at the "
            f"{_pct(cap, 0)} per-asset ceiling, which binds most weeks.</p>"
        )
    return (
        "<table class='book'><thead><tr>"
        "<th>Ticker</th><th>Name</th><th>Bucket</th>"
        "<th class='num'>Target</th><th class='num'>Current</th>"
        "<th class='num'>Delta</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>{note}"
    )


def _bucket_summary(target, buckets) -> str:
    by_bucket: dict[str, float] = {}
    for ticker, w in target.items():
        b = str(buckets.get(ticker, "unknown"))
        by_bucket[b] = by_bucket.get(b, 0.0) + float(w)
    order = ["equity", "fixed_income", "commodity", "cash", "real_estate"]
    ordered = [b for b in order if b in by_bucket]
    ordered += [b for b in by_bucket if b not in order]
    cells = "".join(
        f"<tr><td>{escape(_BUCKET_LABEL.get(b, b))}</td>"
        f"<td class='num'>{_pct(by_bucket[b])}</td></tr>"
        for b in ordered
    )
    return (
        "<h3>By asset class</h3>"
        "<table class='bucket'><tbody>"
        f"{cells}</tbody></table>"
    )


def _trade_section(run) -> str:
    diag = run.diagnostics or {}
    turn = diag.get("turnover", {})
    trade = _series_from(run.trade)

    if trade is None:
        return _card(
            "The Trade",
            "<p class='muted'>No trade generated this run.</p>",
        )

    moved = trade[trade.abs() > 1e-6].sort_values()
    if moved.empty:
        trade_rows = (
            "<tr><td colspan='3' class='muted'>No trade this week "
            "(within the no-trade band).</td></tr>"
        )
    else:
        trade_rows = "".join(
            f"<tr><td class='tk'>{escape(t)}</td>"
            f"<td class='nm'>{escape(_name_of(t))}</td>"
            f"<td class='num {'pos' if v > 0 else 'neg'}'>"
            f"{_signed_pct(v)}</td></tr>"
            for t, v in moved.items()
        )

    wanted = turn.get("wanted_one_way")
    executed = turn.get("executed_one_way")
    cap = turn.get("cap")
    throttled = bool(turn.get("cap_binding"))
    skipped = bool(turn.get("skipped"))

    if throttled:
        verdict = (
            "<p class='callout warn'>The turnover cap throttled this "
            f"trade: the model wanted {_pct(wanted, 2)} one-way but only "
            f"{_pct(executed, 2)} (the {_pct(cap, 0)} cap) executed. The "
            "book will keep converging toward target over the coming "
            "weeks.</p>"
        )
    elif skipped:
        verdict = (
            "<p class='callout'>Trade skipped: the desired move "
            f"({_pct(wanted, 2)}) was inside the no-trade band. The book "
            "holds.</p>"
        )
    else:
        verdict = (
            "<p class='callout'>The full desired trade executed; the cap "
            "did not bind this week.</p>"
        )

    turn_table = (
        "<table class='turn'><tbody>"
        f"<tr><td>Wanted (one-way)</td><td class='num'>{_pct(wanted, 2)}"
        "</td></tr>"
        f"<tr><td>Executed (one-way)</td><td class='num'>{_pct(executed, 2)}"
        "</td></tr>"
        f"<tr><td>Cap</td><td class='num'>{_pct(cap, 0)}</td></tr>"
        "</tbody></table>"
    )

    body = (
        "<table class='trade'><thead><tr><th>Ticker</th><th>Name</th>"
        "<th class='num'>Change</th></tr></thead>"
        f"<tbody>{trade_rows}</tbody></table>"
        f"{turn_table}{verdict}"
    )
    return _card("The Trade", body)


def _drivers_section(run) -> str:
    diag = run.diagnostics or {}
    bl = diag.get("black_litterman", {})
    parts = [
        _regime_block(bl),
        _views_block(bl, diag.get("modules", {})),
        _gate_block(bl),
        _anomaly_block(diag.get("anomaly", {})),
    ]
    return _card("What Drove It", "".join(parts))


def _regime_block(bl) -> str:
    regime = bl.get("regime", {})
    posture = regime.get("posture", "n/a")
    plabel = _POSTURE.get(posture, (posture, ""))[0]
    profiles = regime.get("state_profiles")
    trans = regime.get("transition_row")
    current = regime.get("current_state")

    detail = ""
    if profiles and trans is not None:
        s2p = regime.get("state_to_posture", {})
        rows = []
        for s in sorted(profiles, key=lambda k: int(k)):
            prof = profiles[s]
            p = trans.get(str(s), trans.get(int(s)))
            rows.append(
                f"<tr><td>State {escape(str(s))}"
                f"{' (now)' if str(s) == str(current) else ''}</td>"
                f"<td>{escape(str(s2p.get(str(s), s2p.get(int(s), '-'))))}"
                "</td>"
                f"<td class='num'>{_signed_pct(prof.get('ann_return'))}</td>"
                f"<td class='num'>{_pct(prof.get('ann_vol'))}</td>"
                f"<td class='num'>{_pct(p) if p is not None else 'n/a'}"
                "</td></tr>"
            )
        detail = (
            "<table class='states'><thead><tr><th>State</th>"
            "<th>Posture</th><th class='num'>Ann. ret</th>"
            "<th class='num'>Ann. vol</th>"
            "<th class='num'>P(next)</th></tr></thead>"
            f"<tbody>{''.join(rows)}</tbody></table>"
            "<p class='note'>P(next) is the HMM transition probability "
            "from the current state.</p>"
        )
    else:
        detail = (
            "<p class='muted'>State detail not available (thin history "
            "or a neutral fallback).</p>"
        )
    head = f"<h3>Regime</h3><p>HMM posture: <b>{escape(plabel)}</b>.</p>"
    return head + detail


def _views_block(bl, modules) -> str:
    fired = bl.get("views_fired")
    if fired is None:
        body = "<p class='muted'>View detail not available.</p>"
    elif not fired:
        body = "<p class='muted'>No views fired this week; the posterior "
        body += "equals the equilibrium prior (neutral book).</p>"
    else:
        rows = "".join(
            f"<tr><td>{escape(str(v.get('family')))}</td>"
            f"<td class='tk'>{escape('/'.join(v.get('support', {}).keys()))}"
            "</td>"
            f"<td class='num'>{_signed_pct(v.get('tilt'))}</td>"
            f"<td class='num'>{_num(v.get('confidence'), 2)}</td></tr>"
            for v in fired
        )
        body = (
            "<table class='views'><thead><tr><th>Family</th><th>Asset(s)"
            "</th><th class='num'>Tilt</th><th class='num'>Confidence"
            "</th></tr></thead>"
            f"<tbody>{rows}</tbody></table>"
        )

    disabled = []
    for key, reason in _DISABLED_REASON.items():
        if modules and modules.get(key) is False:
            disabled.append(f"<li>{escape(reason)}</li>")
    if disabled:
        body += (
            "<p class='note'>Disabled families:</p>"
            f"<ul class='disabled'>{''.join(disabled)}</ul>"
        )
    return f"<h3>Views that fired</h3>{body}"


def _gate_block(bl) -> str:
    gate = bl.get("gate", {})
    books = bl.get("candidate_books")
    selected = gate.get("selected")
    if not books:
        return (
            "<h3>The gate</h3><p class='muted'>Candidate-book detail not "
            "available.</p>"
        )
    rows = []
    for name, b in books.items():
        chosen = name == selected
        cls = " class='chosen'" if chosen else ""
        mark = " <span class='pick'>selected</span>" if chosen else ""
        rows.append(
            f"<tr{cls}><td>{escape(name)}{mark}</td>"
            f"<td class='num'>{_signed_pct(b.get('exp_return'))}</td>"
            f"<td class='num'>{_pct(b.get('vol'))}</td>"
            f"<td class='num'>{_num(b.get('utility'), 5)}</td></tr>"
        )
    margin = gate.get("utility_margin")
    margin_txt = "Winner's margin not available."
    if margin is not None:
        margin_txt = (
            "Winner's utility margin over the runner-up: "
            f"<b>{_num(margin, 5)}</b>."
        )
    return (
        "<h3>The gate (where the decision happens)</h3>"
        "<table class='gate'><thead><tr><th>Candidate book</th>"
        "<th class='num'>Exp. excess ret</th><th class='num'>Vol</th>"
        "<th class='num'>Utility</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
        f"<p class='note'>Gate: {escape(str(gate.get('type')))} "
        f"(risk aversion {escape(str(gate.get('risk_aversion')))}). "
        f"{margin_txt}</p>"
    )


def _anomaly_block(anom) -> str:
    if not anom or anom.get("active") is False:
        reason = anom.get("reason", "inactive") if anom else "not available"
        return (
            "<h3>Anomaly override</h3>"
            f"<p class='muted'>Inactive ({escape(str(reason))}).</p>"
        )
    fired = anom.get("fired")
    err = anom.get("recon_error")
    thr = anom.get("threshold")
    if fired:
        verdict = (
            "<p class='callout warn'>Anomaly FIRED: current market "
            "behaviour is outside the normal range; the book was pulled "
            "toward the defensive minimum-variance allocation.</p>"
        )
    else:
        verdict = (
            "<p class='callout'>Normal: reconstruction error is within "
            "the trained range; no override.</p>"
        )
    return (
        "<h3>Anomaly override</h3>"
        f"<p>Reconstruction error <b>{_num(err, 5)}</b> vs threshold "
        f"<b>{_num(thr, 5)}</b>.</p>{verdict}"
    )


def _context_section(run, history) -> str:
    hist = history
    if hist is None or len(hist) == 0:
        return _card(
            "Context",
            _placeholder(0),
        )
    if "book_return" not in hist.columns:
        return _card("Context", _placeholder(len(hist)))

    ok = hist.copy()
    if "status" in ok.columns:
        ok = ok[ok["status"] == _STATUS_OK]
    ok = ok.dropna(subset=["book_return"])
    n = len(ok)
    if n < _MIN_CONTEXT_WEEKS:
        return _card("Context", _placeholder(n))

    try:
        idx = pd.to_datetime(ok["as_of"])
        book = pd.Series(ok["book_return"].to_numpy(), index=idx)
        curves = {"ATLAS": book}
        has_perm = (
            "permanent_return" in ok.columns
            and ok["permanent_return"].notna().any()
        )
        if has_perm:
            curves["Permanent"] = pd.Series(
                ok["permanent_return"].to_numpy(), index=idx
            )
        chart_img = _equity_png(curves)
        nav = (1.0 + book).cumprod()
        dd = nav / nav.cummax() - 1.0
        realized_vol = float(book.std(ddof=1)) * (52.0 ** 0.5)
        stats = (
            "<table class='ctx'><tbody>"
            f"<tr><td>Weeks live</td><td class='num'>{n}</td></tr>"
            f"<tr><td>Current drawdown</td>"
            f"<td class='num'>{_pct(float(dd.iloc[-1]))}</td></tr>"
            f"<tr><td>Realized vol (annualized)</td>"
            f"<td class='num'>{_pct(realized_vol)}</td></tr>"
            "</tbody></table>"
        )
        return _card(
            "Context",
            f"<div class='chart'>{chart_img}</div>{stats}",
            note="Realized, gross of costs, since the live book was funded.",
        )
    except Exception as exc:  # noqa: BLE001
        return _card(
            "Context",
            f"<p class='muted'>Context chart unavailable: "
            f"{escape(str(exc))}</p>",
        )


def _placeholder(n_weeks: int) -> str:
    return (
        "<p class='muted'>Insufficient live history to plot an equity "
        f"curve ({n_weeks} completed week(s); need "
        f"{_MIN_CONTEXT_WEEKS}). This panel populates automatically once "
        "the live book has run long enough.</p>"
    )


def _equity_png(curves: dict[str, pd.Series]) -> str:
    """Render the context equity curve as a base64 PNG <img>. PNG (not
    SVG) so the file carries ZERO http(s) references -- an SVG embeds
    xmlns/DTD/RDF namespace URIs that, while never fetched, would fail a
    strict self-containment check. 160 dpi keeps it crisp when a manager
    prints the note to PDF."""
    fig = plotting.equity_curves(
        curves,
        title="Growth of $1 since inception (live)",
        ylabel="Growth of $1",
        figsize=(8.2, 3.4),
    )
    for ax in fig.axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    encoded = base64.b64encode(buf.getvalue()).decode("ascii")
    return (
        "<img alt='Live equity curve vs Permanent' "
        f"src='data:image/png;base64,{encoded}'>"
    )


def _footer(run) -> str:
    diag = run.diagnostics or {}
    data = diag.get("data", {})
    cfg_hash = escape(str(diag.get("config_hash", "n/a")))
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    prices_asof = escape(str(data.get("prices_as_of", "n/a")))
    rf_asof = escape(str(data.get("rf_as_of", "n/a")))
    return f"""
    <footer>
      <div class="ftrow">
        <span>config {cfg_hash}</span>
        <span>git {escape(_git_sha())}</span>
        <span>generated {escape(generated)}</span>
      </div>
      <div class="ftrow">
        <span>prices as-of {prices_asof}</span>
        <span>rf as-of {rf_asof}</span>
      </div>
    </footer>
    """


def _git_sha() -> str:
    sha = os.environ.get("GITHUB_SHA")
    if sha:
        return sha[:12]
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short=12", "HEAD"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=5,
        )
        if out.returncode == 0:
            return out.stdout.strip() or "unknown"
    except Exception:  # noqa: BLE001
        pass
    return "unknown"


# --------------------------------------------------------------------- #
# Document shell + public entry point
# --------------------------------------------------------------------- #
def _safe(fn, run, *args) -> str:
    try:
        return fn(run, *args)
    except Exception as exc:  # noqa: BLE001 - a section must never break all
        return _card(
            "Section unavailable",
            f"<p class='muted'>This section could not be rendered: "
            f"{escape(str(exc))}</p>",
        )


def render_weekly_report(
    run_result: "RunResult",
    history: pd.DataFrame | None = None,
    out_path: str | Path | None = None,
) -> Path:
    """Render `run_result` to a single self-contained HTML file and
    return its path.

    `history` is the parsed `live/history.csv` (or None): when it holds
    at least `_MIN_CONTEXT_WEEKS` completed weeks, the Context panel
    plots the live equity curve; otherwise it shows a placeholder.
    `out_path` defaults to `live/weekly/<as_of>/report.html`.
    """
    body = (
        _safe(_header, run_result)
        + _safe(_book_section, run_result)
        + _safe(_trade_section, run_result)
        + _safe(_drivers_section, run_result)
        + _safe(_context_section, run_result, history)
        + _safe(_footer, run_result)
    )
    html = _document(body)

    if out_path is not None:
        out = Path(out_path)
    else:
        label = (
            run_result.as_of.date().isoformat()
            if run_result.as_of is not None
            else "unknown"
        )
        out = LIVE_DIR / "weekly" / label / "report.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    return out


def _document(body: str) -> str:
    return (
        "<!DOCTYPE html><html lang='en'><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width,"
        " initial-scale=1'><title>ATLAS Weekly Signal Report</title>"
        f"<style>{_CSS}</style></head><body><main>{body}</main></body></html>"
    )


_CSS = """
:root{
  --ink:#1c2024; --muted:#6b727c; --line:#e3e6ea; --paper:#ffffff;
  --accent:#2f4b7c; --pos:#1a7f4b; --neg:#b3261e; --cap:#fff6e6;
}
*{box-sizing:border-box}
body{margin:0;background:#f4f5f7;color:var(--ink);
  font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,
  Arial,sans-serif;font-size:14px;line-height:1.5;}
main{max-width:900px;margin:0 auto;padding:32px 28px 64px;
  background:var(--paper);}
header{border-bottom:2px solid var(--ink);padding-bottom:16px;
  margin-bottom:8px;}
.titlerow{display:flex;justify-content:space-between;align-items:flex-start;
  gap:16px;flex-wrap:wrap;}
.eyebrow{font-size:11px;letter-spacing:.14em;text-transform:uppercase;
  color:var(--muted);}
h1{font-size:26px;margin:2px 0 0;font-weight:600;letter-spacing:-.01em;}
.asof{text-align:right;}
.asof-date{font-variant-numeric:tabular-nums;font-weight:600;font-size:15px;}
.badges{margin-top:6px;display:flex;gap:6px;justify-content:flex-end;
  flex-wrap:wrap;}
.badge{display:inline-block;padding:3px 9px;border-radius:3px;font-size:11px;
  font-weight:600;letter-spacing:.06em;color:#fff;background:var(--c);}
.disclaimer{margin-top:14px;font-size:11.5px;letter-spacing:.03em;
  color:var(--muted);border:1px solid var(--line);border-left:3px solid
  var(--accent);padding:8px 12px;background:#fafbfc;}
.messages{margin:10px 0 0;padding-left:18px;color:var(--muted);
  font-size:12.5px;}
.card{padding:22px 0;border-bottom:1px solid var(--line);}
.card:last-of-type{border-bottom:none;}
h2{font-size:16px;margin:0 0 12px;font-weight:600;letter-spacing:.01em;}
h3{font-size:13px;margin:18px 0 8px;font-weight:600;color:var(--accent);
  text-transform:uppercase;letter-spacing:.05em;}
p{margin:8px 0;}
.muted{color:var(--muted);}
.note{color:var(--muted);font-size:12px;margin:6px 0;}
table{width:100%;border-collapse:collapse;margin:8px 0;}
th,td{text-align:left;padding:6px 8px;border-bottom:1px solid var(--line);
  vertical-align:top;}
th{font-size:11px;text-transform:uppercase;letter-spacing:.05em;
  color:var(--muted);font-weight:600;border-bottom:1px solid var(--ink);}
td.num,th.num{text-align:right;font-variant-numeric:tabular-nums;
  font-family:"SFMono-Regular",Consolas,"Liberation Mono",Menlo,monospace;
  white-space:nowrap;}
td.tk{font-weight:600;font-family:"SFMono-Regular",Consolas,monospace;}
td.nm{color:var(--muted);}
tr.capped{background:var(--cap);}
.cap-flag,.pick{font-size:9.5px;font-weight:700;letter-spacing:.04em;
  text-transform:uppercase;color:#8a6d1f;background:#f6e6bf;padding:1px 4px;
  border-radius:2px;margin-left:4px;}
.pick{color:#155e3a;background:#d8efe1;}
tr.chosen{background:#f2f8f4;font-weight:600;}
td.pos{color:var(--pos);}
td.neg{color:var(--neg);}
.bucket,.turn,.ctx{max-width:360px;}
.callout{border-left:3px solid var(--accent);padding:8px 12px;
  background:#fafbfc;font-size:13px;}
.callout.warn{border-left-color:#b8860b;background:#fdf9ef;}
.disabled{margin:4px 0 0;padding-left:18px;color:var(--muted);font-size:12px;}
.chart{margin:8px 0;}
.chart img{max-width:100%;height:auto;display:block;}
footer{margin-top:8px;padding-top:14px;border-top:1px solid var(--line);
  color:var(--muted);font-size:11px;}
.ftrow{display:flex;gap:18px;flex-wrap:wrap;font-variant-numeric:tabular-nums;}
@media print{
  body{background:#fff;font-size:11.5px;}
  main{max-width:none;padding:0;}
  .card{break-inside:avoid;}
  table,.callout,.chart{break-inside:avoid;}
}
"""
