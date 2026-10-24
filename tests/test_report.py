"""Weekly HTML report tests (live stage 3).

Fast and offline: the report renderer imports only matplotlib/pandas
(no TensorFlow), and these tests drive it with duck-typed RunResult
stand-ins (SimpleNamespace) plus a hand-built diagnostics dict mirroring
`live._decide`'s output, so nothing here runs the decision stack.
"""

from types import SimpleNamespace

import pandas as pd
import pytest

from atlas import report


def _diagnostics():
    tickers = ["IWM", "QQQ", "SLV", "VEA", "BIL"]
    return {
        "as_of": "2026-06-05",
        "is_funding": False,
        "rf": 0.042,
        "black_litterman": {
            "rf": 0.042,
            "delta": 2.5,
            "regime": {
                "posture": "risk_on",
                "current_state": 1,
                "state_to_posture": {0: "risk_off", 1: "risk_on"},
                "state_profiles": {
                    0: {"ann_return": -0.10, "ann_vol": 0.25},
                    1: {"ann_return": 0.15, "ann_vol": 0.12},
                },
                "transition_row": {0: 0.1, 1: 0.9},
            },
            "equilibrium_prior": {t: 0.05 for t in tickers},
            "views_fired": [
                {
                    "label": "V1_regime_equity",
                    "family": "V1",
                    "q": 0.08,
                    "tilt": 0.03,
                    "confidence": 1.0,
                    "support": {"IWM": 0.5, "QQQ": 0.5},
                }
            ],
            "mu_bl": {t: 0.04 for t in tickers},
            "candidate_books": {
                "black_litterman": {
                    "exp_return": 0.0164,
                    "vol": 0.075,
                    "utility": -0.02922,
                    "weights": {t: 0.2 for t in tickers},
                },
                "gmv": {
                    "exp_return": 0.0122,
                    "vol": 0.040,
                    "utility": -0.02944,
                    "weights": {},
                },
                "risk_parity": {
                    "exp_return": 0.0120,
                    "vol": 0.054,
                    "utility": -0.03092,
                    "weights": {},
                },
            },
            "gate": {
                "type": "utility",
                "risk_aversion": 2,
                "selected": "black_litterman",
                "utility_margin": 0.00021,
            },
        },
        "anomaly": {
            "active": True,
            "fired": False,
            "recon_error": 0.312,
            "threshold": 0.501,
            "fitted_at": "2026-05-01",
        },
        "turnover": {
            "wanted_one_way": 0.0302,
            "executed_one_way": 0.02,
            "cap": 0.02,
            "no_trade_band": 0.005,
            "cap_binding": True,
            "skipped": False,
        },
        "class_bucket": {
            "IWM": "equity",
            "QQQ": "equity",
            "SLV": "commodity",
            "VEA": "equity",
            "BIL": "cash",
        },
        "constraints": {"per_asset_cap": 0.20},
        "modules": {
            "regime_view": True,
            "meanreversion_view": True,
            "technical_view": False,
            "sentiment_view": False,
            "momentum_view": False,
            "anomaly_override": True,
        },
        "performance": {
            "book_return": 0.011,
            "permanent_return": 0.008,
            "anchor": "2026-05-29",
        },
        "data": {"prices_as_of": "2026-06-05", "rf_as_of": "2026-06-05"},
        "config_hash": "abc123def456",
    }


def _ok_run(**overrides):
    target = pd.Series(
        {"IWM": 0.2, "QQQ": 0.2, "SLV": 0.2, "VEA": 0.2, "BIL": 0.2}
    )
    current = pd.Series(
        {"IWM": 0.206, "QQQ": 0.186, "SLV": 0.214, "VEA": 0.190, "BIL": 0.204}
    )
    fields = dict(
        as_of=pd.Timestamp("2026-06-05"),
        status="ok",
        target_weights=target,
        current_weights=current,
        executed_weights=target,
        trade=target - current,
        one_way_turnover_wanted=0.0302,
        one_way_executed=0.02,
        cap_binding=True,
        diagnostics=_diagnostics(),
        messages=[
            "Turnover cap binding: wanted 0.0302 one-way, executed 0.0200."
        ],
    )
    fields.update(overrides)
    return SimpleNamespace(**fields)


def _degraded_run(status, message):
    return SimpleNamespace(
        as_of=pd.Timestamp("2026-06-05"),
        status=status,
        target_weights=None,
        current_weights=None,
        executed_weights=None,
        trade=None,
        one_way_turnover_wanted=None,
        one_way_executed=None,
        cap_binding=False,
        diagnostics={},
        messages=[message],
    )


def _history(n, status="ok"):
    dates = pd.bdate_range("2026-01-02", periods=n, freq="W-FRI")
    rng = pd.Series(range(n))
    return pd.DataFrame(
        {
            "as_of": [d.date().isoformat() for d in dates],
            "status": [status] * n,
            "book_return": (0.002 + 0.001 * (rng % 3 - 1)).to_numpy(),
            "permanent_return": (0.0015 + 0.0008 * (rng % 2)).to_numpy(),
        }
    )


def _render(run, history=None, tmp_path=None):
    out = tmp_path / "report.html"
    path = report.render_weekly_report(run, history=history, out_path=out)
    return path.read_text(encoding="utf-8")


# --------------------------------------------------------------------- #
# Happy path
# --------------------------------------------------------------------- #
def test_renders_ok_without_raising(tmp_path):
    html = _render(_ok_run(), tmp_path=tmp_path)
    assert html.startswith("<!DOCTYPE html>")
    assert "Weekly Signal Report" in html
    assert "NOT A TRADE INSTRUCTION" in html


def test_output_is_self_contained_no_network(tmp_path):
    html = _render(_ok_run(), history=_history(10), tmp_path=tmp_path)
    # No external references of any kind: the file must open offline,
    # over email, in five years.
    assert "http://" not in html
    assert "https://" not in html
    assert "data:image/png;base64," in html  # chart embedded, not linked


def test_numbers_match_the_run(tmp_path):
    html = _render(_ok_run(), tmp_path=tmp_path)
    # Top holding weight and executed turnover must appear verbatim.
    assert "20.0%" in html  # each capped position
    assert "2.00%" in html  # executed one-way turnover
    assert "IWM" in html
    # The gate's winning book and its margin are rendered.
    assert "black_litterman" in html
    assert "selected" in html


def test_cap_binding_is_called_out(tmp_path):
    html = _render(_ok_run(), tmp_path=tmp_path)
    assert "turnover cap throttled" in html.lower()
    assert "cap" in html.lower()


# --------------------------------------------------------------------- #
# Degraded paths must still produce a report
# --------------------------------------------------------------------- #
def test_renders_held_run(tmp_path):
    run = _degraded_run("held", "Stale prices; holding this week.")
    html = _render(run, tmp_path=tmp_path)
    assert "HELD" in html
    assert "Stale prices" in html
    assert "http://" not in html


def test_renders_failed_run(tmp_path):
    run = _degraded_run("failed", "Traceback: boom")
    html = _render(run, tmp_path=tmp_path)
    assert "FAILED" in html
    assert "http://" not in html


# --------------------------------------------------------------------- #
# Header shows both the decision date and the run date (ANALYSIS_V4
# Sec 2.7): a decision-lag bug like Sec 2.2's is invisible otherwise.
# --------------------------------------------------------------------- #
def test_header_shows_decision_and_run_date(tmp_path):
    run = _ok_run(run_utc=pd.Timestamp("2026-06-07 02:15", tz="UTC"))
    html = _render(run, tmp_path=tmp_path)
    assert "2026-06-05" in html  # decision date
    assert "2026-06-07 02:15 UTC" in html  # run date


def test_header_omits_run_date_when_absent(tmp_path):
    # Older/degraded RunResult-like objects without run_utc must not
    # break rendering.
    html = _render(_ok_run(), tmp_path=tmp_path)
    assert "Decision: 2026-06-05" in html
    assert '<div class="run-date">' not in html


def test_delta_column_matches_target_current_precision(tmp_path):
    html = _render(_ok_run(), tmp_path=tmp_path)
    # IWM: target 0.2, current 0.206 -> in the Book table's own row,
    # Target/Current render at 1dp (20.0%/20.6%) and Delta must match
    # (-0.6%), not the 2dp -0.60% it used before this fix. Scope the
    # check to the Book table: "The Trade" section renders deltas at
    # its own (unrelated, untouched) precision.
    book_start = html.index("The Book")
    book_end = html.index("The Trade")
    book_html = html[book_start:book_end]
    assert "20.0%" in book_html and "20.6%" in book_html
    assert "-0.6%" in book_html
    assert "-0.60%" not in book_html


def test_missing_diagnostics_keys_do_not_raise(tmp_path):
    # A run with weights but almost-empty diagnostics (e.g. a partial
    # capture) must render "not available", never raise.
    run = _ok_run(diagnostics={"turnover": {}})
    html = _render(run, tmp_path=tmp_path)
    assert "Weekly Signal Report" in html
    assert "not available" in html.lower() or "n/a" in html.lower()


def test_empty_diagnostics_renders(tmp_path):
    run = _ok_run(diagnostics={})
    html = _render(run, tmp_path=tmp_path)
    assert html.startswith("<!DOCTYPE html>")


# --------------------------------------------------------------------- #
# Context: history handling
# --------------------------------------------------------------------- #
def test_short_history_shows_placeholder(tmp_path):
    html = _render(_ok_run(), history=_history(3), tmp_path=tmp_path)
    assert "Insufficient live history" in html
    assert "data:image/png;base64," not in html  # no misleading 3-pt chart


def test_no_history_shows_placeholder(tmp_path):
    html = _render(_ok_run(), history=None, tmp_path=tmp_path)
    assert "Insufficient live history" in html


def test_enough_history_renders_chart(tmp_path):
    html = _render(_ok_run(), history=_history(10), tmp_path=tmp_path)
    assert "data:image/png;base64," in html
    assert "Weeks live" in html
    assert "Current drawdown" in html


# --------------------------------------------------------------------- #
# Context: net-of-cost NAV (external-review Phase B3). The chart must
# plot and label the net series when present, and fall back to an
# honestly-labelled gross reading for history rows that predate the fix.
# --------------------------------------------------------------------- #
def _history_net(n, status="ok"):
    dates = pd.bdate_range("2026-01-02", periods=n, freq="W-FRI")
    rng = pd.Series(range(n))
    gross = (0.002 + 0.001 * (rng % 3 - 1)).to_numpy()
    return pd.DataFrame(
        {
            "as_of": [d.date().isoformat() for d in dates],
            "status": [status] * n,
            "book_return": gross,
            "book_return_net": gross - 0.0002,
            "permanent_return": (0.0015 + 0.0008 * (rng % 2)).to_numpy(),
        }
    )


def test_context_uses_net_and_labels_it_when_available(tmp_path):
    html = _render(_ok_run(), history=_history_net(10), tmp_path=tmp_path)
    assert "data:image/png;base64," in html
    assert "net of 10 bp" in html.lower()
    assert "gross of costs" not in html.lower()


def test_context_falls_back_to_gross_label_for_old_history(tmp_path):
    html = _render(_ok_run(), history=_history(10), tmp_path=tmp_path)
    assert "gross of costs" in html.lower()
    assert "net of 10 bp" not in html.lower()


def test_disabled_view_families_are_explained(tmp_path):
    html = _render(_ok_run(), tmp_path=tmp_path)
    # V3/V4/V5 are off in the fixture; their one-line reasons must show.
    assert "V3 technical" in html
    assert "live-only" in html


def test_returns_written_path(tmp_path):
    run = _ok_run()
    out = tmp_path / "custom.html"
    path = report.render_weekly_report(run, out_path=out)
    assert path == out and out.exists()
