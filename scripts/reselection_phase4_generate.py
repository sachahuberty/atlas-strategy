"""HISTORICAL RECORD -- reselection task Phase 4, the single OOS look.

This is the exact script that generated
`reports/results/reselection_phase4_daily_returns.pkl`. It was run
ONCE, on 2026-11-01 (per this repo's own git commit metadata for the
commit that introduced that pickle, `56503bb`), and it must NOT be
run again: the pre-registration (docs/RESELECTION_PREREG.md) commits
this task to a single OOS look, and a second run -- even an identical
one -- would be a second look at 2022-2026 data, exactly what the
pre-registration rules out.

Committed AS-IS (external review, following a correction task) for
reproducibility/audit: this is what actually produced the committed
daily-return pickle, not a cleaned-up or path-genericized rewrite.
That means it still has the absolute, machine-specific path below
and the informal elapsed-time prints (later found to be unreliable in
this sandboxed environment, per this project's own session notes) --
preserved deliberately rather than tidied, since the point of
committing it is historical fidelity to what ran, not a reusable
library module. Do not import from this file; do not execute it.
Reusable analysis of its OUTPUT lives in
`scripts/reselection_phase4_stats.py`, which reads only the already-
committed pickle.
"""
import copy
import pickle
import sys
import time
from pathlib import Path

ROOT = Path(r"c:/Users/sacha/OneDrive/IE/Master in Business Analytics and Data Science/Term 3/Algo Trading/atlas-strategy")
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd
from atlas import allocation, data, universe, backtest, strategy, meanreversion

cfg = data.load_config()
posture_cfg = data.load_config(ROOT / "config" / "regime_posture.yaml")

as_of_universe = pd.Timestamp(cfg["general"]["is_end_date"])
universe_df = universe.load_universe(as_of_universe)
tickers = universe_df.index.tolist()
class_bucket = universe_df["class_bucket"]

prices = data.download_prices(tickers, start=cfg["general"]["start_date"])
prices = data.align_calendars(prices)
returns = data.daily_returns(prices)
is_end = pd.Timestamp(cfg["general"]["is_end_date"])
is_prices = prices.loc[:is_end]

rf_raw = data.download_macro([cfg["risk_free"]["fred_series"]], start=cfg["general"]["start_date"])
rf_series = (rf_raw[cfg["risk_free"]["fred_series"]] / 100.0).reindex(returns.index).ffill()

meanrev_eligible = meanreversion.hit_rate_eligible_tickers(is_prices, cfg)

lookback = cfg["optimization"]["lookback_days"]
report_start = pd.Timestamp("2014-01-01")
buffer_start_pos = max(0, returns.index.searchsorted(report_start) - lookback)
full_returns = returns.iloc[buffer_start_pos:]
print(
    "run range:", full_returns.index.min().date(), "to",
    full_returns.index.max().date(),
)

results_dir = ROOT / "reports" / "results"
results_dir.mkdir(parents=True, exist_ok=True)
out_path = results_dir / "reselection_phase4_daily_returns.pkl"

out = {}
if out_path.exists():
    with open(out_path, "rb") as f:
        out = pickle.load(f)

# (a) current config
if "current" not in out:
    t0 = time.time()
    bl_fn = strategy.black_litterman_strategy(
        class_bucket, cfg, posture_cfg, rf_series=rf_series,
        meanrev_eligible=meanrev_eligible,
    )
    anomaly_fn = strategy.with_anomaly_override(bl_fn, cfg)
    result_a = backtest.run(anomaly_fn, full_returns, cfg)
    out["current"] = result_a.daily_returns
    with open(out_path, "wb") as f:
        pickle.dump(out, f)
    print("current config done, elapsed(unreliable):", time.time() - t0)
else:
    print("current config already cached")

# (b) Phase 2 IS-selected config: V5 on, turnover 1%/0.5%, else current
phase2_cfg = copy.deepcopy(cfg)
phase2_cfg["modules"]["momentum_view"] = True
phase2_cfg["rebalance"]["max_weekly_turnover"] = 0.01
phase2_cfg["rebalance"]["no_trade_band"] = 0.005

if "phase2" not in out:
    t0 = time.time()
    bl_fn_b = strategy.black_litterman_strategy(
        class_bucket, phase2_cfg, posture_cfg, rf_series=rf_series,
        meanrev_eligible=meanrev_eligible,
    )
    anomaly_fn_b = strategy.with_anomaly_override(bl_fn_b, phase2_cfg)
    result_b = backtest.run(anomaly_fn_b, full_returns, phase2_cfg)
    out["phase2"] = result_b.daily_returns
    with open(out_path, "wb") as f:
        pickle.dump(out, f)
    print("phase2 config done, elapsed(unreliable):", time.time() - t0)
else:
    print("phase2 config already cached")

# (c) permanent, capped like the strategy
if "permanent" not in out:
    cap = cfg["constraints"]["per_asset_cap"]

    def permanent_fn(as_of, window):
        return allocation.permanent(class_bucket, cap=cap)

    result_c = backtest.run(permanent_fn, full_returns, cfg)
    out["permanent"] = result_c.daily_returns
    with open(out_path, "wb") as f:
        pickle.dump(out, f)
    print("permanent done")
else:
    print("permanent already cached")

print("PHASE 4 RUNS DONE")
