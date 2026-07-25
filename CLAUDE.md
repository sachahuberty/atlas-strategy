# CLAUDE.md — ATLAS Strategy

## First things first

**Read `PROJECT_STRUCTURE.md` before doing any work.** It is the single source of truth for architecture, module responsibilities, notebook plan, and validation rules. If a request conflicts with it, flag the conflict instead of silently diverging.

## What this project is

ATLAS (All-Techniques Layered Allocation Strategy): a weekly-rebalanced, regime-aware, multi-signal global asset allocation strategy. Signals (regime posture, mean reversion, technicals/options, sentiment) become Black-Litterman views; one SLSQP optimizer produces the portfolio; an autoencoder anomaly score can override to defensive. Everything is judged out-of-sample.

## Build discipline

- Follow the **build order in PROJECT_STRUCTURE.md §11 strictly**. One stage per session. Every stage must end runnable.
- Do not start a later stage while the current stage's tests fail.
- Commit at the end of each stage with a descriptive message (`stage 2: classical allocations + weekly backtester`).

## Code conventions

- **All logic lives in `src/atlas/`.** Notebooks are thin narrative drivers that import from the package. Never copy-paste logic between notebooks.
- **Config-driven everything.** No hardcoded tickers, dates, thresholds, costs, or model params — they belong in `config/config.yaml`. Reaching for a magic number means adding a config key.
- Allocation functions return **`pd.Series` of weights indexed by ticker** (course convention from S5). They must satisfy: weights ≥ 0, sum to 1.
- Type hints and docstrings on all public functions. Use `dataclasses` for structured results (e.g., backtest output).
- Python ≥ 3.11, editable install: `pip install -e .`

## Data & leakage rules (non-negotiable)

- **No lookahead:** a decision dated Friday may only use data available at that Friday's close. Signals lag appropriately (sentiment lagged 1 day). Orders execute the next trading day.
- **Chronological splits before scaling.** Scalers/models fit on training windows only; persist per walk-forward fold in `models/`.
- **Regime labels are remapped to postures by cluster profile** (vol/return characteristics), never by cluster index, at every re-fit.
- **Grid search in-sample only.** Selection criterion = stability across IS validation folds, not peak Sharpe. The frozen config runs through OOS walk-forward exactly once. Never tune anything after seeing OOS results.
- yfinance option chains have **no history** — the options-positioning signal is collected live only and excluded from historical backtests (price-level technicals backtest fine).

## Testing

- Run `pytest` after every change to `src/atlas/`. Add tests when adding modules.
- Priority test areas: metrics correctness (known-answer tests), weight constraints (sum, bounds, caps), backtest accounting (costs, turnover, no-lookahead), view construction shapes (P/Q/Ω).
- A cheap but crucial test: shift input prices by one day and confirm strategy decisions change accordingly (lookahead canary).

## Performance claims

- Only out-of-sample, cost-inclusive results count. In-sample numbers are for diagnostics only and must be labeled as such in any output.
- Every signal module has an on/off flag in config; the ablation study (notebook 11) measures marginal OOS contribution per module.

## Environment notes

- Windows + OneDrive-synced folder: if git behaves oddly, suspect OneDrive sync. Prefer small, frequent commits.
- Data downloads are cached in `data/raw/` (parquet, dated filenames); never re-download inside a loop.
- FRED API key expected in environment variable `FRED_API_KEY` (never commit keys).
