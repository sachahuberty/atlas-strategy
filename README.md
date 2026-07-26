# ATLAS — All-Techniques Layered Allocation Strategy

Weekly-rebalanced, regime-aware, multi-signal global asset allocation strategy.
Built as a personal portfolio project on the IE MBDS Algorithmic Trading course material.

**Architecture in one line:** every course technique produces a *view* (regime posture from HMM/K-Means,
mean-reversion z-scores, technical levels, news sentiment) → Black-Litterman fuses views with equilibrium
returns → one SLSQP optimizer produces the portfolio → an autoencoder anomaly score can override to
defensive → everything validated weekly out-of-sample with walk-forward, transaction costs, and an
ablation study per signal.

See `PROJECT_STRUCTURE.md` for the full design document.

## Status

- [x] Stage 1 — scaffold, data layer, universe screener, metrics
- [x] Stage 2 — classical allocations + weekly backtester (baseline)
- [x] Stage 3 — regime detection (HMM primary) + posture switching
- [x] Stage 4 — autoencoder anomaly override
- [x] Stage 5 — mean-reversion views
- [x] Stage 6 — technical/options views + execution timing
- [x] Stage 7 — sentiment views
- [x] Stage 8 — Black-Litterman fusion
- [x] Stage 9 — full walk-forward validation
- [ ] Stage 10 — Monte Carlo + stress testing
- [ ] Stage 11 — ablation study + final report

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
pip install -e .
set FRED_API_KEY=your_key_here  # free key: https://fred.stlouisfed.org/docs/api/api_key.html
pytest
```

## Results

Out-of-sample, cost-inclusive, 2022-01 to 2026-07 (notebook 09), frozen
config after stage 9's IS-only grid search (`meanreversion.lookback_days=20`,
`rebalance.no_trade_band=0.01`, `rebalance.max_weekly_turnover=0.01`):

| Strategy | Sharpe | Max drawdown | Ann. vol |
| --- | --- | --- | --- |
| permanent | 1.10 | -12.6% | 8.3% |
| black_litterman (frozen) | 0.86 | -10.4% | 5.3% |
| risk_parity | 0.84 | -16.5% | 9.8% |
| hrp | 0.82 | -9.2% | 3.6% |
| gmv | 0.74 | -8.1% | 3.2% |
| max_sharpe_static | 0.62 | -13.6% | 5.5% |
| sixty_forty | 0.58 | -22.5% | 11.5% |

The naive `permanent` benchmark still posts the best OOS Sharpe, as it
has since stage 8. Grid-searching mean-reversion and turnover
parameters IS-only, selected for cross-fold stability rather than
peak IS performance, did not improve OOS Sharpe over stage 8's
untuned defaults (0.86 vs. 0.895) -- reported as found, per this
project's overfitting-defense discipline. Full per-quarter breakdown
and methodology notes in `notebooks/09_strategy_backtest.ipynb`.
