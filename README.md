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
- [ ] Stage 3 — regime detection (HMM primary) + posture switching
- [ ] Stage 4 — autoencoder anomaly override
- [ ] Stage 5 — mean-reversion views
- [ ] Stage 6 — technical/options views + execution timing
- [ ] Stage 7 — sentiment views
- [ ] Stage 8 — Black-Litterman fusion
- [ ] Stage 9 — full walk-forward validation
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

_(populated after stage 9 — out-of-sample, cost-inclusive numbers only)_
