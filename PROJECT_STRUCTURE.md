# ATLAS — All-Techniques Layered Allocation Strategy

**Author:** Sacha Huberty
**Type:** Personal portfolio project (GitHub/CV showcase)
**Basis:** IE MBDS — Algorithmic Trading course (every session's techniques combined into one system)

---

## 1. Project Overview

One integrated **multi-signal global allocation strategy** that uses *every* technique taught in the course, each with a defined, non-overlapping role. The architecture prevents "kitchen-sink" chaos with a single organizing principle:

> **Black-Litterman (S4) is the fusion engine.** Every signal-generating technique from class produces a *view* (an expected-return opinion with a confidence level). Black-Litterman blends those views with market-equilibrium returns into one posterior return vector, which feeds one optimizer. One portfolio comes out.

**The six layers:**

| Layer | What it does | Course techniques used |
|---|---|---|
| L0 — Universe | Rule-based screening of any asset in any global market | S6 fund-selection stats + clustering, S4 smart-beta peer logic |
| L1 — Market state | Classify regime, detect anomalies | S7/S8: HMM (primary), K-Means+PCA macro, Autoencoder+GMM |
| L2 — Signals (views) | Generate return views per asset | S10 mean reversion (OU, z-score), S12 technical levels & options positioning, S6/S14 NLP sentiment, S1/S4 regime→factor tilts |
| L3 — Fusion & optimization | Blend views (Black-Litterman) → optimize | S4 BL, S2 SciPy/SLSQP, S3 efficient frontier & max Sharpe, S1/S3 utility function, S5 GMV/Risk Parity, S4 HRP |
| L4 — Risk management | Override/de-risk, stress the book | S8 autoencoder threshold, S7 stress scenarios, S8 Monte Carlo/GBM |
| L5 — Validation & execution | Weekly backtest, walk-forward, cost realism | S5 backtester, S10 IS/OOS + walk-forward + grid search, S4 turnover constraints, S12 execution timing |

**Rebalance schedule:** weekly — signals computed after the Friday close, orders executed on the first trading day of the following week (no-lookahead preserved).

**Universe:** unrestricted — any asset with a Yahoo Finance ticker, worldwide (equities, ETFs, bonds, commodities, FX, crypto), screened by rules, not hand-picked.

**Benchmarks:** Permanent Portfolio 25/25/25/25 (S4) and global 60/40 (S1/S5).

**Design stance: complexity is welcome, returns must be real.** There is no ceiling on model sophistication — deeper autoencoders, more HMM states, more view families, larger universes are all fair game whenever they improve performance. The only non-negotiable is that performance claims come from out-of-sample results: with this many degrees of freedom, in-sample returns can always be manufactured, so OOS is the only scoreboard.

**Overfitting defense (what makes high returns believable):** every signal module has an on/off flag in `config.yaml`. The final notebook runs an **ablation study** — the strategy with each module disabled — measuring each technique's marginal OOS contribution. When a module underperforms, the response is *iterate or deepen it* (better features, better calibration), not necessarily remove it; modules that still don't contribute are documented honestly (that finding is itself a result, per the S10 overfitting lesson).

---

## 2. Core Components (full class-technique map)

| # | Component | Techniques (session) | Role in ATLAS |
|---|---|---|---|
| 1 | Universe screener | Statistical ratios: 1/3/5Y return, vol, Sharpe (S6); correlation vs equity/FI/cash factors (S6); K-Means asset clustering + representative selection (S6); peer/smart-beta logic (S4) | Build investable universe; de-duplicate correlated assets; classify each asset into an asset-class bucket |
| 2 | Metrics engine | HPR, portfolio return/σ, covariance, correlation (S1); Sharpe, drawdown (S3); Sortino, Calmar, CAGR, hit rate (S10); turnover (S5) | Single source of truth for every stat in the project |
| 3 | Macro regime model | FRED indicators (S8); StandardScaler w/ train-only fit (S7/S8); elbow method + K-Means (S8); PCA 2-D + explained variance (S7/S8); regime-profile heatmap (S8) | Slow structural regime: Expansion / Slowdown / Contraction / Recovery |
| 4 | Market regime model | HMM on returns + rolling vol, transition matrix (S7/S8) | **Primary regime switch** — persistent, sequence-aware |
| 5 | Anomaly detector | Stationary feature engineering (S8); 5-day sequences; sequential autoencoder w/ early stopping (S8); 99th-pct reconstruction-error threshold (S8); latent space + GMM clustering (S7/S8) | Risk-off tripwire + independent regime cross-check |
| 6 | Mean-reversion signal | ADF stationarity test (S10); OU process + half-life (S10); detrended log-price, rolling z-score (S10); vectorized signals, volatility filter (S10) | Per-asset tactical view: stretched assets expected to revert |
| 7 | Technical/positioning signal | Pivot highs/lows (S12); K-Means support/resistance zones (S12); event-study sanity check (S12); option chain, OI notional, call/put walls, gamma proxy (S12) | Per-asset view near key levels + execution timing + positioning-based risk sentiment |
| 8 | Sentiment signal | Web scraping with requests/BeautifulSoup (S14); NLP preprocessing (S14); sentiment analysis (S14); wordcloud + LDA topics for reporting (S14) | Slow-moving directional view on broad asset classes from financial news |
| 9 | View builder + Black-Litterman | Equilibrium returns, views + confidence levels, posterior returns (S4) | Fuse layers 3–8 into one posterior expected-return vector |
| 10 | Optimizers | SciPy SLSQP: objective/constraints/bounds/x0/callback (S2); efficient frontier + max Sharpe + CML (S3); tracking-error minimization (S2); GMV (S5); risk contributions + Risk Parity (S5); HRP dendrogram (S4); utility function U = E[r] − ½Aσ² (S1/S3) | Turn posterior returns into weights; defensive books; final selection by utility |
| 11 | Risk overlays | Anomaly override (S8); regime-conditional caps (S7); historical/sensitivity/reverse stress tests (S7); GBM Monte Carlo, scenario histograms, recovery probability (S8) | De-risking logic + weekly forward-looking risk report |
| 12 | Backtester | Weekly scheduler adapted from S5 block logic; position lagging (S10); transaction costs (S10); max weekly turnover 2% + no-trade band (S4); weight-evolution charts (S5); IS/OOS split, walk-forward, parameter grid (S10) | The referee: every design decision judged OOS |

---

## 3. Directory Structure

```
atlas-strategy/
├── README.md                   # Pitch, architecture diagram, results, how to run
├── PROJECT_STRUCTURE.md        # This document
├── requirements.txt
├── pyproject.toml
├── .gitignore
├── config/
│   ├── config.yaml             # Universe rules, base currency, dates, costs, rebalance,
│   │                           #   module on/off flags, model params, BL confidences
│   └── regime_posture.yaml     # Regime → allocation-posture mapping
├── data/
│   ├── raw/                    # Prices, macro, option chains, scraped headlines (dated)
│   └── processed/              # Returns, features, regime labels, sentiment scores,
│                               #   universe_YYYYMMDD.csv (versioned universes)
├── notebooks/
│   ├── 01_universe_screening.ipynb
│   ├── 02_classical_allocations.ipynb
│   ├── 03_regime_detection.ipynb
│   ├── 04_anomaly_autoencoder.ipynb
│   ├── 05_mean_reversion_signals.ipynb
│   ├── 06_technical_options_signals.ipynb
│   ├── 07_sentiment_altdata.ipynb
│   ├── 08_black_litterman_fusion.ipynb
│   ├── 09_strategy_backtest.ipynb
│   ├── 10_montecarlo_stress.ipynb
│   └── 11_final_report_ablation.ipynb
├── src/
│   └── atlas/
│       ├── __init__.py
│       ├── data.py             # yfinance/fredapi/option-chain download, FX conversion,
│       │                       #   calendar alignment, caching
│       ├── universe.py         # Screening rules, S6 stats ratios, clustering de-dup
│       ├── metrics.py          # All return/risk metrics
│       ├── regimes.py          # HMM (primary), K-Means macro, PCA, label→posture mapping
│       ├── anomaly.py          # Autoencoder build/train/score, GMM latent clustering
│       ├── meanreversion.py    # ADF, OU half-life, z-scores, vol filter
│       ├── technicals.py       # Pivots, K-Means S/R zones, option walls, gamma proxy
│       ├── sentiment.py        # Scraper, NLP pipeline, per-asset-class sentiment score
│       ├── views.py            # Convert each signal into BL views (P, Q, Ω)
│       ├── allocation.py       # black_litterman, max_sharpe, gmv, risk_parity, hrp,
│       │                       #   tracking_error_min, utility, benchmarks
│       ├── risk.py             # Anomaly override, caps, stress scenarios
│       ├── strategy.py         # Weekly decision pipeline (glues everything)
│       ├── backtest.py         # Scheduler, costs, lagging, turnover cap, walk-forward
│       ├── simulation.py       # GBM Monte Carlo
│       └── plotting.py         # Standard chart library
├── models/                     # Autoencoder weights, scalers, HMM params (per fold)
├── reports/
│   ├── figures/                # fig01_... convention
│   └── results/                # Backtest CSVs, ablation table, results_summary.json
└── tests/
    ├── test_metrics.py
    ├── test_allocation.py
    ├── test_views.py
    └── test_backtest.py
```

---

## 4. Notebook Plan

Notebooks are the narrative; all logic lives in `src/atlas/`.

1. **01_universe_screening** — Candidate pool → screens (history ≥ 8y, liquidity, missing-data cap) → S6 statistical ratios (1/3/5Y return, vol, Sharpe) → correlations vs equity/FI/cash factors (S6) → K-Means clustering of assets, keep representatives per cluster (S6) → asset-class bucketing → versioned universe file. Calendar alignment + base-currency conversion demonstrated.
2. **02_classical_allocations** — The full S1–S5 toolkit on the screened universe: efficient frontier via SLSQP (S2/S3), Max Sharpe + CML, utility curves for 3 risk-aversion levels (S3), GMV, Risk Parity with risk-contribution math (S5), HRP dendrogram (S4), tracking-error-minimized benchmark tracker (S2), Permanent + 60/40 baselines. Static comparison table.
3. **03_regime_detection** — HMM on market returns/vol: states, transition matrix, persistence (primary switch). K-Means on FRED macro (elbow, PCA, heatmap) for structural context. Posture mapping by cluster *profile*, not index. Regime-colored market chart; factor-style-by-regime discussion (S4/S7).
4. **04_anomaly_autoencoder** — Stationary features → 5-day sequences → sequential autoencoder (early stopping) → reconstruction-error threshold at 99th pct → anomaly flags on chart → latent space + GMM regimes → agreement analysis vs notebook 03 (do HMM and GMM regimes coincide?).
5. **05_mean_reversion_signals** — Per-asset: ADF tests, detrended log-price, OU half-life estimation, rolling z-scores, volatility filter (S10). Output: standardized per-asset reversion score each Friday. Event-study check that extreme z-scores actually predicted reversion in-sample.
6. **06_technical_options_signals** — Per-asset: pivot detection → K-Means support/resistance zones → event-study sanity check (S12). For optionable assets: OI notional, call/put walls, gamma proxy (S12). Output: (a) level-proximity signal, (b) positioning-risk score, (c) execution-timing flags.
7. **07_sentiment_altdata** — Scrape financial headlines (requests/BeautifulSoup, robots.txt-respecting — S14 legal considerations); NLP preprocessing; sentiment scoring per asset class; wordcloud + LDA topics for the report (S14). Output: weekly sentiment score per asset-class bucket, lagged to avoid lookahead.
8. **08_black_litterman_fusion** — The heart. Equilibrium returns from asset-class weights (S4); build P/Q/Ω matrices from the four view sources (regime posture, mean reversion, technicals, sentiment); confidence calibration; posterior returns; compare posterior vs prior vs historical means; sensitivity of final weights to each view family.
9. **09_strategy_backtest** — Full weekly pipeline OOS: costs, lagging, 2% turnover cap, no-trade band; IS/OOS split; walk-forward with model re-fits per fold; parameter grid searched IS, judged on OOS stability (S10 discipline); equity curves vs benchmarks, drawdowns, weight evolution, regime-colored performance, turnover report.
10. **10_montecarlo_stress** — GBM Monte Carlo on strategy returns: 1,000 paths, horizon histograms, P(positive), recovery-time analysis (S8). Historical scenarios (2008, 2020, 2022), sensitivity analysis, reverse stress test: "what kills this strategy?" (S7).
11. **11_final_report_ablation** — Ablation study (each module off, one at a time → OOS Sharpe delta table); executive tearsheet; honest conclusions about which techniques earn their complexity; exported figures.

---

## 5. Strategy Implementation — the weekly pipeline

Every Friday after the close (all data as-of Friday close only):

```
STEP 0 · UNIVERSE (L0)                                   [universe.py]
  Refresh screens on schedule (quarterly; weekly liquidity check).
  Output: asset list + asset-class buckets.

STEP 1 · MARKET STATE (L1)                               [regimes.py, anomaly.py]
  1a. HMM state (primary): Risk-On / Neutral / Risk-Off  → posture P_hmm
  1b. K-Means macro cluster → structural context          → posture P_macro
  1c. Autoencoder reconstruction error e_t vs threshold   → anomaly flag A_t
  1d. GMM latent regime → agreement score with HMM        → confidence scaler

STEP 2 · VIEWS (L2)                                      [meanreversion.py,
  Each produces (asset(s), expected excess return, conf):  technicals.py,
  V1 Regime view: posture tilts asset classes              sentiment.py → views.py]
     (Risk-On: +equities; Risk-Off: +short bonds/gold)
  V2 Mean-reversion view: |z| > threshold & passes vol
     filter → expected reversion scaled by half-life
  V3 Technical/positioning view: price at K-Means S/R
     zone ± option-wall confluence → bounded view;
     gamma-proxy sign sets view confidence
  V4 Sentiment view: asset-class sentiment score
     (small confidence — slow, noisy signal)

STEP 3 · FUSION (L3)                                     [views.py, allocation.py]
  Black-Litterman: equilibrium prior Π + views (P,Q,Ω)
  → posterior expected returns μ_BL
  Covariance: shrunk sample covariance of the universe.

STEP 4 · OPTIMIZATION (L3)                               [allocation.py]
  Maximize Sharpe on (μ_BL, Σ) via SLSQP subject to:
    long-only, Σw = 1, per-asset cap, per-class cap,
    optional tracking-error budget vs Permanent Portfolio.
  Also compute GMV & Risk Parity books (defensive + fallback).
  Final check: utility U = E[r] − ½Aσ² must beat the
  defensive book's utility; otherwise hold defensive.

STEP 5 · RISK OVERRIDES (L4)                             [risk.py]
  If anomaly flag A_t ON        → blend 50/100% toward GMV book
  If HMM = Risk-Off             → enforce defensive class caps
  If gamma proxy deeply negative→ halve tactical view weights
  Weekly mini stress report: current book through 2008/2020/2022
  scenarios + 1,000-path GBM fan chart.

STEP 6 · EXECUTION (L5)                                  [strategy.py, backtest.py]
  No-trade band: skip if ‖w_target − w_drifted‖₁ < band.
  Turnover cap: scale trades to ≤ 2% weekly turnover (S4).
  Timing (S12): entries into assets sitting at resistance /
  call walls may be phased over 2–3 days.
  Orders execute first trading day of next week. Costs in bps.
```

### 5.1 Fusion engine: rationale and alternatives

**v1 fusion engine: Black-Litterman.** Chosen because it natively solves this exact problem — combining multiple opinions of differing reliability about different assets:

- Confidence handling is built-in (Ω matrix); weak signals get small influence automatically.
- Disagreement between views resolves gracefully toward the equilibrium prior — no insane weights from noisy forecasts (the classic failure of raw mean-variance).
- Fully interpretable: every final weight can be traced back to the views that moved it.
- Course material (S4).

**Alternatives considered (documented for v2+ experiments):**

| Approach | Idea | Verdict |
|---|---|---|
| Supervised meta-model (stacking) | Feed all signal outputs as features into GBM/NN predicting next-week returns per asset ("alpha combination") | Most expressive, but ~600–800 weekly observations of a very noisy target → severe overfitting risk; interpretability drops to feature importances. Candidate for v2 with heavy regularization. |
| Online expert weighting (Hedge / exponential weights) | Each signal family = an "expert"; weights multiply by recent forecast accuracy | Simple, theoretically grounded, adapts to signal decay; no cross-asset structure. Cheap experiment. |
| Kalman filter / dynamic linear model | Expected returns as hidden state; each signal = a noisy observation | Elegant, BL-like with time dynamics built in; less standard, harder to explain. |
| Autoencoder as fusion | Compress all signals into a latent representation | **Not a fusion engine** — a latent vector is not a return forecast; still needs a predictor on top. AE's correct role (already in ATLAS): feature extraction + anomaly detection *feeding* the fusion layer. |
| Reinforcement learning | Learn allocation policy end-to-end | Sample-inefficient by orders of magnitude vs weekly financial data. Rejected. |

**Planned upgrade path (v2, ablation-tested):** keep the BL skeleton but *learn* the confidence matrix Ω — a small model sets each view family's confidence from its trailing forecast accuracy, optionally regime-conditional (e.g., trust mean reversion more in Risk-Off, sentiment more in calm regimes). Adaptivity where it matters, while retaining BL's stability and interpretability. Slots into `views.py` with no architectural change.

---

## 6. Data Pipeline

| Stage | Detail |
|---|---|
| Sources | **yfinance**: global prices, ^GSPC, ^VIX, FX rates, option chains (S12). **fredapi**: macro set (S8). **Web scraping**: financial-news headlines (S14), stored raw with timestamps. |
| Universe | Config-driven screens; versioned universe files; re-screened per walk-forward fold with as-of data only (no survivorship peeking). |
| Cleaning | Calendar alignment across exchanges (union of trading days, ffill per-market holidays); base-currency conversion; NA-column drops (S8); day-to-day returns NaN→0 (course convention). |
| Features | Stationary transforms only for models (S8 rule): returns, rolling vol, z-scores, VIX changes, detrended log-prices. |
| Leakage control | Chronological splits **before** scaling; scalers fit on train only (S7/S8); all Friday signals use ≤ Friday data; sentiment lagged one day; scalers/models persisted per walk-forward fold in `models/`. |

---

## 7. Backtesting Framework

- **Scheduler:** weekly (Friday decision → next-trading-day execution), adapted from the S5 block scheduler; weights drift intra-week.
- **Realism:** position lagging (S10), costs = bps × turnover, max weekly turnover 2% + no-trade band (S4), turnover reported per module configuration (S5 §14).
- **Metrics:** total/annualized return, vol, Sharpe, Sortino, Calmar, CAGR, max drawdown, hit rate (S10 §9.1), utility at 3 risk-aversion levels (S1/S3).
- **Two-clock design:**
  - *Signal clock (weekly):* all computed indicators — momentum, z-scores, rolling vol, pivots, K-Means S/R zones, sentiment — are recalculated every Friday on rolling lookbacks ending that Friday, so the latest week is always incorporated. No fitting involved.
  - *Model clock (walk-forward folds):* estimated models (HMM, macro K-Means, autoencoder) are re-fit on a quarterly/semi-annual walk-forward schedule with label-remapping by profile, then *applied* weekly to decode the newest data. The HMM may optionally be re-fit weekly (cheap); the autoencoder stays on the fold schedule (optionally warm-started).
- **Train/test structure (three levels):**
  1. *Global chronological split:* ~2010–2021 in-sample (all design and tuning), 2022→present out-of-sample — run once, at the end, via walk-forward. The strategy is never modified after seeing OOS results.
  2. *Walk-forward folds over OOS:* expanding-window fit → trade next quarter → roll. OOS performance = average over folds, not one split.
  3. *Model-internal splits:* autoencoder train/validation for early stopping, anomaly threshold from training-error distribution only, scalers fit on train only (S7/S8).
- **Hyperparameter grid search** (IS only; winner = most *stable* across IS validation folds, not highest single-period Sharpe — S10 §13):

  | Module | Hyperparameters searched |
  |---|---|
  | Mean reversion | lookback window (20/40/60d), entry z (±1.5/2/2.5), exit z (0/±0.5), max half-life filter |
  | HMM | n_states (2/3/4), features (returns+vol ± VIX) |
  | Macro K-Means | k via elbow (3–6), indicator subset |
  | Autoencoder | latent dim (2/4), depth, sequence length (5/10d), threshold pct (97.5/99) |
  | Black-Litterman | τ, per-view-family confidence scalers |
  | Technicals | pivot order (3/5/8), S/R cluster k (3/5), zone width (25/50 bps) |
  | Execution | no-trade band, per-asset/class caps, anomaly blend fraction (50/100%) |

- **Validation:** after grid selection IS, the *frozen* config runs through the OOS walk-forward exactly once; OOS monitoring checklist per S10.
- **Options-data caveat:** yfinance provides only *current* option chains — no history. Call/put walls and the gamma proxy therefore cannot be backtested retroactively: the backtest uses the price-level component of V3 only, while options positioning is collected live and evaluated in a forward paper-trading phase (documented explicitly in the final report).
- **Ablation:** rerun with each of V1–V4 and each L1 model disabled; report marginal OOS Sharpe per technique.
- **Benchmarks:** Permanent Portfolio, 60/40, static Max Sharpe, GMV, Risk Parity, HRP — the strategy must beat the *best* classical single-method book OOS, not just the Permanent Portfolio.

---

## 8. Dependencies & Libraries

| Library | Role (sessions) |
|---|---|
| pandas, numpy | Everything (all) |
| scipy | SLSQP optimization (S2/S3/S5), stats |
| statsmodels | ADF test, OU half-life regression (S10) |
| scikit-learn | StandardScaler, K-Means, PCA, GMM, covariance shrinkage (S6–S8, S12) |
| hmmlearn | HMM regimes (S8) |
| tensorflow/keras | Sequential autoencoder (S8) |
| yfinance | Prices, FX, option chains (all price sessions, S12) |
| fredapi | Macro data (S8) — free API key |
| requests, beautifulsoup4 | Web scraping (S14) |
| nltk or spaCy, wordcloud, gensim | NLP sentiment, LDA, wordcloud (S14) |
| matplotlib, seaborn, plotly | Charts (all) |
| pyyaml, pyarrow | Config, parquet storage |
| pytest | Tests |

Pinned in `requirements.txt`; dedicated `.venv`; editable install (`pip install -e .`).

---

## 9. Execution Flow

```
config.yaml ──► data.py ──► universe.py ──► data/processed/ + universe file
                                │
        ┌───────────┬───────────┼───────────────┬──────────────┐
        ▼           ▼           ▼               ▼              ▼
   regimes.py   anomaly.py  meanreversion.py  technicals.py  sentiment.py
   (HMM+KMeans) (AE + GMM)  (ADF/OU/z-score)  (S/R + options) (scrape+NLP)
        │           │           │               │              │
        └───────────┴───────────┴───────┬───────┴──────────────┘
                                        ▼
                                    views.py  (P, Q, Ω)
                                        ▼
                                  allocation.py  (Black-Litterman → SLSQP;
                                        │         GMV/RP defensive books)
                                        ▼
                                     risk.py   (anomaly override, caps,
                                        │       stress + Monte Carlo report)
                                        ▼
                                   strategy.py → backtest.py (weekly, costs,
                                        │            turnover cap, walk-forward)
                                        ▼
                          plotting.py → reports/figures + reports/results
                                        (incl. ablation table)
```

---

## 10. Deliverables

1. **GitHub-ready repo** with the structure above and a README featuring the architecture diagram and OOS results.
2. **Eleven executed notebooks** — one per layer, ending with the ablation study.
3. **`src/atlas` package** — tested, importable, every class technique implemented as a reusable module.
4. **Backtest artifacts**: OOS tearsheet, weight/turnover history, regime timeline, ablation table (`reports/results/`), figures in `fig01_...` convention.
5. **Final report** (notebook 11, exportable to PDF ≤ 10 pages, individual-assignment format): methodology, assumptions, OOS results vs all benchmarks, stress results, and an honest account of which techniques added value.

---

## 11. Build Order (each stage ends with something runnable)

1. Scaffold + config + `data.py` + `metrics.py` + `universe.py` → notebook 01
2. `allocation.py` classical methods + `backtest.py` weekly engine → notebooks 02, benchmark books running OOS (baseline!)
3. `regimes.py` → notebook 03; wire V1 regime view + posture switching → first full strategy version
4. `anomaly.py` → notebook 04; add risk override
5. `meanreversion.py` → notebook 05; add V2
6. `technicals.py` → notebook 06; add V3 + execution timing
7. `sentiment.py` → notebook 07; add V4
8. `views.py` + Black-Litterman in `allocation.py` → notebook 08 (until here, views can combine naively; BL formalizes it)
9. Full walk-forward + grid + turnover tuning → notebook 09
10. `simulation.py` + stress → notebook 10; ablation + report → notebook 11

**Rule of the build:** after step 2 you always have a working, measurable strategy. Every later step must *prove* its OOS improvement in the ablation table — that's the difference between "uses everything from class" and "throws everything at the wall."
