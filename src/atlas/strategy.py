"""Weekly decision pipeline (glues L0-L5). See PROJECT_STRUCTURE.md section 5.

Stage 3 first version (regime switching only); grows one view per stage.

Planned public API:
    decide(as_of, state, cfg) -> Decision
        # Steps: universe -> regime + anomaly -> views -> BL -> optimize
        #        -> risk overrides -> no-trade band / turnover cap -> target weights
    run_period(dates, cfg) -> pd.DataFrame       # weekly decisions over a span
"""
