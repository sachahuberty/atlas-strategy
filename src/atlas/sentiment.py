"""News sentiment signal (L2 view V4): scraping + NLP (S6/S14).

Stage 7.
Rules: respect robots.txt (S14 legal considerations); store raw headlines with
timestamps; sentiment lagged 1 day to avoid lookahead; low BL confidence.

Planned public API:
    scrape_headlines(sources, as_of) -> pd.DataFrame
    preprocess(texts) -> list[str]                   # tokenize, stopwords, lemmatize
    sentiment_scores(texts) -> pd.Series
    class_sentiment(headlines, class_map, as_of) -> pd.Series  # per asset-class bucket
    topics_lda(texts, n) / wordcloud_data(texts)     # reporting only
"""
