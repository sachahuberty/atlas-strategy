"""News sentiment signal (L2 view V4): scraping + NLP (S6/S14).

Stage 7. Scrapes financial-news RSS feeds via requests + BeautifulSoup,
checking robots.txt at REQUEST TIME (not just pre-vetted by hand) --
the S14 legal consideration. NLP preprocessing (tokenize, stopwords,
lemmatize), then VADER sentiment per headline, keyword-tagged to an
asset-class bucket, aggregated to a weekly per-bucket score lagged by
`sentiment.lag_days` to avoid lookahead.

Like stage 6's options positioning, this is fundamentally LIVE-ONLY:
free news sources have no historical archive, so there is no way to
backtest sentiment over 2010-present -- only to compute it from
today's headlines and use it forward. `sentiment_view`/
`with_sentiment_view` (strategy.py) exist for that forward use, not
for the OOS backtest.

Public API:
    scrape_headlines(sources, cfg) -> pd.DataFrame
    preprocess(text) -> list[str]
    score_sentiment(text) -> float
    tag_bucket(text) -> str | None
    aggregate_sentiment(headlines, cfg) -> pd.DataFrame  # week x bucket
    sentiment_view(bucket_scores, class_bucket, cfg) -> pd.Series
    build_lda_model(documents, cfg) -> (model, dictionary, corpus)  # reporting
"""

from __future__ import annotations

import urllib.robotparser
import warnings
from urllib.parse import urlparse

import gensim
import nltk
import pandas as pd
import requests
from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning
from nltk import pos_tag
from nltk.corpus import stopwords, wordnet
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# html.parser (stdlib, no lxml dependency) parses RSS's simple, flat
# <item> structure fine; bs4's "this isn't a real XML parser" warning
# doesn't apply to our narrow tag extraction.
warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

_NLTK_RESOURCES = {
    "tokenizers/punkt_tab": "punkt_tab",
    "corpora/stopwords": "stopwords",
    "corpora/wordnet": "wordnet",
    "sentiment/vader_lexicon": "vader_lexicon",
    "taggers/averaged_perceptron_tagger_eng": "averaged_perceptron_tagger_eng",
}

_PENN_TO_WORDNET = {"J": wordnet.ADJ, "V": wordnet.VERB, "R": wordnet.ADV}

BUCKET_KEYWORDS = {
    "fixed_income": [
        "bond", "yield", "treasury", "fed ", "federal reserve", "rate cut",
        "rate hike", "interest rate",
    ],
    "commodity": ["gold", "oil", "crude", "opec", "commodit"],
    "real_estate": ["real estate", "reit", "housing", "mortgage"],
    "equity": [
        "stock", "s&p", "nasdaq", "dow jones", "equit", "shares", "earnings",
    ],
}

_sentiment_analyzer = None


def _ensure_nltk_data() -> None:
    for path, package in _NLTK_RESOURCES.items():
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(package, quiet=True)


def _robots_allowed(url: str, user_agent: str) -> bool:
    """Check robots.txt before scraping (S14 legal consideration):
    fail closed (disallow) if robots.txt can't be verified at all."""
    parsed = urlparse(url)
    robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"
    try:
        resp = requests.get(
            robots_url, timeout=10, headers={"User-Agent": user_agent}
        )
        parser = urllib.robotparser.RobotFileParser()
        parser.parse(resp.text.splitlines())
        return parser.can_fetch(user_agent, url)
    except requests.RequestException:
        return False


def _fetch_rss(url: str, cfg: dict) -> pd.DataFrame:
    """Fetch and parse one RSS feed. Empty DataFrame on any failure
    (network, robots disallow, parse error) rather than crashing --
    scraping the live web is inherently unreliable and one dead/
    disallowed feed shouldn't take down the whole pipeline."""
    scfg = cfg["sentiment"]
    user_agent = scfg["user_agent"]
    empty = pd.DataFrame(
        columns=["title", "description", "link", "published", "source"]
    )

    if not _robots_allowed(url, user_agent):
        return empty

    try:
        resp = requests.get(
            url,
            headers={"User-Agent": user_agent},
            timeout=scfg["request_timeout_seconds"],
        )
        resp.raise_for_status()
    except requests.RequestException:
        return empty

    soup = BeautifulSoup(resp.content, "html.parser")
    rows = []
    for item in soup.find_all("item"):
        title = item.find("title")
        description = item.find("description")
        link = item.find("link")
        # html.parser lowercases all tag names (an HTML convention),
        # so the RSS spec's camelCase <pubDate> must be looked up as
        # "pubdate" -- a subtle, easy-to-miss consequence of using an
        # HTML parser on an XML document.
        pub_date = item.find("pubdate")
        if title is None:
            continue
        rows.append(
            {
                "title": title.get_text(strip=True),
                "description": (
                    description.get_text(strip=True)
                    if description is not None
                    else ""
                ),
                "link": link.get_text(strip=True) if link is not None else "",
                "published": (
                    pub_date.get_text(strip=True)
                    if pub_date is not None
                    else None
                ),
                "source": url,
            }
        )
    if not rows:
        return empty
    return pd.DataFrame(rows)


def scrape_headlines(sources: list[str], cfg: dict) -> pd.DataFrame:
    """Scrape all configured RSS sources (S14), respecting robots.txt
    at request time. Columns: title, description, link, published,
    source."""
    frames = [_fetch_rss(url, cfg) for url in sources]
    frames = [f for f in frames if len(f) > 0]
    if not frames:
        return pd.DataFrame(
            columns=["title", "description", "link", "published", "source"]
        )
    return pd.concat(frames, ignore_index=True)


def preprocess(text: str) -> list[str]:
    """Tokenize, lowercase, drop non-alphabetic tokens and stopwords,
    POS-tag, and lemmatize (S14). POS-tagging matters: WordNet's
    lemmatizer assumes NOUN by default, so without it verb forms like
    "rallying"/"surged" never reduce to their base form ("rally"/
    "surge") -- confirmed directly, this silently degraded topic/
    sentiment quality on financial headlines, which are verb-heavy.
    """
    _ensure_nltk_data()
    tokens = word_tokenize(text.lower())
    tokens = [t for t in tokens if t.isalpha()]
    stop_words = set(stopwords.words("english"))
    tokens = [t for t in tokens if t not in stop_words]
    tagged = pos_tag(tokens)
    lemmatizer = WordNetLemmatizer()
    return [
        lemmatizer.lemmatize(token, _PENN_TO_WORDNET.get(tag[0], wordnet.NOUN))
        for token, tag in tagged
    ]


def score_sentiment(text: str) -> float:
    """VADER compound sentiment score in [-1, 1] (S14): a lexicon/
    rule-based analyzer well suited to short, headline-style text."""
    global _sentiment_analyzer
    _ensure_nltk_data()
    if _sentiment_analyzer is None:
        _sentiment_analyzer = SentimentIntensityAnalyzer()
    return _sentiment_analyzer.polarity_scores(text)["compound"]


def tag_bucket(text: str) -> str | None:
    """Keyword-tag a headline to an asset-class bucket. None if no
    bucket's keywords match (excluded from bucket aggregation)."""
    text_lower = text.lower()
    for bucket, keywords in BUCKET_KEYWORDS.items():
        if any(kw in text_lower for kw in keywords):
            return bucket
    return None


def aggregate_sentiment(headlines: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Per-asset-class weekly mean sentiment score (S14), lagged by
    `sentiment.lag_days` to avoid lookahead. Rows: week (period);
    columns: asset-class bucket."""
    scfg = cfg["sentiment"]
    if len(headlines) == 0:
        return pd.DataFrame()

    df = headlines.copy()
    df["bucket"] = df["title"].map(tag_bucket)
    df["sentiment"] = df["title"].map(score_sentiment)
    df = df.dropna(subset=["bucket", "published"])
    if len(df) == 0:
        return pd.DataFrame()

    published = pd.to_datetime(df["published"], utc=True, format="mixed")
    published = published.dt.tz_localize(None) + pd.Timedelta(
        days=scfg["lag_days"]
    )
    df["published"] = published
    df["week"] = df["published"].dt.to_period("W")
    return df.groupby(["week", "bucket"])["sentiment"].mean().unstack("bucket")


def sentiment_view(
    bucket_scores: pd.Series, class_bucket: pd.Series, cfg: dict
) -> pd.Series:
    """Per-asset V4 view (S14): broadcast each asset's class-bucket
    sentiment score to a small, bounded expected-return view -- "small
    confidence, slow, noisy signal" per the weekly pipeline design."""
    scfg = cfg["sentiment"]
    mapped = class_bucket.map(bucket_scores).fillna(0.0)
    return mapped * scfg["max_view_magnitude"]


def build_lda_model(documents: list[list[str]], cfg: dict):
    """LDA topic model over preprocessed headline tokens, for the
    report only (S14): word clouds and topics are narrative, not
    inputs to sentiment_view."""
    scfg = cfg["sentiment"]
    seed = cfg["general"]["random_seed"]
    dictionary = gensim.corpora.Dictionary(documents)
    corpus = [dictionary.doc2bow(doc) for doc in documents]
    model = gensim.models.LdaModel(
        corpus,
        num_topics=scfg["n_topics"],
        id2word=dictionary,
        random_state=seed,
        passes=10,
    )
    return model, dictionary, corpus
