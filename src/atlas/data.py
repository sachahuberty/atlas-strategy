"""Data layer: downloads, caching, cleaning, FX, calendar alignment.

Stage 1. See PROJECT_STRUCTURE.md sections 3 and 6.

Public API:
    load_config(path) -> dict
    download_prices(tickers, start, end) -> pd.DataFrame
        # cached parquet, dated
    download_fx(pairs, start, end) -> pd.DataFrame
    download_macro(fred_series, start, end) -> pd.DataFrame
        # FRED_API_KEY env var
    download_option_chain(ticker) -> pd.DataFrame
        # live snapshot only (no history)
    to_base_currency(prices, currencies, fx, base) -> pd.DataFrame
    align_calendars(prices) -> pd.DataFrame
        # union of trading days, ffill
    daily_returns(prices) -> pd.DataFrame
        # NaN -> 0 (course convention)
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import yaml
import yfinance as yf

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"


def load_config(path: str | Path = CONFIG_PATH) -> dict:
    """Load the project config.yaml as a plain dict."""
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _load_env_file(path: str | Path = PROJECT_ROOT / ".env") -> None:
    """Populate os.environ from a .env file, no python-dotenv needed.

    Existing environment variables take precedence; a missing file
    is a no-op.
    """
    path = Path(path)
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip())


def _cache_path(
    prefix: str, key: str, as_of: pd.Timestamp | None = None
) -> Path:
    as_of = as_of or pd.Timestamp.today().normalize()
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    stamp = as_of.strftime("%Y%m%d")
    return RAW_DIR / f"{prefix}_{key}_{stamp}.parquet"


def _flatten_close(raw: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    """Extract close prices from a yf.download() result into one column
    per ticker, regardless of single- vs multi-ticker column shape."""
    if isinstance(raw.columns, pd.MultiIndex):
        close = raw["Close"]
    else:
        close = raw[["Close"]].rename(columns={"Close": tickers[0]})
    return close.sort_index()


def download_prices(
    tickers: list[str],
    start: str,
    end: str | None = None,
    use_cache: bool = True,
) -> pd.DataFrame:
    """Adjusted close prices for tickers, cached to data/raw/ as
    dated parquet.

    The cache key includes `start`/`end`: a request for the same
    tickers but a different date range (e.g. a longer window for a
    pre-2010 stress scenario) must not silently return another
    request's cached range.
    """
    key = f"{'-'.join(sorted(tickers))}_{start}_{end or 'latest'}"
    cache_file = _cache_path("prices", key)
    if use_cache and cache_file.exists():
        return pd.read_parquet(cache_file)

    raw = yf.download(
        tickers, start=start, end=end, auto_adjust=True, progress=False
    )
    prices = _flatten_close(raw, tickers)
    prices.to_parquet(cache_file)
    return prices


def download_fx(
    pairs: list[str],
    start: str,
    end: str | None = None,
    use_cache: bool = True,
) -> pd.DataFrame:
    """FX rates for pairs (Yahoo tickers, e.g. 'EURUSD=X'), cached like
    prices (cache key includes the date range, same reasoning)."""
    key = f"{'-'.join(sorted(pairs))}_{start}_{end or 'latest'}"
    cache_file = _cache_path("fx", key)
    if use_cache and cache_file.exists():
        return pd.read_parquet(cache_file)

    raw = yf.download(
        pairs, start=start, end=end, auto_adjust=True, progress=False
    )
    rates = _flatten_close(raw, pairs)
    rates.to_parquet(cache_file)
    return rates


def download_macro(
    fred_series: list[str],
    start: str,
    end: str | None = None,
    use_cache: bool = True,
) -> pd.DataFrame:
    """Macro indicator series from FRED (S8). Requires FRED_API_KEY.

    Cache key includes the date range, same reasoning as
    `download_prices`."""
    key = f"{'-'.join(sorted(fred_series))}_{start}_{end or 'latest'}"
    cache_file = _cache_path("macro", key)
    if use_cache and cache_file.exists():
        return pd.read_parquet(cache_file)

    _load_env_file()
    api_key = os.environ.get("FRED_API_KEY")
    if not api_key:
        raise RuntimeError(
            "FRED_API_KEY is not set. Add it to a local .env file "
            "(gitignored) or export it in your shell."
        )

    from fredapi import Fred

    fred = Fred(api_key=api_key)
    series = {
        name: fred.get_series(
            name, observation_start=start, observation_end=end
        )
        for name in fred_series
    }
    macro = pd.DataFrame(series)
    macro.index.name = "date"
    macro.to_parquet(cache_file)
    return macro


def download_option_chain(ticker: str) -> pd.DataFrame:
    """Live option-chain snapshot for ticker. yfinance has no option
    history, so this is used live-only, never inside a backtest."""
    tk = yf.Ticker(ticker)
    expirations = tk.options
    if not expirations:
        return pd.DataFrame()

    chains = []
    for expiry in expirations:
        chain = tk.option_chain(expiry)
        calls = chain.calls.assign(option_type="call", expiration=expiry)
        puts = chain.puts.assign(option_type="put", expiration=expiry)
        chains.append(pd.concat([calls, puts], ignore_index=True))
    return pd.concat(chains, ignore_index=True)


def to_base_currency(
    prices: pd.DataFrame,
    currencies: dict[str, str],
    fx: pd.DataFrame,
    base: str,
) -> pd.DataFrame:
    """Convert each ticker column of prices into `base` currency.

    `currencies` maps ticker -> its quote currency (e.g.
    {'EWG': 'USD'}). `fx` has one column per pair used, named
    '{ccy}{base}=X' (Yahoo convention), already aligned to
    prices.index. Tickers already quoted in base currency pass
    through unchanged.
    """
    converted = prices.copy()
    for ticker in prices.columns:
        ccy = currencies.get(ticker, base)
        if ccy == base:
            continue
        pair = f"{ccy}{base}=X"
        if pair not in fx.columns:
            raise KeyError(
                f"Missing FX rate column '{pair}' for ticker '{ticker}'"
            )
        rate = fx[pair].reindex(prices.index).ffill()
        converted[ticker] = prices[ticker] * rate
    return converted


def align_calendars(prices: pd.DataFrame) -> pd.DataFrame:
    """Reindex every column onto the union of trading days across
    assets, forward-filling per-market holidays (course convention)."""
    union_index = prices.index
    for col in prices.columns:
        union_index = union_index.union(prices[col].dropna().index)
    return prices.reindex(union_index).sort_index().ffill()


def daily_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Simple daily returns; day-to-day NaN -> 0 (course convention)."""
    return prices.pct_change().fillna(0.0)
