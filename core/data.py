# core/data.py
from __future__ import annotations

from pathlib import Path
import hashlib
from typing import Iterable, List, Optional, Tuple, Union

import pandas as pd
import yfinance as yf


CACHE_DIR = Path("data/cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _cache_key(tickers: Iterable[str], start: str, end: Optional[str], interval: str) -> Path:
    tickers = tuple(sorted(set(t.strip().upper() for t in tickers)))
    raw = f"{tickers}|{start}|{end}|{interval}"
    h = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]
    return CACHE_DIR / f"yf_{h}.parquet"


def get_prices(
    tickers: Union[str, Iterable[str]],
    start: str = "2015-01-01",
    end: Optional[str] = None,
    interval: str = "1d",
    force_refresh: bool = False,
) -> pd.DataFrame:
    """
    Download OHLCV for one or more tickers and return a tidy DataFrame:
        index: DatetimeIndex
        columns: ["ticker", "open","high","low","close","adj_close","volume"]
    Cached to data/cache/*.parquet for speed.
    """
    if isinstance(tickers, str):
        tickers = [tickers]
    tickers = [t.strip().upper() for t in tickers if t and t.strip()]
    if not tickers:
        return pd.DataFrame(columns=["ticker","open","high","low","close","adj_close","volume"])

    cache_path = _cache_key(tickers, start, end, interval)
    if cache_path.exists() and not force_refresh:
        return pd.read_parquet(cache_path)

    df = yf.download(
        tickers,
        start=start,
        end=end,
        interval=interval,
        auto_adjust=False,   # keep raw + adj_close separately
        progress=False,
        group_by="ticker",
        threads=True,
    )

    # Normalize to tidy format
    out_rows = []
    if isinstance(df.columns, pd.MultiIndex):
        # multiple tickers
        for t in sorted(set(lvl for lvl, _ in df.columns)):
            sub = df[t].copy()
            sub.columns = [c.lower() for c in sub.columns]
            sub = sub.rename(columns={"adj close": "adj_close"})
            sub["ticker"] = t
            out_rows.append(sub.reset_index())
        out = pd.concat(out_rows, ignore_index=True)
    else:
        # single ticker
        single = df.copy()
        single.columns = [c.lower() for c in single.columns]
        single = single.rename(columns={"adj close": "adj_close"})
        single["ticker"] = tickers[0]
        out = single.reset_index()

    out = out.rename(columns={"index": "date", "datetime": "date"})
    out["date"] = pd.to_datetime(out["date"])
    out = out[["date","ticker","open","high","low","close","adj_close","volume"]].sort_values(["ticker","date"])
    out.to_parquet(cache_path, index=False)
    return out


def get_close_matrix(
    tickers: Union[str, Iterable[str]],
    start: str = "2015-01-01",
    end: Optional[str] = None,
    interval: str = "1d",
) -> pd.DataFrame:
    """
    Convenience: return a wide dataframe of adjusted closes (dates x tickers).
    """
    prices = get_prices(tickers, start=start, end=end, interval=interval)
    piv = prices.pivot(index="date", columns="ticker", values="adj_close").sort_index()
    return piv


def example_universe(name: str = "core") -> List[str]:
    """
    Small, liquid default universe. Expand later.
    """
    if name == "core":
        return ["SPY", "QQQ", "IWM", "DIA", "XLK", "XLF", "XLE", "XLV"]
    elif name == "mega":
        return ["AAPL","MSFT","NVDA","GOOGL","AMZN","META","TSLA","AVGO","JPM","XOM"]
    return ["SPY"]
