# core/features.py
from __future__ import annotations

from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd


def _rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    roll_up = pd.Series(gain, index=close.index).rolling(window).mean()
    roll_down = pd.Series(loss, index=close.index).rolling(window).mean()
    rs = roll_up / (roll_down.replace(0, np.nan))
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50.0)


def _safe_div(a: pd.Series, b: pd.Series) -> pd.Series:
    return (a / b.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)


def make_features(
    prices_tidy: pd.DataFrame,
    horizons: Iterable[int] = (5, 10, 21),
    add_cross_sectional: bool = True,
) -> pd.DataFrame:
    """
    Input: tidy OHLCV with columns: date, ticker, open, high, low, close, adj_close, volume
    Output: multi-ticker feature table indexed by [date, ticker]
    """
    df = prices_tidy.copy()
    df = df.sort_values(["ticker","date"]).set_index(["date","ticker"])

    # Base series
    close = df["adj_close"]
    vol = df["volume"].astype(float)

    # Trend & momentum
    sma20 = close.groupby(level="ticker").rolling(20).mean().droplevel(0)
    sma50 = close.groupby(level="ticker").rolling(50).mean().droplevel(0)
    sma200 = close.groupby(level="ticker").rolling(200).mean().droplevel(0)
    mom21 = close.groupby(level="ticker").pct_change(21)
    rsi14 = close.groupby(level="ticker").apply(_rsi).droplevel(0)

    # Volatility
    ret1 = close.groupby(level="ticker").pct_change()
    vol21 = ret1.groupby(level="ticker").rolling(21).std().droplevel(0)

    # Liquidity proxy
    vol_z21 = vol.groupby(level="ticker").apply(lambda s: (s - s.rolling(21).mean()) / (s.rolling(21).std()))
    vol_z21 = vol_z21.droplevel(0)

    feats = pd.DataFrame(
        {
            "close": close,
            "sma20_ratio": _safe_div(close, sma20),
            "sma50_ratio": _safe_div(close, sma50),
            "sma200_ratio": _safe_div(close, sma200),
            "mom21": mom21,
            "rsi14": rsi14,
            "vol21": vol21,
            "vol_z21": vol_z21,
        }
    )

    # Cross-sectional ranks (optional): rank each day across tickers
    if add_cross_sectional:
        def cs_rank(s: pd.Series) -> pd.Series:
            return s.groupby(level="date").rank(pct=True)

        feats["cs_rank_mom21"] = cs_rank(feats["mom21"])
        feats["cs_rank_rsi14"] = cs_rank(feats["rsi14"])

    # Forward returns & labels for multiple horizons
    for h in horizons:
        fwd = close.groupby(level="ticker").shift(-h)
        fwd_ret = (fwd / close) - 1.0
        feats[f"fwd_ret_{h}d"] = fwd_ret
        feats[f"label_up_{h}d"] = (fwd_ret > 0).astype(int)

    feats = feats.replace([np.inf, -np.inf], np.nan).dropna(how="any")
    return feats


def build_Xy(
    feats: pd.DataFrame,
    horizon: int = 5,
    feature_cols: Iterable[str] = None,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Extract (X, y) for a given horizon.
    """
    if feature_cols is None:
        # Default: use all non-label, non-close columns
        ignore = {f"fwd_ret_{horizon}d", f"label_up_{horizon}d", "close"}
        feature_cols = [c for c in feats.columns if c not in ignore and not c.startswith("fwd_ret_") and not c.startswith("label_up_")]

    X = feats[feature_cols].copy()
    y = feats[f"label_up_{horizon}d"].copy().astype(int)
    return X, y
