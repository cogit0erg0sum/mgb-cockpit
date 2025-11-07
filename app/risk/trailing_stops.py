# app/risk/trailing_stops.py
import pandas as pd

def _atr(df: pd.DataFrame, length: int = 14) -> pd.Series:
    """Average True Range (simple rolling mean). Requires High/Low/Close."""
    high, low, close = df["High"], df["Low"], df["Close"]
    hl = high - low
    hc = (high - close.shift()).abs()
    lc = (low - close.shift()).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    return tr.rolling(length).mean()

def atr_trailing_line(df: pd.DataFrame, atr_mult: float = 2.5, lookback: int = 14) -> pd.Series:
    """Classic ATR multiple from Close."""
    atr = _atr(df, length=lookback)
    return df["Close"] - atr_mult * atr

def chandelier_exit_long(df: pd.DataFrame, atr_mult: float = 3.0, lookback: int = 22) -> pd.Series:
    """Chandelier Exit for long: Highest(High, N) - k*ATR(N)."""
    hh = df["High"].rolling(lookback).max()
    atr = _atr(df, length=lookback)
    return hh - atr_mult * atr