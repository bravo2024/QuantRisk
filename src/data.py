from __future__ import annotations

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

INDIAN_NSE_12 = [
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS",
    "SBIN.NS", "BHARTIARTL.NS", "KOTAKBANK.NS", "ITC.NS", "AXISBANK.NS",
    "GOLDBEES.NS", "BANKBEES.NS",
]

US_LARGE_6 = ["SPY", "QQQ", "TLT", "GLD", "EEM", "IYR"]

GLOBAL_MIX_8 = [
    "SPY", "QQQ", "TLT", "GLD",
    "RELIANCE.NS", "TCS.NS", "EEM", "BND",
]

NIFTY_50_INDEX = ["^NSEI"]

BANK_NIFTY_INDEX = ["^NSEBANK"]

BANK_NIFTY_STOCKS = [
    "HDFCBANK.NS", "ICICIBANK.NS", "SBIN.NS", "KOTAKBANK.NS",
    "AXISBANK.NS", "INDUSINDBK.NS", "FEDERALBNK.NS", "BANKBARODA.NS",
]

TICKER_PRESETS = {
    "Indian NSE 12": INDIAN_NSE_12,
    "Nifty 50 Index": NIFTY_50_INDEX,
    "Bank Nifty Index": BANK_NIFTY_INDEX,
    "Bank Nifty Stocks": BANK_NIFTY_STOCKS,
    "US Large Cap 6": US_LARGE_6,
    "Global Mix 8": GLOBAL_MIX_8,
}


def fetch_prices(
    tickers: list[str],
    start_date: datetime | str | None = None,
    end_date: datetime | str | None = None,
    period: str = "3y",
) -> pd.DataFrame | None:
    if start_date is None:
        start_date = datetime.now() - timedelta(days=365 * 3)
    if end_date is None:
        end_date = datetime.now()
    tickers = [t.strip().upper() for t in tickers if t.strip()]
    if not tickers:
        return None
    data = yf.download(tickers, start=start_date, end=end_date, progress=False, auto_adjust=True)
    if data is None or (isinstance(data, pd.DataFrame) and data.empty):
        return None
    if isinstance(data.columns, pd.MultiIndex):
        if "Close" in data.columns.get_level_values(0):
            out = data["Close"].copy()
        else:
            return None
    else:
        out = data.copy()
    out = out.dropna(how="all")
    out = out.dropna(axis=1, thresh=int(0.8 * len(out)))
    if isinstance(out, pd.Series):
        out = out.to_frame()
    return out


def compute_returns(prices: pd.DataFrame, freq: str = "D") -> pd.DataFrame:
    if freq == "D":
        ret = prices.pct_change(fill_method=None).dropna()
    else:
        agg = prices.resample(freq).last().dropna()
        ret = agg.pct_change(fill_method=None).dropna()
    return ret


def synthetic_returns(
    tickers: list[str], n_days: int = 756, seed: int = 42
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = len(tickers)
    base_mu = np.full(n, 0.08 / 252)
    base_vol = np.full(n, 0.20 / np.sqrt(252))
    base_mu[: max(1, n // 3)] = 0.03 / 252
    base_vol[: max(1, n // 3)] = 0.05 / np.sqrt(252)
    base_mu[max(1, n // 3) : max(1, 2 * n // 3)] = 0.06 / 252
    base_vol[max(1, n // 3) : max(1, 2 * n // 3)] = 0.16 / np.sqrt(252)
    L = np.linalg.cholesky(0.3 * np.ones((n, n)) + 0.7 * np.eye(n))
    R = base_mu + (rng.normal(size=(n_days, n)) @ L.T) * base_vol
    idx = pd.date_range(end=datetime.now(), periods=n_days, freq="B")
    return pd.DataFrame(R, index=idx, columns=tickers)


def make_synthetic(
    n_days: int = 756,
    n_assets: int = 6,
    seed: int = 42,
) -> dict:
    rng = np.random.default_rng(seed)
    tickers = [f"ASSET_{i}" for i in range(n_assets)]
    rets = synthetic_returns(tickers, n_days, seed)
    prices = (1 + rets).cumprod() * 100.0
    return {"prices": prices, "returns": rets, "tickers": tickers}
