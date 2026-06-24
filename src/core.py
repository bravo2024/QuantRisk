from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm


def portfolio_stats(
    weights: np.ndarray,
    mu: np.ndarray,
    cov: np.ndarray,
    rf: float = 0.06,
    freq: int = 252,
) -> tuple[float, float, float]:
    w = np.asarray(weights, dtype=float)
    ret = float(w @ mu * freq)
    vol = float(np.sqrt(w @ cov @ w) * np.sqrt(freq))
    sharpe = (ret - rf) / vol if vol > 0 else 0.0
    return ret, vol, sharpe


def value_at_risk(returns: pd.Series | np.ndarray, alpha: float = 0.05) -> float:
    return float(-np.quantile(returns, alpha))


def expected_shortfall(returns: pd.Series | np.ndarray, alpha: float = 0.05) -> float:
    q = np.quantile(returns, alpha)
    tail = returns[returns <= q]
    return float(-tail.mean()) if len(tail) else float(-q)


def parametric_var(returns: pd.Series | np.ndarray, alpha: float = 0.05) -> float:
    mu = float(np.mean(returns))
    sigma = float(np.std(returns))
    return float(-(mu + sigma * norm.ppf(alpha)))


def risk_contribution(
    weights: np.ndarray, cov: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    w = np.asarray(weights, dtype=float)
    port_vol = float(np.sqrt(w @ cov @ w))
    if port_vol <= 0:
        return np.zeros(len(w)), np.zeros(len(w)), np.zeros(len(w))
    marginal = cov @ w
    component = w * marginal
    proportional = component / port_vol
    return marginal, component, proportional


def technical_indicators(prices: pd.Series) -> pd.DataFrame:
    out = pd.DataFrame(index=prices.index)
    ret = prices.pct_change()
    out["ret_1"] = ret
    out["ret_5"] = prices.pct_change(5)
    out["ret_21"] = prices.pct_change(21)
    out["vol_21"] = ret.rolling(21).std()
    delta = prices.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / (loss + 1e-9)
    out["rsi_14"] = 100 - 100 / (1 + rs)
    ema12 = prices.ewm(span=12, adjust=False).mean()
    ema26 = prices.ewm(span=26, adjust=False).mean()
    out["macd"] = ema12 - ema26
    out["macd_signal"] = out["macd"].ewm(span=9, adjust=False).mean()
    out["momentum_21"] = prices / prices.shift(21) - 1
    return out.dropna()


def compute_portfolio_metrics(
    returns: pd.DataFrame, weights: np.ndarray, rf: float = 0.06
) -> dict:
    freq = 252
    mu = returns.mean().values
    cov = returns.cov().values
    port_ret = returns @ weights

    r_ann, v_ann, sharpe = portfolio_stats(weights, mu, cov, rf, freq)

    cum = (1 + port_ret).cumprod()
    rolling_max = cum.cummax()
    drawdown = (cum / rolling_max - 1)
    max_dd = float(drawdown.min())

    var_95 = value_at_risk(port_ret, 0.05)
    var_99 = value_at_risk(port_ret, 0.01)
    es_95 = expected_shortfall(port_ret, 0.05)

    cumulative_return = float(cum.iloc[-1] - 1) if len(cum) > 0 else 0.0

    return {
        "annualized_return": r_ann,
        "annualized_volatility": v_ann,
        "sharpe_ratio": sharpe,
        "max_drawdown": max_dd,
        "var_95": var_95,
        "var_99": var_99,
        "cvar_95": es_95,
        "cumulative_return": cumulative_return,
        "n_samples": len(port_ret),
        "n_assets": len(returns.columns),
    }
