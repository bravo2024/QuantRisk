from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

from src.core import portfolio_stats


def max_sharpe_weights(
    mu: np.ndarray,
    cov: np.ndarray,
    rf: float = 0.06,
    freq: int = 252,
    max_w: float = 1.0,
) -> np.ndarray:
    n = len(mu)
    bounds = [(0.0, max_w)] * n
    constraints = [{"type": "eq", "fun": lambda w: w.sum() - 1.0}]

    def neg_sharpe(w: np.ndarray) -> float:
        _, _, s = portfolio_stats(w, mu, cov, rf, freq)
        return -s

    w0 = np.full(n, 1.0 / n)
    res = minimize(
        neg_sharpe, w0, method="SLSQP", bounds=bounds,
        constraints=constraints, options={"maxiter": 500, "ftol": 1e-9},
    )
    if not res.success:
        return w0
    w = np.clip(res.x, 0, None)
    s = w.sum()
    return w / s if s > 0 else w0


def min_vol_weights(
    mu: np.ndarray,
    cov: np.ndarray,
    freq: int = 252,
    max_w: float = 1.0,
) -> np.ndarray:
    n = len(mu)
    bounds = [(0.0, max_w)] * n
    constraints = [{"type": "eq", "fun": lambda w: w.sum() - 1.0}]

    def port_vol(w: np.ndarray) -> float:
        return float(np.sqrt(w @ cov @ w) * np.sqrt(freq))

    w0 = np.full(n, 1.0 / n)
    res = minimize(
        port_vol, w0, method="SLSQP", bounds=bounds,
        constraints=constraints, options={"maxiter": 500, "ftol": 1e-9},
    )
    if not res.success:
        return w0
    w = np.clip(res.x, 0, None)
    s = w.sum()
    return w / s if s > 0 else w0


def risk_parity_weights(
    cov: np.ndarray, freq: int = 252, max_iter: int = 200, tol: float = 1e-7
) -> np.ndarray:
    n = len(cov)
    w = np.full(n, 1.0 / n)
    for _ in range(max_iter):
        port_var = w @ cov @ w
        if port_var <= 0:
            return np.full(n, 1.0 / n)
        marginal = cov @ w
        risk_contrib = w * marginal / np.sqrt(port_var)
        target = float(np.mean(risk_contrib))
        if target <= 0:
            return np.full(n, 1.0 / n)
        ratio = risk_contrib / target
        w_new = w * (1.0 - 0.05) + (w / ratio) * 0.05
        w_new = np.clip(w_new, 1e-6, None)
        w_new = w_new / w_new.sum()
        if np.max(np.abs(w_new - w)) < tol:
            return w_new
        w = w_new
    return w


def efficient_frontier_points(
    mu: np.ndarray,
    cov: np.ndarray,
    n_points: int = 40,
    freq: int = 252,
    max_w: float = 1.0,
):
    n = len(mu)
    bounds = [(0.0, max_w)] * n
    rets_grid = np.linspace(mu.min() * freq, mu.max() * freq * 1.05, n_points)
    out_rets, out_vols, out_w = [], [], []

    for tr in rets_grid:
        cons = [
            {"type": "eq", "fun": lambda w: w.sum() - 1.0},
            {"type": "eq", "fun": lambda w, tr_=tr: w @ mu * freq - tr_},
        ]

        def vol_fn(w: np.ndarray) -> float:
            return float(np.sqrt(w @ cov @ w) * np.sqrt(freq))

        w0 = np.full(n, 1.0 / n)
        res = minimize(
            vol_fn, w0, method="SLSQP", bounds=bounds,
            constraints=cons, options={"maxiter": 500, "ftol": 1e-9},
        )
        if res.success:
            w = np.clip(res.x, 0, None)
            s = w.sum()
            if s > 0:
                w = w / s
                out_rets.append(tr)
                out_vols.append(float(np.sqrt(w @ cov @ w) * np.sqrt(freq)))
                out_w.append(w)
    return (
        np.array(out_rets),
        np.array(out_vols),
        np.array(out_w) if out_w else np.empty((0, n)),
    )


def random_portfolios(
    mu: np.ndarray,
    cov: np.ndarray,
    n: int = 5000,
    freq: int = 252,
    max_w: float = 1.0,
    seed: int = 42,
):
    rng = np.random.default_rng(seed)
    n_assets = len(mu)
    raw = rng.dirichlet(np.ones(n_assets), size=n)
    weights = raw * rng.uniform(0, max_w, size=(n, 1))
    weights = weights / weights.sum(axis=1, keepdims=True)
    rets = weights @ mu * freq
    vols = np.sqrt(np.einsum("ij,jk,ik->i", weights, cov, weights)) * np.sqrt(freq)
    sharpes = np.divide(rets, vols, out=np.full_like(rets, np.nan), where=vols > 0)
    return weights, rets, vols, sharpes
