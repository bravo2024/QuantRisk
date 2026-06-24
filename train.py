from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
import yfinance as yf
from src.data import fetch_prices, compute_returns, INDIAN_NSE_12
from src.core import compute_portfolio_metrics
from src.model import max_sharpe_weights, min_vol_weights, risk_parity_weights

DATA_DIR = Path(__file__).parent / "data"
MODELS_DIR = Path(__file__).parent / "models"
DATA_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)


def main():
    tickers = INDIAN_NSE_12

    print("Fetching prices via yfinance...")
    prices = fetch_prices(tickers)
    if prices is None or prices.empty:
        print("yfinance failed. Generating synthetic data for demo.")
        from src.data import make_synthetic

        syn = make_synthetic(n_days=756, n_assets=6)
        prices = syn["prices"]
        returns = syn["returns"]
        tickers = syn["tickers"]
    else:
        tickers = list(prices.columns)
        print(f"Got {len(tickers)} tickers, {len(prices)} days.")
        returns = compute_returns(prices)

    print("Computing metrics...")
    mu = returns.mean().values
    cov = returns.cov().values

    portfolios = {
        "Equal_Weight": np.full(len(tickers), 1.0 / len(tickers)),
    }
    portfolios["Max_Sharpe"] = max_sharpe_weights(mu, cov)
    portfolios["Min_Volatility"] = min_vol_weights(mu, cov)
    portfolios["Risk_Parity"] = risk_parity_weights(cov)

    results = {}
    for name, w in portfolios.items():
        metrics = compute_portfolio_metrics(returns, w)
        results[name] = {
            "weights": {t: float(w[i]) for i, t in enumerate(tickers)},
            "metrics": {k: float(v) if isinstance(v, (np.floating, float)) else v for k, v in metrics.items()},
        }

    # Benchmark comparison: Nifty 50
    print("Fetching benchmark (^NSEI)...")
    benchmark = None
    try:
        bench_data = yf.download("^NSEI", start=prices.index[0], end=prices.index[-1],
                                  progress=False, auto_adjust=True)
        if bench_data is None or bench_data.empty:
            raise ValueError("No benchmark data")
        if isinstance(bench_data.columns, pd.MultiIndex):
            bench_series = bench_data["Close"].squeeze() if "Close" in bench_data.columns.get_level_values(0) else bench_data.iloc[:, 0].squeeze()
        else:
            bench_series = bench_data["Close"].squeeze() if "Close" in bench_data.columns else bench_data.iloc[:, 0].squeeze()
        bench_series = bench_series.dropna()
        if len(bench_series) < 10:
            raise ValueError("Too few benchmark data points")
        bench_ret = bench_series.pct_change(fill_method=None).dropna()
        bench_ret = bench_ret.reindex(returns.index).ffill().dropna()
        if len(bench_ret) > 50:
            ew_ret = returns.mean(axis=1).reindex(bench_ret.index)
            ew_ret, bench_ret = ew_ret.align(bench_ret, join="inner")
            ew_r = float(ew_ret.mean() * 252)
            bench_r = float(bench_ret.mean() * 252)
            te = float((ew_ret - bench_ret).std() * np.sqrt(252))
            ir = (ew_r - bench_r) / te if te > 0 else 0
            alpha = ew_r - bench_r
            benchmark = {
                "ticker": "^NSEI",
                "name": "Nifty 50",
                "return": round(bench_r, 6),
                "volatility": round(float(bench_ret.std() * np.sqrt(252)), 6),
                "sharpe": round((bench_r - 0.06) / float(bench_ret.std() * np.sqrt(252)) if bench_ret.std() > 0 else 0, 6),
                "equal_weight_return": round(ew_r, 6),
                "alpha": round(alpha, 6),
                "tracking_error": round(te, 6),
                "information_ratio": round(ir, 6),
            }
            print(f"  Nifty 50 return={bench_r:.2%}  Alpha={alpha:+.2%}  TE={te:.2%}  IR={ir:.3f}")
    except Exception:
        pass

    output = {
        "n_tickers": len(tickers),
        "n_samples": len(returns),
        "tickers": tickers,
        "portfolios": results,
        "benchmark": benchmark,
    }

    metrics_path = MODELS_DIR / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Metrics saved to {metrics_path}")

    # Save weights as simple model
    weights_path = MODELS_DIR / "model.pkl"
    import pickle
    with open(weights_path, "wb") as f:
        pickle.dump(
            {
                "tickers": tickers,
                "mu": mu,
                "cov": cov,
                "portfolios": {k: v["weights"] for k, v in results.items()},
            },
            f,
        )
    print(f"Model saved to {weights_path}")

    # Print summary
    print("\n--- Portfolio Comparison ---")
    for name, r in results.items():
        m = r["metrics"]
        print(
            f"{name:20s}  Return={m['annualized_return']:.2%}  "
            f"Vol={m['annualized_volatility']:.2%}  "
            f"Sharpe={m['sharpe_ratio']:.3f}  "
            f"VaR95={m['var_95']:.2%}  "
            f"MaxDD={m['max_drawdown']:.2%}"
        )


if __name__ == "__main__":
    main()
