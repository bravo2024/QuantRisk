# QuantRisk

> Multi-asset portfolio risk management dashboard with yfinance data, mean-variance optimisation, VaR/CVaR, Fama-French factor models, Monte Carlo simulation, and ML return forecasting.

Supports any ticker supported by yfinance — Indian NSE (`.NS`), BSE (`.BO`), US equities/ETFs, indices, and custom inputs. Auto-selects Nifty 50 as benchmark for Indian tickers, SPY otherwise.

## Quickstart

```bash
pip install -r requirements.txt
python train.py
pytest -q
streamlit run app.py
```

## Model Performance

Equal-weight portfolio of 12 Indian NSE large-caps:

| Metric | Value |
|---|---|
| Annualized Return | 9.70% |
| Annualized Volatility | 12.25% |
| Sharpe Ratio | 0.302 |
| VaR (95%) | 1.13% |
| Max Drawdown | −15.00% |

Max-Sharpe portfolio: Sharpe 1.604, Return 28.87%, Vol 14.26%.

## Features

| Tab | What it does |
|---|---|
| **Overview** | Asset universe stats, cumulative returns, correlation matrix, rolling vol |
| **Portfolio Optimization** | Efficient frontier, Max-Sharpe/Min-Vol/Risk-Parity/Equal-Weight, backtest vs benchmark |
| **Risk Analytics** | Historical / Parametric / Monte-Carlo VaR & ES, drawdown, risk contribution, stress scenarios |
| **Factor Model** | Fama-French 5-factor regression per asset, factor betas, rolling market beta |
| **Monte Carlo** | GBM simulation with configurable horizon + paths, percentiles, terminal distribution |
| **ML Forecasting** | Gradient Boosting / Random Forest return prediction, RMSE, direction accuracy, SHAP |
| **Model Card** | Methodology, assumptions, Basel III/IV regulatory context |

## Repo Structure

```
QuantRisk/
  src/         data, core, model modules
  train.py     data fetching + metric computation
  app.py       Streamlit dashboard
  tests/       pytest smoke test
  models/      saved model + metrics (gitignored)
  data/        cached data (gitignored)
```

## Data

Real-time data via yfinance for any ticker. Falls back to synthetic correlated GBM returns if yfinance is unavailable. Indian NSE tickers use `.NS` suffix (e.g., `RELIANCE.NS`), BSE use `.BO`. Benchmark auto-selects `^NSEI` or `SPY`.

## License

MIT
