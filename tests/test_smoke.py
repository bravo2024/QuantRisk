from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
import numpy as np


def test_import_src():
    from src.data import fetch_prices, compute_returns, synthetic_returns, make_synthetic
    from src.core import portfolio_stats, value_at_risk, expected_shortfall, parametric_var
    from src.model import max_sharpe_weights, min_vol_weights, risk_parity_weights

    assert fetch_prices is not None
    assert compute_returns is not None
    assert synthetic_returns is not None
    assert make_synthetic is not None
    assert portfolio_stats is not None
    assert value_at_risk is not None
    assert expected_shortfall is not None
    assert parametric_var is not None
    assert max_sharpe_weights is not None
    assert min_vol_weights is not None
    assert risk_parity_weights is not None


def test_synthetic_data():
    from src.data import make_synthetic

    syn = make_synthetic(n_days=252, n_assets=4, seed=42)
    assert "prices" in syn
    assert "returns" in syn
    assert "tickers" in syn
    assert syn["prices"].shape == (252, 4)
    assert syn["returns"].shape == (252, 4)


def test_portfolio_stats():
    from src.data import make_synthetic
    from src.core import portfolio_stats

    syn = make_synthetic(n_days=252, n_assets=4, seed=42)
    mu = syn["returns"].mean().values
    cov = syn["returns"].cov().values
    w = np.ones(4) / 4
    ret, vol, sharpe = portfolio_stats(w, mu, cov, rf=0.06, freq=252)
    assert isinstance(ret, float)
    assert isinstance(vol, float)
    assert isinstance(sharpe, float)
    assert vol > 0


def test_risk_metrics():
    from src.data import make_synthetic
    from src.core import value_at_risk, expected_shortfall, parametric_var

    rng = np.random.default_rng(42)
    returns = rng.normal(0.001, 0.02, 1000)
    var = value_at_risk(returns, 0.05)
    es = expected_shortfall(returns, 0.05)
    pvar = parametric_var(returns, 0.05)
    assert var > 0
    assert es >= var
    assert pvar > 0


def test_weight_optimization():
    from src.data import make_synthetic
    from src.model import max_sharpe_weights, min_vol_weights, risk_parity_weights

    syn = make_synthetic(n_days=252, n_assets=4, seed=42)
    mu = syn["returns"].mean().values
    cov = syn["returns"].cov().values

    w_ms = max_sharpe_weights(mu, cov)
    w_mv = min_vol_weights(mu, cov)
    w_rp = risk_parity_weights(cov)

    for w, name in [(w_ms, "max_sharpe"), (w_mv, "min_vol"), (w_rp, "risk_parity")]:
        assert len(w) == 4, f"{name} wrong length"
        assert abs(w.sum() - 1.0) < 1e-6, f"{name} not normalized"
        assert np.all(w >= 0), f"{name} has negative weights"


def test_compute_portfolio_metrics():
    from src.data import make_synthetic
    from src.core import compute_portfolio_metrics

    syn = make_synthetic(n_days=252, n_assets=4, seed=42)
    w = np.ones(4) / 4
    metrics = compute_portfolio_metrics(syn["returns"], w, rf=0.06)
    assert "annualized_return" in metrics
    assert "annualized_volatility" in metrics
    assert "sharpe_ratio" in metrics
    assert "max_drawdown" in metrics
    assert "var_95" in metrics
    assert metrics["n_samples"] > 0
    assert metrics["annualized_volatility"] > 0
