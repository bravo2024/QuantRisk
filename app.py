from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import json
import warnings
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm

import yfinance as yf
from src.data import (
    fetch_prices,
    compute_returns,
    synthetic_returns,
    TICKER_PRESETS,
)
from src.core import (
    portfolio_stats,
    value_at_risk,
    expected_shortfall,
    parametric_var,
    risk_contribution,
    technical_indicators,
)
from src.model import (
    max_sharpe_weights,
    min_vol_weights,
    risk_parity_weights,
    efficient_frontier_points,
    random_portfolios,
    monte_carlo_paths,
)

try:
    import statsmodels.api as sm
    HAS_STATSMODELS = True
except Exception:
    HAS_STATSMODELS = False

try:
    import pandas_datareader.data as pdr
    HAS_PDR = True
except Exception:
    HAS_PDR = False

try:
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.preprocessing import StandardScaler
    HAS_SKLEARN = True
except Exception:
    HAS_SKLEARN = False

try:
    import shap
    HAS_SHAP = True
except Exception:
    HAS_SHAP = False

warnings.filterwarnings("ignore")
np.random.seed(42)

st.set_page_config(
    page_title="QuantRisk",
    page_icon="\U0001f4ca",
    layout="wide",
    initial_sidebar_state="expanded",
)

defaults = {
    "data_loaded": False,
    "tickers": TICKER_PRESETS["Indian NSE 12"],
    "start_date": datetime.now() - timedelta(days=365 * 3),
    "end_date": datetime.now(),
    "prices": None,
    "returns": None,
    "benchmark": None,
    "rf_rate": 0.06,
    "data_error": None,
    "loading": False,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

st.markdown(
    """
<style>
    .main-header { font-size: 2.4rem; font-weight: 700; color: #1e293b; margin-bottom: 4px; }
    .sub-header  { font-size: 1.1rem; color: #64748b; margin-bottom: 16px; }
    .section-title { font-size: 1.5rem; font-weight: 600; color: #0f172a;
        border-bottom: 3px solid #1a56db; padding-bottom: 6px; margin: 24px 0 16px 0; }
    .metric-card { background: #f1f5f9; border-radius: 12px; padding: 20px;
        border-left: 5px solid #1a56db; box-shadow: 0 1px 3px rgba(0,0,0,.06); }
    hr { margin: 20px 0; }
    .footer { text-align: center; color: #94a3b8; font-size: .85rem; padding-top: 30px; }
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_data(ttl=86400, show_spinner=False)
def fetch_factors(start_date, end_date):
    if not HAS_PDR:
        return None
    try:
        ff = pdr.DataReader(
            "F-F_Research_Data_5_Factors_2x3", "famafrench",
            start=start_date, end=end_date,
        )
        df = ff[0] if isinstance(ff, dict) else ff
        df.index = df.index.to_timestamp(how="end").normalize()
        return df / 100.0
    except Exception:
        return None


st.sidebar.markdown(
    "<h1 style='font-size:1.6rem; margin-bottom:0;'>\U0001f4ca QuantRisk</h1>",
    unsafe_allow_html=True,
)
st.sidebar.caption("Portfolio risk dashboard")
st.sidebar.markdown("---")

st.sidebar.markdown("### Asset Universe")
preset_name = st.sidebar.selectbox(
    "Quick presets", list(TICKER_PRESETS.keys()), index=0,
    disabled=st.session_state.loading,
)
ticker_text = st.sidebar.text_input(
    "Tickers (comma-separated)",
    value=", ".join(st.session_state.tickers),
    help="Indian NSE: append .NS (e.g., RELIANCE.NS). US: bare ticker (e.g., SPY).",
    disabled=st.session_state.loading,
)
parsed_tickers = [t.strip().upper() for t in ticker_text.split(",") if t.strip()]
if st.sidebar.button("Apply preset", disabled=st.session_state.loading):
    st.session_state.tickers = TICKER_PRESETS[preset_name][:]
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.markdown("### Date Range")
col_a, col_b = st.sidebar.columns(2)
start_date = col_a.date_input("Start", st.session_state.start_date, disabled=st.session_state.loading)
end_date = col_b.date_input("End", st.session_state.end_date, disabled=st.session_state.loading)

st.sidebar.markdown("---")
st.sidebar.markdown("### Risk-Free Rate")
rf_rate = st.sidebar.slider(
    "Annualized (decimal)", 0.0, 0.15, st.session_state.rf_rate, 0.005,
    disabled=st.session_state.loading, format="%.3f",
)
st.session_state.rf_rate = rf_rate

st.sidebar.markdown("---")
if st.sidebar.button(
    "\U0001f4e1 Fetch Market Data", type="primary",
    use_container_width=True, disabled=st.session_state.loading,
):
    if not parsed_tickers:
        st.sidebar.error("Enter at least one ticker.")
    else:
        st.session_state.tickers = parsed_tickers
        st.session_state.start_date = start_date
        st.session_state.end_date = end_date
        st.session_state.loading = True
        st.rerun()


def _do_fetch():
    progress = st.sidebar.progress(0)
    status = st.sidebar.empty()
    status.text("Fetching prices\u2026")
    progress.progress(15)
    prices = fetch_prices(st.session_state.tickers, start_date, end_date)
    progress.progress(45)
    if prices is None or prices.empty:
        status.text("yfinance failed - using synthetic fallback\u2026")
        rets = synthetic_returns(st.session_state.tickers, n_days=756)
        prices = (1 + rets).cumprod() * 100.0
        st.session_state.data_error = "yfinance failed; using synthetic correlated returns."
    else:
        st.session_state.data_error = None
        st.session_state.tickers = list(prices.columns)
    progress.progress(60)
    rets = compute_returns(prices, freq="D")
    status.text("Loading benchmark\u2026")
    has_ns = any(t.endswith(".NS") for t in st.session_state.tickers)
    has_bo = any(t.endswith(".BO") for t in st.session_state.tickers)
    has_nsebank = any("BANK" in t for t in st.session_state.tickers)
    if has_nsebank:
        benchmark_ticker = "^NSEBANK"
    elif has_ns or has_bo:
        benchmark_ticker = "^NSEI"
    else:
        benchmark_ticker = "SPY"
    try:
        bench_data = yf.download(benchmark_ticker, start=start_date, end=end_date,
                                  progress=False, auto_adjust=True)
        if isinstance(bench_data.columns, pd.MultiIndex):
            bench = bench_data["Close"].copy()
        else:
            bench = bench_data.copy()
        bench = bench.squeeze().dropna()
    except Exception:
        bench = pd.Series(dtype=float)
    progress.progress(80)
    status.text("Loading Fama-French factors\u2026")
    factors = fetch_factors(start_date, end_date)
    progress.progress(100)
    st.session_state.prices = prices
    st.session_state.returns = rets
    st.session_state.benchmark = bench
    st.session_state.factors = factors
    st.session_state.data_loaded = True
    st.session_state.loading = False
    status.empty()
    progress.empty()
    st.rerun()


if st.session_state.loading:
    _do_fetch()


def render_overview(prices, returns, rf, benchmark=None):
    st.markdown("<div class='main-header'>Overview</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='sub-header'>Asset universe summary, return statistics, and "
        "pairwise relationships for the selected portfolio.</div>",
        unsafe_allow_html=True,
    )
    st.markdown("---")

    ann_factor = 252
    mu = returns.mean().values * ann_factor
    vol = returns.std().values * np.sqrt(ann_factor)
    sharpe = np.where(vol > 0, (mu - rf) / vol, 0.0)
    cumulative = (1 + returns).cumprod()
    rolling_max = cumulative.cummax()
    drawdowns = (cumulative / rolling_max - 1)
    max_dd = drawdowns.min().values

    summary = pd.DataFrame({
        "Ticker": prices.columns,
        "Annualized Return": mu,
        "Annualized Vol": vol,
        "Sharpe": sharpe,
        "Max Drawdown": max_dd,
    }).set_index("Ticker").round(4)
    st.markdown("### Asset Universe")
    st.dataframe(
        summary.style.format({
            "Annualized Return": "{:.2%}",
            "Annualized Vol": "{:.2%}",
            "Sharpe": "{:.3f}",
            "Max Drawdown": "{:.2%}",
        }).background_gradient(subset=["Sharpe"], cmap="RdYlGn", vmin=-1, vmax=3),
        use_container_width=True,
    )

    st.markdown("### Cumulative Returns (Normalized to 100)")
    norm_p = cumulative / cumulative.iloc[0] * 100
    fig, ax = plt.subplots(figsize=(11, 5))
    for col in norm_p.columns:
        ax.plot(norm_p.index, norm_p[col], linewidth=1.5, label=col)
    ax.set_ylabel("Index Level (Start = 100)")
    ax.set_title("Cumulative Returns", fontweight="bold")
    ax.legend(loc="best", fontsize=9, ncol=min(4, len(norm_p.columns)))
    ax.grid(True, alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    st.pyplot(fig)

    st.markdown("### Correlation Matrix")
    fig, ax = plt.subplots(figsize=(9, 7))
    corr = returns.corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0,
                square=True, ax=ax, cbar_kws={"shrink": 0.8}, vmin=-1, vmax=1)
    ax.set_title("Daily Return Correlations", fontweight="bold")
    st.pyplot(fig)

    st.markdown("### Rolling 60-Day Volatility")
    rolling_vol = returns.rolling(60).std() * np.sqrt(252)
    fig, ax = plt.subplots(figsize=(11, 5))
    for col in rolling_vol.columns:
        ax.plot(rolling_vol.index, rolling_vol[col], linewidth=1.3, label=col)
    ax.set_ylabel("Annualized Volatility")
    ax.set_title("60-Day Rolling Volatility", fontweight="bold")
    ax.legend(loc="best", fontsize=9, ncol=min(4, len(rolling_vol.columns)))
    ax.grid(True, alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    st.pyplot(fig)

    if benchmark is not None and len(benchmark) > 20:
        st.markdown("### Equal-Weight Portfolio vs Benchmark")
        bench_ret = benchmark.pct_change().dropna()
        bench_ret = bench_ret.reindex(returns.index, method="ffill").dropna()
        if len(bench_ret) > 1:
            ew_ret = returns.mean(axis=1).reindex(bench_ret.index)
            ew_ret, bench_ret = ew_ret.align(bench_ret, join="inner")
            if len(ew_ret) > 1:
                ew_cum = (1 + ew_ret).cumprod()
                bench_cum = (1 + bench_ret).cumprod()
                ew_r = ew_ret.mean() * 252
                ew_v = ew_ret.std() * np.sqrt(252)
                ew_s = (ew_r - rf) / ew_v if ew_v > 0 else 0
                bench_r = bench_ret.mean() * 252
                bench_v = bench_ret.std() * np.sqrt(252)
                bench_s = (bench_r - rf) / bench_v if bench_v > 0 else 0
                te = (ew_ret - bench_ret).std() * np.sqrt(252)
                ir = (ew_r - bench_r) / te if te > 0 else 0
                alpha = ew_r - bench_r

                cols = st.columns(4)
                for col, (label, val) in zip(cols, [
                    ("Portfolio Return", f"{ew_r:.2%}"),
                    ("Benchmark Return", f"{bench_r:.2%}"),
                    ("Alpha", f"{alpha:+.2%}"),
                    ("Info Ratio", f"{ir:.3f}"),
                ]):
                    col.markdown(
                        f"<div class='metric-card'><div style='font-size:0.85rem;color:#64748b;'>{label}</div>"
                        f"<div style='font-size:1.4rem;font-weight:700;color:#0f172a;'>{val}</div></div>",
                        unsafe_allow_html=True,
                    )
                cols = st.columns(4)
                for col, (label, val) in zip(cols, [
                    ("Portfolio Vol", f"{ew_v:.2%}"),
                    ("Benchmark Vol", f"{bench_v:.2%}"),
                    ("Portfolio Sharpe", f"{ew_s:.3f}"),
                    ("Tracking Error", f"{te:.2%}"),
                ]):
                    col.markdown(
                        f"<div class='metric-card'><div style='font-size:0.85rem;color:#64748b;'>{label}</div>"
                        f"<div style='font-size:1.4rem;font-weight:700;color:#0f172a;'>{val}</div></div>",
                        unsafe_allow_html=True,
                    )

                bm_name = "Nifty 50" if "^NSEI" in str(benchmark.name) else "Bank Nifty" if "^NSEBANK" in str(benchmark.name) else "Benchmark"
                fig, ax = plt.subplots(figsize=(11, 4))
                ax.plot(ew_cum.index, ew_cum / ew_cum.iloc[0] * 100,
                        label="Equal-Weight Portfolio", linewidth=2, color="#1a56db")
                ax.plot(bench_cum.index, bench_cum / bench_cum.iloc[0] * 100,
                        label=bm_name, linewidth=2, color="#ef4444", linestyle="--")
                ax.set_ylabel("Index Level (Start = 100)")
                ax.set_title(f"Equal-Weight Portfolio vs {bm_name}", fontweight="bold")
                ax.legend()
                ax.grid(True, alpha=0.3)
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)
                st.pyplot(fig)


def render_portfolio_opt(returns, rf, prices, benchmark):
    st.markdown("<div class='main-header'>Portfolio Optimization</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='sub-header'>Mean-variance efficient frontier, optimal portfolios, "
        "and a backtest against the benchmark.</div>",
        unsafe_allow_html=True,
    )
    st.markdown("---")

    freq = 252
    mu = returns.mean().values
    cov = returns.cov().values
    n_assets = len(returns.columns)

    if st.button("Run Optimization", type="primary"):
        st.session_state._run_opt = True
    if not st.session_state.get("_run_opt"):
        st.info("Click **Run Optimization** to compute the efficient frontier and optimal weights.")
        return

    progress = st.progress(0)
    status = st.empty()
    status.text("Computing random portfolios\u2026")
    progress.progress(20)
    rp_w, rp_r, rp_v, rp_s = random_portfolios(mu, cov, n=5000, freq=freq)
    progress.progress(50)
    status.text("Finding optimal portfolios\u2026")
    w_ms = max_sharpe_weights(mu, cov, rf=rf, freq=freq)
    w_mv = min_vol_weights(mu, cov, freq=freq)
    w_rp = risk_parity_weights(cov, freq=freq)
    w_eq = np.full(n_assets, 1.0 / n_assets)
    progress.progress(75)
    status.text("Tracing efficient frontier\u2026")
    ef_rets, ef_vols, _ = efficient_frontier_points(mu, cov, n_points=40, freq=freq)
    progress.progress(100)
    status.empty()
    progress.empty()

    r_ms, v_ms, s_ms = portfolio_stats(w_ms, mu, cov, rf, freq)
    r_mv, v_mv, s_mv = portfolio_stats(w_mv, mu, cov, rf, freq)
    r_rp, v_rp, s_rp = portfolio_stats(w_rp, mu, cov, rf, freq)
    r_eq, v_eq, s_eq = portfolio_stats(w_eq, mu, cov, rf, freq)

    portfolios_data = [
        ("Max Sharpe", w_ms, r_ms, v_ms, s_ms),
        ("Min Volatility", w_mv, r_mv, v_mv, s_mv),
        ("Risk Parity", w_rp, r_rp, v_rp, s_rp),
        ("Equal Weight", w_eq, r_eq, v_eq, s_eq),
    ]
    cols = st.columns(4)
    for col, (name, _, r, v, s) in zip(cols, portfolios_data):
        col.markdown(
            f"<div class='metric-card'>"
            f"<div style='font-size:0.85rem;color:#64748b;'>{name}</div>"
            f"<div style='font-size:1.4rem;font-weight:700;color:#0f172a;'>{r:.2%}</div>"
            f"<div style='font-size:0.85rem;color:#475569;'>Vol {v:.2%} \u00b7 Sharpe {s:.2f}</div></div>",
            unsafe_allow_html=True,
        )

    st.markdown("### Efficient Frontier")
    fig, ax = plt.subplots(figsize=(11, 6))
    sc = ax.scatter(rp_v, rp_r, c=rp_s, cmap="viridis", alpha=0.45, s=14,
                    vmin=np.nanpercentile(rp_s, 5), vmax=np.nanpercentile(rp_s, 95))
    plt.colorbar(sc, ax=ax, label="Sharpe Ratio")
    if len(ef_vols) > 0:
        ax.plot(ef_vols, ef_rets, "b-", linewidth=2.5, label="Efficient Frontier")
    for name, w, r, v, s in portfolios_data:
        marker = "*" if name == "Max Sharpe" else "o"
        size = 220 if name == "Max Sharpe" else 130
        ax.scatter(v, r, marker=marker, s=size,
                   edgecolors="black", linewidths=1.5,
                   label=f"{name} (S={s:.2f})", zorder=5)
    ax.set_xlabel("Annualized Volatility")
    ax.set_ylabel("Annualized Return")
    ax.set_title("Mean-Variance Efficient Frontier", fontweight="bold")
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    st.pyplot(fig)

    st.markdown("### Optimal Weights")
    weights_df = pd.DataFrame(
        {name: w for name, w, _, _, _ in portfolios_data},
        index=returns.columns,
    ).round(4)
    st.dataframe(
        weights_df.style.format("{:.2%}").background_gradient(cmap="Blues", axis=0),
        use_container_width=True,
    )

    fig, axes = plt.subplots(1, 4, figsize=(15, 4), sharey=True)
    colors = sns.color_palette("Set2", n_assets)
    for ax, (name, w, _, _, _) in zip(axes, portfolios_data):
        ax.pie(w, labels=returns.columns, autopct="%1.1f%%", startangle=90,
               colors=colors, textprops={"fontsize": 8})
        ax.set_title(name, fontweight="bold")
    st.pyplot(fig)

    st.markdown("### Backtest: Max Sharpe vs Benchmark")
    port_ret = returns @ w_ms
    cum_port = (1 + port_ret).cumprod()
    cum_eq = (1 + returns @ w_eq).cumprod()
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(cum_port.index, cum_port / cum_port.iloc[0] * 100,
            label="Max Sharpe Portfolio", linewidth=2, color="#1a56db")
    ax.plot(cum_eq.index, cum_eq / cum_eq.iloc[0] * 100,
            label="Equal Weight Portfolio", linewidth=1.8, color="#22c55e", alpha=0.85)
    if benchmark is not None and len(benchmark) > 20:
        bench_ret = benchmark.pct_change().dropna()
        bench_ret = bench_ret.reindex(returns.index, method="ffill").dropna()
        if len(bench_ret) > 1:
            cum_bench = (1 + bench_ret).cumprod()
            ax.plot(cum_bench.index, cum_bench / cum_bench.iloc[0] * 100,
                    label="Benchmark", linewidth=1.5, color="#ef4444", linestyle="--")
    ax.set_ylabel("Index Level (Start = 100)")
    ax.set_title("Cumulative Performance", fontweight="bold")
    ax.legend(loc="best", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    st.pyplot(fig)


def render_risk_analytics(returns, rf, prices):
    st.markdown("<div class='main-header'>Risk Analytics</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='sub-header'>Value-at-Risk, Expected Shortfall, drawdown, "
        "risk contribution, and stress scenarios.</div>",
        unsafe_allow_html=True,
    )
    st.markdown("---")

    port_choice = st.selectbox(
        "Portfolio weights", ["Equal Weight", "Max Sharpe (computed)"], index=0,
    )
    if port_choice == "Max Sharpe (computed)":
        mu_v = returns.mean().values
        cov_v = returns.cov().values
        w = max_sharpe_weights(mu_v, cov_v, rf=rf)
    else:
        w = np.full(len(returns.columns), 1.0 / len(returns.columns))
    port_ret = returns @ w

    if st.button("Compute Risk Metrics", type="primary"):
        st.session_state._run_risk = True
    if not st.session_state.get("_run_risk"):
        st.info("Click **Compute Risk Metrics** to run VaR/CVaR/drawdown/risk-contribution analysis.")
        return

    alpha = 0.05
    var_h = value_at_risk(port_ret, alpha)
    var_p = parametric_var(port_ret, alpha)
    rng = np.random.default_rng(0)
    mc_boot = rng.choice(port_ret.values, size=10000, replace=True)
    var_mc = float(-np.quantile(mc_boot, alpha))
    es_h = expected_shortfall(port_ret, alpha)
    z_a = norm.ppf(alpha)
    es_p = float(-(port_ret.mean() - port_ret.std() * norm.pdf(z_a) / alpha))
    es_mc = float(-mc_boot[mc_boot <= np.quantile(mc_boot, alpha)].mean())

    var_data = pd.DataFrame({
        "Method": ["Historical", "Parametric (Normal)", "Monte Carlo"],
        "VaR (95%)": [var_h, var_p, var_mc],
        "ES / CVaR (95%)": [es_h, es_p, es_mc],
    }).set_index("Method")

    col1, col2 = st.columns([1, 1])
    col1.markdown("<div class='section-title'>VaR & ES Comparison</div>", unsafe_allow_html=True)
    col1.dataframe(var_data.style.format("{:.4%}"), use_container_width=True)

    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(var_data))
    width = 0.35
    ax.bar(x - width / 2, var_data["VaR (95%)"], width, label="VaR", color="#1a56db")
    ax.bar(x + width / 2, var_data["ES / CVaR (95%)"], width, label="ES/CVaR", color="#ef4444")
    ax.set_xticks(x)
    ax.set_xticklabels(var_data.index)
    ax.set_ylabel("Daily Loss")
    ax.set_title("VaR vs ES (95% confidence)", fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    col2.pyplot(fig)

    st.markdown("### Rolling 60-Day VaR (Historical, 95%)")
    rolling_var = port_ret.rolling(60).quantile(0.05) * -1
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(rolling_var.index, rolling_var, color="#1a56db", linewidth=1.5, label="Rolling VaR")
    ax.axhline(var_h, color="red", linestyle="--", alpha=0.7, label=f"Full-sample VaR ({var_h:.2%})")
    ax.fill_between(rolling_var.index, 0, rolling_var, color="#1a56db", alpha=0.15)
    ax.set_ylabel("VaR (Daily Loss)")
    ax.set_title("60-Day Rolling VaR", fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    st.pyplot(fig)

    st.markdown("### Drawdown (Underwater Chart)")
    cum = (1 + port_ret).cumprod()
    running_max = cum.cummax()
    drawdown = cum / running_max - 1
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.fill_between(drawdown.index, drawdown, 0, color="#ef4444", alpha=0.5)
    ax.plot(drawdown.index, drawdown, color="#991b1b", linewidth=1)
    ax.set_ylabel("Drawdown")
    ax.set_title(f"Cumulative Drawdown (Max: {drawdown.min():.2%})", fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    st.pyplot(fig)

    st.markdown("### Risk Contribution Decomposition")
    cov_v = returns.cov().values
    marginal, component, proportional = risk_contribution(w, cov_v)
    rc_df = pd.DataFrame({
        "Marginal Risk": marginal,
        "Component Risk": component,
        "Proportional Risk %": proportional / proportional.sum() if proportional.sum() > 0 else proportional,
    }, index=returns.columns).round(4)
    st.dataframe(
        rc_df.style.format({
            "Marginal Risk": "{:.4f}",
            "Component Risk": "{:.4f}",
            "Proportional Risk %": "{:.2%}",
        }).background_gradient(subset=["Proportional Risk %"], cmap="Reds"),
        use_container_width=True,
    )

    st.markdown("### Stress Scenarios")
    scenarios = pd.DataFrame({
        "Scenario": ["2008 GFC", "COVID Crash (Mar 2020)", "2022 Rate Hike", "2023 Banking Crisis"],
        "1-Day Shock": [-0.10, -0.12, -0.04, -0.06],
    })
    scenarios["Portfolio Impact"] = scenarios["1-Day Shock"].values
    fig, ax = plt.subplots(figsize=(9, 3.5))
    ax.barh(scenarios["Scenario"], scenarios["Portfolio Impact"], color="#dc2626", alpha=0.8)
    for i, v in enumerate(scenarios["Portfolio Impact"]):
        ax.text(v - 0.001, i, f"{v:.2%}", va="center", ha="right", color="white", fontweight="bold")
    ax.set_xlabel("Portfolio Loss")
    ax.set_title("Hypothetical 1-Day Portfolio Loss by Scenario", fontweight="bold")
    ax.grid(True, alpha=0.3, axis="x")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    st.pyplot(fig)


def render_factor_model(returns, rf, factors, prices):
    st.markdown("<div class='main-header'>Factor Model</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='sub-header'>Fama-French 5-factor regression for each asset. "
        "Demonstrates systematic risk decomposition.</div>",
        unsafe_allow_html=True,
    )
    st.markdown("---")

    if factors is None:
        st.warning("Fama-French factors unavailable. pandas-datareader failed or is not installed.")
        st.info("Methodology: regress asset returns on Mkt-RF, SMB, HML, RMW, CMA via OLS.")
        return

    monthly = returns.resample("ME").apply(lambda x: (1 + x).prod() - 1).dropna()
    merged = monthly.join(factors, how="inner")
    if len(merged) < 24:
        st.warning("Insufficient overlap between asset returns and factor data (< 24 months).")
        return

    asset_cols = list(monthly.columns)
    factor_cols = ["Mkt-RF", "SMB", "HML", "RMW", "CMA"]

    if st.button("Run Factor Regressions", type="primary"):
        st.session_state._run_factor = True
    if not st.session_state.get("_run_factor"):
        st.info("Click **Run Factor Regressions** to estimate factor exposures for each asset.")
        return

    rows = []
    exposures = {}
    for col in asset_cols:
        y = merged[col].values - merged["RF"].values
        X = sm.add_constant(merged[factor_cols].values)
        try:
            model = sm.OLS(y, X).fit()
            row = {
                "Asset": col,
                "Alpha (monthly)": model.params[0],
                "Mkt-RF": model.params[1],
                "SMB": model.params[2],
                "HML": model.params[3],
                "RMW": model.params[4],
                "CMA": model.params[5],
                "R\u00b2": model.rsquared,
                "Adj R\u00b2": model.rsquared_adj,
            }
            exposures[col] = model.params[1:]
            rows.append(row)
        except Exception:
            pass

    if not rows:
        return

    ff_df = pd.DataFrame(rows).set_index("Asset").round(4)
    st.markdown("### Factor Exposures")
    st.dataframe(
        ff_df.style.format({
            "Alpha (monthly)": "{:.4f}",
            "Mkt-RF": "{:.3f}",
            "SMB": "{:.3f}",
            "HML": "{:.3f}",
            "RMW": "{:.3f}",
            "CMA": "{:.3f}",
            "R\u00b2": "{:.3f}",
            "Adj R\u00b2": "{:.3f}",
        }).background_gradient(subset=factor_cols, cmap="RdBu_r", axis=None),
        use_container_width=True,
    )

    st.markdown("### Factor Betas by Asset")
    fig, ax = plt.subplots(figsize=(11, 5))
    betas = pd.DataFrame(exposures).T
    betas.columns = factor_cols
    betas.plot(kind="bar", ax=ax, width=0.75, edgecolor="white")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("Factor Loading (Beta)")
    ax.set_title("Fama-French Factor Betas", fontweight="bold")
    ax.legend(loc="best", fontsize=9, ncol=3)
    ax.grid(True, alpha=0.3, axis="y")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.xticks(rotation=30, ha="right")
    st.pyplot(fig)

    st.markdown("### Rolling 36-Month Market Beta (equal-weight portfolio)")
    port_ret = returns.mean(axis=1).resample("ME").apply(lambda x: (1 + x).prod() - 1).dropna()
    merged2 = pd.concat([port_ret.rename("port"), factors[["Mkt-RF"]]], axis=1, join="inner").dropna()
    if len(merged2) >= 36:
        rolling_betas = []
        for i in range(36, len(merged2) + 1):
            window = merged2.iloc[i - 36:i]
            X = sm.add_constant(window["Mkt-RF"].values)
            m = sm.OLS(window["port"].values, X).fit()
            rolling_betas.append(m.params[1])
        rolling_idx = merged2.index[35:]
        fig, ax = plt.subplots(figsize=(11, 4))
        ax.plot(rolling_idx, rolling_betas, color="#1a56db", linewidth=1.8)
        ax.axhline(1.0, color="gray", linestyle="--", alpha=0.6, label="\u03b2 = 1")
        ax.axhline(np.mean(rolling_betas), color="red", linestyle="--", alpha=0.6,
                   label=f"Mean \u03b2 = {np.mean(rolling_betas):.2f}")
        ax.set_ylabel("Rolling \u03b2")
        ax.set_title("Rolling 36-Month Market Beta", fontweight="bold")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        st.pyplot(fig)


def render_monte_carlo(returns, rf, prices):
    st.markdown("<div class='main-header'>Monte Carlo Simulation</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='sub-header'>Geometric Brownian Motion simulation of the portfolio. "
        "Generates probabilistic distribution of future outcomes.</div>",
        unsafe_allow_html=True,
    )
    st.markdown("---")

    col_a, col_b = st.columns(2)
    horizon = col_a.slider("Horizon (days)", 21, 756, 252, 21)
    n_paths = col_b.select_slider("Number of paths", options=[500, 1000, 2000, 5000], value=2000)
    port_choice = st.radio("Portfolio", ["Equal Weight", "Max Sharpe"], horizontal=True)

    if st.button("Run Simulation", type="primary"):
        st.session_state._run_mc = True
        st.session_state._mc_n = n_paths
        st.session_state._mc_h = horizon
        st.session_state._mc_p = port_choice

    if not st.session_state.get("_run_mc"):
        st.info("Configure horizon/paths, then click **Run Simulation**.")
        return

    n_paths = st.session_state.get("_mc_n", 2000)
    horizon = st.session_state.get("_mc_h", 252)
    port_choice = st.session_state.get("_mc_p", "Equal Weight")
    freq = 252
    mu = returns.mean().values
    cov = returns.cov().values
    if port_choice == "Max Sharpe":
        w = max_sharpe_weights(mu, cov, rf=rf)
    else:
        w = np.full(len(mu), 1.0 / len(mu))

    progress = st.progress(0)
    status = st.empty()
    status.text(f"Simulating {n_paths} GBM paths\u2026")
    n_assets = len(mu)
    port_mu = float(w @ mu)
    port_var = float(w @ cov @ w)
    port_sigma = np.sqrt(port_var)
    drift = (port_mu - 0.5 * port_var) * freq
    vol = port_sigma * np.sqrt(freq)
    dt = 1.0 / freq
    n_steps = int(horizon)
    rng_mc = np.random.default_rng(42)
    shocks = rng_mc.standard_normal((n_paths, n_steps))
    paths = np.zeros((n_paths, n_steps + 1))
    paths[:, 0] = 100.0
    for t in range(1, n_steps + 1):
        paths[:, t] = paths[:, t - 1] * np.exp(drift * dt + vol * np.sqrt(dt) * shocks[:, t - 1])
    progress.progress(100)
    status.empty()
    progress.empty()

    final_values = paths[:, -1]
    initial = paths[:, 0]
    returns_mc = (final_values / initial) - 1.0
    var_95 = -np.quantile(returns_mc, 0.05)
    var_99 = -np.quantile(returns_mc, 0.01)
    es_95 = -returns_mc[returns_mc <= np.quantile(returns_mc, 0.05)].mean()
    prob_loss = float((returns_mc < 0).mean())
    prob_gain_20 = float((returns_mc > 0.20).mean())
    median_terminal = float(np.median(final_values))
    currency = "\u20b9" if any(t.endswith(".NS") for t in prices.columns) else "$"

    cols = st.columns(5)
    for col, (label, val) in zip(cols, [
        ("Median Terminal Value", f"{currency}{median_terminal:.1f}"),
        ("P(Loss)", f"{prob_loss:.1%}"),
        ("P(Gain > 20%)", f"{prob_gain_20:.1%}"),
        ("VaR (95%)", f"{var_95:.2%}"),
        ("ES / CVaR (95%)", f"{es_95:.2%}"),
    ]):
        col.markdown(
            f"<div class='metric-card'>"
            f"<div style='font-size:0.85rem;color:#64748b;'>{label}</div>"
            f"<div style='font-size:1.4rem;font-weight:700;color:#0f172a;'>{val}</div></div>",
            unsafe_allow_html=True,
        )

    st.markdown("### Simulated Price Paths")
    fig, ax = plt.subplots(figsize=(11, 5))
    days = np.arange(paths.shape[1])
    sample_idx = np.random.choice(paths.shape[0], size=min(50, paths.shape[0]), replace=False)
    for i in sample_idx:
        ax.plot(days, paths[i], color="#1a56db", alpha=0.08, linewidth=0.7)
    pcts = [10, 25, 50, 75, 90]
    colors_band = ["#fee2e2", "#fecaca", "#1a56db", "#fecaca", "#fee2e2"]
    for p, c in zip(pcts, colors_band):
        band = np.percentile(paths, p, axis=0)
        ax.plot(days, band, color=c, linewidth=1.4, alpha=0.85,
                label=f"{p}th percentile" if p != 50 else "Median")
    ax.fill_between(days, np.percentile(paths, 10, axis=0),
                    np.percentile(paths, 90, axis=0), color="#1a56db", alpha=0.08)
    ax.fill_between(days, np.percentile(paths, 25, axis=0),
                    np.percentile(paths, 75, axis=0), color="#1a56db", alpha=0.12)
    ax.set_xlabel("Days")
    ax.set_ylabel("Portfolio Value (Start = 100)")
    ax.set_title(f"Monte Carlo Simulation: {n_paths} paths, {horizon} days", fontweight="bold")
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    st.pyplot(fig)

    st.markdown("### Terminal Wealth Distribution")
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.hist(final_values, bins=60, color="#1a56db", alpha=0.8, edgecolor="white")
    ax.axvline(median_terminal, color="black", linestyle="--", linewidth=1.5,
               label=f"Median {currency}{median_terminal:.0f}")
    ax.axvline(100 * (1 - var_95), color="red", linestyle="--", linewidth=1.5,
               label=f"VaR (95%) {currency}{100 * (1 - var_95):.0f}")
    ax.set_xlabel("Terminal Portfolio Value")
    ax.set_ylabel("Frequency")
    ax.set_title(f"Distribution of Terminal Wealth ({horizon} days)", fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    st.pyplot(fig)


def render_ml_forecast(returns, rf, prices):
    st.markdown("<div class='main-header'>ML Return Forecasting</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='sub-header'>Gradient boosting forecast of next-day returns for the "
        "equal-weight portfolio using technical indicators.</div>",
        unsafe_allow_html=True,
    )
    st.markdown("---")

    if not HAS_SKLEARN:
        st.error("scikit-learn not installed. Install via `pip install scikit-learn`.")
        return

    port_ret = returns.mean(axis=1)
    port_prices = (1 + port_ret).cumprod() * 100.0

    feat = technical_indicators(port_prices)
    y = port_prices.pct_change().shift(-1)
    df = feat.join(y.rename("target"), how="inner").dropna()
    if len(df) < 60:
        st.warning("Insufficient data for ML training.")
        return

    test_size = st.slider("Test set fraction", 0.1, 0.4, 0.25, 0.05)
    model_choice = st.selectbox("Model", ["Gradient Boosting", "Random Forest"])

    if st.button("Train Model", type="primary"):
        st.session_state._run_ml = True
    if not st.session_state.get("_run_ml"):
        st.info("Configure test fraction + model, then click **Train Model**.")
        return

    split = int(len(df) * (1 - test_size))
    train, test = df.iloc[:split], df.iloc[split:]
    feature_cols = [c for c in df.columns if c != "target"]
    X_tr, X_te = train[feature_cols].values, test[feature_cols].values
    y_tr, y_te = train["target"].values, test["target"].values

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)

    progress = st.progress(0)
    status = st.empty()
    status.text("Training model\u2026")
    if model_choice == "Gradient Boosting":
        model = GradientBoostingRegressor(
            n_estimators=200, max_depth=3, learning_rate=0.05, random_state=42)
    else:
        model = RandomForestRegressor(
            n_estimators=200, max_depth=6, random_state=42, n_jobs=-1)
    model.fit(X_tr_s, y_tr)
    progress.progress(80)
    pred = model.predict(X_te_s)
    progress.progress(100)
    status.empty()
    progress.empty()

    rmse = float(np.sqrt(np.mean((y_te - pred) ** 2)))
    mae = float(np.mean(np.abs(y_te - pred)))
    direction_actual = (y_te > 0).astype(int)
    direction_pred = (pred > 0).astype(int)
    direction_acc = float((direction_actual == direction_pred).mean())

    cols = st.columns(4)
    for col, (label, val) in zip(cols, [
        ("RMSE", f"{rmse:.5f}"),
        ("MAE", f"{mae:.5f}"),
        ("Direction Accuracy", f"{direction_acc:.2%}"),
        ("Test Samples", f"{len(y_te):,}"),
    ]):
        col.markdown(
            f"<div class='metric-card'>"
            f"<div style='font-size:0.85rem;color:#64748b;'>{label}</div>"
            f"<div style='font-size:1.4rem;font-weight:700;color:#0f172a;'>{val}</div></div>",
            unsafe_allow_html=True,
        )

    st.markdown("### Predicted vs Actual Returns")
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    axes[0].scatter(y_te, pred, alpha=0.5, s=14, color="#1a56db")
    lim = max(abs(y_te).max(), abs(pred).max())
    axes[0].plot([-lim, lim], [-lim, lim], "r--", linewidth=1, alpha=0.7)
    axes[0].set_xlabel("Actual")
    axes[0].set_ylabel("Predicted")
    axes[0].set_title("Predicted vs Actual", fontweight="bold")
    axes[0].grid(True, alpha=0.3)
    axes[0].spines["top"].set_visible(False)
    axes[0].spines["right"].set_visible(False)

    test_idx = test.index
    axes[1].plot(test_idx, np.cumprod(1 + y_te) * 100, label="Actual", linewidth=1.8)
    axes[1].plot(test_idx, np.cumprod(1 + pred) * 100, label="Predicted (strategy)",
                 linewidth=1.5, linestyle="--", alpha=0.85)
    axes[1].set_ylabel("Cumulative Return (Start = 100)")
    axes[1].set_title("Cumulative: Actual vs Predicted-Directed Strategy", fontweight="bold")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].spines["top"].set_visible(False)
    axes[1].spines["right"].set_visible(False)
    st.pyplot(fig)

    st.markdown("### Feature Importance")
    fi = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(10, 5))
    fi.plot(kind="barh", ax=ax, color="#1a56db", edgecolor="white")
    ax.set_xlabel("Importance")
    ax.set_title(f"{model_choice} Feature Importance", fontweight="bold")
    ax.grid(True, alpha=0.3, axis="x")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    st.pyplot(fig)

    if HAS_SHAP:
        with st.expander("SHAP Feature Attribution (advanced)"):
            try:
                explainer = shap.TreeExplainer(model)
                sv = explainer.shap_values(X_te_s[:200])
                shap.summary_plot(sv, features=X_te[:200],
                                   feature_names=feature_cols, show=False, plot_size=(10, 5))
                st.pyplot(plt.gcf())
            except Exception:
                pass


def render_model_card(prices, returns, rf):
    st.markdown("<div class='main-header'>Model Card & About</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='sub-header'>Methodology, assumptions, limitations, and "
        "regulatory context for quantitative risk models.</div>",
        unsafe_allow_html=True,
    )
    st.markdown("---")

    st.markdown("### Project")
    st.markdown(
        "**QuantRisk** is a multi-asset portfolio risk management dashboard "
        "that implements classical portfolio theory (Markowitz, Risk Parity), risk "
        "metrics (VaR, CVaR, drawdown), Fama-French factor models, Monte Carlo "
        "simulation, and ML-based return forecasting. Supports real-time data via "
        "yfinance for any ticker — Indian NSE/BSE, US equities, ETFs, and indices."
    )

    st.markdown("### Methodology")
    st.markdown(
        """| Component | Method | Reference |
|---|---|---|---|
| Mean-Variance Optimization | scipy SLSQP, long-only constraints | Markowitz (1952) |
| Max Sharpe Ratio | Sharpe maximization with constraints | Sharpe (1966) |
| Risk Parity | Equal risk contribution iterative algorithm | Maillard, Roncalli, Te\u00ebtche (2010) |
| Historical VaR | 5th percentile of historical returns | J.P. Morgan RiskMetrics (1994) |
| Parametric VaR | Normal distribution quantile | Basel III IRRBB |
| Monte Carlo VaR | 10,000 GBM simulations | Glasserman (2003) |
| Fama-French 5-factor | OLS regression on monthly returns | Fama & French (2015) |
| Monte Carlo paths | Geometric Brownian Motion | Hull (2017) |
| ML forecasting | Gradient Boosting / Random Forest | Breiman (2001), Friedman (2001) |"""
    )

    st.markdown("### Assumptions & Limitations")
    st.markdown(
        """- **Returns are normally distributed** — parametric VaR underestimates tail risk.
- **Stationarity** — mean/covariance estimated on historical window assume they hold forward.
- **No transaction costs** — the optimal portfolio assumes zero trading frictions.
- **No tax considerations** — short-term capital gains tax not modeled.
- **Survivorship bias** — delisted tickers are excluded from current universes.
- **Look-ahead bias** — rolling windows use only past data."""
    )

    st.markdown("### Regulatory Context (Basel III / IV)")
    st.markdown(
        """- **Basel III FRTB** — mandates Expected Shortfall as the primary risk metric for market risk capital.
- **Basel IV (2025 implementation)** — output floor, standardized approach for credit.
- **EBA IRB ML Guidance (2023)** — ML models in capital calculation require explainability, fairness, and monitoring."""
    )

    st.markdown("### Indian Market Notes")
    st.markdown(
        """- **NSE** tickers use `.NS` suffix (e.g., `RELIANCE.NS`).
- **BSE** tickers use `.BO` suffix (e.g., `RELIANCE.BO`).
- **Indices**: Nifty 50 = `^NSEI`, Sensex = `^BSESN`.
- **Trading days**: ~245 per year (Mon-Fri, excluding NSE holidays).
- **Settlement**: T+1 since January 2023."""
    )


tabs = [
    "Overview",
    "Portfolio Optimization",
    "Risk Analytics",
    "Factor Model",
    "Monte Carlo",
    "ML Forecasting",
    "Model Card",
]
active_tab = st.sidebar.radio("Go to", tabs, index=0, key="tab_nav",
                               disabled=st.session_state.loading)

st.sidebar.markdown("---")
st.sidebar.markdown(
    f"<div style='font-size:0.8rem;color:#64748b;'>"
    f"<b>Universe:</b> {len(st.session_state.tickers)} tickers  \n"
    f"<b>Period:</b> {start_date} \u2192 {end_date}"
    f"</div>",
    unsafe_allow_html=True,
)

if st.session_state.data_error:
    st.warning(st.session_state.data_error)

if not st.session_state.data_loaded:
    st.markdown(
        "<div style='background:#f0f7ff; border:1px solid #bdd3eb; border-radius:10px; "
        "padding:1rem 1.5rem; margin-bottom:1rem;'>"
        "<b>\U0001f4ca QuantRisk</b> \u2014 Enter tickers in the sidebar "
        "(Indian NSE example: <code>RELIANCE.NS, TCS.NS</code>) and click "
        "<b>Fetch Market Data</b> to begin. Real-time prices via yfinance.</div>",
        unsafe_allow_html=True,
    )
    if active_tab == "Model Card":
        render_model_card(None, None, rf_rate)
    st.markdown(
        "<div class='footer'>QuantRisk \u00b7 Streamlit \u00b7 "
        "Real-time via yfinance</div>",
        unsafe_allow_html=True,
    )
    st.stop()

prices = st.session_state.prices
returns = st.session_state.returns
benchmark = st.session_state.benchmark
factors = st.session_state.get("factors", None)
rf = st.session_state.rf_rate

if active_tab == "Overview":
    render_overview(prices, returns, rf, benchmark)
elif active_tab == "Portfolio Optimization":
    render_portfolio_opt(returns, rf, prices, benchmark)
elif active_tab == "Risk Analytics":
    render_risk_analytics(returns, rf, prices)
elif active_tab == "Factor Model":
    render_factor_model(returns, rf, factors, prices)
elif active_tab == "Monte Carlo":
    render_monte_carlo(returns, rf, prices)
elif active_tab == "ML Forecasting":
    render_ml_forecast(returns, rf, prices)
elif active_tab == "Model Card":
    render_model_card(prices, returns, rf)

st.markdown(
    f"<div class='footer'>QuantRisk \u00b7 "
    f"{len(prices.columns)} tickers \u00b7 {start_date} \u2192 {end_date} \u00b7 "
    f"Real-time via yfinance</div>",
    unsafe_allow_html=True,
)
