import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests as crequests
import yfinance as yf
import pandas_datareader.data as web
from scipy.optimize import minimize
from scipy.stats import chi2 as chi2_dist
import warnings
warnings.filterwarnings("ignore")

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Portfolio Optimizer — Black-Litterman",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Theme ─────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=DM+Mono:wght@400;500&display=swap');
  html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
  .stApp { background-color: #0a1628; color: #c8d4e0; }

  /* ── Sidebar ── */
  section[data-testid="stSidebar"] {
    background-color: #f8fafc;
    border-right: 1px solid #dde3ea;
  }
  section[data-testid="stSidebar"] * { color: #111827 !important; }
  section[data-testid="stSidebar"] input { color: #111827 !important; background: #fff !important; }
  section[data-testid="stSidebar"] label { color: #111827 !important; }
  section[data-testid="stSidebar"] p { color: #374151 !important; }
  section[data-testid="stSidebar"] span { color: #111827 !important; }
  section[data-testid="stSidebar"] .section-header {
    color: #6b7280 !important;
    border-bottom-color: #dde3ea !important;
  }

  /* ── Metric cards ── */
  .metric-card {
    background: #0e1e35; border: 1px solid rgba(78,158,200,0.2);
    border-radius: 4px; padding: 16px 20px; text-align: center; margin-bottom: 8px;
  }
  .metric-label { font-size: 10px; font-weight: 600; letter-spacing: 2px;
    text-transform: uppercase; color: rgba(200,212,224,0.5); margin-bottom: 6px; }
  .metric-value { font-family: 'DM Mono', monospace; font-size: 22px; font-weight: 500; color: #78bdd8; }
  .metric-value.positive { color: #4ec87a; }
  .metric-value.negative { color: #e05c5c; }
  .metric-value.neutral  { color: #f5c842; }

  /* ── Section headers ── */
  .section-header { font-size: 9px; font-weight: 600; letter-spacing: 3px;
    text-transform: uppercase; color: rgba(200,212,224,0.4);
    border-bottom: 1px solid rgba(200,212,224,0.08);
    padding-bottom: 8px; margin-bottom: 6px; margin-top: 28px; }

  /* ── Explain text below section headers ── */
  .explain { font-size: 12px; color: rgba(200,212,224,0.55);
    margin: 0 0 14px 0; line-height: 1.5; }

  /* ── Buttons ── */
  .stButton>button { background: #4e9ec8; color: #fff; border: none; border-radius: 2px;
    font-size: 11px; font-weight: 700; letter-spacing: 2px; text-transform: uppercase;
    padding: 12px 24px; width: 100%; transition: background 0.2s; }
  .stButton>button:hover { background: #78bdd8; }

  /* ── Headings ── */
  h1 { font-size: 22px !important; font-weight: 600 !important; color: #fff !important; }
  h2 { font-size: 14px !important; font-weight: 600 !important; color: #c8d4e0 !important; }

  /* ── Alert boxes ── */
  .warning-box { background: rgba(245,200,66,0.08); border: 1px solid rgba(245,200,66,0.3);
    border-radius: 4px; padding: 12px 16px; font-size: 13px; color: #f5c842; margin-top: 8px; }
  .info-box { background: rgba(78,158,200,0.08); border: 1px solid rgba(78,158,200,0.25);
    border-radius: 4px; padding: 12px 16px; font-size: 13px; color: #78bdd8; margin-top: 8px; }
</style>
""", unsafe_allow_html=True)

PLOT_BG   = "#0a1628"
PLOT_CARD = "#0e1e35"
GRID_CLR  = "rgba(200,212,224,0.05)"
TEXT_CLR  = "#c8d4e0"
BLUE      = "#4e9ec8"
GREEN     = "#4ec87a"
RED       = "#e05c5c"
GOLD      = "#f5c842"

def plot_layout(height=380, r=10, **kw):
    return dict(
        paper_bgcolor=PLOT_BG, plot_bgcolor=PLOT_CARD,
        font=dict(color=TEXT_CLR, family="DM Sans"),
        xaxis=dict(showgrid=True, gridcolor=GRID_CLR, color=TEXT_CLR, zeroline=False),
        yaxis=dict(showgrid=True, gridcolor=GRID_CLR, color=TEXT_CLR, zeroline=False),
        legend=dict(bgcolor="rgba(14,30,53,0.85)", bordercolor="rgba(200,212,224,0.15)",
                    borderwidth=1, font=dict(color=TEXT_CLR, size=12)),
        margin=dict(t=30, b=40, l=10, r=r), height=height, **kw
    )

def explain(text):
    st.markdown(f'<p class="explain">{text}</p>', unsafe_allow_html=True)

# ── HTTP headers (Yahoo/FRED reject requests without a real UA from cloud IPs) ─
HTTP_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml,application/json;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}

# ── Risk-free rate (FRED CSV → fallback hardcode) ─────────────────────────────
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_risk_free_rate():
    """Returns (rate_annual_decimal, label_string)."""
    try:
        url = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DGS3MO"
        r   = crequests.get(url, timeout=10, headers=HTTP_HEADERS)
        lines = [l for l in r.text.strip().split("\n") if l.strip()]
        for line in reversed(lines[1:]):
            parts = line.split(",")
            if len(parts) == 2 and parts[1].strip() not in (".", ""):
                rate  = float(parts[1].strip()) / 100
                date  = parts[0].strip()
                return rate, f"T-Bill 3m FRED ({date})"
    except Exception:
        pass
    return 0.043, "default 4.30%"

# ── Data (cached) ─────────────────────────────────────────────────────────────
def _fetch_stooq(ticker: str, start: str) -> pd.Series | None:
    """Fetch adjusted close from Stooq (works from cloud IPs, no auth needed)."""
    try:
        stooq_ticker = ticker + ".US" if not ticker.startswith("^") else ticker
        df = web.DataReader(stooq_ticker, "stooq", start=start)
        if df.empty:
            return None
        close = df["Close"].sort_index()
        close.name = ticker
        close.index = pd.to_datetime(close.index).normalize()
        return close[~close.index.duplicated(keep="last")]
    except Exception:
        return None

def _fetch_yfinance(ticker: str, start: str) -> pd.Series | None:
    """Fallback: yfinance Ticker.history()."""
    try:
        t = yf.Ticker(ticker)
        df = t.history(start=start, auto_adjust=True, actions=False)
        if df.empty:
            return None
        close = df["Close"]
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
        close.name = ticker
        close.index = pd.to_datetime(close.index).tz_localize(None).normalize()
        return close[~close.index.duplicated(keep="last")]
    except Exception:
        return None

@st.cache_data(show_spinner=False)
def download_prices(tickers: tuple, start: str):
    """Adjusted close prices. Primary: Stooq (cloud-safe). Fallback: yfinance."""
    series = {}
    for ticker in tickers:
        close = _fetch_stooq(ticker, start) or _fetch_yfinance(ticker, start)
        if close is not None:
            series[ticker] = close
    if not series:
        return pd.DataFrame()
    return pd.DataFrame(series).dropna(how="all")

@st.cache_data(ttl=300, show_spinner=False)
def fetch_realtime_prices(tickers: tuple) -> dict:
    """Devuelve {ticker: último precio cierre} via Stooq, fallback yfinance."""
    from datetime import datetime, timedelta
    prices = {}
    start = (datetime.today() - timedelta(days=10)).strftime("%Y-%m-%d")
    for ticker in tickers:
        close = _fetch_stooq(ticker, start)
        if close is not None and not close.empty:
            prices[ticker] = float(close.iloc[-1])
            continue
        try:
            t = yf.Ticker(ticker)
            hist = t.history(period="5d")
            if not hist.empty:
                prices[ticker] = float(hist["Close"].iloc[-1])
        except Exception:
            pass
    return prices

# ── Black-Litterman (true implementation) ────────────────────────────────────
def black_litterman_posterior(mean_returns, cov_matrix, views_annual, confidences, tau=0.025):
    """
    Posterior BL: μ_BL = [(τΣ)⁻¹ + P'Ω⁻¹P]⁻¹ [(τΣ)⁻¹π + P'Ω⁻¹Q]
    Ω_ii = (1/conf − 1) · τ · P_i Σ P_i'
    """
    assets = list(mean_returns.index)
    n      = len(assets)
    pi     = mean_returns.values.copy()
    Sigma  = cov_matrix.values.copy()

    view_assets = [a for a in assets if a in views_annual and confidences.get(a, 0) > 0.001]
    k = len(view_assets)
    if k == 0:
        return mean_returns.copy()

    P = np.zeros((k, n))
    Q = np.zeros(k)
    for i, asset in enumerate(view_assets):
        idx      = assets.index(asset)
        P[i, idx] = 1.0
        Q[i]     = (1 + views_annual[asset]) ** (1/252) - 1

    Omega = np.diag([
        max((1 / max(confidences[a], 1e-6) - 1) * tau * float(P[i] @ Sigma @ P[i]), 1e-10)
        for i, a in enumerate(view_assets)
    ])

    tSigma_inv = np.linalg.inv(tau * Sigma + np.eye(n) * 1e-12)
    Omega_inv  = np.linalg.inv(Omega + np.eye(k) * 1e-12)
    M_inv      = tSigma_inv + P.T @ Omega_inv @ P
    M          = np.linalg.inv(M_inv + np.eye(n) * 1e-12)
    mu_bl      = M @ (tSigma_inv @ pi + P.T @ Omega_inv @ Q)

    return pd.Series(mu_bl, index=assets)

# ── Quantitative signals ──────────────────────────────────────────────────────
def compute_quant_views(prices, method="momentum_12_1"):
    """
    Returns annualized expected return per asset from a quantitative signal.

    momentum_12_1  : (Jegadeesh & Titman) return from t-12m to t-1m, annualized.
                     Skips last month to avoid short-term reversal.
    ewma_blend     : 70 % long-term historical + 30 % 3-month EWMA.
                     Smoother, less reactive to recent noise.
    risk_adj_mom   : Momentum divided by trailing volatility (Sharpe-like score
                     rescaled back to return space via vol).
                     Penalizes high-vol assets for the same raw momentum.
    """
    log_ret = np.log(prices / prices.shift(1)).dropna()

    if method == "momentum_12_1":
        skip, lb = 21, 252          # skip 1 month, 12-month lookback
        min_needed = lb + skip + 2
        if len(prices) < min_needed:
            return log_ret.mean() * 252
        p_skip = prices.iloc[-(skip + 1)]
        p_lb   = prices.iloc[-(lb + skip + 1)]
        ann    = 252 / (lb - skip)  # annualize the 11-month return
        return (p_skip / p_lb) ** ann - 1

    elif method == "ewma_blend":
        hist   = log_ret.mean() * 252
        ewma3m = log_ret.ewm(span=63, adjust=False).mean().iloc[-1] * 252
        return 0.70 * hist + 0.30 * ewma3m

    elif method == "risk_adj_mom":
        skip, lb = 21, 252
        if len(prices) < lb + skip + 2:
            return log_ret.mean() * 252
        p_skip  = prices.iloc[-(skip + 1)]
        p_lb    = prices.iloc[-(lb + skip + 1)]
        mom_raw = (p_skip / p_lb) - 1                      # raw 11-month return
        vol_ann = log_ret.tail(lb).std() * np.sqrt(252)    # trailing vol
        # Signal = mom / vol (dimensionless Sharpe-like score)
        # Re-express as return: sign(signal) * vol * |signal|^0.5
        signal  = mom_raw / (vol_ann + 1e-8)
        return np.sign(signal) * vol_ann * np.sqrt(signal.abs())

    return log_ret.mean() * 252  # fallback: historical mean


def compute_quant_conf(daily_returns, prices, method="sharpe_12m"):
    """
    Returns confidence per asset in [0, 1] from a quantitative measure.

    sharpe_12m     : |Sharpe ratio| over the last 12 months.
                     Sharpe=1 → 58%, Sharpe=1.5 → 76%, Sharpe=2 → 88%.
                     Uses tanh(|SR| / 1.5) so it saturates gracefully.
    trend_r2       : R² of a linear regression on log-prices over 252 days.
                     High R² = strong, consistent trend = high confidence.
    vol_stability  : Recent vol (63d) relative to historical vol.
                     Contracted vol (ratio < 1) → regime is stable → more confident.
    """
    if method == "sharpe_12m":
        r  = daily_returns.tail(252)
        sh = r.mean() / (r.std() + 1e-8) * np.sqrt(252)
        return sh.abs().apply(lambda x: float(np.tanh(x / 1.5))).clip(0.01, 0.99)

    elif method == "trend_r2":
        conf = {}
        for col in prices.columns:
            p = np.log(prices[col].dropna().values[-252:])
            if len(p) < 30:
                conf[col] = 0.50; continue
            x      = np.arange(len(p), dtype=float)
            p_hat  = np.polyval(np.polyfit(x, p, 1), x)
            ss_res = np.sum((p - p_hat) ** 2)
            ss_tot = np.sum((p - p.mean()) ** 2)
            conf[col] = float(np.clip(1 - ss_res / (ss_tot + 1e-12), 0.01, 0.99))
        return pd.Series(conf)

    elif method == "vol_stability":
        vol_rec  = daily_returns.tail(63).std()
        vol_hist = daily_returns.std()
        ratio    = vol_rec / (vol_hist + 1e-8)           # < 1 = vol contracted
        inv      = (1 / ratio).clip(0.2, 5.0)
        mn, mx   = inv.min(), inv.max()
        if mx - mn < 1e-8:
            return pd.Series(0.50, index=daily_returns.columns)
        return ((inv - mn) / (mx - mn) * 0.88 + 0.05).clip(0.05, 0.95)

    return pd.Series(0.50, index=daily_returns.columns)  # fallback


# ── Risk metrics ──────────────────────────────────────────────────────────────
def sortino_ratio(weights, returns_df, rf_daily):
    port_returns = returns_df.values @ weights
    excess       = port_returns - rf_daily
    downside     = excess[excess < 0]
    downside_std = np.std(downside) * np.sqrt(252) if len(downside) > 1 else 1e-6
    ann_excess   = np.mean(excess) * 252
    return ann_excess / downside_std if downside_std > 1e-8 else 0.0

def max_drawdown(cum_returns):
    roll_max = np.maximum.accumulate(cum_returns)
    dd       = (cum_returns - roll_max) / roll_max
    return float(dd.min())

def calmar_ratio(ann_return, mdd):
    return ann_return / abs(mdd) if abs(mdd) > 1e-6 else 0.0

# ── Risk Parity ───────────────────────────────────────────────────────────────
def risk_parity_weights(cov_matrix):
    """
    Equal Risk Contribution: cada activo aporta el mismo riesgo al portafolio total.
    Minimiza la dispersión de las contribuciones al riesgo relativas.
    """
    n = cov_matrix.shape[0]

    def risk_contributions(w):
        vol   = np.sqrt(w.T @ cov_matrix @ w)
        mrc   = cov_matrix @ w          # marginal risk contribution
        rc    = w * mrc / (vol + 1e-12) # absolute risk contribution per asset
        return rc / rc.sum()            # relative (sums to 1)

    def objective(w):
        rc     = risk_contributions(w)
        target = 1.0 / n
        return float(np.sum((rc - target) ** 2))

    w0   = np.array([1.0/n] * n)
    cons = {"type": "eq", "fun": lambda x: np.sum(x) - 1}
    bnds = [(1e-4, 1.0)] * n           # small lower bound for numerical stability

    res = minimize(objective, w0, method="SLSQP", bounds=bnds, constraints=cons,
                   options={"ftol": 1e-14, "maxiter": 2000})
    w = np.clip(res.x, 0, 1)
    return w / w.sum()

# ── Optimization ──────────────────────────────────────────────────────────────
def neg_sortino(weights, daily_returns_df, rf_daily):
    return -sortino_ratio(weights, daily_returns_df, rf_daily)

def neg_sharpe(weights, adj_returns, cov_matrix, rf_daily):
    ret = np.dot(weights, adj_returns)
    vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    return -(ret - rf_daily) / vol if vol > 1e-8 else 0.0

def optimize_portfolio(adj_returns_daily, cov_matrix, daily_returns_df,
                       rf_daily, bounds, objective="sharpe"):
    n    = len(adj_returns_daily)
    w0   = np.array([1/n] * n)
    cons = {"type": "eq", "fun": lambda x: np.sum(x) - 1}
    if objective == "sharpe":
        fn, args = neg_sharpe, (adj_returns_daily, cov_matrix, rf_daily)
    else:
        fn, args = neg_sortino, (daily_returns_df, rf_daily)
    res = minimize(fn, w0, args=args, method="SLSQP",
                   bounds=bounds, constraints=cons,
                   options={"ftol": 1e-12, "maxiter": 1000})
    return res.x

# ── Monte Carlo (multivariate t) ──────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def monte_carlo(bl_returns_arr, cov_arr, weights_arr, horizon, n_sim, initial_investment, df, seed=42):
    """Multivariate t-distribution — fatter tails than normal."""
    np.random.seed(seed)
    n  = len(bl_returns_arr)
    Z  = np.random.multivariate_normal(np.zeros(n), cov_arr, (n_sim, horizon))
    chi2 = chi2_dist.rvs(df, size=(n_sim, horizon, 1)) / df
    T    = Z / np.sqrt(chi2)
    scale = np.sqrt((df - 2) / df)
    sim_ret    = bl_returns_arr + T * scale
    port_daily = np.dot(sim_ret, weights_arr)
    values     = initial_investment * np.cumprod(1 + port_daily, axis=1)
    return values

# ── Efficient Frontier ────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def efficient_frontier(bl_returns_arr, cov_arr, rf_daily, n_points=1200):
    n  = len(bl_returns_arr)
    rs, vs, srs = [], [], []
    for _ in range(n_points):
        w  = np.random.dirichlet(np.ones(n))
        r  = np.dot(w, bl_returns_arr) * 252
        v  = np.sqrt(w.T @ cov_arr @ w) * np.sqrt(252)
        rs.append(r); vs.append(v)
        srs.append((r - rf_daily*252) / v if v > 1e-8 else 0)
    return np.array(rs), np.array(vs), np.array(srs)

# ═════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## Portfolio Optimizer")
    st.markdown("Black-Litterman · Monte Carlo · Markowitz")
    st.markdown("---")

    # ── Capital ──
    st.markdown('<div class="section-header">Capital</div>', unsafe_allow_html=True)
    initial_investment = st.number_input(
        "Capital inicial (USD)",
        min_value=1000.0, value=100000.0, step=1000.0, format="%.0f",
        help="Monto total a invertir en dólares. Se usa en el backtest, Monte Carlo y CAPM."
    )

    # ── Horizonte ──
    st.markdown('<div class="section-header">Horizonte de inversión</div>', unsafe_allow_html=True)
    time_unit = st.selectbox(
        "Unidad de tiempo",
        ["Años", "Meses", "Días"],
        help="Período futuro que quieres proyectar con Monte Carlo."
    )
    if time_unit == "Años":
        horizon_val      = st.slider("Cantidad de años", 1, 30, 10,
                                      help="Cada año equivale a ~252 días hábiles de mercado.")
        investment_horizon = horizon_val * 252
    elif time_unit == "Meses":
        horizon_val      = st.slider("Cantidad de meses", 1, 120, 24,
                                      help="Cada mes equivale a ~21 días hábiles.")
        investment_horizon = horizon_val * 21
    else:
        horizon_val      = st.slider("Días hábiles", 21, 7560, 252,
                                      help="Días hábiles de mercado (252 ≈ 1 año).")
        investment_horizon = horizon_val

    # ── Tasa libre de riesgo ──
    st.markdown('<div class="section-header">Tasa libre de riesgo</div>', unsafe_allow_html=True)
    rf_live, rf_label = fetch_risk_free_rate()
    rf_default = float(round(rf_live * 100 * 4) / 4)   # round to nearest 0.25%
    rf_annual = st.slider(
        "Tasa anual (%)", 0.0, 15.0, rf_default, step=0.25,
        help=(
            "Retorno de un activo sin riesgo. Se usa para calcular Sharpe, Sortino "
            "y la línea de mercado de capitales (CML).\n\n"
            "Valor cargado automáticamente desde FRED (Reserva Federal de EE.UU.): "
            "T-Bill a 3 meses — el proxy estándar para la tasa libre de riesgo en "
            "portfolios de corto/mediano plazo.\n\n"
            f"Fuente actual: {rf_label}. Se actualiza cada hora."
        )
    ) / 100
    st.caption(f"Fuente: {rf_label}")
    rf_daily = rf_annual / 252

    # ── Activos ──
    st.markdown('<div class="section-header">Activos</div>', unsafe_allow_html=True)
    input_method = st.radio(
        "Ingreso de tickers",
        ["Manual", "Importar CSV"],
        help="Ingresá los símbolos de mercado (ej: AAPL, SPY) o importá una lista desde CSV."
    )
    if input_method == "Manual":
        tickers_raw  = st.text_input(
            "Tickers (separados por coma)", "XLP, XLK, IYK, XLU, XLV",
            help="Símbolos de Yahoo Finance. ETFs, acciones, índices. Mínimo 2 activos."
        )
        assets_input = [t.strip().upper() for t in tickers_raw.split(",") if t.strip()]
    else:
        uploaded = st.file_uploader("CSV con columna 'ticker'", type="csv",
                                     help="El archivo debe tener una columna llamada 'ticker' con los símbolos.")
        if uploaded:
            df_up = pd.read_csv(uploaded)
            col   = "ticker" if "ticker" in df_up.columns else df_up.columns[0]
            assets_input = df_up[col].str.upper().tolist()
            st.success(f"{len(assets_input)} tickers cargados")
        else:
            assets_input = ["XLP", "XLK", "IYK", "XLU", "XLV"]

    start_date = st.date_input(
        "Inicio de datos históricos",
        value=pd.to_datetime("2015-01-01"),
        help="Fecha desde la cual se descargan precios para calcular retornos históricos, "
             "correlaciones y el prior del modelo."
    )

    # ── Modo de optimización ──
    st.markdown('<div class="section-header">Modo de optimización</div>', unsafe_allow_html=True)
    port_mode = st.radio(
        "Modo",
        ["bl_puro", "bl_views", "risk_parity"],
        format_func=lambda x: {
            "bl_views":     "BL + Señales",
            "bl_puro":      "BL Puro",
            "risk_parity":  "Risk Parity",
        }[x],
        help=(
            "BL + Señales — Black-Litterman con señales cuantitativas de precio. "
            "El modelo calcula un retorno esperado por activo usando momentum o tendencia, "
            "y lo combina con el histórico según la confianza en la señal. "
            "Usar cuando creés que existe predictibilidad en los precios de tu universo "
            "(más común en carteras de acciones individuales, funciona parcialmente en ETFs sectoriales).\n\n"
            "BL Puro — Black-Litterman sin señales adicionales. "
            "Usa el retorno histórico de cada activo como única estimación de retorno futuro "
            "y la matriz de covarianza para el riesgo. "
            "Más estable que MVO clásico porque BL suaviza el impacto del error en las estimaciones. "
            "Usar cuando no querés asumir que el pasado reciente predice el futuro, "
            "o cuando tu universo son ETFs y no tenés convicción en las señales.\n\n"
            "Risk Parity — No estima retornos futuros. "
            "Asigna pesos para que cada activo contribuya igual al riesgo total del portafolio. "
            "Activos más volátiles reciben menos peso automáticamente. "
            "Muy usado en fondos multi-asset (Bridgewater, AQR). "
            "Usar cuando el universo mezcla activos con volatilidades muy distintas "
            "y no querés que los más volátiles dominen el riesgo."
        )
    )

    view_method = "momentum_12_1"
    conf_method = "sharpe_12m"

    if port_mode == "bl_views":
        view_method = st.selectbox(
            "Señal de retorno esperado",
            ["momentum_12_1", "ewma_blend", "risk_adj_mom"],
            format_func=lambda x: {
                "momentum_12_1": "Momentum 12-1",
                "ewma_blend":    "EWMA Blend",
                "risk_adj_mom":  "Momentum / Vol",
            }[x],
            help=(
                "Momentum 12-1 — Retorno del precio entre hace 12 meses y hace 1 mes "
                "(saltea el último mes para evitar la reversión de corto plazo documentada por "
                "Jegadeesh & Titman 1993). Si un activo subió fuerte en ese período, el modelo "
                "espera que continúe. Es la señal con más respaldo empírico en finanzas. "
                "Mejor para universos donde el momentum histórico tiene evidencia.\n\n"
                "EWMA Blend — Combina el retorno histórico promedio (70%) con la media "
                "exponencialmente ponderada de los últimos 3 meses (30%). Da más peso a lo "
                "reciente sin ser tan reactivo como el momentum puro. Señal más suave, "
                "menos sensible a un mes particular, más estable.\n\n"
                "Momentum / Vol — Mismo cálculo que Momentum 12-1 pero divide la señal por "
                "la volatilidad del activo. Un ETF que subió 10% con volatilidad 10% tiene "
                "mejor señal que uno que subió 10% con volatilidad 25%. Premia la calidad "
                "del movimiento, no solo su magnitud. Penaliza activos que subieron por ruido."
            )
        )
        conf_method = st.selectbox(
            "Señal de confianza",
            ["sharpe_12m", "trend_r2", "vol_stability"],
            format_func=lambda x: {
                "sharpe_12m":    "Sharpe 12m",
                "trend_r2":      "Trend R²",
                "vol_stability": "Estabilidad de vol",
            }[x],
            help=(
                "Sharpe 12m — Mide el retorno ajustado por riesgo de los últimos 12 meses. "
                "Un Sharpe alto significa que el activo generó buen retorno por unidad de "
                "volatilidad de forma consistente. Fórmula: tanh(|Sharpe| / 1.5). "
                "Sharpe=1 → 58% de confianza. Sharpe=2 → 88%. Sharpe=0.5 → 32%. "
                "Funciona bien cuando la tendencia reciente fue sostenida.\n\n"
                "Trend R² — Ajusta una línea recta al logaritmo del precio de los últimos "
                "252 días y mide qué tan bien se ajusta (R²). R²=1 = tendencia perfectamente "
                "lineal y limpia. R²=0 = movimiento completamente aleatorio. "
                "Alta R² significa que el momentum fue consistente, no un zigzag. "
                "Más robusto que el Sharpe ante meses atípicos.\n\n"
                "Estabilidad de vol — Compara la volatilidad reciente (63 días = 3 meses) "
                "con la volatilidad histórica. Si la volatilidad se contrajo, el mercado "
                "está en un régimen estable y predecible → más confianza en la señal. "
                "Si la volatilidad explotó (crisis, shock), reduce la confianza "
                "automáticamente. Útil como señal de régimen de mercado."
            )
        )

    # ── Parámetros avanzados ──
    if port_mode in ("bl_views", "bl_puro"):
        with st.expander("⚙️ Parámetros avanzados (tau)"):
            st.markdown("""
**tau** calibra la incertidumbre del retorno histórico como punto de partida.

- En **BL Puro**: tau bajo = el histórico es muy confiable, el optimizer lo sigue de cerca.
- En **BL + Señales**: tau + confianza determinan juntos cuánto pesa la señal vs el histórico.

Valor estándar: 0.025. No modificar salvo razón específica.
""")
            tau = st.slider("tau", 0.005, 0.10, 0.025, step=0.005,
                            help="0.025 es el valor estándar de la literatura BL.")
    else:
        tau = 0.025  # unused in risk parity but kept for code consistency

    # ── Objetivo ──
    st.markdown('<div class="section-header">Objetivo de optimización</div>', unsafe_allow_html=True)
    objective = st.radio(
        "Maximizar",
        ["Sharpe Ratio", "Sortino Ratio"],
        help="Sharpe: divide el exceso de retorno por la volatilidad total (sube y baja). "
             "Sortino: divide solo por la volatilidad negativa — más apropiado si te preocupan "
             "más las pérdidas que la volatilidad en general."
    )
    obj_key = "sharpe" if objective == "Sharpe Ratio" else "sortino"

    # ── Monte Carlo ──
    st.markdown('<div class="section-header">Monte Carlo</div>', unsafe_allow_html=True)
    mc_df = st.slider(
        "Grados de libertad (t-Student)", 3, 30, 7,
        help=(
            "Controla el grosor de las colas: cuánta probabilidad le das a eventos extremos "
            "(crashes, rallies). A menor df → colas más gruesas → simulación más conservadora.\n\n"

            "POR QUÉ NO SE USA LA NORMAL:\n"
            "La distribución normal asume que un crash como 2008 o 2020 ocurre una vez cada "
            "miles de años. Empíricamente (Mandelbrot 1963, Fama 1965) ocurren cada década. "
            "La t-Student corrige eso.\n\n"

            "─── QUÉ VALOR USAR SEGÚN TU CARTERA ───\n\n"

            "df = 3  │ Cripto (BTC, ETH, altcoins)\n"
            "        │ Volatilidad anual >60%. Crashes del 80% en meses.\n"
            "        │ La más conservadora disponible.\n\n"

            "df = 4  │ Acciones individuales de alta beta\n"
            "        │ Ej: TSLA, NVDA, MELI, biotechs.\n"
            "        │ Movimientos de ±20% en días son posibles.\n\n"

            "df = 5  │ ★ ACCIONES INDIVIDUALES — ESTÁNDAR\n"
            "        │ Ej: AAPL, MSFT, AMZN, carteras de 5-15 acciones.\n"
            "        │ Fama (1965) estimó df≈4-6 para acciones de EE.UU.\n"
            "        │ Un evento 3-sigma ocurre ~4x más que en la normal.\n\n"

            "df = 7  │ ETFs sectoriales concentrados\n"
            "        │ Ej: XLK (tech), XLE (energía), XLF (financials).\n"
            "        │ Más diversificados que acciones, pero siguen 1 sector.\n\n"

            "df = 10 │ ETFs multi-sector / índices amplios\n"
            "        │ Ej: SPY, QQQ, IWM, VTI.\n"
            "        │ Diversificación interna reduce las colas extremas.\n\n"

            "df = 15 │ Cartera muy diversificada (multi-asset)\n"
            "        │ Ej: acciones + bonos + commodities + REIT.\n"
            "        │ Los activos descorrelacionados suavizan los shocks.\n\n"

            "df = 20 │ Bonos del Tesoro / renta fija de alta calidad\n"
            "        │ Ej: TLT, IEF, SHY, bonos investment grade.\n"
            "        │ Colas casi normales, shocks más raros y acotados.\n\n"

            "df = 30 │ Equivale a distribución normal.\n"
            "        │ Solo para renta fija de muy corto plazo o Money Market.\n"
            "        │ Subestima el riesgo si tenés acciones o ETFs."
        )
    )
    n_sim = st.select_slider(
        "Cantidad de simulaciones",
        [1000, 5000, 10000, 20000, 50000],
        value=10000,
        help=(
            "Número de escenarios futuros a simular.\n\n"
            "POR QUÉ NO SE OFRECEN MÁS DE 50.000:\n"
            "La precisión del VaR al 95% depende de cuántas observaciones caen en el 5% "
            "peor. Con 10.000 simulaciones hay ~500 en esa cola — suficiente para estimar "
            "el VaR con un error estándar de ~±1.5% del capital. Con 50.000 el error baja "
            "a ~±0.7%. Más allá de 50.000 la ganancia en precisión es marginal (<0.3%) "
            "pero el costo de memoria y tiempo sube linealmente.\n\n"
            "GUÍA PRÁCTICA:\n"
            "1.000 → exploración rápida, resultados aproximados.\n"
            "10.000 → uso estándar, buena precisión.\n"
            "50.000 → análisis final antes de invertir. Más lento (~30 seg)."
        )
    )

    var_confidence = st.select_slider(
        "Nivel de confianza VaR / CVaR",
        [90, 95, 99],
        value=95,
        help=(
            "Define cuán conservador es el VaR y CVaR.\n\n"
            "VaR 90% → el 10% peor de los escenarios cae por debajo de este valor.\n"
            "VaR 95% → el 5% peor. Estándar en la industria (Basel III, regulación bancaria).\n"
            "VaR 99% → el 1% peor. Más exigente; usado en fondos y bancos con alta exposición.\n\n"
            "IMPORTANTE: no cambia la media ni la mediana, solo el corte de la cola."
        )
    )

    # ── CAPM / Capital Market Line ──
    st.markdown('<div class="section-header">Línea de mercado de capitales</div>',
                unsafe_allow_html=True)
    capm_target_pct = st.slider(
        "Retorno anual objetivo (%)",
        0.0, 30.0, 0.0, step=0.5,
        key="sidebar_capm",
        help=(
            "Cuánto riesgo querés asumir combinando el portafolio óptimo con el bono libre de riesgo.\n\n"
            "0% → todo en bono (sin riesgo de mercado).\n"
            "= retorno del portafolio → 100% en portafolio, 0% en bono.\n"
            "> retorno del portafolio → apalancamiento (pedís prestado para invertir más).\n\n"
            "Este valor se usa en el Monte Carlo: la proyección refleja exactamente "
            "cuánto capital va al portafolio y cuánto al bono."
        )
    )

    run_btn = st.button("▶  OPTIMIZAR")

# ── Session state: persist run across reruns caused by widget changes ─────────
if run_btn:
    st.session_state["triggered"] = True
    st.session_state["reset_views"] = True   # reset view defaults on new run

if not st.session_state.get("triggered", False):
    st.markdown("# Portfolio Optimizer")
    st.markdown("### Black-Litterman · Monte Carlo t-Student · Markowitz")
    st.markdown(
        '<div class="info-box">Configurá los parámetros en el panel izquierdo y presioná '
        '<strong>▶ OPTIMIZAR</strong> para comenzar.</div>',
        unsafe_allow_html=True
    )
    st.stop()

# ═════════════════════════════════════════════════════════════════════════════
# MAIN — only runs after OPTIMIZAR
# ═════════════════════════════════════════════════════════════════════════════
st.markdown("# Portfolio Optimizer")
st.markdown("### Black-Litterman · Monte Carlo t-Student · Markowitz")

# ── Download (cached by tickers + start_date) ─────────────────────────────────
with st.spinner("Descargando datos de mercado..."):
    # Always include SPY as benchmark reference (excluded from optimization)
    tickers_with_spy = tuple(sorted(set(list(assets_input) + ["SPY"])))
    data_all = download_prices(tickers_with_spy, str(start_date))
    assets   = [t for t in assets_input if t in data_all.columns]
    data     = data_all[[a for a in assets]] if assets else pd.DataFrame()

if data.empty or len(assets) < 2:
    st.error("No se pudieron obtener datos. Verificá los tickers.")
    st.stop()

daily_returns = np.log(data / data.shift(1)).dropna()
mean_returns  = daily_returns.mean()
cov_matrix    = daily_returns.cov()
ann_returns   = mean_returns * 252
ann_vol       = np.sqrt(np.diag(cov_matrix.values)) * np.sqrt(252)

# ── SPY benchmark (separate — not part of optimization) ───────────────────────
spy_in_portfolio = "SPY" in assets  # user explicitly included SPY in their portfolio
spy_available = "SPY" in data_all.columns
if spy_available:
    spy_prices  = data_all["SPY"].reindex(daily_returns.index).dropna()
    spy_rets    = np.log(spy_prices / spy_prices.shift(1)).dropna()
    spy_aligned = spy_rets.reindex(daily_returns.index).fillna(0)
    spy_ann_ret = float(spy_rets.mean() * 252)
    spy_ann_vol = float(spy_rets.std() * np.sqrt(252))
    spy_sharpe  = (spy_ann_ret - rf_annual) / spy_ann_vol if spy_ann_vol > 1e-8 else 0.0
    spy_cum     = float((spy_prices.iloc[-1] / spy_prices.iloc[0]) - 1)
    # Sortino para SPY
    _spy_exc = spy_aligned.values - rf_daily
    _spy_dn  = _spy_exc[_spy_exc < 0]
    _spy_dn_std = np.std(_spy_dn) * np.sqrt(252) if len(_spy_dn) > 1 else 1e-6
    spy_sortino = float(np.mean(_spy_exc) * 252 / _spy_dn_std) if _spy_dn_std > 1e-8 else 0.0
    # MDD y Calmar para SPY
    spy_dd_val  = max_drawdown(np.cumprod(1 + spy_aligned.values))
    spy_calmar  = calmar_ratio(spy_ann_ret, spy_dd_val)
else:
    spy_ann_ret = spy_ann_vol = spy_sharpe = spy_cum = None
    spy_sortino = spy_dd_val = spy_calmar = None

# ── Historical stats ──────────────────────────────────────────────────────────
st.markdown('<div class="section-header">Estadísticas históricas</div>', unsafe_allow_html=True)
explain("Retornos y riesgo calculados sobre precios reales del período seleccionado. "
        "El Sharpe histórico mide cuánto retorno extra se obtuvo por cada unidad de volatilidad asumida.")

stats_df = pd.DataFrame({
    "Retorno anualizado (%)": (ann_returns * 100).round(2),
    "Volatilidad anual (%)":  pd.Series(ann_vol * 100, index=assets).round(2),
    "Sharpe histórico":       ((ann_returns - rf_annual) / pd.Series(ann_vol, index=assets)).round(2),
})
# Append SPY benchmark row if available and not already in portfolio
if spy_available and not spy_in_portfolio:
    spy_row = pd.DataFrame({
        "Retorno anualizado (%)": [round(spy_ann_ret * 100, 2)],
        "Volatilidad anual (%)":  [round(spy_ann_vol * 100, 2)],
        "Sharpe histórico":       [round(spy_sharpe, 2)],
    }, index=["SPY (benchmark)"])
    stats_df = pd.concat([stats_df, spy_row])
st.dataframe(stats_df, use_container_width=True)

# ── Correlation heatmap ───────────────────────────────────────────────────────
st.markdown('<div class="section-header">Correlación entre activos</div>', unsafe_allow_html=True)
explain("Escala de −1 a +1. Valores cercanos a +1 indican activos que se mueven juntos (poca diversificación). "
        "Valores cercanos a −1 indican movimiento opuesto (máxima diversificación). "
        "El objetivo es combinar activos con baja correlación entre sí.")

corr = daily_returns.corr()
fig_corr = go.Figure(go.Heatmap(
    z=corr.values, x=assets, y=assets,
    colorscale=[[0, RED], [0.5, PLOT_CARD], [1, BLUE]],
    zmid=0, zmin=-1, zmax=1,
    text=corr.round(2).values,
    texttemplate="%{text}", textfont=dict(size=11),
    hovertemplate="%{y} / %{x}: %{z:.2f}<extra></extra>",
))
fig_corr.update_layout(**plot_layout(height=300 + len(assets)*10))
st.plotly_chart(fig_corr, use_container_width=True)

VIEW_LABELS = {
    "momentum_12_1": "Momentum 12-1",
    "ewma_blend":    "EWMA Blend",
    "risk_adj_mom":  "Momentum / Vol",
}
CONF_LABELS = {
    "sharpe_12m":    "Sharpe 12m",
    "trend_r2":      "Trend R²",
    "vol_stability": "Estabilidad de vol",
}
MODE_LABELS = {
    "bl_views":    "BL + Señales",
    "bl_puro":     "BL Puro",
    "risk_parity": "Risk Parity",
}

# ── Max weight form (shared across modes) ─────────────────────────────────────
st.markdown('<div class="section-header">Límite máximo de exposición por activo</div>', unsafe_allow_html=True)
explain("Único input manual: peso máximo por activo para controlar concentración. "
        "Recomendado: 40% (≈ 2× equal weight para 5 activos). "
        "Si no querés un activo, sacalo directamente de la lista de tickers.")

if st.session_state.get("reset_views", False):
    for asset in assets:
        if f"wmax_{asset}" in st.session_state:
            del st.session_state[f"wmax_{asset}"]
    st.session_state["reset_views"] = False

default_max = float(min(100.0, round(200.0 / len(assets) / 5) * 5))  # 2/N rounded to 5%
weight_bounds = {}
with st.form("bounds_form"):
    cols_per_row = min(5, len(assets))
    rows = [assets[i:i+cols_per_row] for i in range(0, len(assets), cols_per_row)]
    for row in rows:
        cols = st.columns(len(row))
        for col, asset in zip(cols, row):
            with col:
                st.markdown(f"**{asset}**")
                w_max = st.number_input(
                    f"Máx % — {asset}", 0.0, 100.0, default_max, step=5.0,
                    key=f"wmax_{asset}",
                    help=f"Peso máximo permitido para {asset}. "
                         f"Default: {default_max:.0f}% (≈ 2× equal weight). "
                         "100% = sin restricción."
                ) / 100
                weight_bounds[asset] = (0.0, w_max)
    st.form_submit_button("🔄  Re-optimizar")

# ── Signals / BL / Risk Parity (conditional on mode) ─────────────────────────
bounds = [weight_bounds[a] for a in assets]

if port_mode == "bl_views":
    # ── Compute quantitative signals ──
    st.markdown('<div class="section-header">Señales cuantitativas</div>', unsafe_allow_html=True)
    explain(f"Views calculados con {VIEW_LABELS[view_method]} · "
            f"Confianza calculada con {CONF_LABELS[conf_method]}. "
            "Todo automático — sin inputs discrecionales.")

    quant_views_raw = compute_quant_views(data[assets], method=view_method)
    quant_conf_raw  = compute_quant_conf(daily_returns, data[assets], method=conf_method)
    views_annual    = {a: float(quant_views_raw[a]) for a in assets}
    confidences     = {a: float(quant_conf_raw[a])  for a in assets}

    signals_preview = pd.DataFrame({
        f"View — {VIEW_LABELS[view_method]} (% anual)":
            pd.Series({a: f"{views_annual[a]*100:+.2f}%" for a in assets}),
        "Prior histórico (% anual)":
            pd.Series({a: f"{float(ann_returns[a])*100:.2f}%" for a in assets}),
        f"Confianza — {CONF_LABELS[conf_method]}":
            pd.Series({a: f"{confidences[a]*100:.1f}%" for a in assets}),
    })
    st.dataframe(signals_preview, use_container_width=True)

    bl_returns = black_litterman_posterior(
        mean_returns, cov_matrix, views_annual, confidences, tau=tau
    )

    with st.expander("Ver detalle: prior → señal → posterior BL"):
        st.caption("Cómo la señal desplaza el retorno esperado desde el prior histórico.")
        compare_df = pd.DataFrame({
            "Prior histórico (% anual)": (mean_returns * 252 * 100).round(2),
            f"View — {VIEW_LABELS[view_method]} (% anual)":
                pd.Series({a: round(views_annual[a]*100, 2) for a in assets}),
            f"Confianza — {CONF_LABELS[conf_method]} (%)":
                pd.Series({a: round(confidences[a]*100, 1) for a in assets}),
            "Posterior BL (% anual)": (bl_returns * 252 * 100).round(2),
            "Ajuste neto (pp)":       ((bl_returns - mean_returns) * 252 * 100).round(2),
        })
        st.dataframe(compare_df, use_container_width=True)

    with st.spinner("Optimizando portafolio (BL + Señales)..."):
        optimal_weights = optimize_portfolio(
            bl_returns.values, cov_matrix.values, daily_returns,
            rf_daily, bounds, objective=obj_key
        )

elif port_mode == "bl_puro":
    # BL without views: all confidences = 0 → posterior = prior
    st.markdown('<div class="section-header">BL Puro — sin señales adicionales</div>',
                unsafe_allow_html=True)
    explain("El modelo usa el retorno histórico de cada activo como única estimación. "
            "No se incorporan señales de momentum ni tendencia. "
            "BL aporta estabilidad numérica respecto al MVO clásico.")

    bl_returns = mean_returns.copy()   # posterior = prior when confidence = 0

    with st.expander("Ver retornos usados en la optimización"):
        st.dataframe(pd.DataFrame({
            "Retorno histórico anualizado (%)": (mean_returns * 252 * 100).round(2),
            "Volatilidad anual (%)": pd.Series(ann_vol * 100, index=assets).round(2),
            "Sharpe histórico": ((ann_returns - rf_annual) /
                                  pd.Series(ann_vol, index=assets)).round(2),
        }), use_container_width=True)

    with st.spinner("Optimizando portafolio (BL Puro)..."):
        optimal_weights = optimize_portfolio(
            bl_returns.values, cov_matrix.values, daily_returns,
            rf_daily, bounds, objective=obj_key
        )

else:  # risk_parity
    st.markdown('<div class="section-header">Risk Parity — igual contribución al riesgo</div>',
                unsafe_allow_html=True)
    explain("No se estiman retornos futuros. Cada activo aporta el mismo riesgo al portafolio. "
            "Los límites máximos se respetan como restricción adicional.")

    bl_returns = mean_returns.copy()   # used only for display metrics

    with st.spinner("Calculando pesos Risk Parity..."):
        # Apply max bounds: clip and renormalize iteratively
        rp_weights_raw = risk_parity_weights(cov_matrix.values)
        # Enforce max bounds (clip + renormalize up to 10 iterations)
        w = rp_weights_raw.copy()
        max_arr = np.array([weight_bounds[a][1] for a in assets])
        for _ in range(10):
            clipped = np.clip(w, 0, max_arr)
            if clipped.sum() < 1e-8:
                clipped = np.ones(len(assets)) / len(assets)
            w = clipped / clipped.sum()
            if np.all(w <= max_arr + 1e-6):
                break
        optimal_weights = w

    # Show risk contribution breakdown
    vol_port  = np.sqrt(optimal_weights.T @ cov_matrix.values @ optimal_weights)
    mrc       = cov_matrix.values @ optimal_weights
    rc        = optimal_weights * mrc / (vol_port + 1e-12)
    rc_pct    = rc / rc.sum() * 100
    with st.expander("Ver contribución al riesgo por activo"):
        st.caption("En Risk Parity ideal todas las contribuciones son iguales (100%/N activos).")
        rc_df = pd.DataFrame({
            "Peso (%)":                    [f"{w*100:.1f}%" for w in optimal_weights],
            "Contribución al riesgo (%)":  [f"{r:.1f}%" for r in rc_pct],
            "Volatilidad anual (%)":       [f"{float(cov_matrix.loc[a,a]**0.5)*np.sqrt(252)*100:.1f}%"
                                            for a in assets],
        }, index=assets)
        st.dataframe(rc_df, use_container_width=True)

# ── Portfolio metrics (pre-compute before tabs) ───────────────────────────────
port_daily_ret = daily_returns.values @ optimal_weights
cum_ret        = np.exp(np.cumsum(np.log1p(port_daily_ret)))
ann_ret_opt    = float(bl_returns.values @ optimal_weights) * 252
ann_vol_opt    = float(np.sqrt(optimal_weights.T @ cov_matrix.values @ optimal_weights)) * np.sqrt(252)
sharpe_opt     = (ann_ret_opt - rf_annual) / ann_vol_opt if ann_vol_opt > 1e-8 else 0.0
sortino_opt    = sortino_ratio(optimal_weights, daily_returns, rf_daily)
mdd            = max_drawdown(cum_ret)
calmar         = calmar_ratio(ann_ret_opt, mdd)

w_df = pd.DataFrame({"Activo": assets, "Peso": optimal_weights})
w_df = w_df[w_df["Peso"] > 0.001].sort_values("Peso", ascending=False)

# ── CAPM split (computed once, used in Tab 1 display and Monte Carlo) ─────────
_capm_target   = st.session_state.get("sidebar_capm", 0.0)
_exp_hor       = ann_ret_opt * investment_horizon / 252
_denom         = _exp_hor - rf_daily * investment_horizon
w_p            = (_capm_target * investment_horizon / 252 / 100 - rf_daily * investment_horizon) / _denom \
                 if abs(_denom) > 1e-8 else 1.0
# If slider is at 0 (default / not yet set), treat as 100% portfolio
if _capm_target == 0.0:
    w_p = 1.0
w_rf           = 1.0 - w_p

# ═════════════════════════════════════════════════════════════════════════════
# TABS
# ═════════════════════════════════════════════════════════════════════════════
HDR_STYLE = ("font-size:9px;font-weight:700;letter-spacing:2px;text-transform:uppercase;"
             "color:rgba(200,212,224,0.45);padding:6px 0 4px 0;text-align:center")

def metric_row_3(label, values, classes, border):
    cols = st.columns([1.1, 1, 1, 1, 1, 1, 1])
    with cols[0]:
        st.markdown(
            f'<div class="metric-card" style="border-color:{border};text-align:left;padding:12px 10px">'
            f'<div class="metric-label" style="font-size:9px;letter-spacing:1.5px">{label}</div>'
            f'</div>', unsafe_allow_html=True)
    for col, (val, cls) in zip(cols[1:], zip(values, classes)):
        with col:
            st.markdown(
                f'<div class="metric-card" style="border-color:{border}">'
                f'<div class="metric-value {cls}" style="font-size:18px">{val}</div>'
                f'</div>', unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["  Portafolio  ", "  Análisis histórico  ", "  Proyección  "])

# ═══════════════════════════════════════════════════════════════════════════
# TAB 1 — Portafolio: Métricas · Pesos · CAPM
# ═══════════════════════════════════════════════════════════════════════════
with tab1:

    # ── Métricas 3-filas ────────────────────────────────────────────────────
    st.markdown(f'<div class="section-header">Métricas — {MODE_LABELS[port_mode]}</div>',
                unsafe_allow_html=True)
    explain("Sharpe y Sortino miden eficiencia (retorno por riesgo). "
            "Max Drawdown es la peor caída histórica acumulada con estos pesos. "
            "Calmar es el retorno anualizado dividido el Max Drawdown.")

    hdr_cols = st.columns([1.1, 1, 1, 1, 1, 1, 1])
    with hdr_cols[0]:
        st.markdown(f'<div style="{HDR_STYLE}"></div>', unsafe_allow_html=True)
    for col, lbl in zip(hdr_cols[1:], ["Ret. anual", "Volatilidad", "Sharpe", "Sortino", "Max DD", "Calmar"]):
        with col:
            st.markdown(f'<div style="{HDR_STYLE}">{lbl}</div>', unsafe_allow_html=True)

    port_vals = [
        f"{ann_ret_opt*100:.2f}%", f"{ann_vol_opt*100:.2f}%",
        f"{sharpe_opt:.2f}", f"{sortino_opt:.2f}",
        f"{mdd*100:.1f}%", f"{calmar:.2f}",
    ]
    port_cls = [
        "positive" if ann_ret_opt > 0 else "negative", "neutral",
        "positive" if sharpe_opt > 1 else "neutral" if sharpe_opt > 0 else "negative",
        "positive" if sortino_opt > 1 else "neutral" if sortino_opt > 0 else "negative",
        "negative" if mdd < -0.1 else "neutral",
        "positive" if calmar > 0.5 else "neutral",
    ]
    metric_row_3("PORTFOLIO", port_vals, port_cls, "rgba(78,158,200,0.3)")
    st.markdown('<div style="border-top:1px solid rgba(200,212,224,0.1);margin:4px 0"></div>',
                unsafe_allow_html=True)

    if spy_available and not spy_in_portfolio:
        spy_vals = [
            f"{spy_ann_ret*100:.2f}%", f"{spy_ann_vol*100:.2f}%",
            f"{spy_sharpe:.2f}", f"{spy_sortino:.2f}",
            f"{spy_dd_val*100:.1f}%", f"{spy_calmar:.2f}",
        ]
        spy_cls = [
            "positive" if spy_ann_ret > 0 else "negative", "neutral",
            "positive" if spy_sharpe > 1 else "neutral" if spy_sharpe > 0 else "negative",
            "positive" if spy_sortino > 1 else "neutral" if spy_sortino > 0 else "negative",
            "negative" if spy_dd_val < -0.1 else "neutral",
            "positive" if spy_calmar > 0.5 else "neutral",
        ]
        metric_row_3("SPY", spy_vals, spy_cls, "rgba(245,200,66,0.2)")
        st.markdown(
            '<div style="border-top:2px solid rgba(200,212,224,0.15);margin:4px 0">'
            '<div style="font-size:8px;letter-spacing:2px;color:rgba(200,212,224,0.3);'
            'text-align:center;margin-top:2px">DIFERENCIA (PORTFOLIO − SPY)</div></div>',
            unsafe_allow_html=True)
        d_ret = ann_ret_opt - spy_ann_ret; d_vol = ann_vol_opt - spy_ann_vol
        d_sharpe = sharpe_opt - spy_sharpe; d_sortino = sortino_opt - spy_sortino
        d_mdd = mdd - spy_dd_val; d_calmar = calmar - spy_calmar
        metric_row_3("DIFERENCIA",
            [f"{d_ret*100:+.2f}pp", f"{d_vol*100:+.2f}pp", f"{d_sharpe:+.2f}",
             f"{d_sortino:+.2f}", f"{d_mdd*100:+.1f}pp", f"{d_calmar:+.2f}"],
            ["positive" if d_ret > 0 else "negative",
             "positive" if d_vol < 0 else "negative",
             "positive" if d_sharpe > 0 else "negative",
             "positive" if d_sortino > 0 else "negative",
             "positive" if d_mdd > 0 else "negative",
             "positive" if d_calmar > 0 else "negative"],
            "rgba(78,200,120,0.15)")
        explain("Retorno anualizado = proyección BL (expectativa futura). "
                "SPY en retorno histórico realizado — misma unidad para comparar directo.")

    # ── Pesos óptimos y montos ───────────────────────────────────────────────
    st.markdown('<div class="section-header">Pesos óptimos y montos a invertir</div>',
                unsafe_allow_html=True)
    explain(f"Distribución del capital de ${initial_investment:,.0f} que maximiza el {objective}. "
            "La tabla muestra exactamente cuánto destinar a cada activo.")

    inv_data = []
    for _, row in w_df.iterrows():
        monto = row["Peso"] * initial_investment
        inv_data.append({
            "Activo":              row["Activo"],
            "Peso":                f"{row['Peso']*100:.1f}%",
            "Monto a invertir":    f"${monto:,.0f}",
            "Retorno BL (anual)":  f"{float(bl_returns[row['Activo']])*252*100:.2f}%",
            "Volatilidad (anual)": f"{float(cov_matrix.loc[row['Activo'], row['Activo']]**0.5)*np.sqrt(252)*100:.2f}%",
        })
    st.dataframe(pd.DataFrame(inv_data), use_container_width=True, hide_index=True)

    fig_w = go.Figure(go.Bar(
        x=w_df["Activo"], y=w_df["Peso"]*100,
        marker=dict(color=BLUE, line=dict(color="#78bdd8", width=1)),
        text=w_df.apply(lambda r: f"{r['Peso']*100:.1f}%<br>${r['Peso']*initial_investment:,.0f}", axis=1),
        textposition="outside", textfont=dict(color=TEXT_CLR, size=11),
    ))
    fig_w.update_layout(**plot_layout(height=280), yaxis_title="Peso (%)", xaxis_title="")
    st.plotly_chart(fig_w, use_container_width=True)

    # ── CAPM / Capital Market Line ───────────────────────────────────────────
    st.markdown('<div class="section-header">Línea de mercado de capitales (CAPM)</div>',
                unsafe_allow_html=True)
    explain("El slider de retorno objetivo está en el panel izquierdo. "
            "Ajustalo antes de correr OPTIMIZAR para que el Monte Carlo use la asignación correcta. "
            "La tabla debajo muestra exactamente cuánto comprar de cada activo y del bono libre de riesgo.")

    _target_ret_pct_display = _capm_target if _capm_target > 0 else ann_ret_opt * 100
    c1, c2, c3, c4 = st.columns(4)
    for col, (lbl, val, cls) in zip([c1, c2, c3, c4], [
        ("Peso en portafolio",        f"{w_p*100:.1f}%",                    "neutral"),
        ("Peso en bono libre riesgo", f"{w_rf*100:.1f}%",                   "neutral"),
        ("Retorno esperado",          f"{_target_ret_pct_display:.1f}%",    "positive"),
        ("Volatilidad esperada",      f"{w_p*ann_vol_opt*100:.2f}%",        "neutral"),
    ]):
        with col:
            st.markdown(f'<div class="metric-card"><div class="metric-label">{lbl}</div>'
                        f'<div class="metric-value {cls}">{val}</div></div>', unsafe_allow_html=True)

    if w_p > 1:
        st.markdown('<div class="warning-box">⚠️ Apalancamiento — pedís prestado para invertir '
                    'más del 100% en el portafolio. El riesgo aumenta proporcionalmente.</div>',
                    unsafe_allow_html=True)
    elif w_p < 0:
        st.markdown('<div class="warning-box">⚠️ Posición corta en el portafolio.</div>',
                    unsafe_allow_html=True)

    # ── Tabla de composición final incluyendo bono ───────────────────────────
    st.markdown('<div class="section-header">Composición final de la cartera combinada</div>',
                unsafe_allow_html=True)
    explain("Exactamente cuánto comprar de cada activo y cuánto destinar al bono libre de riesgo, "
            "dado el retorno objetivo seleccionado. "
            "Peso portafolio = peso dentro del portafolio óptimo. "
            "Peso total = peso dentro de la cartera completa (portafolio + bono).")

    # Precios en tiempo real
    rt_prices = fetch_realtime_prices(tuple(w_df["Activo"].tolist()))

    capm_rows  = []
    total_cash = 0.0
    for _, row in w_df.iterrows():
        ticker     = row["Activo"]
        peso_total = w_p * row["Peso"]
        monto      = peso_total * initial_investment
        precio     = rt_prices.get(ticker)
        if precio and precio > 0:
            cantidad      = int(monto // precio)
            total_compra  = cantidad * precio
            residual      = monto - total_compra
            total_cash   += residual
            capm_rows.append({
                "Activo":             ticker,
                "Peso portafolio":    f"{row['Peso']*100:.1f}%",
                "Peso total":         f"{peso_total*100:.1f}%",
                "Monto asignado":     f"${monto:,.0f}",
                "Precio":             f"${precio:,.2f}",
                "Cantidad":           cantidad,
                "Total compra":       f"${total_compra:,.0f}",
            })
        else:
            capm_rows.append({
                "Activo":             ticker,
                "Peso portafolio":    f"{row['Peso']*100:.1f}%",
                "Peso total":         f"{peso_total*100:.1f}%",
                "Monto asignado":     f"${monto:,.0f}",
                "Precio":             "—",
                "Cantidad":           "—",
                "Total compra":       "—",
            })

    # Bono libre de riesgo (compra exacta, sin residual)
    monto_rf       = w_rf * initial_investment
    rf_label_short = rf_label.split("(")[0].strip() if "(" in rf_label else rf_label
    capm_rows.append({
        "Activo":          f"Bono libre de riesgo ({rf_label_short})",
        "Peso portafolio": "—",
        "Peso total":      f"{w_rf*100:.1f}%",
        "Monto asignado":  f"${monto_rf:,.0f}",
        "Precio":          "—",
        "Cantidad":        "—",
        "Total compra":    f"${monto_rf:,.0f}",
    })

    # Cash / Money Market (residual de redondeo)
    if total_cash > 0.005:
        capm_rows.append({
            "Activo":          "Cash / Money Market",
            "Peso portafolio": "—",
            "Peso total":      f"{total_cash/initial_investment*100:.2f}%",
            "Monto asignado":  "—",
            "Precio":          "—",
            "Cantidad":        "—",
            "Total compra":    f"${total_cash:,.2f}",
        })

    capm_rows.append({
        "Activo":          "TOTAL",
        "Peso portafolio": "100%",
        "Peso total":      "100%",
        "Monto asignado":  f"${initial_investment:,.0f}",
        "Precio":          "—",
        "Cantidad":        "—",
        "Total compra":    f"${initial_investment:,.0f}",
    })
    st.dataframe(pd.DataFrame(capm_rows), use_container_width=True, hide_index=True)

# ═══════════════════════════════════════════════════════════════════════════
# TAB 2 — Análisis histórico: Frontera + Backtest
# ═══════════════════════════════════════════════════════════════════════════
with tab2:

    # ── Frontera eficiente ───────────────────────────────────────────────────
    st.markdown('<div class="section-header">Frontera eficiente</div>', unsafe_allow_html=True)
    explain("Cada punto representa un portafolio aleatorio con diferentes combinaciones de pesos. "
            "El color indica el Sharpe Ratio (verde = mejor relación retorno/riesgo). "
            "La estrella dorada es el portafolio óptimo. "
            "Los círculos azules son los activos individuales sin diversificación.")

    with st.spinner("Calculando frontera eficiente..."):
        ef_r, ef_v, ef_sr = efficient_frontier(
            tuple(bl_returns.values.tolist()),
            tuple(map(tuple, cov_matrix.values.tolist())),
            rf_daily
        )
    fig_ef = go.Figure()
    fig_ef.add_trace(go.Scatter(
        x=ef_v*100, y=ef_r*100, mode="markers",
        marker=dict(color=ef_sr, colorscale=[[0, RED], [0.5, GOLD], [1, GREEN]],
                    size=4, opacity=0.6,
                    colorbar=dict(title=dict(text="Sharpe", font=dict(color=TEXT_CLR)),
                                  tickfont=dict(color=TEXT_CLR))),
        name="Portafolios aleatorios",
        hovertemplate="Vol: %{x:.1f}% | Ret: %{y:.1f}%<extra></extra>",
    ))
    fig_ef.add_trace(go.Scatter(
        x=[ann_vol_opt*100], y=[ann_ret_opt*100], mode="markers",
        marker=dict(color=GOLD, size=14, symbol="star", line=dict(color="white", width=1)),
        name="Portafolio óptimo",
        hovertemplate=f"Vol: {ann_vol_opt*100:.1f}% | Ret: {ann_ret_opt*100:.1f}%<extra></extra>",
    ))
    for i, asset in enumerate(assets):
        fig_ef.add_trace(go.Scatter(
            x=[ann_vol[i]*100], y=[ann_returns[asset]*100],
            mode="markers+text", text=[asset], textposition="top center",
            textfont=dict(size=10, color=TEXT_CLR),
            marker=dict(color=BLUE, size=8, symbol="circle"),
            name=asset, showlegend=False,
            hovertemplate=f"{asset}<br>Vol: {ann_vol[i]*100:.1f}% | Ret: {ann_returns[asset]*100:.1f}%<extra></extra>",
        ))
    fig_ef.update_layout(**plot_layout(height=420),
                         xaxis_title="Volatilidad anual (%)", yaxis_title="Retorno anual (%)")
    st.plotly_chart(fig_ef, use_container_width=True)

    # ── Backtest histórico ───────────────────────────────────────────────────
    st.markdown('<div class="section-header">Backtest histórico (pesos fijos)</div>',
                unsafe_allow_html=True)
    explain("Simulación de cómo habría rendido este portafolio con los pesos calculados, mantenidos fijos. "
            "Línea azul: portafolio óptimo. "
            "Línea amarilla: benchmark — mismo capital dividido en partes iguales entre todos los activos (sin optimización). "
            "Línea naranja: SPY (S&P 500). "
            "No considera rebalanceo, costos de transacción ni dividendos.")

    port_cum = initial_investment * np.exp(np.cumsum(np.log1p(port_daily_ret)))
    bench_eq = initial_investment * np.exp(np.cumsum(np.log1p(daily_returns.mean(axis=1).values)))
    if spy_available and not spy_in_portfolio:
        spy_cum_bt    = initial_investment * np.exp(np.cumsum(np.log1p(spy_aligned.values)))
        spy_ret_total = (spy_cum_bt[-1] - initial_investment) / initial_investment * 100
    else:
        spy_cum_bt = None

    fig_bt = go.Figure()
    fig_bt.add_trace(go.Scatter(x=daily_returns.index, y=port_cum, mode="lines",
        name="Portafolio óptimo", line=dict(color=BLUE, width=2),
        hovertemplate="$%{y:,.0f}<extra>Portafolio óptimo</extra>"))
    fig_bt.add_trace(go.Scatter(x=daily_returns.index, y=bench_eq, mode="lines",
        name="Pesos iguales (sin optimizar)", line=dict(color=GOLD, width=1.5, dash="dash"),
        hovertemplate="$%{y:,.0f}<extra>Pesos iguales</extra>"))
    if spy_cum_bt is not None:
        fig_bt.add_trace(go.Scatter(x=daily_returns.index, y=spy_cum_bt, mode="lines",
            name="SPY (benchmark)", line=dict(color="#ff8c42", width=1.5, dash="dot"),
            hovertemplate="$%{y:,.0f}<extra>SPY</extra>"))
    fig_bt.add_hline(y=initial_investment, line=dict(color=RED, dash="dot", width=1),
                      annotation_text=f"Capital inicial ${initial_investment:,.0f}",
                      annotation_font_color=RED)
    port_ret_total  = (port_cum[-1]  - initial_investment) / initial_investment * 100
    bench_ret_total = (bench_eq[-1]  - initial_investment) / initial_investment * 100
    bt_r = 170 if spy_cum_bt is not None else 140
    fig_bt.add_annotation(x=daily_returns.index[-1], y=port_cum[-1],
        text=f"  ${port_cum[-1]:,.0f} ({port_ret_total:+.1f}%)",
        showarrow=False, xanchor="left", font=dict(color=BLUE, size=11))
    fig_bt.add_annotation(x=daily_returns.index[-1], y=bench_eq[-1],
        text=f"  ${bench_eq[-1]:,.0f} ({bench_ret_total:+.1f}%)",
        showarrow=False, xanchor="left", font=dict(color=GOLD, size=11))
    if spy_cum_bt is not None:
        fig_bt.add_annotation(x=daily_returns.index[-1], y=spy_cum_bt[-1],
            text=f"  SPY ${spy_cum_bt[-1]:,.0f} ({spy_ret_total:+.1f}%)",
            showarrow=False, xanchor="left", font=dict(color="#ff8c42", size=11))
    fig_bt.update_layout(**plot_layout(height=380, r=bt_r),
                         yaxis_title="Valor (USD)", yaxis_tickformat="$,.0f",
                         hovermode="x unified")
    st.plotly_chart(fig_bt, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════
# TAB 3 — Proyección: Monte Carlo · Distribución · VaR · Percentiles
# ═══════════════════════════════════════════════════════════════════════════
with tab3:

    # ── Monte Carlo ──────────────────────────────────────────────────────────
    _rf_bond_label = f" · {w_rf*100:.0f}% bono" if w_rf > 0.001 else ""
    st.markdown(
        f'<div class="section-header">Monte Carlo — {n_sim:,} escenarios · t-Student df={mc_df}'
        f' · {w_p*100:.0f}% portafolio{_rf_bond_label}</div>',
        unsafe_allow_html=True)
    explain(f"Proyección futura de {horizon_val} {time_unit.lower()} con distribución t-Student "
            f"(df={mc_df}). "
            f"${w_p*initial_investment:,.0f} en el portafolio risky · "
            f"${w_rf*initial_investment:,.0f} en bono libre de riesgo ({rf_label}). "
            f"La banda azul oscura cubre el 50% central de escenarios; la clara cubre el 90%.")

    # Capital risky simulado con Monte Carlo; bono crece a tasa libre de riesgo diaria
    risky_capital = w_p * initial_investment
    bond_final    = w_rf * initial_investment * ((1 + rf_daily) ** investment_horizon)

    with st.spinner("Corriendo simulación Monte Carlo..."):
        mc_risky = monte_carlo(
            tuple(bl_returns.values.tolist()),
            tuple(map(tuple, cov_matrix.values.tolist())),
            tuple(optimal_weights.tolist()),
            investment_horizon, n_sim, risky_capital, mc_df
        )

    # Valor total = portafolio simulado + bono compuesto
    # El bono crece linealmente día a día para reflejarlo en la trayectoria
    bond_trajectory = w_rf * initial_investment * ((1 + rf_daily) ** np.arange(1, investment_horizon + 1))
    mc_values = mc_risky + bond_trajectory[np.newaxis, :]

    final_vals    = mc_values[:, -1]
    losses        = final_vals - initial_investment
    var_pct_low   = 100 - var_confidence          # e.g. 95% conf → 5th percentile
    var_val       = float(np.percentile(losses, var_pct_low))
    cvar_val      = float(np.mean(losses[losses <= var_val]))
    x_axis     = np.arange(investment_horizon)
    pct        = {p: np.percentile(mc_values, p, axis=0) for p in [5, 25, 50, 75, 95]}

    fig_mc = go.Figure()
    fig_mc.add_trace(go.Scatter(x=x_axis, y=pct[95], line=dict(width=0),
                                showlegend=False, hoverinfo="skip"))
    fig_mc.add_trace(go.Scatter(
        x=x_axis, y=pct[5], fill="tonexty",
        fillcolor="rgba(78,158,200,0.07)", line=dict(width=0), name="P5–P95",
        customdata=pct[95],
        hovertemplate="P5: $%{y:,.0f}  |  P95: $%{customdata:,.0f}<extra>Banda 90%</extra>"))
    fig_mc.add_trace(go.Scatter(x=x_axis, y=pct[75], line=dict(width=0),
                                showlegend=False, hoverinfo="skip"))
    fig_mc.add_trace(go.Scatter(
        x=x_axis, y=pct[25], fill="tonexty",
        fillcolor="rgba(78,158,200,0.15)", line=dict(width=0), name="P25–P75",
        customdata=pct[75],
        hovertemplate="P25: $%{y:,.0f}  |  P75: $%{customdata:,.0f}<extra>Banda central 50%</extra>"))
    for p, lbl, clr in [(5, "P5", RED), (50, "Mediana", BLUE), (95, "P95", GREEN)]:
        fig_mc.add_trace(go.Scatter(x=x_axis, y=pct[p], mode="lines",
            line=dict(color=clr, width=1.8), name=lbl,
            hovertemplate=f"{lbl}: $%{{y:,.0f}}<extra></extra>"))
    fig_mc.add_hline(y=initial_investment, line=dict(color=GOLD, dash="dash", width=1.5),
                      annotation_text=f"Capital inicial ${initial_investment:,.0f}",
                      annotation_font_color=GOLD)
    last_x = investment_horizon - 1
    for p_val, lbl, clr in [(95, "P95", GREEN), (50, "Mediana", BLUE), (5, "P5", RED)]:
        fig_mc.add_annotation(x=last_x, y=pct[p_val][-1],
            text=f"  {lbl}: ${pct[p_val][-1]:,.0f}",
            showarrow=False, xanchor="left", font=dict(color=clr, size=11))
    fig_mc.update_layout(**plot_layout(height=420, r=160),
                         xaxis_title="Días", yaxis_title="Valor (USD)",
                         yaxis_tickformat="$,.0f", hovermode="x unified")
    st.plotly_chart(fig_mc, use_container_width=True)

    # ── Tabla resumen Monte Carlo ─────────────────────────────────────────────
    mc_summary = pd.DataFrame([
        {
            "Escenario":        "P5  (pesimista)",
            "Valor final":      f"${pct[5][-1]:,.0f}",
            "Ganancia / Pérdida": f"${pct[5][-1] - initial_investment:,.0f}",
            "Retorno total":    f"{(pct[5][-1]/initial_investment - 1)*100:.1f}%",
        },
        {
            "Escenario":        "P25 (banda baja)",
            "Valor final":      f"${pct[25][-1]:,.0f}",
            "Ganancia / Pérdida": f"${pct[25][-1] - initial_investment:,.0f}",
            "Retorno total":    f"{(pct[25][-1]/initial_investment - 1)*100:.1f}%",
        },
        {
            "Escenario":        "P50  Mediana",
            "Valor final":      f"${pct[50][-1]:,.0f}",
            "Ganancia / Pérdida": f"${pct[50][-1] - initial_investment:,.0f}",
            "Retorno total":    f"{(pct[50][-1]/initial_investment - 1)*100:.1f}%",
        },
        {
            "Escenario":        "P75 (banda alta)",
            "Valor final":      f"${pct[75][-1]:,.0f}",
            "Ganancia / Pérdida": f"${pct[75][-1] - initial_investment:,.0f}",
            "Retorno total":    f"{(pct[75][-1]/initial_investment - 1)*100:.1f}%",
        },
        {
            "Escenario":        "P95 (optimista)",
            "Valor final":      f"${pct[95][-1]:,.0f}",
            "Ganancia / Pérdida": f"${pct[95][-1] - initial_investment:,.0f}",
            "Retorno total":    f"{(pct[95][-1]/initial_investment - 1)*100:.1f}%",
        },
    ])
    st.dataframe(mc_summary, use_container_width=True, hide_index=True)

    # ── Distribución + VaR/CVaR ──────────────────────────────────────────────
    st.markdown('<div class="section-header">Distribución de valores finales</div>',
                unsafe_allow_html=True)
    explain(f"Histograma de los valores finales al término del horizonte. "
            f"La línea amarilla es el capital inicial. "
            f"La línea roja es el umbral del VaR {var_confidence}%: "
            f"el {var_pct_low}% de los escenarios peores cae debajo.")

    fig_hist = go.Figure()
    fig_hist.add_trace(go.Histogram(
        x=final_vals, nbinsx=100,
        marker=dict(color=BLUE, opacity=0.7, line=dict(color=PLOT_CARD, width=0.3)),
        name="Valor final",
        hovertemplate="$%{x:,.0f}: %{y} escenarios<extra></extra>",
    ))
    fig_hist.add_vline(x=initial_investment, line=dict(color=GOLD, dash="dash", width=2),
                        annotation_text="Capital inicial", annotation_font_color=GOLD)
    fig_hist.add_vline(x=initial_investment + var_val,
                        line=dict(color=RED, dash="dot", width=1.5),
                        annotation_text=f"VaR {var_confidence}%", annotation_font_color=RED)
    fig_hist.update_layout(**plot_layout(height=300),
                            xaxis_title="Valor final (USD)", xaxis_tickformat="$,.0f",
                            yaxis_title="Frecuencia")
    st.plotly_chart(fig_hist, use_container_width=True)

    v1, v2 = st.columns(2)
    with v1:
        cls = "positive" if var_val >= 0 else "negative"
        st.markdown(f'<div class="metric-card"><div class="metric-label">VaR {var_confidence}% (Valor en Riesgo)</div>'
                    f'<div class="metric-value {cls}">${var_val:,.0f}</div>'
                    f'<div style="font-size:11px;color:rgba(200,212,224,0.45);margin-top:6px">'
                    f'Pérdida máxima en el {var_confidence}% de los escenarios</div></div>',
                    unsafe_allow_html=True)
    with v2:
        cls = "positive" if cvar_val >= 0 else "negative"
        st.markdown(f'<div class="metric-card"><div class="metric-label">CVaR {var_confidence}% (Expected Shortfall)</div>'
                    f'<div class="metric-value {cls}">${cvar_val:,.0f}</div>'
                    f'<div style="font-size:11px;color:rgba(200,212,224,0.45);margin-top:6px">'
                    f'Pérdida promedio en el {var_pct_low}% de peores escenarios</div></div>',
                    unsafe_allow_html=True)

    # ── Tabla de percentiles ─────────────────────────────────────────────────
    st.markdown('<div class="section-header">Distribución de resultados por percentil</div>',
                unsafe_allow_html=True)
    explain("Cada fila muestra el resultado al finalizar el horizonte para ese percentil. "
            "Percentil 10 = el 10% de los escenarios terminó por debajo de ese valor.")

    target_pcts = [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 99]
    pct_vals    = np.percentile(losses, target_pcts)
    pct_df      = pd.DataFrame({
        "Percentil":         target_pcts,
        "P&G (USD)":         [f"${v:,.0f}" for v in pct_vals],
        "Valor final (USD)": [f"${initial_investment+v:,.0f}" for v in pct_vals],
        "Retorno total (%)": [f"{v/initial_investment*100:.1f}%" for v in pct_vals],
    })
    st.dataframe(pct_df, use_container_width=True, hide_index=True)
