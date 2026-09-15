"""
Gold Price Analysis & Forecasting — Streamlit Dashboard
"""

import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix,
)
from xgboost import XGBRegressor
import warnings
warnings.filterwarnings("ignore")

# ============================================================================
# PAGE CONFIG + THEME
# ============================================================================
st.set_page_config(page_title="Gold Price Forecasting", page_icon="🪙", layout="wide")

GOLD = "#C9A227"
GOLD_LIGHT = "#E8C766"
BG = "#141210"
PANEL = "#1D1A16"
INK = "#EDE6D6"
MUTED = "#9C9384"
UP = "#4C9A6B"
DOWN = "#B5473C"

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,600&family=Inter:wght@400;500;600&display=swap');

html, body, [class*="css"] {{
    font-family: 'Inter', sans-serif;
}}
.stApp {{
    background-color: {BG};
    color: {INK};
}}
h1, h2, h3 {{
    font-family: 'Fraunces', serif !important;
    color: {INK} !important;
    font-weight: 600 !important;
}}
[data-testid="stSidebar"] {{
    background-color: {PANEL};
    border-right: 1px solid #33301F;
}}
[data-testid="stMetric"] {{
    background-color: {PANEL};
    border: 1px solid #33301F;
    border-radius: 6px;
    padding: 14px 16px 10px 16px;
}}
[data-testid="stMetricLabel"] {{
    color: {MUTED} !important;
}}
[data-testid="stMetricValue"] {{
    color: {GOLD_LIGHT} !important;
}}
.stTabs [data-baseweb="tab-list"] {{
    gap: 4px;
}}
.stTabs [data-baseweb="tab"] {{
    background-color: {PANEL};
    border-radius: 4px 4px 0 0;
    color: {MUTED};
}}
.stTabs [aria-selected="true"] {{
    background-color: {GOLD} !important;
    color: {BG} !important;
}}
.hero {{
    padding: 28px 32px;
    background: linear-gradient(120deg, #1D1A16 0%, #26211A 100%);
    border: 1px solid #33301F;
    border-left: 4px solid {GOLD};
    border-radius: 6px;
    margin-bottom: 22px;
}}
.hero h1 {{
    margin: 0 0 4px 0;
    font-size: 2.1rem;
}}
.hero p {{
    color: {MUTED};
    margin: 0;
    font-size: 0.95rem;
}}
</style>
""", unsafe_allow_html=True)

PLOTLY_LAYOUT = dict(
    paper_bgcolor=PANEL,
    plot_bgcolor=PANEL,
    font=dict(color=INK, family="Inter"),
    colorway=[GOLD, "#7C9DB5", "#B5473C", "#4C9A6B", "#9C7BB5"],
    xaxis=dict(gridcolor="#2C2820", zerolinecolor="#2C2820"),
    yaxis=dict(gridcolor="#2C2820", zerolinecolor="#2C2820"),
    legend=dict(bgcolor="rgba(0,0,0,0)"),
    margin=dict(t=40, l=10, r=10, b=10),
)

# ============================================================================
# DATA LOADING (real, public CSVs — see notebook for full source documentation)
# ============================================================================
GOLD_URL = "https://raw.githubusercontent.com/datasets/gold-prices/main/data/monthly.csv"
OIL_URL = "https://raw.githubusercontent.com/datasets/oil-prices/main/data/brent-daily.csv"
SP500_URL = "https://raw.githubusercontent.com/datasets/s-and-p-500/main/data/data.csv"
FX_URL = "https://raw.githubusercontent.com/datasets/exchange-rates/main/data/daily.csv"

USDX_WEIGHTS = {
    "Euro": 0.576, "Japan": 0.136, "United Kingdom": 0.119,
    "Canada": 0.091, "Sweden": 0.042, "Switzerland": 0.036,
}

# Approximate historical USD/PKR annual average rates (State Bank of Pakistan /
# widely published benchmarks). No free no-login direct-CSV exists for PKR
# history, so we linearly interpolate between these year-anchors to build a
# monthly series. This is clearly an APPROXIMATION for illustration, not a
# precise daily rate.
PKR_ANCHORS = {
    2000: 53, 2002: 60, 2004: 58, 2006: 60, 2008: 70, 2010: 85, 2012: 94,
    2014: 98, 2016: 104, 2017: 105, 2018: 124, 2019: 155, 2020: 160,
    2021: 162, 2022: 205, 2023: 280, 2024: 278, 2025: 280, 2026: 285,
}

GRAMS_PER_OZ = 31.1035
GRAMS_PER_TOLA = 11.6638

# Major gold-market events for chart annotations
GOLD_EVENTS = [
    ("2001-09-01", "9/11 Attacks"),
    ("2008-09-01", "Global Financial Crisis"),
    ("2011-08-01", "Gold hits record high (~$1900)"),
    ("2013-04-01", "\"Gold Crash\" / Taper Tantrum"),
    ("2016-06-01", "Brexit Vote"),
    ("2018-06-01", "US-China Trade War Escalates"),
    ("2020-03-01", "COVID-19 Crash & Safe-Haven Surge"),
    ("2022-02-01", "Russia-Ukraine War Begins"),
    ("2023-03-01", "US Banking Crisis (SVB Collapse)"),
]


def get_usd_pkr_series(index):
    """Builds a monthly USD/PKR rate series over the given DatetimeIndex by
    linearly interpolating between known annual-average benchmark rates."""
    years = sorted(PKR_ANCHORS.keys())
    anchor_dates = pd.to_datetime([f"{y}-06-01" for y in years])
    anchor_rates = [PKR_ANCHORS[y] for y in years]
    anchor_series = pd.Series(anchor_rates, index=anchor_dates)
    full_index = index.union(anchor_series.index).sort_values()
    interpolated = anchor_series.reindex(full_index).interpolate(method="time")
    return interpolated.reindex(index).ffill().bfill()


@st.cache_data(show_spinner="Fetching live data...", ttl=3600)
def load_gold_dataset(start="2000-01-01"):
    gold = pd.read_csv(GOLD_URL)
    gold["Date"] = pd.to_datetime(gold["Date"], errors="coerce")
    gold = gold.set_index("Date").rename(columns={"Price": "Gold"})[["Gold"]]

    oil = pd.read_csv(OIL_URL)
    oil["Date"] = pd.to_datetime(oil["Date"])
    oil = oil.set_index("Date").rename(columns={"Price": "Crude_Oil"})
    oil = oil["Crude_Oil"].resample("MS").mean().to_frame()

    sp500 = pd.read_csv(SP500_URL)
    sp500["Date"] = pd.to_datetime(sp500["Date"])
    sp500 = sp500.set_index("Date")[["SP500"]]
    sp500.index = sp500.index.to_period("M").to_timestamp()

    fx = pd.read_csv(FX_URL)
    fx["Date"] = pd.to_datetime(fx["Date"])
    fx = fx[fx["Country"].isin(USDX_WEIGHTS.keys())]
    pivot = fx.pivot(index="Date", columns="Country", values="Exchange rate")
    monthly_fx = pivot.resample("MS").mean()
    available_currencies = [country for country in USDX_WEIGHTS if country in monthly_fx]
    if not available_currencies:
        raise ValueError("The FX dataset contains none of the currencies required for the USD Index proxy.")
    monthly_fx = monthly_fx[available_currencies].ffill().bfill()
    rebased = monthly_fx.div(monthly_fx.iloc[0]).mul(100)
    total_weight = sum(USDX_WEIGHTS[country] for country in available_currencies)
    usd_index = sum(rebased[country] * USDX_WEIGHTS[country] for country in available_currencies) / total_weight
    usd = usd_index.rename("USD_Index").to_frame()

    data = gold.join([usd, oil, sp500], how="inner")
    data = data.loc[start:]
    data = data.dropna()
    data.index.name = "Date"

    pkr_rate = get_usd_pkr_series(data.index)
    data["USD_PKR"] = pkr_rate
    data["Gold_PKR_per_tola"] = data["Gold"] / GRAMS_PER_OZ * GRAMS_PER_TOLA * data["USD_PKR"]
    return data


def add_technical_indicators(data, price_col="Gold"):
    data = data.copy()
    data["SMA_3"] = data[price_col].rolling(3).mean()
    data["SMA_12"] = data[price_col].rolling(12).mean()
    data["EMA_3"] = data[price_col].ewm(span=3, adjust=False).mean()
    data["EMA_12"] = data[price_col].ewm(span=12, adjust=False).mean()

    delta = data[price_col].diff()
    gain = delta.clip(lower=0).rolling(6).mean()
    loss = (-delta.clip(upper=0)).rolling(6).mean()
    rs = gain / loss.replace(0, np.nan)
    data["RSI_6"] = 100 - (100 / (1 + rs))

    ema6 = data[price_col].ewm(span=6, adjust=False).mean()
    ema12 = data[price_col].ewm(span=12, adjust=False).mean()
    data["MACD"] = ema6 - ema12
    data["MACD_signal"] = data["MACD"].ewm(span=3, adjust=False).mean()

    data["BB_mid"] = data[price_col].rolling(12).mean()
    bb_std = data[price_col].rolling(12).std()
    data["BB_upper"] = data["BB_mid"] + 2 * bb_std
    data["BB_lower"] = data["BB_mid"] - 2 * bb_std
    data["Volatility_6"] = data[price_col].pct_change().rolling(6).std()
    return data


def add_lag_features(data, price_col="Gold", lags=(1, 2, 3, 6, 12)):
    data = data.copy()
    for lag in lags:
        data[f"{price_col}_lag{lag}"] = data[price_col].shift(lag)
    return data


def add_macro_features(data):
    data = data.copy()
    for col in ["USD_Index", "Crude_Oil", "SP500"]:
        data[f"{col}_pct_change"] = data[col].pct_change()
    return data


def add_target(data, price_col="Gold", horizon=1):
    data = data.copy()
    data["Target_Price"] = data[price_col].shift(-horizon)
    data["Target_Direction"] = (data["Target_Price"] > data[price_col]).astype(int)
    return data


FEATURE_COLS = [
    "USD_Index", "Crude_Oil", "SP500", "SMA_3", "SMA_12", "EMA_3", "EMA_12",
    "RSI_6", "MACD", "MACD_signal", "BB_upper", "BB_lower", "Volatility_6",
    "Gold_lag1", "Gold_lag2", "Gold_lag3", "Gold_lag6", "Gold_lag12",
    "USD_Index_pct_change", "Crude_Oil_pct_change", "SP500_pct_change",
]


def build_feature_set(data):
    data = add_technical_indicators(data)
    data = add_lag_features(data)
    data = add_macro_features(data)
    data = add_target(data)
    return data.dropna()


def time_split(data, test_size=0.2):
    n = len(data)
    split_idx = int(n * (1 - test_size))
    return data.iloc[:split_idx], data.iloc[split_idx:]


def evaluate_regression(y_true, y_pred, name):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    return {"Model": name, "RMSE": rmse, "MAE": mae, "R2": r2, "MAPE (%)": mape}


@st.cache_resource(show_spinner="Training models...")
def train_all_models(feat_df):
    train, test = time_split(feat_df, test_size=0.2)
    X_train, y_train = train[FEATURE_COLS], train["Target_Price"]
    X_test, y_test = test[FEATURE_COLS], test["Target_Price"]

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    results, predictions, models = [], {}, {}

    lr = LinearRegression().fit(X_train_s, y_train)
    pred = lr.predict(X_test_s)
    results.append(evaluate_regression(y_test.values, pred, "Linear Regression"))
    predictions["Linear Regression"] = pred
    models["Linear Regression"] = lr

    rf = RandomForestRegressor(n_estimators=300, max_depth=8, random_state=42)
    rf.fit(X_train, y_train)
    pred = rf.predict(X_test)
    results.append(evaluate_regression(y_test.values, pred, "Random Forest"))
    predictions["Random Forest"] = pred
    models["Random Forest"] = rf

    xgb = XGBRegressor(n_estimators=400, max_depth=5, learning_rate=0.03, random_state=42)
    xgb.fit(X_train, y_train)
    pred = xgb.predict(X_test)
    results.append(evaluate_regression(y_test.values, pred, "XGBoost"))
    predictions["XGBoost"] = pred
    models["XGBoost"] = xgb

    try:
        gold_series = train["Gold"].asfreq("MS").ffill()
        arima_model = ARIMA(gold_series, order=(5, 1, 0)).fit()
        arima_pred = arima_model.forecast(steps=len(test))
        results.append(evaluate_regression(test["Gold"].values, arima_pred.values, "ARIMA"))
        predictions["ARIMA"] = arima_pred.values
    except Exception:
        pass

    results_df = pd.DataFrame(results).sort_values("RMSE").reset_index(drop=True)

    X_train_c, y_train_c = train[FEATURE_COLS], train["Target_Direction"]
    X_test_c, y_test_c = test[FEATURE_COLS], test["Target_Direction"]
    clf = RandomForestClassifier(n_estimators=300, max_depth=6, random_state=42, class_weight="balanced")
    clf.fit(X_train_c, y_train_c)
    clf_pred = clf.predict(X_test_c)
    clf_metrics = {
        "Accuracy": accuracy_score(y_test_c, clf_pred),
        "Precision": precision_score(y_test_c, clf_pred),
        "Recall": recall_score(y_test_c, clf_pred),
        "F1": f1_score(y_test_c, clf_pred),
    }
    cm = confusion_matrix(y_test_c, clf_pred)

    return {
        "train": train, "test": test, "results_df": results_df,
        "predictions": predictions, "models": models, "rf": rf, "xgb": xgb,
        "scaler": scaler, "clf": clf, "clf_metrics": clf_metrics, "cm": cm,
        "clf_pred": clf_pred,
    }


# ============================================================================
# LOAD DATA + TRAIN
# ============================================================================
CORE_COLS = ["Gold", "USD_Index", "Crude_Oil", "SP500"]

with st.sidebar:
    st.markdown("### Settings")
    start_year = st.slider("Data start year", min_value=1975, max_value=2022, value=2000, step=1)
    show_pkr = st.checkbox("💰 Show Gold price in PKR (per tola)", value=False)
    st.markdown("---")
    st.caption(
        "Data loaded live from public open datasets — gold price, Brent crude oil, "
        "S&P 500, and a USD Index proxy built from real Federal Reserve FX rates. "
        "Monthly frequency. USD/PKR is an interpolated benchmark series (see caption "
        "in the PKR view) — treat it as illustrative, not an exact historical quote."
    )

df = load_gold_dataset(start=f"{start_year}-01-01")
feat_df = build_feature_set(df)
bundle = train_all_models(feat_df)

# ============================================================================
# HERO
# ============================================================================
latest = df.iloc[-1]
prev = df.iloc[-2]
mom_change = (latest["Gold"] - prev["Gold"]) / prev["Gold"] * 100

st.markdown(f"""
<div class="hero">
<h1>🪙 Gold Price Analysis & Forecasting</h1>
<p>Macro-aware forecasting — gold priced against the US Dollar, Crude Oil, and the S&P 500,
not in isolation. Data range: {df.index.min().strftime('%b %Y')} → {df.index.max().strftime('%b %Y')}.</p>
</div>
""", unsafe_allow_html=True)

if show_pkr:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Gold (USD/oz)", f"${latest['Gold']:,.0f}", f"{mom_change:+.1f}% MoM")
    c2.metric("Gold (PKR/tola)", f"₨{latest['Gold_PKR_per_tola']:,.0f}")
    c3.metric("USD/PKR (approx.)", f"₨{latest['USD_PKR']:,.1f}")
    c4.metric("Crude Oil", f"${latest['Crude_Oil']:.1f}")
else:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Gold (latest)", f"${latest['Gold']:,.0f}", f"{mom_change:+.1f}% MoM")
    c2.metric("USD Index (proxy)", f"{latest['USD_Index']:.1f}")
    c3.metric("Crude Oil", f"${latest['Crude_Oil']:.1f}")
    c4.metric("S&P 500", f"{latest['SP500']:,.0f}")

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
    ["📈 Trends", "📅 Year/Month Explorer", "🔗 Correlations", "🤖 Forecasting Models",
     "🔮 Live Prediction", "💼 Backtest"]
)

# ---------------- TAB 1: TRENDS ----------------
with tab1:
    price_col = "Gold_PKR_per_tola" if show_pkr else "Gold"
    price_label = "Gold Price (PKR/tola)" if show_pkr else "Gold Price (USD/oz)"
    currency_prefix = "₨" if show_pkr else "$"

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df[price_col], name="Gold", line=dict(color=GOLD, width=2)))
    fig.add_trace(go.Scatter(x=df.index, y=df[price_col].rolling(3).mean(), name="3-mo MA",
                              line=dict(color=GOLD_LIGHT, width=1, dash="dot")))
    fig.add_trace(go.Scatter(x=df.index, y=df[price_col].rolling(12).mean(), name="12-mo MA",
                              line=dict(color="#7C9DB5", width=1, dash="dot")))
    for event_date, label in GOLD_EVENTS:
        event_ts = pd.to_datetime(event_date)
        if df.index.min() <= event_ts <= df.index.max():
            fig.add_vline(x=event_ts, line_dash="dot", line_color=MUTED, opacity=0.5)
    fig.update_layout(**PLOTLY_LAYOUT, title=f"{price_label} with Moving Averages", height=420)
    st.plotly_chart(fig, width='stretch')
    st.caption("Dotted vertical lines mark major macro/geopolitical events (hover near them for context in the Correlations tab discussion).")

    if show_pkr:
        st.info(
            "USD/PKR uses interpolated benchmark rates (not a daily quoted series, since no "
            "free no-login direct-CSV source exists for full PKR history) — treat the PKR "
            "view as illustrative of the *trend*, not an exact historical price.",
            icon="ℹ️",
        )

    col_a, col_b = st.columns(2)
    with col_a:
        vol = df["Gold"].pct_change().rolling(6).std() * 100
        fig2 = go.Figure(go.Scatter(x=vol.index, y=vol, fill="tozeroy",
                                     line=dict(color=DOWN), fillcolor="rgba(181,71,60,0.25)"))
        fig2.update_layout(**PLOTLY_LAYOUT, title="6-Month Rolling Volatility (%)", height=320)
        st.plotly_chart(fig2, width='stretch')
    with col_b:
        normalized = df[CORE_COLS] / df[CORE_COLS].iloc[0] * 100
        fig3 = go.Figure()
        for col in normalized.columns:
            fig3.add_trace(go.Scatter(x=normalized.index, y=normalized[col], name=col))
        fig3.update_layout(**PLOTLY_LAYOUT, title="Normalized Comparison (Base=100)", height=320)
        st.plotly_chart(fig3, width='stretch')

# ---------------- TAB 2: YEAR/MONTH EXPLORER ----------------
with tab2:
    price_col = "Gold_PKR_per_tola" if show_pkr else "Gold"
    currency_prefix = "₨" if show_pkr else "$"
    unit_label = "PKR/tola" if show_pkr else "USD/oz"

    st.subheader("Monthly Returns Heatmap — Every Year at a Glance")
    monthly_returns = df["Gold"].pct_change() * 100
    heat = monthly_returns.to_frame("Return")
    heat["Year"] = heat.index.year
    heat["Month"] = heat.index.strftime("%b")
    month_order = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    pivot_heat = heat.pivot(index="Year", columns="Month", values="Return").reindex(columns=month_order)

    fig_heat = px.imshow(
        pivot_heat, color_continuous_scale=["#B5473C", "#1D1A16", "#4C9A6B"],
        aspect="auto", labels=dict(color="Return %"),
        zmin=-8, zmax=8,
    )
    fig_heat.update_layout(**PLOTLY_LAYOUT, title="Gold Monthly Returns (%) by Year", height=520)
    fig_heat.update_xaxes(side="top")
    st.plotly_chart(fig_heat, width='stretch')
    st.caption("Green = gold gained that month · Red = gold fell that month. Scan across a row to see a full year; down a column to compare the same month across years. (Returns are in USD terms regardless of the PKR toggle, since PKR gains mix in currency depreciation.)")

    st.markdown("---")
    st.subheader(f"Pick a Year — Month-by-Month Breakdown ({unit_label})")
    years_available = sorted(df.index.year.unique(), reverse=True)
    picked_year = st.selectbox("Year", years_available)

    year_data = df[df.index.year == picked_year]
    if len(year_data) > 0:
        col1, col2 = st.columns([2, 1])
        with col1:
            fig_year = go.Figure(go.Bar(
                x=year_data.index.strftime("%b"), y=year_data[price_col],
                marker_color=GOLD, marker_line_color=GOLD_LIGHT, marker_line_width=1,
            ))
            fig_year.update_layout(**PLOTLY_LAYOUT, title=f"Gold Price by Month — {picked_year} ({unit_label})", height=380)
            st.plotly_chart(fig_year, width='stretch')
        with col2:
            yr_start, yr_end = year_data[price_col].iloc[0], year_data[price_col].iloc[-1]
            yr_change = (yr_end - yr_start) / yr_start * 100
            st.metric(f"{picked_year} Change", f"{currency_prefix}{yr_end:,.0f}", f"{yr_change:+.1f}%")
            st.metric("Year High", f"{currency_prefix}{year_data[price_col].max():,.0f}")
            st.metric("Year Low", f"{currency_prefix}{year_data[price_col].min():,.0f}")

        st.markdown("---")
        st.subheader(f"Compare Gold, USD Index & Crude Oil — {picked_year}")
        st.caption("Each series is shown on its own scale (Gold, USD Index, and Crude Oil aren't comparable in raw units).")

        series_specs = [
            ("Gold", price_col, GOLD, unit_label),
            ("USD Index (proxy)", "USD_Index", "#7C9DB5", "index level"),
            ("Crude Oil", "Crude_Oil", "#B5473C", "USD/barrel"),
        ]
        cols = st.columns(3)
        for c, (label, col_name, color, unit) in zip(cols, series_specs):
            with c:
                fig_mini = go.Figure(go.Bar(
                    x=year_data.index.strftime("%b"), y=year_data[col_name],
                    marker_color=color,
                ))
                fig_mini.update_layout(**{**PLOTLY_LAYOUT, "margin": dict(t=36, l=10, r=10, b=10)},
                                        title=f"{label} — {picked_year}", height=280)
                st.plotly_chart(fig_mini, width='stretch')
                s_start, s_end = year_data[col_name].iloc[0], year_data[col_name].iloc[-1]
                s_change = (s_end - s_start) / s_start * 100
                st.caption(f"{label}: {s_end:,.1f} {unit} ({s_change:+.1f}% over the year)")
    else:
        st.info("No data for this year in the selected range.")

# ---------------- TAB 3: CORRELATIONS ----------------
with tab3:
    corr = df.corr()
    fig_corr = px.imshow(
        corr, text_auto=".2f", color_continuous_scale=["#B5473C", "#1D1A16", "#4C9A6B"],
        zmin=-1, zmax=1, aspect="auto",
    )
    fig_corr.update_layout(**PLOTLY_LAYOUT, title="Correlation Matrix", height=440)
    st.plotly_chart(fig_corr, width='stretch')

    st.markdown("""
**How to read this:** Gold typically shows a *negative* correlation with the USD Index
(gold is priced in dollars) and a *positive* correlation with Crude Oil (both react to
inflation expectations). The S&P 500 relationship is often weak on a monthly basis, but gold
tends to spike specifically during equity sell-offs — a single correlation number can
understate that "flight to safety" effect.
""")

    st.subheader("Technical Indicators")
    ind_choice = st.selectbox("Choose indicator", ["RSI (6-month)", "MACD", "Bollinger Bands"])
    if ind_choice == "RSI (6-month)":
        fig_ind = go.Figure(go.Scatter(x=feat_df.index, y=feat_df["RSI_6"], line=dict(color=GOLD)))
        fig_ind.add_hline(y=70, line_dash="dash", line_color=DOWN)
        fig_ind.add_hline(y=30, line_dash="dash", line_color=UP)
        fig_ind.update_layout(**PLOTLY_LAYOUT, title="RSI (6-month)", height=340)
    elif ind_choice == "MACD":
        fig_ind = go.Figure()
        fig_ind.add_trace(go.Scatter(x=feat_df.index, y=feat_df["MACD"], name="MACD", line=dict(color=GOLD)))
        fig_ind.add_trace(go.Scatter(x=feat_df.index, y=feat_df["MACD_signal"], name="Signal", line=dict(color="#7C9DB5")))
        fig_ind.update_layout(**PLOTLY_LAYOUT, title="MACD", height=340)
    else:
        fig_ind = go.Figure()
        fig_ind.add_trace(go.Scatter(x=feat_df.index, y=feat_df["Gold"], name="Gold", line=dict(color=GOLD)))
        fig_ind.add_trace(go.Scatter(x=feat_df.index, y=feat_df["BB_upper"], name="Upper Band", line=dict(color=MUTED, dash="dot")))
        fig_ind.add_trace(go.Scatter(x=feat_df.index, y=feat_df["BB_lower"], name="Lower Band", line=dict(color=MUTED, dash="dot")))
        fig_ind.update_layout(**PLOTLY_LAYOUT, title="Bollinger Bands", height=340)
    st.plotly_chart(fig_ind, width='stretch')

# ---------------- TAB 4: MODELS ----------------
with tab4:
    st.subheader("Model Comparison (Test Set)")
    st.dataframe(
        bundle["results_df"].style.highlight_min(subset=["RMSE", "MAE", "MAPE (%)"], color=f"{GOLD}44")
        .format({"RMSE": "{:.2f}", "MAE": "{:.2f}", "R2": "{:.3f}", "MAPE (%)": "{:.2f}"}),
        width='stretch',
    )
    st.caption("Lower RMSE / MAE / MAPE = better. This is a regression problem (predicting a price level), so accuracy doesn't apply here.")

    test = bundle["test"]
    fig_pred = go.Figure()
    fig_pred.add_trace(go.Scatter(x=test.index, y=test["Target_Price"], name="Actual",
                                    line=dict(color=INK, width=2)))
    for name, pred in bundle["predictions"].items():
        fig_pred.add_trace(go.Scatter(x=test.index[:len(pred)], y=pred, name=name, opacity=0.75))
    fig_pred.update_layout(**PLOTLY_LAYOUT, title="Actual vs Predicted Gold Price (Test Period)", height=420)
    st.plotly_chart(fig_pred, width='stretch')

    col1, col2 = st.columns(2)
    with col1:
        importances = pd.Series(bundle["rf"].feature_importances_, index=FEATURE_COLS).sort_values(ascending=False).head(10)
        fig_imp = go.Figure(go.Bar(x=importances.values, y=importances.index, orientation="h", marker_color=GOLD))
        fig_imp.update_layout(**{**PLOTLY_LAYOUT, "yaxis": dict(autorange="reversed")},
                               title="Top 10 Feature Importances (Random Forest)", height=380)
        st.plotly_chart(fig_imp, width='stretch')
    with col2:
        st.markdown("##### Direction Classifier (Up/Down)")
        m = bundle["clf_metrics"]
        mcol1, mcol2 = st.columns(2)
        mcol1.metric("Accuracy", f"{m['Accuracy']*100:.1f}%")
        mcol2.metric("Precision", f"{m['Precision']*100:.1f}%")
        mcol1.metric("Recall", f"{m['Recall']*100:.1f}%")
        mcol2.metric("F1 Score", f"{m['F1']*100:.1f}%")

        fig_cm = px.imshow(bundle["cm"], text_auto=True, color_continuous_scale=["#1D1A16", GOLD],
                            x=["Pred: Down", "Pred: Up"], y=["Actual: Down", "Actual: Up"])
        fig_cm.update_layout(**PLOTLY_LAYOUT, title="Confusion Matrix", height=280)
        st.plotly_chart(fig_cm, width='stretch')

# ---------------- TAB 5: LIVE PREDICTION ----------------
with tab5:
    st.subheader("Predict Next-Month Gold Price")
    st.caption("Defaults are the most recent available data — adjust any input and predict.")

    latest_features = feat_df[FEATURE_COLS].iloc[-1]
    cols = st.columns(3)
    user_input = {}
    for i, feature in enumerate(FEATURE_COLS):
        with cols[i % 3]:
            user_input[feature] = st.number_input(feature, value=float(latest_features[feature]), format="%.4f")

    input_df = pd.DataFrame([user_input])[FEATURE_COLS]

    if st.button("Predict", type="primary"):
        price_pred = bundle["xgb"].predict(input_df)[0]
        direction_pred = bundle["clf"].predict(input_df)[0]
        direction_proba = bundle["clf"].predict_proba(input_df)[0]

        r1, r2 = st.columns(2)
        r1.metric("Predicted Next-Month Price", f"${price_pred:,.2f}")
        r2.metric("Predicted Direction", "📈 Up" if direction_pred == 1 else "📉 Down",
                   f"{max(direction_proba)*100:.1f}% confidence")

        st.warning(
            "⚠️ Research/demo tool, not financial advice. Gold prices react to geopolitical "
            "and macro events no model can fully anticipate."
        )

# ---------------- TAB 6: BACKTEST ----------------
with tab6:
    st.subheader("Simple Trading Strategy Backtest")
    st.caption(
        "Strategy: each month, hold gold ONLY if the direction classifier predicted "
        "'Up' for that month; otherwise hold cash (0% return). Compared against a "
        "plain Buy & Hold approach over the same test period."
    )

    test = bundle["test"]
    clf_pred = bundle["clf_pred"]

    realized_return = test["Target_Price"] / test["Gold"] - 1
    strategy_return = realized_return * clf_pred
    buyhold_return = realized_return

    strategy_equity = (1 + strategy_return).cumprod() * 100
    buyhold_equity = (1 + buyhold_return).cumprod() * 100

    fig_bt = go.Figure()
    fig_bt.add_trace(go.Scatter(x=test.index, y=strategy_equity, name="Direction-Signal Strategy",
                                 line=dict(color=GOLD, width=2)))
    fig_bt.add_trace(go.Scatter(x=test.index, y=buyhold_equity, name="Buy & Hold",
                                 line=dict(color="#7C9DB5", width=2, dash="dot")))
    fig_bt.update_layout(**PLOTLY_LAYOUT, title="Portfolio Value Over Time (Start = 100)", height=420)
    st.plotly_chart(fig_bt, width='stretch')

    def sharpe_ratio(returns):
        if returns.std() == 0:
            return 0.0
        return (returns.mean() / returns.std()) * np.sqrt(12)

    def max_drawdown(equity):
        running_max = equity.cummax()
        drawdown = (equity - running_max) / running_max
        return drawdown.min() * 100

    strat_total_return = (strategy_equity.iloc[-1] / 100 - 1) * 100
    bh_total_return = (buyhold_equity.iloc[-1] / 100 - 1) * 100
    strat_sharpe = sharpe_ratio(strategy_return)
    bh_sharpe = sharpe_ratio(buyhold_return)
    strat_dd = max_drawdown(strategy_equity)
    bh_dd = max_drawdown(buyhold_equity)
    win_rate = (clf_pred == test["Target_Direction"]).mean() * 100

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Strategy Total Return", f"{strat_total_return:+.1f}%", f"vs Buy&Hold {bh_total_return:+.1f}%")
    col2.metric("Strategy Sharpe Ratio", f"{strat_sharpe:.2f}", f"vs Buy&Hold {bh_sharpe:.2f}")
    col3.metric("Strategy Max Drawdown", f"{strat_dd:.1f}%", f"vs Buy&Hold {bh_dd:.1f}%")
    col4.metric("Direction Win Rate", f"{win_rate:.1f}%")

    st.markdown("""
**How to read this:**
- **Sharpe Ratio** measures return per unit of risk (annualized here since data is monthly) —
  higher is better; a strategy beating Buy & Hold on Sharpe is delivering smoother, more
  risk-adjusted gains, not just bigger swings.
- **Max Drawdown** is the worst peak-to-trough decline the strategy would have experienced —
  a smaller (less negative) number means a gentler ride for whoever's capital is on the line.
- **Direction Win Rate** is simply how often the classifier's Up/Down call matched what
  actually happened next month.

**Honest caveat:** this backtest ignores transaction costs, slippage, and taxes, and assumes
the strategy can move fully in or out of gold every month with zero delay — none of which
holds in real trading. Treat this as a research illustration of the model's practical value,
not a validated trading system.
""")

st.markdown("---")
st.caption("Gold Price Analysis & Forecasting · Built with Streamlit, Plotly, scikit-learn, XGBoost & statsmodels")
