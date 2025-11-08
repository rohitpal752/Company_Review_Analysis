import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from io import StringIO
from datetime import datetime

# =============================
# App Config & Styling
# =============================
st.set_page_config(
    page_title="Company Performance & EDA — Pro Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---- Light custom CSS for a bigger, polished UI ----
st.markdown(
    """
    <style>
    .big-metric {font-size: 28px; font-weight: 700;}
    .subtle {color:#6b7280}
    .stTabs [data-baseweb="tab-list"] {gap: 8px}
    .stTabs [data-baseweb="tab"] {height: 54px; padding: 10px 18px; border-radius: 14px; background: #f8fafc}
    .kpi-card {padding:18px; border:1px solid #e5e7eb; border-radius:16px; background:white;}
    .section {padding:12px 4px}
    .metric-title {font-size:13px; color:#64748b;}
    .metric-value {font-size:24px; font-weight:700;}
    .badge {display:inline-block; padding:4px 10px; border-radius:999px; background:#eef2ff; color:#4338ca; font-size:12px;}
    </style>
    """,
    unsafe_allow_html=True,
)

# =============================
# Helpers
# =============================

def _norm_cols(df: pd.DataFrame):
    return {c.lower().strip(): c for c in df.columns}

COMMON_CHANNEL_COLS = [
    "channel", "platform", "source", "medium", "campaign", "campaign name",
    "channel_name", "platform_name", "publisher", "network"
]

COMMON_DATE_COLS = [
    "date", "day", "dt", "timestamp", "report_date", "calendar_date"
]

COMMON_SPEND_COLS = ["spend", "cost", "amount_spent"]
COMMON_CLICKS_COLS = ["clicks", "click"]
COMMON_IMPR_COLS = ["impressions", "impr"]
COMMON_CONV_COLS = ["conversions", "purchases", "orders", "leads"]


def guess_col(df: pd.DataFrame, candidates):
    norm = _norm_cols(df)
    for cand in candidates:
        if cand in norm:
            return norm[cand]
    # fuzzy: try startswith/contains
    for c in df.columns:
        cl = c.lower().strip()
        if any(cl == x or cl.replace(" ", "") == x.replace(" ", "") for x in candidates):
            return c
        if any(cl.startswith(x) or x in cl for x in candidates):
            return c
    return None


def kpi_card(title: str, value, help_text: str = ""):
    col = st.container()
    with col:
        st.markdown(f"<div class='kpi-card'><div class='metric-title'>{title}</div><div class='metric-value'>{value}</div><div class='subtle'>{help_text}</div></div>", unsafe_allow_html=True)


def percent(n):
    try:
        return f"{n:.2f}%"
    except Exception:
        return "-"


def currency(n):
    try:
        return f"₹{n:,.0f}"
    except Exception:
        try:
            return f"{n:,.0f}"
        except Exception:
            return "-"


# =============================
# Sidebar — Data Upload & Global Filters
# =============================
with st.sidebar:
    st.title("⚙️ Controls")
    st.caption("Upload both datasets. CSV recommended.")
    eda_file = st.file_uploader("Upload EDA dataset (.csv)", type=["csv"])
    comp_file = st.file_uploader("Upload Company dataset (.csv)", type=["csv"])

    st.divider()
    st.caption("Optional: choose theme scale")
    ui_scale = st.slider("UI scale", 90, 130, 100, help="Zoom the UI without browser zoom")
    st.markdown(f"<style>html {{ zoom: {ui_scale}% }}</style>", unsafe_allow_html=True)

if not (eda_file and comp_file):
    st.info("⬅️ Please upload both CSVs to start.")
    st.stop()

eda_df = pd.read_csv(eda_file)
company_df = pd.read_csv(comp_file)

st.success("✅ Datasets loaded successfully.")

# =============================
# Intelligent Column Detection & User Overrides
# =============================
channel_col_guess = guess_col(company_df, COMMON_CHANNEL_COLS) or company_df.columns[0]
date_col_guess = guess_col(company_df, COMMON_DATE_COLS)
spend_col = guess_col(company_df, COMMON_SPEND_COLS)
clicks_col = guess_col(company_df, COMMON_CLICKS_COLS)
impr_col = guess_col(company_df, COMMON_IMPR_COLS)
conv_col = guess_col(company_df, COMMON_CONV_COLS)

with st.sidebar:
    st.subheader("🔧 Column Mapping (Company Data)")
    channel_col = st.selectbox("Channel / Platform column", options=company_df.columns, index=list(company_df.columns).index(channel_col_guess) if channel_col_guess in company_df.columns else 0)
    date_col = st.selectbox("Date column (optional)", options=["<none>"] + list(company_df.columns), index=( ["<none>"] + list(company_df.columns) ).index(date_col_guess) if date_col_guess else 0)
    spend_col = st.selectbox("Spend column", options=["<none>"] + list(company_df.columns), index=( ["<none>"] + list(company_df.columns) ).index(spend_col) if spend_col else 0)
    clicks_col = st.selectbox("Clicks column", options=["<none>"] + list(company_df.columns), index=( ["<none>"] + list(company_df.columns) ).index(clicks_col) if clicks_col else 0)
    impr_col = st.selectbox("Impressions column", options=["<none>"] + list(company_df.columns), index=( ["<none>"] + list(company_df.columns) ).index(impr_col) if impr_col else 0)
    conv_col = st.selectbox("Conversions column", options=["<none>"] + list(company_df.columns), index=( ["<none>"] + list(company_df.columns) ).index(conv_col) if conv_col else 0)

# Derived columns safely
has_spend = spend_col != "<none>" and spend_col in company_df.columns
has_clicks = clicks_col != "<none>" and clicks_col in company_df.columns
has_impr = impr_col != "<none>" and impr_col in company_df.columns
has_conv = conv_col != "<none>" and conv_col in company_df.columns

company_work = company_df.copy()

# Convert date
if date_col and date_col != "<none>" and date_col in company_work.columns:
    with st.sidebar:
        st.caption("Date Filter (if date column selected)")
    try:
        company_work[date_col] = pd.to_datetime(company_work[date_col], errors='coerce')
        min_d, max_d = company_work[date_col].min(), company_work[date_col].max()
        if pd.notna(min_d) and pd.notna(max_d):
            date_range = st.sidebar.date_input(
                "Select date range", value=(min_d.date(), max_d.date()),
                min_value=min_d.date(), max_value=max_d.date()
            )
            if isinstance(date_range, tuple) and len(date_range) == 2:
                start_d, end_d = date_range
                mask = (company_work[date_col].dt.date >= start_d) & (company_work[date_col].dt.date <= end_d)
                company_work = company_work.loc[mask].copy()
    except Exception:
        pass

# Channel filter
unique_channels = sorted(company_work[channel_col].dropna().astype(str).unique()) if channel_col in company_work.columns else []
selected_channels = st.sidebar.multiselect("Filter channels", options=unique_channels, default=unique_channels[: min(6, len(unique_channels))] if unique_channels else [])
if selected_channels:
    company_work = company_work[company_work[channel_col].astype(str).isin(selected_channels)]

# =============================
# Tabs
# =============================
main_tabs = st.tabs(["🏠 Overview", "🧪 EDA", "📈 Marketing Analytics", "🧠 Insights & Recommendations"]) 

# =============================
# Tab 1 — Overview
# =============================
with main_tabs[0]:
    st.markdown("### Overview")
    c1, c2 = st.columns([1, 1])

    # KPI grid for EDA
    with c1:
        st.markdown("#### EDA Dataset Snapshot")
        k1, k2, k3, k4 = st.columns(4)
        k1.markdown("<div class='kpi-card'><div class='metric-title'>Rows</div><div class='metric-value'>{}</div></div>".format(eda_df.shape[0]), unsafe_allow_html=True)
        k2.markdown("<div class='kpi-card'><div class='metric-title'>Columns</div><div class='metric-value'>{}</div></div>".format(eda_df.shape[1]), unsafe_allow_html=True)
        num_cols_eda = eda_df.select_dtypes(include=np.number).shape[1]
        non_null_ratio = (1 - eda_df.isna().mean().mean()) * 100
        k3.markdown("<div class='kpi-card'><div class='metric-title'>Numeric Columns</div><div class='metric-value'>{}</div></div>".format(num_cols_eda), unsafe_allow_html=True)
        k4.markdown("<div class='kpi-card'><div class='metric-title'>Data Completeness</div><div class='metric-value'>{:.1f}%</div></div>".format(non_null_ratio), unsafe_allow_html=True)
        st.dataframe(eda_df.head(20))

    # KPI grid for Company
    with c2:
        st.markdown("#### Company Dataset Snapshot")
        k1, k2, k3, k4 = st.columns(4)
        k1.markdown("<div class='kpi-card'><div class='metric-title'>Rows</div><div class='metric-value'>{}</div></div>".format(company_work.shape[0]), unsafe_allow_html=True)
        k2.markdown("<div class='kpi-card'><div class='metric-title'>Columns</div><div class='metric-value'>{}</div></div>".format(company_work.shape[1]), unsafe_allow_html=True)
        nunique_channels = company_work[channel_col].nunique() if channel_col in company_work.columns else 0
        k3.markdown("<div class='kpi-card'><div class='metric-title'>Channels</div><div class='metric-value'>{}</div></div>".format(nunique_channels), unsafe_allow_html=True)
        metric_ready = has_spend or has_clicks or has_impr or has_conv
        k4.markdown("<div class='kpi-card'><div class='metric-title'>Perf. Fields Present</div><div class='metric-value'>{}</div></div>".format("Yes" if metric_ready else "No"), unsafe_allow_html=True)
        st.dataframe(company_work.head(20))

# =============================
# Tab 2 — EDA
# =============================
with main_tabs[1]:
    st.markdown("### Exploratory Data Analysis")

    eda_left, eda_right = st.columns([1, 2])

    with eda_left:
        st.subheader("Column Summary")
        target_col = st.selectbox("Select a column", eda_df.columns)
        st.write(eda_df[target_col].describe(include='all'))
        st.write("Missing values:", int(eda_df[target_col].isna().sum()))
        st.write("Unique values:", eda_df[target_col].nunique())

    with eda_right:
        st.subheader("Distribution")
        if pd.api.types.is_numeric_dtype(eda_df[target_col]):
            fig = px.histogram(eda_df, x=target_col, marginal="box", nbins=40, title=f"Distribution of {target_col}")
            st.plotly_chart(fig, use_container_width=True)
        else:
            vc = eda_df[target_col].astype(str).value_counts().head(30).reset_index()
            vc.columns = [target_col, "count"]
            fig = px.bar(vc, x=target_col, y="count", title=f"Top {min(30, len(vc))} categories — {target_col}")
            st.plotly_chart(fig, use_container_width=True)

    st.divider()

    num_df = eda_df.select_dtypes(include=np.number)
    if num_df.shape[1] >= 2:
        st.subheader("Correlation Heatmap")
        corr = num_df.corr(numeric_only=True)
        fig = px.imshow(corr, text_auto=False, aspect="auto", title="Feature Correlations")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Not enough numeric columns for a correlation heatmap.")

# =============================
# Tab 3 — Marketing Analytics
# =============================
with main_tabs[2]:
    st.markdown("### Marketing Performance by Channel")

    # Safely compute derived metrics if possible
    high_ctr = None
    high_cvr = None

    perf_ready = has_clicks and has_impr
    conv_ready = has_conv and has_clicks

    if perf_ready:
        try:
            if has_spend and has_clicks:
                company_work["CPC"] = (company_work[spend_col] / company_work[clicks_col]).replace([np.inf, -np.inf], np.nan)
            if has_spend and has_conv:
                company_work["CPA"] = (company_work[spend_col] / company_work[conv_col]).replace([np.inf, -np.inf], np.nan)
            company_work["CTR"] = (company_work[clicks_col] / company_work[impr_col] * 100).replace([np.inf, -np.inf], np.nan)
            if conv_ready:
                company_work["CVR"] = (company_work[conv_col] / company_work[clicks_col] * 100).replace([np.inf, -np.inf], np.nan)
        except Exception as e:
            st.warning(f"Could not compute some metrics automatically: {e}")

    # KPI row
    k1, k2, k3, k4 = st.columns(4)
    if has_spend:
        k1.markdown(f"<div class='kpi-card'><div class='metric-title'>Total Spend</div><div class='metric-value'>{currency(company_work[spend_col].sum())}</div></div>", unsafe_allow_html=True)
    else:
        k1.markdown("<div class='kpi-card'><div class='metric-title'>Total Spend</div><div class='metric-value'>-</div></div>", unsafe_allow_html=True)

    if has_clicks:
        k2.markdown(f"<div class='kpi-card'><div class='metric-title'>Total Clicks</div><div class='metric-value'>{int(company_work[clicks_col].sum())}</div></div>", unsafe_allow_html=True)
    else:
        k2.markdown("<div class='kpi-card'><div class='metric-title'>Total Clicks</div><div class='metric-value'>-</div></div>", unsafe_allow_html=True)

    if has_impr:
        k3.markdown(f"<div class='kpi-card'><div class='metric-title'>Total Impressions</div><div class='metric-value'>{int(company_work[impr_col].sum())}</div></div>", unsafe_allow_html=True)
    else:
        k3.markdown("<div class='kpi-card'><div class='metric-title'>Total Impressions</div><div class='metric-value'>-</div></div>", unsafe_allow_html=True)

    if has_conv:
        k4.markdown(f"<div class='kpi-card'><div class='metric-title'>Total Conversions</div><div class='metric-value'>{int(company_work[conv_col].sum())}</div></div>", unsafe_allow_html=True)
    else:
        k4.markdown("<div class='kpi-card'><div class='metric-title'>Total Conversions</div><div class='metric-value'>-</div></div>", unsafe_allow_html=True)

    st.divider()

    # Group by channel and visualize
    if channel_col in company_work.columns:
        group_cols = [channel_col]
        agg_dict = {}
        for col, func in [
            (spend_col, 'sum'), (clicks_col, 'sum'), (impr_col, 'sum'), (conv_col, 'sum'),
            ("CPC", 'mean'), ("CPA", 'mean'), ("CTR", 'mean'), ("CVR", 'mean')
        ]:
            if col and col in company_work.columns:
                agg_dict[col] = func
        chan = company_work.groupby(group_cols).agg(agg_dict).reset_index()

        st.subheader("Channel Summary")
        st.dataframe(chan)

        # Top channels by conversions or clicks
        metric_pick = st.selectbox("Bar metric", options=[c for c in [conv_col, clicks_col, spend_col, impr_col, "CTR", "CVR", "CPC", "CPA"] if c and ((isinstance(c, str) and (c in chan.columns)) or c in ["CTR","CVR","CPC","CPA"])])
        fig_bar = px.bar(chan.sort_values(metric_pick, ascending=True), x=metric_pick, y=channel_col, orientation='h', title=f"{metric_pick} by {channel_col}")
        st.plotly_chart(fig_bar, use_container_width=True)

        # CTR vs CVR bubble
        if "CTR" in company_work.columns and "CVR" in company_work.columns and has_spend:
            bubble = company_work.groupby(channel_col).agg({"CTR":"mean","CVR":"mean", spend_col:"sum"}).reset_index()
            fig_sc = px.scatter(bubble, x="CTR", y="CVR", size=spend_col, color=channel_col, hover_name=channel_col,
                                title="Engagement vs Conversion Efficiency (CTR vs CVR)")
            st.plotly_chart(fig_sc, use_container_width=True)

        # Trend over time
        if date_col and date_col != "<none>" and date_col in company_work.columns:
            st.subheader("Trend Over Time")
            trend_metric = st.selectbox("Trend metric", options=[m for m in [spend_col, clicks_col, impr_col, conv_col] if m and m in company_work.columns])
            if trend_metric:
                daily = company_work.groupby([date_col, channel_col])[trend_metric].sum().reset_index()
                fig_line = px.line(daily, x=date_col, y=trend_metric, color=channel_col, markers=True, title=f"{trend_metric} trend by {channel_col}")
                st.plotly_chart(fig_line, use_container_width=True)

        # Compute best channels for insights
        if "CTR" in company_work.columns:
            try:
                high_ctr = company_work.loc[company_work["CTR"].idxmax(), channel_col]
            except Exception:
                high_ctr = None
        if "CVR" in company_work.columns:
            try:
                high_cvr = company_work.loc[company_work["CVR"].idxmax(), channel_col]
            except Exception:
                high_cvr = None

    else:
        st.warning("Channel column not found; please map it correctly in the sidebar.")

# =============================
# Tab 4 — Insights & Recommendations
# =============================
with main_tabs[3]:
    st.markdown("### Auto Insights & Recommendations")

    bullets = []

    # Spend-driven insights
    if has_spend:
        total_spend = company_work[spend_col].sum()
        bullets.append(f"Total spend: {currency(total_spend)}")
        if channel_col in company_work.columns:
            spend_by_ch = company_work.groupby(channel_col)[spend_col].sum().sort_values(ascending=False)
            if not spend_by_ch.empty:
                bullets.append(f"Top spend channel: **{spend_by_ch.index[0]}** ({currency(spend_by_ch.iloc[0])}).")

    # CTR/CVR highlights
    if high_ctr is not None:
        bullets.append(f"Highest engagement (CTR): **{high_ctr}**.")
    if high_cvr is not None:
        bullets.append(f"Best conversion efficiency (CVR): **{high_cvr}**.")

    # Efficiency metrics
    if "CPC" in company_work.columns:
        bullets.append(f"Average CPC: **{company_work['CPC'].mean():.2f}**")
    if "CPA" in company_work.columns:
        bullets.append(f"Average CPA: **{company_work['CPA'].mean():.2f}**")

    # Data quality
    eda_missing = eda_df.isna().mean().mean()
    if eda_missing > 0:
        bullets.append(f"EDA dataset missingness: **{eda_missing*100:.1f}%**. Consider data cleaning for robust modeling.")

    if not bullets:
        st.info("No automated insights available yet — ensure columns are mapped and metrics computed in the previous tab.")
    else:
        for b in bullets:
            st.markdown(f"- {b}")

    st.divider()

    # Download insights text
    insights_text = "### Key Insights Summary\n\n" + "\n".join([f"- {b}" for b in bullets])
    st.download_button("📥 Download Insights (TXT)", insights_text.encode(), file_name="insights_summary.txt")

    with st.expander("Show compact data dictionaries"):
        st.write("**EDA Columns**")
        st.dataframe(pd.DataFrame({"column": eda_df.columns, "dtype": eda_df.dtypes.astype(str)}))
        st.write("**Company Columns**")
        st.dataframe(pd.DataFrame({"column": company_df.columns, "dtype": company_df.dtypes.astype(str)}))

st.caption("Built with ❤️ Streamlit + Plotly + Pandas. Big UI mode enabled.")
