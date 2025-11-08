import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime

# =====================================================
# App Config & Big UI Styling
# =====================================================
st.set_page_config(
    page_title="All-in-One EDA + Marketing Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      .kpi {padding:18px;border:1px solid #e5e7eb;border-radius:16px;background:#fff}
      .kpi-title{font-size:12px;color:#64748b;margin-bottom:4px}
      .kpi-value{font-size:22px;font-weight:700}
      .stTabs [data-baseweb="tab-list"]{gap:8px}
      .stTabs [data-baseweb="tab"]{height:50px;padding:10px 16px;border-radius:14px;background:#f8fafc}
      .small{color:#6b7280;font-size:12px}
    </style>
    """,
    unsafe_allow_html=True,
)

# =====================================================
# Helpers
# =====================================================

def norm_map(df: pd.DataFrame):
    return {c.lower().strip(): c for c in df.columns}

COMMON_CHANNEL = ["channel","platform","source","medium","campaign","campaign name","publisher","network"]
COMMON_DATE    = ["date","day","dt","timestamp","report_date","calendar_date"]
COMMON_SPEND   = ["spend","cost","amount_spent"]
COMMON_CLICKS  = ["clicks","click"]
COMMON_IMPR    = ["impressions","impr"]
COMMON_CONV    = ["conversions","purchases","orders","leads"]


def guess_col(df: pd.DataFrame, candidates):
    nmap = norm_map(df)
    for c in candidates:
        if c in nmap: return nmap[c]
    # soft match
    for col in df.columns:
        cl = col.lower().strip()
        if any(cl == x or cl.replace(" ","") == x.replace(" ","") for x in candidates):
            return col
        if any(cl.startswith(x) or x in cl for x in candidates):
            return col
    return None


def currency(x):
    try:
        return f"₹{x:,.0f}"
    except Exception:
        try:
            return f"{x:,.0f}"
        except Exception:
            return "-"


# =====================================================
# Sidebar — SINGLE upload + global controls
# =====================================================
with st.sidebar:
    st.title("⚙️ Controls")
    data_file = st.file_uploader("Upload a single CSV", type=["csv"])
    ui_scale = st.slider("UI scale", 90, 130, 100, help="Zoom the UI without browser zoom")
    st.markdown(f"<style>html {{ zoom: {ui_scale}% }}</style>", unsafe_allow_html=True)

if not data_file:
    st.info("⬅️ Please upload one CSV. The app will run EDA and, if columns exist, Marketing Analytics too.")
    st.stop()

# Load once
DF = pd.read_csv(data_file)
DF_raw = DF.copy()

st.success("✅ Dataset loaded.")

# =====================================================
# Auto-detect marketing columns (with safe fallbacks)
# =====================================================
channel_col = guess_col(DF, COMMON_CHANNEL) or (DF.columns[0] if len(DF.columns) else None)
date_col    = guess_col(DF, COMMON_DATE)
spend_col   = guess_col(DF, COMMON_SPEND)
clicks_col  = guess_col(DF, COMMON_CLICKS)
impr_col    = guess_col(DF, COMMON_IMPR)
conv_col    = guess_col(DF, COMMON_CONV)

has_spend = bool(spend_col in DF.columns) if spend_col else False
has_clicks= bool(clicks_col in DF.columns) if clicks_col else False
has_impr  = bool(impr_col in DF.columns) if impr_col else False
has_conv  = bool(conv_col in DF.columns) if conv_col else False

# Optional manual mapping in an expander
with st.sidebar:
    with st.expander("🔧 Optional: Remap columns (if detection looks wrong)"):
        channel_col = st.selectbox("Channel / Platform", options=["<none>"]+list(DF.columns), index=( ["<none>"]+list(DF.columns) ).index(channel_col) if channel_col in DF.columns else 0)
        date_col    = st.selectbox("Date", options=["<none>"]+list(DF.columns), index=( ["<none>"]+list(DF.columns) ).index(date_col) if date_col in DF.columns else 0)
        spend_col   = st.selectbox("Spend", options=["<none>"]+list(DF.columns), index=( ["<none>"]+list(DF.columns) ).index(spend_col) if spend_col in DF.columns else 0)
        clicks_col  = st.selectbox("Clicks", options=["<none>"]+list(DF.columns), index=( ["<none>"]+list(DF.columns) ).index(clicks_col) if clicks_col in DF.columns else 0)
        impr_col    = st.selectbox("Impressions", options=["<none>"]+list(DF.columns), index=( ["<none>"]+list(DF.columns) ).index(impr_col) if impr_col in DF.columns else 0)
        conv_col    = st.selectbox("Conversions", options=["<none>"]+list(DF.columns), index=( ["<none>"]+list(DF.columns) ).index(conv_col) if conv_col in DF.columns else 0)

# Normalize "<none>" → None for logic
for name in ["channel_col","date_col","spend_col","clicks_col","impr_col","conv_col"]:
    if isinstance(locals()[name], str) and locals()[name] == "<none>":
        locals()[name] = None

# Filter table copy
work = DF.copy()

# Date filter (if present)
if date_col and date_col in work.columns:
    try:
        work[date_col] = pd.to_datetime(work[date_col], errors="coerce")
        dmin, dmax = work[date_col].min(), work[date_col].max()
        if pd.notna(dmin) and pd.notna(dmax):
            picked = st.sidebar.date_input(
                "Date range", value=(dmin.date(), dmax.date()), min_value=dmin.date(), max_value=dmax.date()
            )
            if isinstance(picked, tuple) and len(picked) == 2:
                mask = (work[date_col].dt.date >= picked[0]) & (work[date_col].dt.date <= picked[1])
                work = work.loc[mask].copy()
    except Exception:
        pass

# Channel filter (if present)
if channel_col and channel_col in work.columns:
    channels = sorted(work[channel_col].dropna().astype(str).unique())
    sel_channels = st.sidebar.multiselect("Filter channels", channels, default=channels[: min(6, len(channels))])
    if sel_channels:
        work = work[work[channel_col].astype(str).isin(sel_channels)]

# =====================================================
# Tabs
# =====================================================
TAB_OVERVIEW, TAB_EDA, TAB_MKT, TAB_INSIGHTS = st.tabs([
    "🏠 Overview", "🧪 EDA", "📈 Marketing", "🧠 Insights"
])

# =====================================================
# Overview
# =====================================================
with TAB_OVERVIEW:
    c1, c2 = st.columns(2)

    with c1:
        st.markdown("#### Dataset Snapshot")
        r1, r2, r3, r4 = st.columns(4)
        r1.markdown(f"<div class='kpi'><div class='kpi-title'>Rows</div><div class='kpi-value'>{work.shape[0]}</div></div>", unsafe_allow_html=True)
        r2.markdown(f"<div class='kpi'><div class='kpi-title'>Columns</div><div class='kpi-value'>{work.shape[1]}</div></div>", unsafe_allow_html=True)
        num_cols = work.select_dtypes(include=np.number).shape[1]
        r3.markdown(f"<div class='kpi'><div class='kpi-title'>Numeric Cols</div><div class='kpi-value'>{num_cols}</div></div>", unsafe_allow_html=True)
        comp = (1 - work.isna().mean().mean()) * 100
        r4.markdown(f"<div class='kpi'><div class='kpi-title'>Completeness</div><div class='kpi-value'>{comp:.1f}%</div></div>", unsafe_allow_html=True)
        st.dataframe(work.head(20))

    with c2:
        st.markdown("#### Quick Glance (Auto)")
        # If marketing fields exist, show mini KPIs
        s1, s2, s3, s4 = st.columns(4)
        if has_spend and spend_col in work.columns:
            s1.markdown(f"<div class='kpi'><div class='kpi-title'>Total Spend</div><div class='kpi-value'>{currency(work[spend_col].sum())}</div></div>", unsafe_allow_html=True)
        else:
            s1.markdown("<div class='kpi'><div class='kpi-title'>Total Spend</div><div class='kpi-value'>-</div></div>", unsafe_allow_html=True)
        if has_clicks and clicks_col in work.columns:
            s2.markdown(f"<div class='kpi'><div class='kpi-title'>Total Clicks</div><div class='kpi-value'>{int(work[clicks_col].sum())}</div></div>", unsafe_allow_html=True)
        else:
            s2.markdown("<div class='kpi'><div class='kpi-title'>Total Clicks</div><div class='kpi-value'>-</div></div>", unsafe_allow_html=True)
        if has_impr and impr_col in work.columns:
            s3.markdown(f"<div class='kpi'><div class='kpi-title'>Total Impr.</div><div class='kpi-value'>{int(work[impr_col].sum())}</div></div>", unsafe_allow_html=True)
        else:
            s3.markdown("<div class='kpi'><div class='kpi-title'>Total Impr.</div><div class='kpi-value'>-</div></div>", unsafe_allow_html=True)
        if has_conv and conv_col in work.columns:
            s4.markdown(f"<div class='kpi'><div class='kpi-title'>Total Conv.</div><div class='kpi-value'>{int(work[conv_col].sum())}</div></div>", unsafe_allow_html=True)
        else:
            s4.markdown("<div class='kpi'><div class='kpi-title'>Total Conv.</div><div class='kpi-value'>-</div></div>", unsafe_allow_html=True)

# =====================================================
# EDA
# =====================================================
with TAB_EDA:
    st.markdown("### Exploratory Data Analysis")

    left, right = st.columns([1, 2])
    with left:
        col_pick = st.selectbox("Select a column", work.columns)
        st.write("**Summary**")
        st.write(work[col_pick].describe(include='all'))
        st.write("Missing:", int(work[col_pick].isna().sum()))
        st.write("Unique:", work[col_pick].nunique())

    with right:
        if pd.api.types.is_numeric_dtype(work[col_pick]):
            fig = px.histogram(work, x=col_pick, nbins=40, marginal="box", title=f"Distribution — {col_pick}")
            st.plotly_chart(fig, use_container_width=True)
        else:
            vc = work[col_pick].astype(str).value_counts().head(30).reset_index()
            vc.columns = [col_pick, "count"]
            fig = px.bar(vc, x=col_pick, y="count", title=f"Top {min(30,len(vc))} categories — {col_pick}")
            st.plotly_chart(fig, use_container_width=True)

    # Correlation
    num = work.select_dtypes(include=np.number)
    if num.shape[1] >= 2:
        st.markdown("#### Correlation Heatmap")
        corr = num.corr(numeric_only=True)
        fig = px.imshow(corr, text_auto=False, aspect="auto", title="Feature Correlations")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Not enough numeric columns for a heatmap.")

# =====================================================
# Marketing (auto-enables when metrics exist)
# =====================================================
with TAB_MKT:
    st.markdown("### Marketing Performance")

    # Compute derived metrics when possible
    perf_ready = has_clicks and has_impr and (clicks_col in work.columns) and (impr_col in work.columns)
    conv_ready = has_conv and has_clicks and (conv_col in work.columns) and (clicks_col in work.columns)

    if perf_ready:
        try:
            if has_spend and spend_col in work.columns and clicks_col in work.columns:
                work["CPC"] = (work[spend_col] / work[clicks_col]).replace([np.inf,-np.inf], np.nan)
            if has_spend and spend_col in work.columns and conv_ready:
                work["CPA"] = (work[spend_col] / work[conv_col]).replace([np.inf,-np.inf], np.nan)
            work["CTR"] = (work[clicks_col] / work[impr_col] * 100).replace([np.inf,-np.inf], np.nan)
            if conv_ready:
                work["CVR"] = (work[conv_col] / work[clicks_col] * 100).replace([np.inf,-np.inf], np.nan)
        except Exception as e:
            st.warning(f"Metric computation warning: {e}")

    # KPI row
    k1,k2,k3,k4 = st.columns(4)
    k1.markdown(f"<div class='kpi'><div class='kpi-title'>Total Spend</div><div class='kpi-value'>{currency(work[spend_col].sum()) if has_spend and spend_col in work.columns else '-'}</div></div>", unsafe_allow_html=True)
    k2.markdown(f"<div class='kpi'><div class='kpi-title'>Total Clicks</div><div class='kpi-value'>{int(work[clicks_col].sum()) if has_clicks and clicks_col in work.columns else '-'}</div></div>", unsafe_allow_html=True)
    k3.markdown(f"<div class='kpi'><div class='kpi-title'>Total Impr.</div><div class='kpi-value'>{int(work[impr_col].sum()) if has_impr and impr_col in work.columns else '-'}</div></div>", unsafe_allow_html=True)
    k4.markdown(f"<div class='kpi'><div class='kpi-title'>Total Conv.</div><div class='kpi-value'>{int(work[conv_col].sum()) if has_conv and conv_col in work.columns else '-'}</div></div>", unsafe_allow_html=True)

    st.divider()

    # Channel summary & visuals only if channel exists
    high_ctr = None
    high_cvr = None

    if channel_col and channel_col in work.columns:
        agg_dict = {}
        for col, func in [
            (spend_col,'sum'), (clicks_col,'sum'), (impr_col,'sum'), (conv_col,'sum'),
            ("CPC",'mean'), ("CPA",'mean'), ("CTR",'mean'), ("CVR",'mean')
        ]:
            if isinstance(col, str) and col in work.columns:
                agg_dict[col] = func
        if agg_dict:
            chan = work.groupby([channel_col]).agg(agg_dict).reset_index()
            st.subheader("Channel Summary")
            st.dataframe(chan)

            metric_opts = [m for m in [conv_col, clicks_col, spend_col, impr_col, "CTR","CVR","CPC","CPA"] if (isinstance(m,str) and m in chan.columns) or m in ["CTR","CVR","CPC","CPA"] and m in chan.columns]
            if metric_opts:
                pick = st.selectbox("Bar metric", options=metric_opts)
                figb = px.bar(chan.sort_values(pick, ascending=True), x=pick, y=channel_col, orientation='h', title=f"{pick} by {channel_col}")
                st.plotly_chart(figb, use_container_width=True)

            if set(["CTR","CVR"]).issubset(work.columns) and has_spend and spend_col in work.columns:
                bub = work.groupby(channel_col).agg({"CTR":"mean","CVR":"mean", spend_col:"sum"}).reset_index()
                figc = px.scatter(bub, x="CTR", y="CVR", size=spend_col, color=channel_col, hover_name=channel_col,
                                  title="CTR vs CVR — Engagement vs Conversion Efficiency")
                st.plotly_chart(figc, use_container_width=True)

            # Best channels
            if "CTR" in work.columns and not work["CTR"].dropna().empty:
                try:
                    high_ctr = work.loc[work["CTR"].idxmax(), channel_col]
                except Exception:
                    high_ctr = None
            if "CVR" in work.columns and not work["CVR"].dropna().empty:
                try:
                    high_cvr = work.loc[work["CVR"].idxmax(), channel_col]
                except Exception:
                    high_cvr = None
        else:
            st.warning("No valid numeric metrics to aggregate. Map columns from the sidebar expander if needed.")
    else:
        st.info("No channel/platform column detected. Use the sidebar expander to map one if available.")

    # Trend over time if date present
    if date_col and date_col in work.columns:
        st.subheader("Trend Over Time")
        metric_for_trend = st.selectbox("Metric", options=[m for m in [spend_col,clicks_col,impr_col,conv_col] if isinstance(m,str) and m in work.columns])
        if metric_for_trend:
            daily = work.groupby([date_col, channel_col] if channel_col and channel_col in work.columns else [date_col])[metric_for_trend].sum().reset_index()
            figt = px.line(daily, x=date_col, y=metric_for_trend, color=channel_col if channel_col in daily.columns else None, markers=True,
                           title=f"{metric_for_trend} trend" + (f" by {channel_col}" if channel_col in daily.columns else ""))
            st.plotly_chart(figt, use_container_width=True)

# =====================================================
# Insights (bullet points + download)
# =====================================================
with TAB_INSIGHTS:
    st.markdown("### Auto Insights")

    bullets = []
    # Data health
    completeness = (1 - work.isna().mean().mean()) * 100
    bullets.append(f"Data completeness: **{completeness:.1f}%**")

    # Spend/efficiency
    if has_spend and spend_col in work.columns:
        bullets.append(f"Total spend: **{currency(work[spend_col].sum())}**")
    if "CPC" in work.columns and not work["CPC"].dropna().empty:
        bullets.append(f"Average CPC: **{work['CPC'].mean():.2f}**")
    if "CPA" in work.columns and not work["CPA"].dropna().empty:
        bullets.append(f"Average CPA: **{work['CPA'].mean():.2f}**")

    # Best channels
    if channel_col and channel_col in work.columns:
        if "CTR" in work.columns and not work["CTR"].dropna().empty:
            top_ctr_ch = work.loc[work["CTR"].idxmax(), channel_col]
            bullets.append(f"Highest engagement (CTR): **{top_ctr_ch}**")
        if "CVR" in work.columns and not work["CVR"].dropna().empty:
            top_cvr_ch = work.loc[work["CVR"].idxmax(), channel_col]
            bullets.append(f"Best conversion efficiency (CVR): **{top_cvr_ch}**")

    # Surface potential data issues
    if has_clicks and has_impr and clicks_col in work.columns and impr_col in work.columns:
        zero_impr = (work[impr_col] == 0).sum()
        if zero_impr > 0:
            bullets.append(f"Found **{zero_impr}** rows with zero impressions — check tracking or filters.")

    for b in bullets:
        st.markdown(f"- {b}")

    st.divider()
txt = "### Key Insights\n\n" + "\n".join(["- point 1", "- point 2"])
print(txt)

st.caption("Built with ❤️ Streamlit + Plotly + Pandas — Single Upload Edition.")
