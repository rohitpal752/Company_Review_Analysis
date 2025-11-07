import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import io

st.set_page_config(page_title="Company & EDA Analytics", layout="wide")

st.title("📊 Company Performance & EDA Dashboard")
st.markdown("This dashboard combines exploratory data insights with company-level analytics for better decision-making.")

# ===========================
# Data Upload Section
# ===========================
st.sidebar.header("📂 Upload Your Datasets")
eda_file = st.sidebar.file_uploader("Upload EDA Dataset (.csv)", type=['csv'])
company_file = st.sidebar.file_uploader("Upload Company Project Dataset (.csv)", type=['csv'])

if eda_file and company_file:
    eda_df = pd.read_csv(eda_file)
    company_df = pd.read_csv(company_file)
    st.success("✅ Both datasets loaded successfully!")
else:
    st.warning("Please upload both datasets to continue.")
    st.stop()

# ===========================
# Overview KPIs
# ===========================
st.header("🔹 Overview Metrics")

col1, col2, col3, col4 = st.columns(4)
col1.metric("EDA Rows", eda_df.shape[0])
col2.metric("EDA Columns", eda_df.shape[1])
col3.metric("Company Rows", company_df.shape[0])
col4.metric("Company Columns", company_df.shape[1])

st.divider()

# ===========================
# Exploratory Data Analysis
# ===========================
st.header("🔍 Exploratory Data Analysis")

selected_col = st.selectbox("Select a column for EDA", eda_df.columns)

col1, col2 = st.columns(2)
with col1:
    st.write("**Basic Statistics**")
    st.write(eda_df[selected_col].describe())

with col2:
    st.write("**Missing Values**")
    st.write(eda_df[selected_col].isna().sum())

# Plot distribution
fig = px.histogram(eda_df, x=selected_col, title=f"Distribution of {selected_col}", marginal="box")
st.plotly_chart(fig, use_container_width=True)

# Correlation Heatmap
st.subheader("Correlation Heatmap (EDA Dataset)")
corr = eda_df.select_dtypes(include=[np.number]).corr()
fig = px.imshow(corr, text_auto=True, aspect="auto", title="Feature Correlations")
st.plotly_chart(fig, use_container_width=True)

# ===========================
# Company Project Analysis
# ===========================
st.header("🏢 Company Project Performance")

metric_options = ["Spend", "Clicks", "Impressions", "Conversions"]
chosen_metric = st.selectbox("Select Metric to Analyze", metric_options)

if chosen_metric.lower() in company_df.columns.str.lower():
    col = [c for c in company_df.columns if c.lower() == chosen_metric.lower()][0]
    fig = px.bar(company_df, x=company_df.columns[0], y=col, color=col,
                 title=f"{chosen_metric} by {company_df.columns[0]}")
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("No matching metric column found, please check column names.")

# ===========================
# Derived Metrics & Insights
# ===========================
st.header("📈 Derived Metrics and Insights")

# Try deriving CPC, CPA, CTR, CVR if relevant columns exist
try:
    company_df["CPC"] = company_df["Spend"] / company_df["Clicks"]
    company_df["CPA"] = company_df["Spend"] / company_df["Conversions"]
    company_df["CTR"] = (company_df["Clicks"] / company_df["Impressions"]) * 100
    company_df["CVR"] = (company_df["Conversions"] / company_df["Clicks"]) * 100

    st.write("### Key Channel Metrics")
    st.dataframe(company_df[["CPC", "CPA", "CTR", "CVR"]].describe().T)

    fig = px.scatter(company_df, x="CTR", y="CVR", color=company_df.columns[0],
                     title="CTR vs CVR by Channel/Platform", size="Spend")
    st.plotly_chart(fig, use_container_width=True)

    # Insights
    high_ctr = company_df.loc[company_df["CTR"].idxmax(), company_df.columns[0]]
    high_cvr = company_df.loc[company_df["CVR"].idxmax(), company_df.columns[0]]

    st.success(f"💡 **Insight:** {high_ctr} shows the highest CTR, indicating strong ad engagement.")
    st.success(f"💡 **Insight:** {high_cvr} achieves the best conversion efficiency (CVR).")
except Exception as e:
    st.warning(f"Could not derive performance metrics: {e}")

# ===========================
# Downloadable Insights
# ===========================
st.header("🧾 Export Insights")

insights_text = f"""
### Key Insights Summary
- Highest CTR Platform: {high_ctr}
- Highest CVR Platform: {high_cvr}
- Average CPC: {company_df['CPC'].mean():.2f}
- Average CPA: {company_df['CPA'].mean():.2f}

EDA Summary:
- Total Columns: {eda_df.shape[1]}
- Strongest Correlation: {corr.unstack().sort_values(ascending=False).drop_duplicates().head(2).index[1]}
"""

st.download_button("📥 Download Insights as TXT", insights_text.encode(), "insights_summary.txt")

st.markdown("---")
st.caption("Built with ❤️ using Streamlit, Plotly & Pandas.")
