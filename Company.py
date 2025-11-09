import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud, STOPWORDS
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import warnings
warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")

@st.cache_data
def load_data(path="Glassdoor companies review dataset.csv"):
    df = pd.read_csv(path, low_memory=False)
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    return df

def safe_plotly_heatmap(df_numeric):
    corr = df_numeric.corr()
    fig = px.imshow(corr, text_auto=False, aspect="auto", title="Correlation Heatmap (numeric features)")
    return fig

def generate_wordcloud(text_series, max_words=150):
    text = " ".join(text_series.dropna().astype(str).tolist())
    stopwords = set(STOPWORDS)
    wc = WordCloud(width=800, height=400, stopwords=stopwords, background_color="white", max_words=max_words).generate(text)
    return wc

st.set_page_config(page_title="Glassdoor Advanced Dashboard", layout="wide")
st.title("📊 Glassdoor Company Reviews — Advanced Interactive Dashboard")

with st.spinner("Loading dataset..."):
    try:
        df = load_data()
        st.success("Dataset loaded successfully.")
    except FileNotFoundError:
        st.error("Dataset not found. Please upload 'Glassdoor companies review dataset.csv' using the sidebar file uploader.")
        df = None

st.sidebar.header("Controls & Uploads")
uploaded = st.sidebar.file_uploader("Upload CSV (optional)", type=["csv"])
if uploaded is not None:
    df = pd.read_csv(uploaded, low_memory=False)
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    st.sidebar.success("Uploaded dataset will be used.")

if df is None:
    st.stop()

if 'timestamp' in df.columns:
    try:
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    except:
        pass
if 'ratings_overall' in df.columns:
    df['ratings_overall'] = pd.to_numeric(df['ratings_overall'], errors='coerce')

st.sidebar.subheader("Filters")
country_list = ['All'] + sorted(df['country_code'].dropna().unique().astype(str).tolist()) if 'country_code' in df.columns else ['All']
country = st.sidebar.selectbox("Country", country_list)
company_type_list = ['All'] + sorted(df['company_type'].dropna().unique().astype(str).tolist()) if 'company_type' in df.columns else ['All']
company_type = st.sidebar.selectbox("Company Type", company_type_list)
industry_list = ['All'] + sorted(df['details_industry'].dropna().unique().astype(str).tolist()) if 'details_industry' in df.columns else ['All']
industry = st.sidebar.selectbox("Industry", industry_list)

min_rating = float(df['ratings_overall'].dropna().min()) if 'ratings_overall' in df.columns else 0.0
max_rating = float(df['ratings_overall'].dropna().max()) if 'ratings_overall' in df.columns else 5.0
rating_range = st.sidebar.slider("Overall Rating Range", min_value=min_rating, max_value=max_rating, value=(min_rating, max_rating))

df_filtered = df.copy()
if country != 'All' and 'country_code' in df_filtered.columns:
    df_filtered = df_filtered[df_filtered['country_code'].astype(str) == country]
if company_type != 'All' and 'company_type' in df_filtered.columns:
    df_filtered = df_filtered[df_filtered['company_type'].astype(str) == company_type]
if industry != 'All' and 'details_industry' in df_filtered.columns:
    df_filtered = df_filtered[df_filtered['details_industry'].astype(str) == industry]
if 'ratings_overall' in df_filtered.columns:
    df_filtered = df_filtered[df_filtered['ratings_overall'].between(rating_range[0], rating_range[1])]

st.markdown("## Key Metrics")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Reviews", f"{len(df_filtered):,}")
col2.metric("Unique Companies", f"{df_filtered['company'].nunique() if 'company' in df_filtered.columns else 'N/A'}")
col3.metric("Avg Rating", f"{df_filtered['ratings_overall'].mean():.2f}" if 'ratings_overall' in df_filtered.columns else "N/A")
col4.metric("Avg Reviews/Company", f"{len(df_filtered)/df_filtered['company'].nunique():.2f}" if 'company' in df_filtered.columns else "N/A")

st.markdown("## Visualizations")

with st.expander("Ratings Distribution"):
    if 'ratings_overall' in df_filtered.columns:
        fig = px.histogram(df_filtered, x='ratings_overall', nbins=20, title="Distribution of Overall Ratings", marginal="box")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.write("No 'ratings_overall' column found.")

with st.expander("Top Companies"):
    if 'company' in df_filtered.columns:
        top = df_filtered.groupby('company').agg(review_count=('id','count') if 'id' in df_filtered.columns else ('company','count'),
                                                 avg_rating=('ratings_overall','mean') if 'ratings_overall' in df_filtered.columns else ('company','count')).reset_index()
        top = top.sort_values(by='review_count', ascending=False).head(20)
        fig = px.bar(top, x='company', y='review_count', hover_data=['avg_rating'], title="Top Companies by Review Count")
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(top.head(50))
    else:
        st.write("No company column found.")

with st.expander("Ratings by Type/Size"):
    cols = []
    if 'company_type' in df_filtered.columns:
        cols.append('company_type')
    if 'details_size' in df_filtered.columns:
        cols.append('details_size')
    if len(cols)>0 and 'ratings_overall' in df_filtered.columns:
        fig = px.box(df_filtered, x=cols[0], y='ratings_overall', color=cols[0], title=f"Ratings by {cols[0]}")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.write("Required columns for this view not found.")

with st.expander("Industry Performance"):
    if 'details_industry' in df_filtered.columns and 'ratings_overall' in df_filtered.columns:
        ind = df_filtered.groupby('details_industry')['ratings_overall'].agg(['mean','count']).reset_index().sort_values('mean', ascending=False)
        fig = px.bar(ind.head(20), x='details_industry', y='mean', title="Top Industries by Avg Rating", text_auto='.2f')
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.write("Industry or rating column missing.")

with st.expander("Correlation Heatmap"):
    nums = df_filtered.select_dtypes(include=[np.number])
    if not nums.empty:
        fig = safe_plotly_heatmap(nums)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.write("No numeric columns to correlate.")

with st.expander("Wordcloud (Reviews)"):
    text_col = None
    for c in ['review_text','reviews','pros','review','comments','description']:
        if c in df_filtered.columns:
            text_col = c
            break
    if text_col is not None:
        wc = generate_wordcloud(df_filtered[text_col].astype(str))
        plt.figure(figsize=(12,5))
        plt.imshow(wc, interpolation='bilinear')
        plt.axis('off')
        st.pyplot(plt)
    else:
        st.write("No review text column found to build wordcloud.")

with st.expander("Sentiment Analysis"):
    use_sentiment = st.checkbox("Compute sentiment (VADER) for review text (may take time)", value=False)
    if use_sentiment:
        try:
            from nltk.sentiment.vader import SentimentIntensityAnalyzer
            import nltk
            nltk.download('vader_lexicon', quiet=True)
            sia = SentimentIntensityAnalyzer()
            if text_col is not None:
                df_filtered['__clean_text'] = df_filtered[text_col].fillna('').astype(str)
                df_filtered['sentiment_compound'] = df_filtered['__clean_text'].apply(lambda x: sia.polarity_scores(x)['compound'])
                st.write("Sentiment computed. Sample:")
                st.dataframe(df_filtered[['company', text_col, 'sentiment_compound']].head())
                if 'ratings_overall' in df_filtered.columns:
                    fig = px.scatter(df_filtered, x='sentiment_compound', y='ratings_overall', color='company', title='Sentiment vs Rating', hover_data=[text_col])
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.write("No text column to analyze.")
        except Exception as e:
            st.error(f"Sentiment step failed: {e}")

with st.expander("Company Clustering"):
    perform_cluster = st.checkbox("Perform company clustering (KMeans)", value=True)
    if perform_cluster and 'company' in df_filtered.columns:
        agg = df_filtered.groupby('company').agg(avg_rating=('ratings_overall','mean') if 'ratings_overall' in df_filtered.columns else ('id','count'),
                                                  reviews_count=('id','count') if 'id' in df_filtered.columns else ('company','count')).reset_index()
        agg[['avg_rating','reviews_count']] = agg[['avg_rating','reviews_count']].fillna(0)
        scaler = StandardScaler()
        X = scaler.fit_transform(agg[['avg_rating','reviews_count']])
        k = st.slider("Number of clusters (k)", 2, 8, 3)
        km = KMeans(n_clusters=k, random_state=42).fit(X)
        agg['cluster'] = km.labels_
        fig = px.scatter(agg, x='avg_rating', y='reviews_count', color='cluster', hover_data=['company'], title='Company Clusters by Avg Rating & Review Count')
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(agg.sort_values('avg_rating', ascending=False).head(50))
    else:
        st.write("Clustering skipped or company column missing.")

with st.expander("Predictive Modeling (Rating)"):
    do_model = st.checkbox("Train a RandomForest model to predict high/low rating", value=False)
    if do_model and 'ratings_overall' in df_filtered.columns:
        df_model = df_filtered.copy()
        df_model = df_model.dropna(subset=['ratings_overall'])
        df_model['target_high'] = (df_model['ratings_overall'] >= 4).astype(int)
        features = []
        if '__clean_text' in df_model.columns:
            features.append('sentiment_compound')
        for c in ['company_type','details_size','details_industry','country_code']:
            if c in df_model.columns and df_model[c].nunique()<200:
                df_model[c] = df_model[c].fillna('NA').astype(str)
                le = LabelEncoder()
                df_model[c+'_enc'] = le.fit_transform(df_model[c])
                features.append(c+'_enc')
        if len(features)==0:
            st.write("Not enough features to train model.")
        else:
            X = df_model[features]
            y = df_model['target_high']
            X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=42)
            rf = RandomForestClassifier(n_estimators=200, random_state=42)
            rf.fit(X_train, y_train)
            preds = rf.predict(X_test)
            st.write("Classification report:")
            st.text(classification_report(y_test, preds))
            fi = pd.Series(rf.feature_importances_, index=features).sort_values(ascending=False)
            fig = px.bar(fi.reset_index().rename(columns={'index':'feature',0:'importance'}), x='feature', y=0, title='Feature Importances')
            st.plotly_chart(fig, use_container_width=True)

with st.expander("Forecasting (Prophet)"):
    do_forecast = st.checkbox("Run Prophet forecasting on aggregated monthly review counts", value=False)
    if do_forecast:
        try:
            from prophet import Prophet
            if 'timestamp' in df_filtered.columns:
                ts = df_filtered.groupby(pd.Grouper(key='timestamp', freq='M')).size().reset_index(name='count')
                ts = ts.rename(columns={'timestamp':'ds','count':'y'})
                if len(ts)>12:
                    m = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
                    m.fit(ts)
                    periods = st.slider("Months to forecast", 1, 24, 6)
                    future = m.make_future_dataframe(periods=periods, freq='M')
                    forecast = m.predict(future)
                    fig = px.line(forecast, x='ds', y='yhat', title='Forecast (yhat)')
                    st.plotly_chart(fig, use_container_width=True)
                    st.dataframe(forecast[['ds','yhat','yhat_lower','yhat_upper']].tail(periods))
                else:
                    st.write("Not enough time series history (need >12 periods).")
            else:
                st.write("No 'timestamp' column found to build time series.")
        except Exception as e:
            st.error(f"Prophet forecasting failed: {e}")

with st.expander("Download Summaries"):
    if 'company' in df_filtered.columns and 'ratings_overall' in df_filtered.columns:
        comp_sum = df_filtered.groupby('company').agg(avg_rating=('ratings_overall','mean'), reviews_count=('id','count') if 'id' in df_filtered.columns else ('company','count')).reset_index()
        csv = comp_sum.to_csv(index=False).encode('utf-8')
        st.download_button("Download company summary CSV", data=csv, file_name="company_summary.csv", mime="text/csv")
    else:
        st.write("No company/rating to summarize.")

st.markdown('---')
st.caption("Built with ❤ — Advanced Glassdoor Dashboard. Customize further as needed.")
