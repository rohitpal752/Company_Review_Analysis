import pandas as pd
import streamlit as st
import numpy as np
import plotly.express as px

# --- Custom Data Processing Functions ---

def apply_rating_decision(y):
    """
    Classifies 'ratings_recommend_to_friend' into 'Yes' (>0.5) or 'No' (<=0.5).
    """
    try:
        y = float(y)
        if 0.0 <= y <= 0.5:
            return "No"
        elif y > 0.5:
            return "Yes"
        else:
            return np.nan
    except:
        return np.nan

def apply_interview_difficulty(z):
    """
    Classifies 'interview_difficulty' into 'Easy', 'Moderate', or 'Difficult'.
    """
    try:
        z = float(z)
        if 0.0 <= z <= 2.0:
            return "Easy"
        elif 2.0 < z <= 3.5:
            return "Moderate"
        elif z > 3.5:
            return "Difficult"
        else:
            return np.nan
    except:
        return np.nan

# --- Data Loading and Preprocessing (Cached) ---

def load_data(file_path):
    """Loads and preprocesses the data."""
    df = pd.read_csv(file_path)

    key_rating_cols = ['ratings_overall', 'ratings_recommend_to_friend', 'interview_difficulty',
                       'ratings_career_opportunities', 'ratings_compensation_benefits',
                       'ratings_cutlure_values', 'ratings_senior_management', 'ratings_work_life_balance']

    for col in key_rating_cols:
        # Clean 'null' strings and convert to numeric
        df[col] = df[col].replace(['null', 'None', '-1', ''], np.nan)
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Apply custom classification functions
    df['Recommended'] = df['ratings_recommend_to_friend'].apply(apply_rating_decision)
    df['Difficulty_Category'] = df['interview_difficulty'].apply(apply_interview_difficulty)

    # Clean categorical columns
    df['company_type'] = df['company_type'].str.replace('Company - ', '', regex=False).str.replace('null', 'Unknown')
    df['details_industry'] = df['details_industry'].replace('null', 'Unknown')

    # Ensure core metric is available
    df.dropna(subset=['ratings_overall'], inplace=True)

    return df

@st.cache_data
def get_data(file):
    """Caches the data loading process."""
    return load_data(file)

# --- Page 1: Overview ---

def page_overview(df):
    """Streamlit page for Data Overview and Key Metrics."""
    st.title("📊 Data Overview")
    st.markdown("---")

    st.subheader(f"Filtered Data Summary (Reviews: {df.shape[0]})")

    # Display key metrics at the top
    col1, col2, col3 = st.columns(3)
    col1.metric("Average Overall Rating", f"{df['ratings_overall'].mean():.2f} / 5.0")
    col2.metric("Median Interview Difficulty", f"{df['interview_difficulty'].median():.2f}")
    col3.metric("Recommended Rate", f"{df['Recommended'].value_counts(normalize=True).get('Yes', 0) * 100:.1f}%")

    st.markdown("### Raw Data Sample")
    st.dataframe(df.head(10))

    st.markdown("### Descriptive Statistics for Ratings")
    rating_cols = ['ratings_overall', 'ratings_career_opportunities', 'ratings_compensation_benefits',
                   'ratings_cutlure_values', 'ratings_senior_management', 'ratings_work_life_balance']
    st.dataframe(df[rating_cols].describe().T)

# --- Page 2: Ratings Analysis ---

def page_ratings_analysis(df):
    """Streamlit page for Detailed Ratings Analysis with comparative insight."""
    st.title("🌟 Detailed Ratings Analysis")
    st.markdown("---")
    
    # 1. Overall Rating Distribution
    st.subheader("Overall Rating Distribution")
    fig_overall_rating = px.histogram(
        df,
        x='ratings_overall',
        nbins=10,
        title='Distribution of Overall Company Ratings in Filtered Data',
        labels={'ratings_overall': 'Overall Rating (1.0 to 5.0)'}
    )
    fig_overall_rating.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig_overall_rating, use_container_width=True)

    # 2. Comparative Analysis of Rating Factors by Company Type
    st.subheader("Comparative Core Rating Analysis")
    st.markdown("Average core ratings (Compensation, Culture, etc.) compared across different company types.")

    df_grouped = df.groupby('company_type')[['ratings_career_opportunities', 'ratings_compensation_benefits',
                                             'ratings_cutlure_values', 'ratings_senior_management',
                                             'ratings_work_life_balance']].mean().reset_index()

    df_melt = df_grouped.melt(id_vars='company_type', var_name='Rating_Factor', value_name='Average_Score')
    df_melt['Rating_Factor'] = df_melt['Rating_Factor'].str.replace('ratings_', '', regex=False).str.replace('_', ' ').str.title()
    
    # Filter out low-count types for meaningful comparison
    valid_types = df['company_type'].value_counts()
    valid_types = valid_types[valid_types > 5].index.tolist()
    df_melt = df_melt[df_melt['company_type'].isin(valid_types)]
    
    if df_melt.empty:
        st.warning("No sufficient data points for comparison with current filters.")
    else:
        fig_comparison = px.bar(
            df_melt,
            x='Rating_Factor',
            y='Average_Score',
            color='company_type',
            barmode='group',
            title='Average Core Ratings Comparison by Company Type',
            labels={'Average_Score': 'Average Rating (1.0 - 5.0)', 'Rating_Factor': 'Rating Factor'},
            height=500
        )
        st.plotly_chart(fig_comparison, use_container_width=True)

# --- Page 3: Recommendation & Difficulty ---

def page_recommendation_and_difficulty(df):
    """Streamlit page for Recommendation and Interview Difficulty Analysis."""
    st.title("👍 Recommendation & Interview Insights")
    st.markdown("---")

    col1, col2 = st.columns(2)

    # 1. Recommendation Status by Company Type (Treemap)
    with col1:
        st.subheader("Recommendation Status by Company Type")
        
        recommend_company_counts = df.groupby(['Recommended', 'company_type']).size().reset_index(name='Count')
        recommend_company_counts = recommend_company_counts.dropna(subset=['Recommended'])
        recommend_company_counts = recommend_company_counts[recommend_company_counts['company_type'] != 'Unknown']
        
        type_counts = recommend_company_counts.groupby('company_type')['Count'].sum()
        top_types = type_counts[type_counts > 10].index.tolist() 
        recommend_company_counts = recommend_company_counts[recommend_company_counts['company_type'].isin(top_types)]

        if recommend_company_counts.empty:
            st.warning("No data points available for Treemap with current filters.")
        else:
            fig_recommend_treemap = px.treemap(
                recommend_company_counts,
                path=['Recommended', 'company_type'],
                values='Count',
                color='Recommended',
                title='Recommendation Status by Company Type (Filtered)',
                color_discrete_map={'Yes': 'rgb(44, 160, 44)', 'No': 'rgb(214, 39, 40)'}
            )
            st.plotly_chart(fig_recommend_treemap, use_container_width=True)

    # 2. Interview Difficulty Analysis (Pie Chart)
    with col2:
        st.subheader("Interview Difficulty Distribution")

        difficulty_counts = df['Difficulty_Category'].value_counts(normalize=True).mul(100).reset_index()
        difficulty_counts.columns = ['Difficulty', 'Percentage']
        difficulty_order = ['Easy', 'Moderate', 'Difficult']
        difficulty_counts['Difficulty'] = pd.Categorical(difficulty_counts['Difficulty'], categories=difficulty_order, ordered=True)
        difficulty_counts = difficulty_counts.sort_values('Difficulty')

        if difficulty_counts.empty:
            st.warning("No data points available for Pie Chart with current filters.")
        else:
            fig_difficulty_pie = px.pie(
                difficulty_counts,
                values='Percentage',
                names='Difficulty',
                title='Interview Difficulty Categories (Filtered)',
                color='Difficulty',
                color_discrete_map={'Easy': 'lightgreen', 'Moderate': 'gold', 'Difficult': 'red'},
                hole=.3
            )
            fig_difficulty_pie.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig_difficulty_pie, use_container_width=True)

# --- Page 4: Advanced Relationships & Industry Benchmarking (NEW) ---

def page_advanced_relationships(df):
    """Streamlit page for Advanced Relationship Analysis."""
    st.title("📈 Advanced Relationships & Benchmarking")
    st.markdown("---")

    all_rating_cols = [
        'ratings_overall', 'ratings_career_opportunities', 'ratings_compensation_benefits',
        'ratings_cutlure_values', 'ratings_senior_management', 'ratings_work_life_balance'
    ]
    display_names = {col: col.replace('ratings_', '').replace('_', ' ').title() for col in all_rating_cols}

    # 1. Interactive Scatter Plot for Correlation
    st.subheader("1. Interactive Rating Correlation Analysis")
    st.info("Select two different rating factors to visualize their relationship and correlation coefficient.")

    col_x, col_y = st.columns(2)

    x_axis = col_x.selectbox(
        'Select X-Axis Rating',
        options=all_rating_cols,
        format_func=lambda x: display_names[x],
        index=0 # ratings_overall
    )

    y_axis = col_y.selectbox(
        'Select Y-Axis Rating',
        options=all_rating_cols,
        format_func=lambda x: display_names[x],
        index=3 # ratings_cutlure_values
    )
    
    # Calculate Correlation Coefficient
    corr_value = df[[x_axis, y_axis]].corr().iloc[0, 1]
    st.markdown(f"**Correlation Coefficient between {display_names[x_axis]} and {display_names[y_axis]}:** `{corr_value:.2f}`")

    # Create the scatter plot
    fig_scatter = px.scatter(
        df.dropna(subset=[x_axis, y_axis]),
        x=x_axis,
        y=y_axis,
        title=f'Relationship between {display_names[x_axis]} and {display_names[y_axis]}',
        labels={x_axis: display_names[x_axis], y_axis: display_names[y_axis]},
        hover_data=['company', 'company_type', 'details_industry']
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

    st.markdown("---")

    # 2. Industry Benchmarking
    st.subheader("2. Industry Benchmarking")
    st.markdown("Compare the average score for a selected factor across top industries in the filtered data.")

    bench_factor = st.selectbox(
        'Select Rating Factor for Benchmarking',
        options=all_rating_cols,
        format_func=lambda x: display_names[x],
        index=5 # Work Life Balance
    )
    
    # Calculate average rating per industry, filter out small industries/Unknown
    industry_counts = df['details_industry'].value_counts()
    top_industries = industry_counts[(industry_counts > 10) & (industry_counts.index != 'Unknown')].index.tolist()
    
    df_industry = df[df['details_industry'].isin(top_industries)]
    df_industry = df_industry.groupby('details_industry')[bench_factor].mean().reset_index()
    df_industry = df_industry.sort_values(by=bench_factor, ascending=False)
    
    fig_bench = px.bar(
        df_industry,
        x=bench_factor,
        y='details_industry',
        orientation='h',
        color=bench_factor,
        title=f'Average {display_names[bench_factor]} Score by Industry (Top Industries)',
        labels={bench_factor: f'Average {display_names[bench_factor]} Score', 'details_industry': 'Industry'},
        color_continuous_scale=px.colors.sequential.Viridis,
        height=600
    )
    st.plotly_chart(fig_bench, use_container_width=True)

# --- Main Application Logic ---
def app():
    # Set the overall page configuration
    st.set_page_config(layout="wide", page_title="Advanced Glassdoor Review Dashboard")

    file_path = "Glassdoor companies review dataset.csv"

    try:
        full_df = get_data(file_path)
    except FileNotFoundError:
        st.error(f"Error: File not found at 'Glassdoor companies review dataset.csv'. Please ensure the file is in the current working directory.")
        return

    # --- Sidebar Filters & Navigation ---
    
    st.sidebar.title("Navigation & Filters")
    
    # 1. Page Selector
    page = st.sidebar.selectbox(
        "Select an Analysis Slide:",
        ["Overview", "Ratings Analysis", "Recommendation & Difficulty", "Advanced Relationships"]
    )
    
    st.sidebar.markdown("---")

    # 2. Interactive Filters (Applied to all pages)
    st.sidebar.subheader("Data Filters")

    # Filter 1: Overall Rating Range
    min_rating = full_df['ratings_overall'].min()
    max_rating = full_df['ratings_overall'].max()
    rating_range = st.sidebar.slider(
        'Overall Rating Range',
        min_rating,
        max_rating,
        (min_rating, max_rating),
        step=0.1
    )

    # Filter 2: Company Type (Multi-select)
    company_types = sorted([t for t in full_df['company_type'].unique() if t != 'Unknown'])
    selected_types = st.sidebar.multiselect(
        'Select Company Types',
        options=company_types,
        default=company_types
    )

    # Apply Filters
    df_filtered = full_df[
        (full_df['ratings_overall'] >= rating_range[0]) &
        (full_df['ratings_overall'] <= rating_range[1]) &
        (full_df['company_type'].isin(selected_types))
    ]
    
    st.sidebar.markdown(f"**Filtered Records:** {df_filtered.shape[0]} of {full_df.shape[0]}")
    st.sidebar.markdown("---")
    
    if df_filtered.empty:
        st.error("No data matches the current filters. Please adjust the sidebar settings.")
        return

    # --- Page Content Router ---
    if page == "Overview":
        page_overview(df_filtered)
    elif page == "Ratings Analysis":
        page_ratings_analysis(df_filtered) 
    elif page == "Recommendation & Difficulty":
        page_recommendation_and_difficulty(df_filtered)
    elif page == "Advanced Relationships":
        page_advanced_relationships(df_filtered)

# Execute the main function
if __name__ == '__main__':
    app()
