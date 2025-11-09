import pandas as pd
import streamlit as st
import numpy as np
import plotly.express as px

# --- Custom Data Processing Functions (Inferred from Company_Project.ipynb) ---

def apply_rating_decision(y):
    """
    Applies the binary 'decision' logic from the notebook (0.0-0.5 is No, >0.5 is Yes).
    Used to create a binary 'Recommended' column from 'ratings_recommend_to_friend'.
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
    Applies the categorical 'difficulty' logic from the notebook.
    Used for 'interview_difficulty'.
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

# --- Data Loading and Preprocessing ---

def load_data(file_path):
    """Loads and preprocesses the data."""
    df = pd.read_csv(file_path)

    # 1. Clean 'null' strings and convert key columns to numeric
    key_rating_cols = ['ratings_overall', 'ratings_recommend_to_friend', 'interview_difficulty',
                       'ratings_career_opportunities', 'ratings_compensation_benefits',
                       'ratings_cutlure_values', 'ratings_senior_management', 'ratings_work_life_balance']

    for col in key_rating_cols:
        # Replace string representations of missing values and errors
        df[col] = df[col].replace(['null', 'None', '-1', ''], np.nan)
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # 2. Apply custom classification functions
    df['Recommended'] = df['ratings_recommend_to_friend'].apply(apply_rating_decision)
    df['Difficulty_Category'] = df['interview_difficulty'].apply(apply_interview_difficulty)

    # 3. Simplify 'company_type' for better visualization
    df['company_type'] = df['company_type'].str.replace('Company - ', '', regex=False).str.replace('null', 'Unknown')
    df['details_industry'] = df['details_industry'].replace('null', 'Unknown')

    return df

@st.cache_data
def get_data(file):
    """Caches the data loading process for better performance."""
    return load_data(file)

# --- Slide 1: Overview ---

def page_overview(df):
    """Streamlit page for Data Overview."""
    st.title("📊 Glassdoor Review Analysis: Data Overview")
    st.markdown("---")

    st.subheader("Raw Data Sample")
    st.dataframe(df.head())

    st.subheader("Data Dimensions")
    st.write(f"Number of Rows: **{df.shape[0]}**")
    st.write(f"Number of Columns: **{df.shape[1]}**")

    st.subheader("Descriptive Statistics for Key Ratings")
    rating_cols = ['ratings_overall', 'ratings_career_opportunities', 'ratings_compensation_benefits',
                   'ratings_cutlure_values', 'ratings_senior_management', 'ratings_work_life_balance']
    st.dataframe(df[rating_cols].describe().T)

# --- Slide 2: Ratings Analysis ---

def page_ratings_analysis(df):
    """Streamlit page for Detailed Ratings Analysis."""
    st.title("🌟 Detailed Ratings Analysis")
    st.markdown("---")

    # Overall Rating Distribution
    st.subheader("Overall Rating Distribution")
    fig_overall_rating = px.histogram(
        df.dropna(subset=['ratings_overall']),
        x='ratings_overall',
        nbins=10,
        title='Distribution of Overall Company Ratings',
        labels={'ratings_overall': 'Overall Rating (1.0 to 5.0)'}
    )
    fig_overall_rating.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig_overall_rating, use_container_width=True)

    # Average Core Rating Factors (Bar Chart)
    st.subheader("Average Core Ratings")
    core_ratings = ['ratings_career_opportunities', 'ratings_compensation_benefits',
                    'ratings_cutlure_values', 'ratings_senior_management', 'ratings_work_life_balance']
    avg_ratings = df[core_ratings].mean().sort_values(ascending=False).reset_index()
    avg_ratings.columns = ['Rating_Type', 'Average_Score']
    # Clean up column names for display
    avg_ratings['Rating_Type'] = avg_ratings['Rating_Type'].str.replace('ratings_', '', regex=False).str.replace('_', ' ').str.title()

    fig_avg_ratings = px.bar(
        avg_ratings,
        x='Average_Score',
        y='Rating_Type',
        orientation='h',
        title='Average Score Across Different Rating Factors',
        color='Average_Score',
        color_continuous_scale=px.colors.sequential.Plotly3,
        labels={'Average_Score': 'Average Rating (1.0 - 5.0)', 'Rating_Type': 'Rating Factor'}
    )
    st.plotly_chart(fig_avg_ratings, use_container_width=True)

# --- Slide 3: Recommendation & Difficulty ---

def page_recommendation_and_difficulty(df):
    """Streamlit page for Recommendation and Interview Difficulty Analysis."""
    st.title("👍 Recommendation & Interview Insights")
    st.markdown("---")

    # 1. Recommendation by Company Type (Inspired by Treemap in EDA.ipynb)
    st.subheader("Recommendation Status by Company Type")

    recommend_company_counts = df.groupby(['Recommended', 'company_type']).size().reset_index(name='Count')
    recommend_company_counts = recommend_company_counts.dropna(subset=['Recommended'])
    recommend_company_counts = recommend_company_counts[recommend_company_counts['company_type'] != 'Unknown']

    # Filter for types with sufficient data to keep the treemap readable
    type_counts = recommend_company_counts.groupby('company_type')['Count'].sum()
    top_types = type_counts[type_counts > 50].index.tolist() # Only showing company types with > 50 total reviews
    recommend_company_counts = recommend_company_counts[recommend_company_counts['company_type'].isin(top_types)]


    fig_recommend_treemap = px.treemap(
        recommend_company_counts,
        path=['Recommended', 'company_type'],
        values='Count',
        color='Recommended',
        title='Recommendation Status by Company Type (Top Types)',
        color_discrete_map={'Yes': 'rgb(44, 160, 44)', 'No': 'rgb(214, 39, 40)'} # Green for Yes, Red for No
    )
    st.plotly_chart(fig_recommend_treemap, use_container_width=True)

    # 2. Interview Difficulty Analysis
    st.subheader("Interview Difficulty Distribution")

    difficulty_counts = df['Difficulty_Category'].value_counts(normalize=True).mul(100).reset_index()
    difficulty_counts.columns = ['Difficulty', 'Percentage']
    # Define order for sorting
    difficulty_order = ['Easy', 'Moderate', 'Difficult']
    difficulty_counts['Difficulty'] = pd.Categorical(difficulty_counts['Difficulty'], categories=difficulty_order, ordered=True)
    difficulty_counts = difficulty_counts.sort_values('Difficulty')

    fig_difficulty_pie = px.pie(
        difficulty_counts,
        values='Percentage',
        names='Difficulty',
        title='Distribution of Interview Difficulty Categories',
        color='Difficulty',
        color_discrete_map={'Easy': 'lightgreen', 'Moderate': 'gold', 'Difficult': 'red'},
        hole=.3
    )
    fig_difficulty_pie.update_traces(textposition='inside', textinfo='percent+label')
    st.plotly_chart(fig_difficulty_pie, use_container_width=True)


# --- Main Application Logic ---

def main():
    # Set the overall page configuration
    st.set_page_config(layout="wide", page_title="Glassdoor Review Analysis")

    # File path (assuming the CSV is in the same directory)
    file_path = "Glassdoor companies review dataset.csv"

    try:
        df = get_data(file_path)
    except FileNotFoundError:
        st.error(f"Error: File not found at 'Glassdoor companies review dataset.csv'. Please ensure the file is in the current working directory.")
        return

    # --- Sidebar Navigation (The 'Multi-slide' feature) ---
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Select an Analysis Slide:",
        ["Overview", "Ratings Analysis", "Recommendation & Difficulty"]
    )
    st.sidebar.markdown("---")
    st.sidebar.info(f"Data source: `{file_path}`")
    st.sidebar.markdown(
        """
        **Interactive Data Analysis Project**
        This multi-slide application visualizes and analyzes
        the Glassdoor company review dataset.
        """
    )


    # --- Page Content Router ---
    if page == "Overview":
        page_overview(df)
    elif page == "Ratings Analysis":
        page_ratings_analysis(df)
    elif page == "Recommendation & Difficulty":
        page_recommendation_and_difficulty(df)

# Execute the main function
if __name__ == '__main__':
    main()
