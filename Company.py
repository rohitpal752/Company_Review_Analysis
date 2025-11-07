import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# --- Configuration ---
st.set_page_config(layout="wide", page_title="Company Fit Dashboard")

# --- 1. Mock Data Generation (Simulating your actual data) ---
def load_mock_data():
    """Generates a mock dataset based on implied structure from EDA.ipynb."""
    np.random.seed(42)
    companies = [
        "TechNova Solutions", "Global Finance Hub", "Agile Innovators", 
        "Creative Digital Labs", "Industrial Dynamics Inc.", "Green Energy Future", 
        "Local Retail Chain", "E-commerce Startup"
    ]
    data = {
        'Company Name': companies,
        'company_type': np.random.choice(['Startup', 'Medium', 'Corporate', 'Large'], size=len(companies)),
        'Avg_Rating': np.round(np.random.uniform(2.5, 4.8, size=len(companies)), 1),
        'Work_Life_Balance_Score': np.random.randint(2, 5, size=len(companies)), # 1 (Bad) to 5 (Excellent)
        'Culture_Score': np.random.randint(2, 5, size=len(companies)),
        'Review_Snippet': [
            "Fast-paced but rewarding.", "Stable environment, good benefits.", 
            "High pressure, rapid growth.", "Excellent creative freedom.",
            "Traditional structure, clear roles.", "Mission-driven and collaborative.",
            "Entry-level focus, shift work.", "Exciting but chaotic.",
        ],
        'Recommend': np.random.choice(['Yes', 'No', 'Neutral'], size=len(companies), p=[0.5, 0.2, 0.3])
    }
    return pd.DataFrame(data)

df = load_mock_data()

# --- 2. Custom Prediction Logic ---
def predict_fit(user_prefs, company_data):
    """
    Predicts if the company is a good fit based on user preferences and company attributes.
    This is a simple rule-based model for demonstration.
    """
    
    # 1. Calculate Score based on User-Company alignment
    score = 0
    feedback = []

    # Alignment 1: Work-Life Balance Preference
    company_wlb = company_data['Work_Life_Balance_Score']
    wlb_diff = abs(company_wlb - user_prefs['wlb'])
    score += (5 - wlb_diff) * 10  # Max 50 points (if wlb_diff is 0)
    
    if company_wlb >= 4 and user_prefs['wlb'] >= 4:
        feedback.append("High alignment on **Work-Life Balance**.")
    elif company_wlb <= 2 and user_prefs['wlb'] >= 4:
        feedback.append("Potential mismatch in **Work-Life Balance** (Company WLB is low).")

    # Alignment 2: Company Type Preference
    company_type = company_data['company_type']
    if user_prefs['company_size'] == 'Startup' and company_type == 'Startup':
        score += 30
        feedback.append("Matches your preference for a **Startup** environment.")
    elif user_prefs['company_size'] == 'Large' and company_type in ['Corporate', 'Large']:
        score += 30
        feedback.append("Matches your preference for a **Corporate/Large** environment.")
    elif user_prefs['company_size'] == 'Medium' and company_type == 'Medium':
        score += 30
        feedback.append("Good match for your preference for a **Medium**-sized company.")
    else:
        feedback.append(f"Company type ({company_type}) may differ from your preference ({user_prefs['company_size']}).")

    # Alignment 3: General Rating (Weighting towards high-rated companies)
    if company_data['Avg_Rating'] >= 4.0:
        score += 20  # Max 20 points
        feedback.append(f"Company has a strong overall rating: **{company_data['Avg_Rating']:.1f}/5.0**.")
    
    # Alignment 4: Company Recommendation Status
    if company_data['Recommend'] == 'Yes':
        score += 20
        feedback.append("High number of employees **recommend** working here.")

    # Total Score ranges from ~20 (worst fit) to ~120 (best fit)
    
    # Determine Final Verdict
    if score >= 100:
        verdict = ("A perfect match! This company strongly aligns with your stated preferences.", "green")
    elif score >= 70:
        verdict = ("A strong candidate! This company could be a great fit for you.", "blue")
    elif score >= 40:
        verdict = ("A potential fit, but review the details carefully for minor mismatches.", "orange")
    else:
        verdict = ("This company may not be the best cultural or lifestyle fit based on your profile.", "red")
        
    return score, verdict, feedback

# --- 3. Streamlit UI Layout ---

st.title("Company Fit and Analysis Dashboard")
st.markdown("Use this tool to explore company data and get a personalized prediction on whether a company is the right fit for you.")

tab_analysis, tab_prediction = st.tabs(["📊 Data Analysis (EDA)", "🔮 Personalized Fit Prediction"])

with tab_analysis:
    st.header("Exploratory Data Analysis Overview")
    st.markdown("This tab showcases key insights from the initial data cleaning and analysis steps performed in the Jupyter notebook.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Raw Data Sample")
        st.dataframe(df.head(5), use_container_width=True)
        st.caption(f"Showing 5 rows of the {len(df)} records in the mock dataset.")

    with col2:
        st.subheader("Recommendation Breakdown by Company Type")
        
        # Simulating the Plotly Treemap from EDA.ipynb
        recommend_counts = df.groupby(['Recommend', 'company_type']).size().reset_index(name='Count')
        
        try:
            fig = px.treemap(
                recommend_counts,
                path=['Recommend', 'company_type'],
                values='Count',
                color='Recommend',
                title='Recommendations by Company Type',
                color_discrete_map={
                    'Yes': 'green', 'No': 'red', 'Neutral': 'blue'
                }
            )
            fig.update_layout(margin=dict(t=30, l=0, r=0, b=0))
            st.plotly_chart(fig, use_container_width=True)
        except Exception:
            st.warning("Could not render Plotly chart due to dependency issues. Showing raw counts instead.")
            st.dataframe(recommend_counts, use_container_width=True)

with tab_prediction:
    st.header("Find Your Perfect Company Match")
    
    # --- Sidebar for User Profiling ---
    st.sidebar.header("👤 Your Preferences")
    st.sidebar.markdown("Answer these questions to help us understand your ideal workplace.")

    # Q1: Work-Life Balance
    user_wlb = st.sidebar.slider(
        "1. How important is Work-Life Balance (WLB)?",
        min_value=1, max_value=5, value=4,
        help="1 = Not important, work is life; 5 = Very important, need strict balance."
    )

    # Q2: Preferred Company Size/Type
    user_company_size = st.sidebar.radio(
        "2. What is your preferred company environment?",
        ('Startup (Fast-paced, high risk/reward)', 'Medium (Stable, flexible)', 'Large (Corporate, stable structure)'),
        index=2
    ).split(' ')[0] # Extract just the type for logic

    # Q3: Required Minimum Rating
    user_min_rating = st.sidebar.select_slider(
        "3. Minimum acceptable Overall Company Rating (out of 5)?",
        options=np.arange(3.0, 5.1, 0.1).round(1),
        value=3.5
    )
    
    user_prefs = {
        'wlb': user_wlb,
        'company_size': user_company_size,
        'min_rating': user_min_rating
    }

    # --- Main Section for Company Selection & Prediction ---
    st.subheader("Select a Company for Personalized Review")
    
    company_names = df['Company Name'].tolist()
    selected_company = st.selectbox(
        "Choose a Company:",
        company_names
    )
    
    if selected_company:
        # Get the row data for the selected company
        company_row = df[df['Company Name'] == selected_company].iloc[0]
        
        st.markdown("---")
        
        # Display Company Review and Metrics
        col_metrics, col_review = st.columns([1, 2])
        
        with col_metrics:
            st.metric(label="Overall Rating", value=f"{company_row['Avg_Rating']}/5.0")
            st.metric(label="WLB Score", value=f"{company_row['Work_Life_Balance_Score']}/5")
            st.metric(label="Culture Score", value=f"{company_row['Culture_Score']}/5")
            st.metric(label="Employee Recommendation", value=company_row['Recommend'])
        
        with col_review:
            st.markdown(f"**Company Type:** {company_row['company_type']}")
            st.info(f"**Review Snippet:** \"{company_row['Review_Snippet']}\"")
        
        st.markdown("---")
        
        # --- Run Prediction ---
        st.subheader("Your Personalized Fit Prediction")
        
        if company_row['Avg_Rating'] < user_prefs['min_rating']:
            st.error(
                f"**🛑 Unacceptable Rating.** The company's overall rating of "
                f"{company_row['Avg_Rating']:.1f} is below your required minimum of {user_prefs['min_rating']:.1f}."
            )
        else:
            score, verdict, feedback = predict_fit(user_prefs, company_row)
            
            # Display Verdict
            if verdict[1] == 'green':
                st.balloons()
                st.success(f"**🎉 Verdict: Yes, This Company Belongs to You!**")
            elif verdict[1] == 'red':
                st.error(f"**❌ Verdict: Not the Best Fit.**")
            elif verdict[1] == 'blue':
                 st.info(f"**👍 Verdict: Strong Potential Fit.**")
            else: # Orange/Warning
                st.warning(f"**⚠️ Verdict: Proceed with Caution.**")

            st.write(verdict[0])
            st.progress(score / 120, text=f"Fit Score: {score}/120")
            
            # Display Feedback
            with st.expander("See Detailed Feedback on Alignment"):
                st.markdown("The following factors contributed to your fit score:")
                for item in feedback:
                    st.markdown(f"- {item}")
        
# --- Footer ---
st.markdown("---")
st.caption("Dashboard powered by Streamlit and analysis based on EDA insights.")
