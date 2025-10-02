# src/app.py - The Integrated Streamlit Dashboard (Final Version)
import streamlit as st
import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np

# Import the core logic and ML utilities
from utils import (
    process_and_score_data, 
    detect_alerts, 
    get_top_n_names
)
from ml_utils import train_and_evaluate_model

# --- Configuration ---
st.set_page_config(layout="wide", page_title="Productivity Monitoring System")
# This file serves as your simple database for the MVP
HISTORICAL_FILE = "../data/processed/scores_historical.csv" 

# --- Utility Functions for Data Management (CRUD on Historical CSV) ---

def load_historical_data():
    """Loads or creates the historical scores DataFrame."""
    # Ensure the directory exists
    os.makedirs(os.path.dirname(HISTORICAL_FILE), exist_ok=True)
    
    if os.path.exists(HISTORICAL_FILE):
        df = pd.read_csv(HISTORICAL_FILE)
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        return df.dropna(subset=['date'])
    return pd.DataFrame(columns=['Name', 'Target', 'Achieved', 'hours_worked', 'achievement_ratio', 'efficiency', 'score', 'source', 'date'])

def save_new_data(new_df):
    """Appends new scores to the historical CSV."""
    df_hist = load_historical_data()
    
    # Check for duplicate dates (prevent running the same data twice in one day)
    today_str = new_df['date'].iloc[0]
    if today_str in df_hist['date'].astype(str).unique():
        st.warning(f"Data for {today_str} already exists in historical records. Skipping save.")
        return df_hist
    
    df_hist = pd.concat([df_hist, new_df], ignore_index=True)
    df_hist.to_csv(HISTORICAL_FILE, index=False)
    st.success(f"New scores for {today_str} successfully processed and saved to history!")
    return df_hist

# --- Utility Function for Dynamic Plotting ---

def plot_historical_trends(df, n_top, time_period='3M'):
    """Plots trend for top N employees over a dynamic time period."""
    
    df['date'] = pd.to_datetime(df['date'])
    
    if df['date'].empty:
        st.info("No data available for plotting.")
        return
        
    end_date = df['date'].max()
    
    if time_period == '1W':
        start_date = end_date - pd.Timedelta(days=7)
        title = "Last 7 Days"
    elif time_period == '1M':
        start_date = end_date - pd.Timedelta(days=30)
        title = "Last 30 Days"
    else: # Default 3M
        start_date = end_date - pd.Timedelta(days=90)
        title = "Last 3 Months"

    df_filtered = df[df['date'] >= start_date].copy()
    
    if df_filtered.empty:
        st.info("Not enough data to plot the selected period.")
        return

    # Identify the top N performers based on the *overall* average score
    top_n_names = get_top_n_names(df, n=n_top)
    
    # Group the filtered data by name and date to ensure one point per day/employee (fixes duplicate points)
    df_plot_grouped = df_filtered[df_filtered['Name'].isin(top_n_names)].groupby(['Name', 'date'])['score'].mean().reset_index()
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for name in top_n_names:
        employee_data = df_plot_grouped[df_plot_grouped['Name'] == name]
        ax.plot(employee_data['date'], employee_data['score'], label=name, marker='o', linestyle='-')

    ax.set_title(f'Top {n_top} Employee Productivity Trends ({title})', fontsize=16)
    ax.set_xlabel('Date')
    ax.set_ylabel('Productivity Score (0-100)')
    ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1.05))
    ax.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig)


# --- 4. Main Dashboard UI ---

def main():
    st.title("🎯 Employee Productivity & Anomaly Detection System")
    
    # --- Sidebar for Security MVP and Data Upload ---
    st.sidebar.header("System Controls (Admin Only)")
    
    # Simple Security MVP: Password prompt
    password = st.sidebar.text_input("Enter Admin Password", type="password")
    if password != "admin123": # Dummy password
        st.error("🔒 Please enter the correct password to access controls.")
        st.stop()
        
    st.sidebar.success("✅ Access Granted: You can now upload and process data.")

    uploaded_tdm = st.sidebar.file_uploader("Upload TDM_clean.csv", type="csv")
    uploaded_quebec = st.sidebar.file_uploader("Upload Quebec_clean.csv", type="csv")

    # --- SIMULATION/FLUCTUATION CONTROL (For Testing) ---
    st.sidebar.header("Simulation Control (Demo)")
    st.sidebar.markdown("Use this to simulate a score change for testing alerts.")
    
    emp_to_change = st.sidebar.selectbox("Employee to Fluctuate", ["Lavanya", "VaniU E", "Dharshini J", "Vinodhini Jv", "None"])
    fluctuation_pct = st.sidebar.slider("Score Change %", -30.0, 30.0, 0.0, 5.0)
    
    fluctuations = None
    if emp_to_change != "None":
        fluctuations = {emp_to_change: fluctuation_pct}
        
    # --- Process Data Button ---
    if st.sidebar.button("Process & Save New Scores"):
        if uploaded_tdm and uploaded_quebec:
            tdm_df = pd.read_csv(uploaded_tdm)
            quebec_df = pd.read_csv(uploaded_quebec)
            
            # This calls the full scoring pipeline
            new_scores_df = process_and_score_data(tdm_df, quebec_df, fluctuations)
            
            if not new_scores_df.empty:
                # This saves and updates the historical data file
                st.session_state.df_hist = save_new_data(new_scores_df)
                st.balloons()
            else:
                st.error("Failed to process data. Check file formats in your input CSVs.")
        else:
            st.error("Please upload both TDM and Quebec CSV files.")
    
    # --- Main Dashboard Logic ---
    
    # Load historical data (use session_state to avoid reloading on every interaction)
    if 'df_hist' not in st.session_state:
        st.session_state.df_hist = load_historical_data()

    df_hist = st.session_state.df_hist
    
    if df_hist.empty:
        st.info("No historical data available. Please upload and process the initial files.")
        st.stop()

    # --- 1. Alert System & Change Detection ---
    alerts_df = detect_alerts(df_hist, days=7, threshold_pct=10, n_top=5)
    
    st.header("1. Anomaly Detection Engine")
    
    if not alerts_df.empty:
        st.error(f"🚨 ALERT! {len(alerts_df)} Significant Score Drops Detected (Top 5 Focus):")
        st.dataframe(alerts_df, use_container_width=True)
    else:
        st.success("✅ System Status: No significant 7-day score drops detected among Top 5 Performers.")

    st.markdown("---")
    
    # --- 2. Productivity Dashboard & Comparisons ---
    st.header("2. Productivity Dashboard & Trend Analysis")
    
    # Tabbed Interface for Comparison
    tab1, tab2, tab3 = st.tabs(["Top 5 Performers", "Trend Visualization", "ML Model Accuracy"])
    
    with tab1:
        st.subheader("Current Top 5 Employees by Average Score")
        top_5_names = get_top_n_names(df_hist, n=5)
        
        # Filter for the latest day
        latest_date = df_hist['date'].max()
        
        # 1. Filter for the Top 5 Names only
        top_5_data = df_hist[df_hist['Name'].isin(top_5_names)].copy()
        
        # 2. Filter for the latest day
        top_5_latest = top_5_data[top_5_data['date'] == latest_date].copy()
        
        # 3. FIX: Group the latest data by Name and aggregate to combine TDM/Quebec scores for the day
        top_5_deduplicated = top_5_latest.groupby('Name').agg({
            'score': 'mean',
            'Target': 'sum',
            'Achieved': 'sum',
            'hours_worked': 'sum',
            'source': lambda x: ', '.join(x.unique()) # Show sources (TDM, Quebec)
        }).reset_index().sort_values(by='score', ascending=False)
        
        if not top_5_deduplicated.empty:
            st.dataframe(top_5_deduplicated[['Name', 'score', 'Target', 'Achieved', 'hours_worked', 'source']], use_container_width=True)
        else:
            st.info("Top 5 calculation pending more data.")

    with tab2:
        st.subheader("Historical Score Trend")
        time_selection = st.radio(
            "Select Comparison Period:",
            ('3M', '1M', '1W'),
            index=0,
            horizontal=True,
            help="Compare current performance against 3 Months, 1 Month, or 1 Week of history."
        )
        # The plot function also includes the fix for duplicate daily entries
        plot_historical_trends(df_hist, n_top=5, time_period=time_selection)
        
    # --- 3. ML Model Accuracy (New Requirement) ---
    with tab3:
        st.subheader("Machine Learning Model: Performance Prediction")
        
        # Train and evaluate the model using all available historical data
        with st.spinner('Training and evaluating model...'):
            ml_results = train_and_evaluate_model(df_hist)

        st.metric("Model Accuracy (High Performer Classification)", 
                  f"{ml_results['accuracy']:.2%}", 
                  help="Accuracy of predicting whether an employee is a 'High Performer' (Score >= 60).")
        
        st.markdown("---")
        st.subheader("Feature Importance")
        st.info("This shows which factors the model believes are most important for predicting high performance.")
        
        if 'feature_importance' in ml_results:
            st.bar_chart(ml_results['feature_importance'])
        else:
            st.info("Feature importance data is not available (run with more historical samples).")


if __name__ == "__main__":
    main()