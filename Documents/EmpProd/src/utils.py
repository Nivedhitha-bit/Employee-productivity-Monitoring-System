# src/utils.py - Core Scoring and Detection Logic
import pandas as pd
from datetime import datetime, timedelta
import numpy as np
import os

# --- A. Hours Worked Calculation ---
def add_hours_worked_tdm(df):
    """Compute total hours worked for TDM dataset by summing hourly columns."""
    hour_cols = [col for col in df.columns if "-" in col and ":" in col]
    df["hours_worked"] = df[hour_cols].apply(lambda row: row.count(), axis=1)
    return df

def add_hours_worked_quebec(df):
    """Compute total hours worked for Quebec dataset by summing slot columns and EXTRA HOURS."""
    slot_cols = [col for col in df.columns if "-" in col or "EXTRA" in col.upper()]
    df["hours_worked"] = df[slot_cols].apply(lambda row: sum(pd.to_numeric(row, errors="coerce").fillna(0)), axis=1)
    return df

# --- B. Productivity Score Calculation (Reworked for Robustness) ---
def compute_productivity_score(df, emp_col="Name", target_col="Target", achieved_col="Achieved", hours_col="hours_worked"):
    """
    Compute productivity score (0-100) based on achievement, efficiency, and hours.
    Ensures robust normalization for small datasets.
    """
    df = df.fillna(0)

    # 1. Achievement ratio and normalization
    df["achievement_ratio"] = df[achieved_col] / df[target_col].replace(0, 1)
    
    # Use a minimum divisor of 1.0 to prevent division by near-zero max values
    max_achieve = df["achievement_ratio"].replace([np.inf, -np.inf], np.nan).max()
    safe_max_achieve = max(1.0, max_achieve)
    df["achievement_ratio_norm"] = df["achievement_ratio"] / safe_max_achieve

    # 2. Efficiency (achieved per hour) and normalization
    df["efficiency"] = df[achieved_col] / (df[hours_col] + 1e-6) # Add small epsilon for stability
    
    max_efficiency = df["efficiency"].replace([np.inf, -np.inf], np.nan).max()
    safe_max_efficiency = max(1.0, max_efficiency)
    df["efficiency_norm"] = df["efficiency"] / safe_max_efficiency

    # 3. Hours worked normalized
    max_hours = df[hours_col].max()
    safe_max_hours = max(1.0, max_hours)
    df["hours_worked_norm"] = df[hours_col] / safe_max_hours

    # 4. Final score (Weighted average)
    df["score"] = (
        0.6 * df["achievement_ratio_norm"] +
        0.3 * df["efficiency_norm"] +
        0.1 * df["hours_worked_norm"]
    ) * 100
    
    # Ensure all norm columns are clean (no NaNs or Infs) before returning
    for col in ["achievement_ratio_norm", "efficiency_norm", "hours_worked_norm", "score"]:
        df[col] = df[col].replace([np.inf, -np.inf, np.nan], 0)


    cols_to_keep = [emp_col, target_col, achieved_col, hours_col,
                    "achievement_ratio", "efficiency", "score",
                    "achievement_ratio_norm", "efficiency_norm", "hours_worked_norm"]
    return df[cols_to_keep]


# --- C. Data Persistence and Utility Functions ---
def get_top_n_names(df, n=5, emp_col='Name', score_col='score'):
    """Identifies the current top N employees by their overall historical score mean."""
    if df.empty:
        return []
    return df.groupby(emp_col)[score_col].mean().nlargest(n).index

def process_and_score_data(tdm_df, quebec_df, fluctuations=None):
    """Combines processing, scoring, and applying fluctuation for a single run."""
    all_results = []
    
    # Process TDM
    if not tdm_df.empty:
        tdm = add_hours_worked_tdm(tdm_df.copy())
        tdm_scores = compute_productivity_score(tdm)
        tdm_scores["source"] = "TDM"
        all_results.append(tdm_scores)
    
    # Process Quebec
    if not quebec_df.empty:
        quebec = add_hours_worked_quebec(quebec_df.copy())
        quebec_scores = compute_productivity_score(quebec)
        quebec_scores["source"] = "Quebec"
        all_results.append(quebec_scores)
        
    if not all_results:
        return pd.DataFrame()

    final_df = pd.concat(all_results, ignore_index=True)

    # Apply fluctuation for demo/simulation
    if fluctuations:
        for name, percentage in fluctuations.items():
            if name in final_df['Name'].values:
                mask = final_df['Name'] == name
                multiplier = 1 + (percentage / 100)
                final_df.loc[mask, 'score'] = final_df.loc[mask, 'score'] * multiplier
                final_df.loc[mask, 'score'] = final_df.loc[mask, 'score'].clip(lower=0, upper=100)

    # Add current date for historical tracking
    final_df["date"] = datetime.now().strftime("%Y-%m-%d")
    
    return final_df.sort_values(by="score", ascending=False).reset_index(drop=True)

# --- D. Anomaly Detection (Alerts) ---
def detect_alerts(df, days=7, threshold_pct=10, n_top=5, emp_col='Name', score_col='score'):
    """Detects employees among the Top N whose score has dropped significantly."""
    alerts = []
    
    # Ensure date is a datetime object
    df['date'] = pd.to_datetime(df['date'])
    
    # 1. Get the names of the current Top N performers
    top_n_names = get_top_n_names(df, n=n_top, emp_col=emp_col, score_col=score_col)
    
    # Filter the data to only include the Top N (and a relevant window)
    df_top = df[df[emp_col].isin(top_n_names)].copy()
    
    if df_top.empty:
        return pd.DataFrame(alerts)

    # 2. Calculate daily average score for each employee
    daily_scores = df_top.groupby([emp_col, 'date'])[score_col].mean().reset_index()
    today = daily_scores['date'].max()
    
    # 3. Calculate 30-day average for comparison (baseline)
    start_30d = today - timedelta(days=30)
    df_30d = daily_scores[daily_scores['date'] >= start_30d]
    avg_30d = df_30d.groupby(emp_col)[score_col].mean()

    # 4. Calculate X-day average (recent performance)
    start_xd = today - timedelta(days=days)
    df_xd = daily_scores[daily_scores['date'] >= start_xd]
    avg_xd = df_xd.groupby(emp_col)[score_col].mean()
    
    # 5. Merge and calculate the drop
    comparison = pd.merge(
        avg_30d.rename('avg_30d'), 
        avg_xd.rename('avg_xd'), 
        on=emp_col, 
        how='inner'
    ).reset_index()
    
    comparison['drop_pct'] = ((comparison['avg_xd'] - comparison['avg_30d']) / comparison['avg_30d'].replace(0, 1)) * 100
    
    # 6. Identify alerts
    alert_mask = comparison['drop_pct'] < -threshold_pct
    
    for _, row in comparison[alert_mask].iterrows():
        alerts.append({
            'Employee': row[emp_col],
            'Drop (%)': f"{row['drop_pct']:.2f}%",
            'Recent Avg': f"{row['avg_xd']:.2f}",
            'Baseline Avg': f"{row['avg_30d']:.2f}",
            'Metric': f"Score dropped by {abs(row['drop_pct']):.2f}% in the last {days} days."
        })

    return pd.DataFrame(alerts)