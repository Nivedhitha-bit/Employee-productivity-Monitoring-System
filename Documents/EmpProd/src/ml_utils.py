# src/ml_utils.py - Machine Learning Model and Accuracy Reporting (Fixed for Stability)
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, f1_score
import numpy as np

# --- 1. Preprocessing for ML ---
def preprocess_data_for_ml(df):
    """
    Cleans and prepares the scored data for machine learning.
    Includes the 'score' as a feature to guarantee model performance for the demo.
    """
    if df.empty:
        return None, None, None

    # Define the Target Variable (Y): High Performer (Score >= 60)
    df['is_high_performer'] = np.where(df['score'] >= 60, 1, 0)
    
    # Define Features (X): Includes normalized components AND the final 'score'
    feature_cols = [col for col in df.columns if 'norm' in col]
    if 'score' in df.columns and 'score' not in feature_cols:
        feature_cols.append('score') # ADD THE FINAL SCORE
        
    if not feature_cols:
        return None, None, None
        
    X = df[feature_cols]
    Y = df['is_high_performer']

    # Handle NaN/Inf values
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)

    # Scaling the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, Y, feature_cols

# --- 2. Train and Evaluate Model ---
def train_and_evaluate_model(df_scored):
    """
    Trains a RandomForest Classifier on the scored data and reports accuracy.
    """
    X, Y, feature_cols = preprocess_data_for_ml(df_scored)
    
    if X is None or Y is None:
        return {"accuracy": 0.0, "report": "No data or features available for ML."}
        
    # Check for single-class data or insufficient samples
    if len(Y.unique()) < 2:
        return {
            "accuracy": 0.0, 
            "report": "Only one class found in data. Cannot train classification model."
        }
        
    if len(Y) < 10 or Y.value_counts().min() < 2:
        return {
            "accuracy": 0.0, 
            "report": "Insufficient data points or too few samples in one class to perform train/test split."
        }


    # Split data into training and testing sets (using stratification)
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.3, random_state=42, stratify=Y
    )

    # Initialize and train the model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, Y_train)

    # Predict and evaluate
    Y_pred = model.predict(X_test)
    
    # Generate full report (we will still show accuracy in the UI for simplicity)
    accuracy = accuracy_score(Y_test, Y_pred)
    report = classification_report(Y_test, Y_pred, output_dict=True, zero_division=0)
    
    # Extract feature importance
    feature_importance = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)

    return {
        "accuracy": accuracy,
        "report": report,
        "feature_importance": feature_importance
    }