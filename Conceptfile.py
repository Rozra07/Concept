import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

###############################################################################
# Step 1: Load Model, Scaler, and Feature Columns from local directory
###############################################################################
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("logistic_regression_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("feature_columns.pkl", "rb") as f:
    feature_columns = pickle.load(f)

###############################################################################
# Step 2: Rule-Based Scoring with Extreme Case Adjustments (Compounding Effect)
###############################################################################
def compute_weighted_attrition(employee):
    score = 0
    extreme_factors = 0

    # Hasn't been promoted beyond 2x the min promotion cycle
    if employee["Hasn't been promoted"] >= 2 * employee["Minimum Promotion Cycle"]:
        score += 30
        extreme_factors += 1

    # Last Performance Rating logic
    if employee["Last Performance Rating"] == 1:
        score += 40
        extreme_factors += 1
    elif employee["Last Performance Rating"] == 5:
        score -= 15  # High performance reduces attrition risk

    # Compa Ratio below 70%
    if employee["Compa Ratio"] < 70:
        score += 35
        extreme_factors += 1

    # Low Retention for College Tier, Industry, Company Type
    if employee["College Tier Retention"] < 15:
        score += 15
        extreme_factors += 1
    if employee["Industry Retention"] < 15:
        score += 15
        extreme_factors += 1
    if employee["Company Type Retention"] < 15:
        score += 15
        extreme_factors += 1

    # Compounding effect for multiple extreme factors
    if extreme_factors >= 3:
        multiplier = 1.3 if extreme_factors == 3 else (1.5 if extreme_factors == 4 else 1.8)
        score = min(100, score * multiplier)

    # Baseline correction
    score = max(0, score - 20)

    return min(100, score)

###############################################################################
# Step 3: Machine Learning Prediction (Logistic Regression)
###############################################################################
def predict_attrition(employee_data):
    # Convert input data into DataFrame
    df_input = pd.DataFrame([employee_data])

    # Drop inputs that are not available in the trained model
    df_input = df_input[[col for col in df_input.columns if col in feature_columns]]

    # Encode categorical variables
    df_input = pd.get_dummies(df_input)
    df_input = df_input.reindex(columns=feature_columns, fill_value=0)

    # Scale input features
    X_scaled = scaler.transform(df_input)

    # Predict probability of attrition from logistic regression
    ml_probability = model.predict_proba(X_scaled)[:, 1][0] * 100

    # Get rule-based attrition score
    rule_probability = compute_weighted_attrition(employee_data)

    # Combine the two probabilities, with heavier weight on rule-based
    combined_score = 0.4 * ml_probability + 0.6 * rule_probability
    return combined_score

###############################################################################
# Step 4: Streamlit UI (All Inputs Included)
###############################################################################
st.title("Employee Attrition Prediction Tool")

with st.form("attrition_form"):
    employee_data = {
        "Employee Age": st.slider("Employee Age", 18, 65, 30),
        "Average Employee Age": st.slider("Avg Employee Age", 18, 65, 35),
        "Tenure (Months)": st.slider("Tenure (Months)", 0, 240, 36),
        "Hasn't been promoted": st.slider("Months Since Last Promotion", 0, 60, 12),
        "Minimum Promotion Cycle": st.slider("Min Promotion Cycle (Months)", 12, 60, 24),
        "College Tier Retention": st.slider("College Tier Retention (%)", 10, 100, 60),
        "Industry Retention": st.slider("Industry Retention (%)", 10, 100, 60),
        "Company Type Retention": st.slider("Company Type Retention (%)", 10, 100, 60),
        "Last Performance Rating": st.slider("Last Performance Rating", 1, 5, 3),
        "Compa Ratio": st.slider("Compa Ratio (%)", 50, 150, 100)
    }
    
    submit_button = st.form_submit_button("Predict")

if submit_button:
    prediction = predict_attrition(employee_data)
    st.write(f"**Estimated Attrition Probability**: {prediction:.2f}%")
