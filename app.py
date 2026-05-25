import streamlit as st
import pandas as pd
import joblib
import json
from catboost import Pool

# ============================================================================
# APP TITLE
# ============================================================================
st.title(" AI-Powered Customer Retention & Churn Prediction System")

# ============================================================================
# LOAD MODEL + METADATA
# ============================================================================
model = joblib.load("churn_prediction_model.pkl")

with open("churn_model_metadata.json") as f:
    config = json.load(f)

feature_names = config["input_features"]
cat_features = config["categorical_inputs"]

# ============================================================================
# FILE UPLOAD
# ============================================================================
uploaded_file = st.file_uploader("Upload customer data (CSV)", type=["csv"])

if uploaded_file:

    # Load data
    df = pd.read_csv(uploaded_file)

    st.write("### Raw Data Preview")
    st.dataframe(df.head())

    # =========================================================================
    # FEATURE ALIGNMENT (CRITICAL FIX)
    # =========================================================================

    missing_cols = [col for col in feature_names if col not in df.columns]

    if missing_cols:
        st.error(f"Missing required columns: {missing_cols}")
    else:

        # Reorder columns exactly as training
        df = df[feature_names]

        # =========================================================================
        # CATBOOST SAFE PREDICTION
        # =========================================================================

        pool = Pool(df, cat_features=cat_features)

        preds = model.predict(pool)
        proba = model.predict_proba(pool)[:, 1]

        # Add results
        df["Churn_Prediction"] = preds
        df["Churn_Probability"] = proba

        # =========================================================================
        # RESULTS DISPLAY
        # =========================================================================

        st.write("### Prediction Results")
        st.dataframe(df)

        # =========================================================================
        # BUSINESS SUMMARY
        # =========================================================================

        total = len(df)
        churners = int(sum(preds))

        st.write("### Business Summary")

        st.metric("Total Customers", total)
        st.metric("Predicted Churners", churners)
        st.metric("Churn Rate (%)", round(churners / total * 100, 2))

        # =========================================================================
        # RISK SEGMENTATION
        # =========================================================================

        st.write("###  Risk Segments")

        def risk_level(p):
            if p >= 0.7:
                return "HIGH RISK"
            elif p >= 0.4:
                return "MEDIUM RISK"
            else:
                return "LOW RISK"

        df["Risk_Segment"] = df["Churn_Probability"].apply(risk_level)

        st.dataframe(df[["Churn_Probability", "Risk_Segment"]])