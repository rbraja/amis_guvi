import streamlit as st
import pandas as pd
import numpy as np

from src.ml_regression import train_regression_model

st.title(
    "Phase 3 - Resolution Time Prediction"
)

df = pd.read_csv(
    "data/integrated/master_dataset_featured.csv"
)

if st.button(
    "Train Regression Model"
):

    model, feature_columns, mae, rmse, r2 = train_regression_model(df)

    st.session_state["resolution_model"] = model
    st.session_state["resolution_features"] = feature_columns

    st.metric("MAE", round(mae, 2))
    st.metric("RMSE", round(rmse, 2))
    st.metric("R2", round(r2, 2))


st.subheader("Predict resolution time for a single complaint")

category = st.selectbox("Category", options=sorted(df["category"].dropna().unique()))
priority = st.selectbox("Priority", options=sorted(df["priority"].dropna().unique()))
season_flag = st.selectbox("Season Flag", options=sorted(df["season_flag"].dropna().unique()))
complaint_hour = st.number_input("Complaint Hour (0-23)", min_value=0, max_value=23, value=12)
asset_failure_count = st.number_input("Asset Failure Count", min_value=0, value=0)

if st.button("Predict Resolution Time"):

    if "resolution_model" not in st.session_state:
        model, feature_columns, _, _, _ = train_regression_model(df)
    else:
        model = st.session_state["resolution_model"]
        feature_columns = st.session_state.get("resolution_features")

    input_row = pd.DataFrame([
        {
            "category": category,
            "priority": priority,
            "season_flag": season_flag,
            "complaint_hour": complaint_hour,
            "asset_failure_count": asset_failure_count,
        }
    ])

    input_dummies = pd.get_dummies(input_row)
    # align columns with training features
    for col in feature_columns:
        if col not in input_dummies.columns:
            input_dummies[col] = 0

    input_dummies = input_dummies[feature_columns]

    pred_log = model.predict(input_dummies)[0]
    pred_hours = float(np.expm1(pred_log))

    st.success(f"Predicted Resolution Time (hours): {pred_hours:.2f}")