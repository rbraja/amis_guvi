import streamlit as st
import pandas as pd

from src.ml_priority import train_priority_model

st.title("Phase 3 - Priority Prediction")

df = pd.read_csv(
    "data/integrated/master_dataset_featured.csv"
)

st.write("Dataset Shape")

st.write(df.shape)

if st.button("Train Priority Model"):

    model, accuracy, report = train_priority_model(df)

    st.success("Model Training Completed")

    st.metric("Accuracy", round(accuracy, 4))

    st.text(report)

    st.session_state["priority_model"] = model
st.subheader("Predict Priority for a single complaint")

category = st.selectbox("Category", options=sorted(df["category"].dropna().unique()))

subcategory = st.selectbox("Subcategory", options=sorted(df["subcategory"].dropna().unique()))

season_flag = st.selectbox("Season Flag", options=sorted(df["season_flag"].dropna().unique()))

complaint_hour = st.number_input("Complaint Hour (0-23)", min_value=0, max_value=23, value=12)

complaint_text = st.text_area("Complaint Text")

if st.button("Predict Priority"):

    if "priority_model" not in st.session_state:
        model, _, _ = train_priority_model(df)
    else:
        model = st.session_state["priority_model"]

    input_df = pd.DataFrame([
        {
            "category": category,
            "subcategory": subcategory,
            "season_flag": season_flag,
            "complaint_hour": complaint_hour,
            "complaint_text": complaint_text,
        }
    ])

    pred = model.predict(input_df)[0]

    st.success(f"Predicted Priority: {pred}")