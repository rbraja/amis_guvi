import streamlit as st
import pandas as pd

from src.category_classifier import (
    train_category_model,
    predict_category
)

st.title(
    "Complaint Category Classifier"
)

df = pd.read_csv(
    "data/integrated/master_dataset_featured.csv"
)

if st.button(
    "Train Category Model"
):

    accuracy, f1 = train_category_model(
        df
    )

    st.success(
        "Training Complete"
    )

    st.write(
        "Accuracy:",
        round(accuracy,4)
    )

    st.write(
        "F1:",
        round(f1,4)
    )

complaint = st.text_area(
    "Complaint Description"
)

if st.button(
    "Predict Category"
):

    prediction = predict_category(
        complaint
    )

    st.success(
        prediction
    )