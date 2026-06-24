import streamlit as st
import pandas as pd

from src.sentiment_baseline import (
    train_sentiment_model,
    predict_sentiment
)

from src.distilbert_sentiment import (
    predict_distilbert
)

st.title(
    "Sentiment Analysis"
)

df = pd.read_csv(
    "data/integrated/master_dataset_featured.csv"
)

if st.button(
    "Train ML Sentiment Model"
):

    accuracy, f1 = train_sentiment_model(
        df
    )

    st.success(
        "Model Trained"
    )

    st.write(
        "Accuracy:",
        round(accuracy, 4)
    )

    st.write(
        "F1:",
        round(f1, 4)
    )

text = st.text_area(
    "Enter Feedback Text"
)

if st.button(
    "Predict Sentiment"
):

    ml_pred = predict_sentiment(
        text
    )

    dl_pred = predict_distilbert(
        text
    )

    st.subheader(
        "ML Prediction"
    )

    st.write(
        ml_pred
    )

    st.subheader(
        "DistilBERT Prediction"
    )

    st.write(
        dl_pred
    )