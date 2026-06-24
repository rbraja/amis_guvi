import streamlit as st
import pandas as pd

from src.ml_recommendation import recommend_technicians

st.title("Technician Recommendation")

df = pd.read_csv("data/integrated/master_dataset_featured.csv")

st.subheader("Recommend technicians for a category")

category = st.selectbox("Category", options=sorted(df["category"].dropna().unique()))

if st.button("Get Recommendations"):
    recs = recommend_technicians(df, category)

    if recs.empty:
        st.info("No recommendations found for selected category.")
    else:
        st.table(recs)