import streamlit as st
import pandas as pd
import plotly.express as px

from src.feature_engineering import (
    add_season_flag,
    add_complaint_hour,
    add_repeat_complaint,
    add_asset_failure_count
)

st.title(
    "Phase 2 - EDA & Feature Engineering"
)

master_df = pd.read_csv(
    "data/integrated/master_dataset.csv"
)

asset_failure_df = pd.read_csv(
    "data/cleaned_by_you/asset_failure_history.csv"
)

master_df = add_season_flag(
    master_df
)

master_df = add_complaint_hour(
    master_df
)

master_df = add_repeat_complaint(
    master_df
)

master_df = add_asset_failure_count(
    master_df,
    asset_failure_df
)

st.subheader(
    "Feature Engineered Dataset"
)

st.dataframe(master_df.head())
st.subheader("Complaints By Tower")

tower_df = (
    master_df
    .groupby("tower_id_x")
    .size()
    .reset_index(name="complaints")
)

fig = px.bar(
    tower_df,
    x="tower_id_x",
    y="complaints"
)

st.plotly_chart(fig)
st.subheader("Complaints By Floor")

floor_df = (
    master_df
    .groupby("floor_number")
    .size()
    .reset_index(name="complaints")
)

fig = px.bar(
    floor_df,
    x="floor_number",
    y="complaints"
)

st.plotly_chart(fig)
st.subheader("Complaints By Season")

season_df = (
    master_df
    .groupby("season_flag")
    .size()
    .reset_index(name="complaints")
)

fig = px.pie(
    season_df,
    names="season_flag",
    values="complaints"
)

st.plotly_chart(fig)
st.subheader("Complaints By Category")

category_df = (
    master_df
    .groupby("category")
    .size()
    .reset_index(name="complaints")
)

fig = px.bar(
    category_df,
    x="category",
    y="complaints"
)

st.plotly_chart(fig)

master_df.to_csv(
    "data/integrated/master_dataset_featured.csv",
    index=False
)

st.success(
    "master_dataset_featured.csv saved successfully"
)