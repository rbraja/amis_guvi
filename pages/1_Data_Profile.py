import streamlit as st
import pandas as pd
import os

from src.data_loader import load_all_tables
from src.profiler import profile_tables

st.title("📊 Day 1 - Data Profiling")

tables = load_all_tables(
    "data/raw"
)

if len(tables) == 0:

    st.warning(
        "No CSV files found in data/raw"
    )

    st.stop()

profile_df = profile_tables(tables)
st.subheader("Dataset Summary")

st.dataframe(
    profile_df,
    use_container_width=True
)

# Save Report

csv = profile_df.to_csv(index=False)

st.download_button(
    "Download Profile Report",
    csv,
    "data_profile.csv",
    "text/csv"
)
selected_table = st.selectbox(

    "Select Table",

    list(tables.keys())

)

df = tables[selected_table]

# Preview

st.subheader(
    f"Preview - {selected_table}"
)

st.dataframe(
    df.head()
)

# Column Details

column_df = pd.DataFrame({

    "Column Name":
        df.columns,

    "Data Type":
        df.dtypes.astype(str),

    "Missing Values":
        df.isnull().sum().values

})

st.subheader(
    "Column Analysis"
)

st.dataframe(
    column_df,
    use_container_width=True
)

