import streamlit as st
import pandas as pd
import os

from src.data_loader import load_all_tables

from src.validator import (
    compare_row_counts,
    validate_targets
)

st.title(
    "Day 3 - Validation"
)

raw_tables = load_all_tables(
    "data/raw"
)

cleaned_tables = load_all_tables(
    "data/cleaned_by_you"
)

row_df = compare_row_counts(
    raw_tables,
    cleaned_tables
)

st.subheader(
    "Raw vs Cleaned"
)

st.dataframe(row_df)

expected_rows = {

    "residents":6500,

    "apartments":2400,

    "maintenance_requests":21600

}

target_df = validate_targets(
    cleaned_tables,
    expected_rows
)

st.subheader(
    "Assignment Validation"
)

st.dataframe(target_df)

target_df.to_csv(
    "reports/validation_report.csv",
    index=False
)

if os.path.exists(
    "reports/cleaning_log.csv"
):

    log_df = pd.read_csv(
        "reports/cleaning_log.csv"
    )

    st.subheader(
        "Cleaning Log"
    )

    st.dataframe(log_df)