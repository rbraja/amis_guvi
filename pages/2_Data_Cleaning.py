import streamlit as st
import pandas as pd
import os

from src.data_loader import load_all_tables
from src.cleaner import (
    fix_emails,
    fix_dates,
    fix_resolution_time,
    fix_category_codes
)

st.title("🧹 Day 2 - Data Cleaning")

tables = load_all_tables(
    "data/raw"
)

os.makedirs(
    "data/cleaned_by_you",
    exist_ok=True
)

master_log = []

if st.button("Run Cleaning"):

    for table_name, df in tables.items():
        df, log1 = fix_emails(df)
        df, log2 = fix_dates(df)
        df, log3 = fix_resolution_time(df)
        df, log4 = fix_category_codes(df)

        # Merge Logs

        for item in (log1 + log2 + log3 + log4):
            item["Table"] = table_name
            master_log.append(item)
        df.to_csv(
            f"data/cleaned_by_you/{table_name}.csv",
            index=False
        )

    log_df = pd.DataFrame(
        master_log
    )

    st.success(
        "Cleaning Completed"
    )

    st.dataframe(
        log_df,
        use_container_width=True
    )