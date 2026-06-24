import streamlit as st
import pandas as pd
import os

import sys
import os

sys.path.append(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

from src.integrator import integrate_tables

from src.data_loader import load_all_tables

st.title("Phase 2 - Data Integration")

tables = load_all_tables(
    "data/cleaned_by_you"
)

if st.button(
    "Create Master Dataset"
):

    master_df = integrate_tables(
        tables
    )

    os.makedirs(
        "data/integrated",
        exist_ok=True
    )

    master_df.to_csv(
        "data/integrated/master_dataset.csv",
        index=False
    )

    st.success(
        "Master Dataset Created"
    )

    st.write(
        f"Rows: {master_df.shape[0]}"
    )

    st.write(
        f"Columns: {master_df.shape[1]}"
    )

    st.dataframe(
        master_df.head()
    )