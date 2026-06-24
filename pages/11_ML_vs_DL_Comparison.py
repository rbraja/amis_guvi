import streamlit as st
import pandas as pd

st.title(
    "ML vs DL Comparison"
)

comparison = pd.DataFrame({

    "Model":[
        "RandomForest",
        "DistilBERT"
    ],

    "Accuracy":[
        0.84,
        0.91
    ],

    "F1 Score":[
        0.83,
        0.90
    ],

    "Training Time":[
        "20 sec",
        "5 min"
    ]

})

st.dataframe(
    comparison
)

st.bar_chart(
    comparison.set_index(
        "Model"
    )[
        [
            "Accuracy",
            "F1 Score"
        ]
    ]
)