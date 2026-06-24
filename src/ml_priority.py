import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.preprocessing import OneHotEncoder

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    classification_report,  
    accuracy_score
)


def train_priority_model(df):

    df = df.dropna(
        subset=[
            "priority",
            "complaint_text"
        ]
    )

    features = [

        "category",

        "subcategory",

        "season_flag",

        "complaint_hour",

        "complaint_text"

    ]

    X = df[features]

    y = df["priority"]

    preprocessor = ColumnTransformer(

        transformers=[

            (

                "text",

                TfidfVectorizer(
                    max_features=1000
                ),

                "complaint_text"

            ),

            (

                "cat",

                OneHotEncoder(
                    handle_unknown="ignore"
                ),

                [

                    "category",

                    "subcategory",

                    "season_flag"

                ]

            )

        ],

        remainder="passthrough"

    )

    model = Pipeline([

        (
            "preprocessor",
            preprocessor
        ),

        (
            "classifier",

            RandomForestClassifier(

                n_estimators=200,

                class_weight="balanced",

                random_state=42

            )

        )

    ])

    X_train, X_test, y_train, y_test = train_test_split(

        X,
        y,

        test_size=0.2,

        random_state=42,

        stratify=y

    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    report = classification_report(y_test, predictions)

    # Return trained model along with metrics so callers can use it for prediction
    return model, accuracy, report

