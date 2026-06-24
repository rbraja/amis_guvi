import joblib

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    f1_score
)

from src.config import (
    CATEGORY_MODEL_PATH
)


def train_category_model(df):

    data = df[
        [
            "complaint_text",
            "category"
        ]
    ].dropna()

    X = data["complaint_text"]
    y = data["category"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    model = Pipeline([

        (
            "tfidf",
            TfidfVectorizer(
                max_features=10000
            )
        ),

        (
            "rf",
            RandomForestClassifier(
                n_estimators=300,
                random_state=42
            )
        )

    ])

    model.fit(
        X_train,
        y_train
    )

    pred = model.predict(
        X_test
    )

    accuracy = accuracy_score(
        y_test,
        pred
    )

    f1 = f1_score(
        y_test,
        pred,
        average="weighted"
    )

    joblib.dump(
        model,
        CATEGORY_MODEL_PATH
    )

    return accuracy, f1


def predict_category(text):

    model = joblib.load(
        CATEGORY_MODEL_PATH
    )

    prediction = model.predict(
        [text]
    )

    return prediction[0]