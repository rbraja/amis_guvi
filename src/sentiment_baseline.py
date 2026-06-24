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
    SENTIMENT_MODEL_PATH
)


def train_sentiment_model(df):

    data = df[
        [
            "feedback_text",
            "sentiment_label"
        ]
    ].dropna()

    X = data["feedback_text"]
    y = data["sentiment_label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    model = Pipeline([

        (
            "tfidf",
            TfidfVectorizer(
                max_features=5000
            )
        ),

        (
            "rf",
            RandomForestClassifier(
                n_estimators=200,
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
        SENTIMENT_MODEL_PATH
    )

    return accuracy, f1


def predict_sentiment(text):

    model = joblib.load(
        SENTIMENT_MODEL_PATH
    )

    prediction = model.predict(
        [text]
    )

    return prediction[0]