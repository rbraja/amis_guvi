import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

def train_regression_model(df):

    df = df.dropna(
        subset=[
            "resolution_time_hours"
        ]
    )

    df["target"] = np.log1p(
        df["resolution_time_hours"]
    )

    features = [

        "category",

        "priority",

        "season_flag",

        "complaint_hour",

        "asset_failure_count"

    ]

    X = pd.get_dummies(
        df[features]
    )

    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(

        X,

        y,

        test_size=0.2,

        random_state=42

    )

    model = RandomForestRegressor(n_estimators=300, random_state=42)

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    predictions = np.expm1(predictions)

    actual = np.expm1(y_test)

    mae = mean_absolute_error(actual, predictions)

    rmse = float(np.sqrt(mean_squared_error(actual, predictions)))

    r2 = r2_score(actual, predictions)

    # Return trained model and feature columns for downstream prediction
    feature_columns = X.columns.tolist()
    return model, feature_columns, mae, rmse, r2