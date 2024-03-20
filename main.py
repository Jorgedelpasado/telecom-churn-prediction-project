import pandas as pd
from pandas import DataFrame, Series
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier


def feature_engineering(raw_df: DataFrame) -> DataFrame:
    df = raw_df.copy()
    df.drop(columns=["Status", "Age Group"], inplace=True)

    # String join is used instead of replace due to some columns having more than one blank space between words.
    cols = ["_".join(column.split()).lower() for column in df.columns]
    df.columns = cols

    # tariff_plan has binary values with 1 and 2, changing into 1 and 0
    df["tariff_plan"] = df["tariff_plan"].replace(2, 0)

    # Use average per month metrics
    df["use_per_month"] = df.seconds_of_use / df.subscription_length
    df["calls_per_month"] = df.frequency_of_use / df.subscription_length
    df["sms_per_month"] = df.frequency_of_sms / df.subscription_length
    df["dist_nums_per_month"] = df.distinct_called_numbers / df.subscription_length

    df.drop(
        columns=[
            "seconds_of_use",
            "frequency_of_use",
            "frequency_of_sms",
            "distinct_called_numbers",
        ],
        inplace=True,
    )
    return df


def fit_model(X_train: DataFrame, y_train: Series) -> XGBClassifier:
    model = XGBClassifier()
    model.fit(X_train, y_train)

    return model


def eval_model(model: XGBClassifier, X_test: DataFrame, y_test: Series) -> dict:
    predictions = model.predict(X_test)

    return {
        "accuracy": accuracy_score(predictions, y_test),
        "precision": precision_score(predictions, y_test),
        "recall": recall_score(predictions, y_test),
        "f1": f1_score(predictions, y_test),
    }


if __name__ == "__main__":
    print("Loading data...")
    df = pd.read_csv("data/customer_churn.csv")
    train_df, test_df = train_test_split(df, test_size=0.20)
    del df

    train_df = feature_engineering(train_df)
    test_df = feature_engineering(test_df)

    print("\nFitting model...")
    X_train = train_df.drop(["churn"], axis=1)
    y_train = train_df["churn"]
    model = fit_model(X_train, y_train)

    train_metrics = eval_model(model, X_train, y_train)
    print("\nTrain metrics:")
    print(train_metrics)

    X_test = test_df.drop(["churn"], axis=1)
    y_test = test_df["churn"]
    test_metrics = eval_model(model, X_test, y_test)
    print("\nTest metrics:")
    print(test_metrics)
