import argparse
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
import joblib

# Consts
CLASS_LABEL = "churn"
train_df_path = "data/train.csv.zip"
test_df_path = "data/test.csv.zip"


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


def fit_model(
    X_train: DataFrame, y_train: Series, random_state: int = 6
) -> XGBClassifier:
    model = XGBClassifier(random_state=random_state)
    model.fit(X_train, y_train)

    return model


def eval_model(model: XGBClassifier, X: DataFrame, y: Series) -> dict:
    predictions = model.predict(X)

    return {
        "accuracy": accuracy_score(y, predictions),
        "precision": precision_score(y, predictions),
        "recall": recall_score(y, predictions),
        "f1": f1_score(y, predictions),
    }


def split(random_state=6):
    print("Loading data...")
    df = pd.read_csv("data/customer_churn.csv")

    train_df, test_df = train_test_split(
        df, random_state=random_state, stratify=df["Churn"]
    )

    print("Saving split data...")
    train_df.to_csv(train_df_path)
    test_df.to_csv(test_df_path)


def train():
    print("Loading data...")
    train_df = pd.read_csv(train_df_path)
    test_df = pd.read_csv(test_df_path)

    print("Engineering features...")
    train_df = feature_engineering(train_df)
    test_df = feature_engineering(test_df)

    print("Training model...")
    X_train = train_df.drop(["churn"], axis=1)
    y_train = train_df[CLASS_LABEL]
    model = fit_model(X_train, y_train)

    print("Saving trained model...")
    joblib.dump(model, "outputs/model.joblib")

    print("Evaluating model...")

    train_metrics = eval_model(model, X_train, y_train)
    print("Train metrics:")
    print(train_metrics)

    X_test = test_df.drop(["churn"], axis=1)
    y_test = test_df[CLASS_LABEL]
    test_metrics = eval_model(model, X_test, y_test)
    print("Test metrics:")
    print(test_metrics)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(title="Split or Train step:", dest="step")
    subparsers.required = True
    split_parser = subparsers.add_parser("split")
    split_parser.set_defaults(func=split)
    train_parser = subparsers.add_parser("train")
    train_parser.set_defaults(func=train)
    parser.parse_args().func()
