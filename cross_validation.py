from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score
import pandas as pd
from pandas import DataFrame, Series

params = {
    "random_state": 6,
}


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


def cross_validation():
    print("Loading data...")
    df = pd.read_csv("data/customer_churn.csv")

    print("Engineering features...")
    df = feature_engineering(df)

    X = df.drop(["churn"], axis=1)
    y = df["churn"]

    model = XGBClassifier(**params)
    scores = cross_val_score(model, X, y, cv=10, scoring="recall")
    print(scores)
    print(sum(scores) / len(scores))


if __name__ == "__main__":
    cross_validation()
