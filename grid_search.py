import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score, recall_score
from pandas import DataFrame, Series


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


print("Loading data...")
train_df = pd.read_csv("data/train.csv.zip")
val_df = pd.read_csv("data/test.csv.zip")

print("Engineering features...")
train_df = feature_engineering(train_df)
val_df = feature_engineering(val_df)

model = XGBClassifier()

param_grid = {
    "n_estimators": [100, 200, 300],
    "max_depth": [3, 5, 7],
    "learning_rate": [0.1, 0.01, 0.001],
}

X_train = train_df.drop(["churn"], axis=1)
y_train = train_df["churn"]

X_val = val_df.drop(["churn"], axis=1)
y_val = val_df["churn"]

grid_search = GridSearchCV(
    estimator=model, param_grid=param_grid, scoring="recall", cv=3, verbose=2
)

grid_search.fit(X_train, y_train)

best_params = grid_search.best_params_
best_model = grid_search.best_estimator_

y_pred_val = best_model.predict(X_val)
recall = recall_score(y_val, y_pred_val)
accuracy = accuracy_score(y_val, y_pred_val)

print(f"recall score: {recall}")
print(f"accuracy score: {accuracy}")
print(f"best parameters: {best_params}")
