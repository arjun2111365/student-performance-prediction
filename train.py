import os
import joblib
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import GradientBoostingRegressor


FEATURE_COLS = [
    "StudyHours",
    "AttendancePercentage",
    "PreviousExamScore",
    "AssignmentsCompleted",
    "SleepHours",
    "ExtracurricularHours",
    "ClassParticipation",
    "PreviousBacklogs"
]


def clean_data(df):
    df = df.copy()

    df.loc[
        (df["StudyHours"] < 0) | (df["StudyHours"] > 24),
        "StudyHours"
    ] = pd.NA

    df.loc[
        (df["AttendancePercentage"] < 0) | (df["AttendancePercentage"] > 100),
        "AttendancePercentage"
    ] = pd.NA

    df.loc[
        (df["PreviousExamScore"] < 0) | (df["PreviousExamScore"] > 100),
        "PreviousExamScore"
    ] = pd.NA

    df.loc[
        (df["AssignmentsCompleted"] < 0) | (df["AssignmentsCompleted"] > 100),
        "AssignmentsCompleted"
    ] = pd.NA

    df.loc[
        (df["SleepHours"] < 0) | (df["SleepHours"] > 24),
        "SleepHours"
    ] = pd.NA

    df.loc[
        df["PreviousBacklogs"] < 0,
        "PreviousBacklogs"
    ] = pd.NA

    df = df[
        (df["FinalExamScore"] >= 0) &
        (df["FinalExamScore"] <= 100)
    ].copy()

    return df


def main():
    train_df = pd.read_csv("data/train.csv")

    print("Original training rows:", len(train_df))

    train_df = clean_data(train_df)

    print("Clean training rows:", len(train_df))

    X = train_df[FEATURE_COLS]
    y = train_df["FinalExamScore"]

    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", GradientBoostingRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=2,
            random_state=42
        ))
    ])

    model.fit(X, y)

    os.makedirs("models", exist_ok=True)

    joblib.dump(model, "models/final_model.joblib")

    print("Final model trained successfully.")
    print("Model saved to models/final_model.joblib")


if __name__ == "__main__":
    main()