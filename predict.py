
import argparse
import joblib
import pandas as pd


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

    return df


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)

    args = parser.parse_args()

    model = joblib.load("models/final_model.joblib")

    df = pd.read_csv(args.input)

    required_columns = ["ID"] + FEATURE_COLS

    missing_columns = [
        col
        for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing columns: {missing_columns}"
        )

    df = clean_data(df)

    X = df[FEATURE_COLS]

    predictions = model.predict(X)

    submission = pd.DataFrame({
        "ID": df["ID"],
        "FinalExamScore": predictions
    })

    submission.to_csv(args.output, index=False)

    print(f"Predictions saved to: {args.output}")
    print(f"Rows predicted: {len(submission)}")


if __name__ == "__main__":
    main()

