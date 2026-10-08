import streamlit as st
import pandas as pd
import joblib


st.title("TEST - Student Performance Prediction")
st.write(" Streamlit is executing app.py.")

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


model = joblib.load("models/final_model.joblib")


st.title("Student Performance Prediction")

st.write(
    "Predict a student's final exam score using pre-exam academic and behavioral features."
)


study_hours = st.number_input(
    "Study Hours",
    min_value=0.0,
    max_value=24.0,
    value=5.0
)

attendance = st.number_input(
    "Attendance Percentage",
    min_value=0.0,
    max_value=100.0,
    value=80.0
)

previous_score = st.number_input(
    "Previous Exam Score",
    min_value=0.0,
    max_value=100.0,
    value=70.0
)

assignments = st.number_input(
    "Assignments Completed (%)",
    min_value=0.0,
    max_value=100.0,
    value=80.0
)

sleep_hours = st.number_input(
    "Sleep Hours",
    min_value=0.0,
    max_value=24.0,
    value=7.0
)

extracurricular_hours = st.number_input(
    "Extracurricular Hours",
    min_value=0.0,
    max_value=24.0,
    value=2.0
)

participation = st.number_input(
    "Class Participation",
    min_value=0.0,
    max_value=100.0,
    value=70.0
)

backlogs = st.number_input(
    "Previous Backlogs",
    min_value=0,
    max_value=20,
    value=0
)


if st.button("Predict Final Exam Score"):

    input_data = pd.DataFrame([{
        "StudyHours": study_hours,
        "AttendancePercentage": attendance,
        "PreviousExamScore": previous_score,
        "AssignmentsCompleted": assignments,
        "SleepHours": sleep_hours,
        "ExtracurricularHours": extracurricular_hours,
        "ClassParticipation": participation,
        "PreviousBacklogs": backlogs
    }])

    prediction = model.predict(input_data)[0]

    st.success(f"Predicted Final Exam Score: {prediction:.2f}")