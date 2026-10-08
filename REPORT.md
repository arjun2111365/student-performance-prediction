# Student Performance Prediction

## 1. Problem Statement

The objective of this project is to predict a student's final examination score using information that is available before the final examination.

The target variable is `FinalExamScore`.


---

## 2. Dataset

The dataset contains student's academic performance and behavioral information (such as attendence).

### Target

- `FinalExamScore` — final examination score to be predicted.

### Features Used

The following features were used because they can reasonably be known before the final examination:

- `StudyHours`
- `AttendancePercentage`
- `PreviousExamScore`
- `AssignmentsCompleted`
- `SleepHours`
- `ExtracurricularHours`
- `ClassParticipation`
- `PreviousBacklogs`

### Features Excluded

#### ID

`ID` was excluded because it is an identifier and does not represent a meaningful student characteristic.

#### PostExamConfidence

`PostExamConfidence` was excluded because its name indicates that it is measured after the examination. Using it for prediction would risk target leakage because it would not be reliably available at the time the prediction is intended to be made.

---

## 3. Data Audit and Cleaning

The training dataset initially contained 1,000 rows.

The audit checked:

- Dataset shape
- Missing values
- Invalid values
- Target distribution
- Potential leakage
- Feature relationships with the target

Several input features contained missing values.

Invalid values were identified using correct ranges.For example:

- Study hours:0–24
- Attendance percentage: 0–100
- Previous exam score:0–100
- Assignment completion: 0–100
- Sleep hours:0–24
- Previous backlogs: non-negative

Invalid feature values were converted to missing values and handled using median imputation inside the machine-learning pipeline.

Two training rows contained invalid target values outside the valid score range of 0–100 and were removed.

After cleaning:

- Original training rows: 1,000
- Clean training rows: 998
- Test rows: 200

---

## 4. Data Leakage Prevention

All learned preprocessing is performed inside the scikit-learn pipeline.

Median imputation is therefore fitted only on the training portion of each cross-validation fold.

The `PostExamConfidence` feature was excluded before model training because it may represent information unavailable before the final examination.

The test dataset was never used for model selection.

---

## 5. Model Evaluation

A shuffled 5-fold cross-validation strategy was used:

- Strategy: `KFold`
- Number of folds: 5
- Shuffle: Yes
- Random seed: 42

The evaluation metric was Root Mean Squared Error (RMSE).

Lower RMSE indicates better predictive performance.

### Model Comparison

| Approach | Mean RMSE | Standard Deviation |
|---|---:|---:|
| Mean baseline | 13.989 | 1.194 |
| Ridge Regression | 7.720 | 0.741 |
| Random Forest | 7.402 | 0.817 |
| Gradient Boosting | 6.797 | 0.501 |

The mean baseline provides a reference point by predicting the average training target.

Ridge Regression substantially improves over the baseline, showing that the available features contain useful predictive information.

Random Forest performs better than Ridge Regression.

Gradient Boosting achieved the lowest mean RMSE and also had the lowest standard deviation among the evaluated models.

Therefore, Gradient Boosting was selected as the final model.

---

## 6. Final Model

The selected model is:

`GradientBoostingRegressor`

Parameters:

- `n_estimators = 200`
- `learning_rate = 0.05`
- `max_depth = 2`
- `random_state = 42`

Missing values are handled using median imputation inside the pipeline.

The final model was retrained on the complete cleaned training dataset and saved as:

`models/final_model.joblib`

---

## 7. Residual Analysis

Out-of-fold predictions were generated using the selected Gradient Boosting model.

The residual is defined as:

`Residual = Actual Score - Predicted Score`

Results:

- Mean residual: approximately `-0.014`
- Mean absolute error: approximately `5.33`
- Largest positive residual: approximately `27.82`
- Largest negative residual: approximately `-28.66`

The mean residual being close to zero suggests that there is no strong overall prediction bias.

However, some individual students have considerably larger errors. This indicates that the available pre-exam features cannot fully explain every student's final examination performance.

The residual plot showed most predictions clustered around zero residual, with some larger individual errors.

A mild pattern was observed across previous-exam-score groups: students with lower previous scores had somewhat higher average absolute errors. However, the difference was not large enough to conclude that this is a strong error segment.

The diagonal boundary visible in the residual plot is largely explained by the fact that the actual score cannot exceed 100.

---

## 8. Inference Pipeline

The project includes a reusable inference script:

`predict.py`

It:

1. Loads the saved model.
2. Reads input data from a CSV file.
3. Validates required columns.
4. Applies the same rule-based data cleaning.
5. Generates predictions.
6. Saves predictions as a CSV file.

Example:

```bash
python predict.py --input data/test.csv --output submission_from_script.csv
```

---

## 9. Submission

The final submission file is `submission.csv`.

It contains exactly two columns:

* `ID`
* `FinalExamScore`

The file contains 200 predictions with no missing values.

---

## 10. Interactive Demo

A Streamlit application is provided in `app.py`.

Users can enter a student's pre-exam information and receive a predicted final examination score.

The deployed application uses the saved model from `models/final_model.joblib`.

---

## 11. Project Structure

```text
student-performance-prediction/
├── data/
│   ├── train.csv
│   └── test.csv
├── models/
│   └── final_model.joblib
├── data_exploration.ipynb
├── train.py
├── predict.py
├── app.py
├── submission.csv
├── requirements.txt
└── REPORT.md
```

---

## 12. AI Usage Declaration

AI tools were used as a development assistant for code guidance, debugging, documentation structure, and explanation of machine-learning concepts.

The dataset analysis, feature selection, model evaluation, residual analysis, model selection, and final project decisions were performed and verified as part of the project workflow.

The final code and results were tested locally, and the deployed application was verified.
