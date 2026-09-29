# Medical Appointment No-Show Prediction

A machine learning project that predicts whether a patient is likely to
miss a scheduled medical appointment. The project uses patient
demographics, appointment details, health indicators, and available
environmental information to estimate no-show risk.

## Project Objective

Patient no-shows can waste appointment slots and make clinic planning
more difficult. This project aims to help identify appointments that may
need additional reminders or follow-up.

The model provides a probability score for the no-show class.
Predictions should support, not replace, clinical and operational
judgment.

## Project Scope

This repository focuses on **No-Show Prediction (binary
classification)**.

-   **Target:** `no_show`
-   **Target encoding:** `0 = attended`, `1 = no-show`
-   **Model output:** predicted class and probability of no-show
-   **Evaluation:** F1-score, ROC-AUC, precision, recall, and confusion
    matrix

Demand forecasting is outside the current scope.

## Workflow

1.  Load the appointment dataset.
2.  Inspect data quality and missing values.
3.  Clean text and categorical values.
4.  Handle missing values and duplicate rows.
5.  Explore the data with EDA and visualizations.
6.  Prepare features and encode categorical variables.
7.  Split the data into training and test sets.
8.  Train classification models.
9.  Evaluate models on the held-out test set.
10. Tune the decision threshold using validation data where possible.
11. Save the trained model and use it for predictions.

## Models

Models explored in the project include:

-   Logistic Regression
-   Random Forest
-   XGBoost

### Current Evaluation Results

The following results were recorded during experimentation:

  Model                   F1-Score   ROC-AUC
  --------------------- ---------- ---------
  Logistic Regression        0.134     0.659
  Random Forest              0.287     0.741
  XGBoost                    0.384     0.751

**Current status:** XGBoost has the highest F1-score among these
recorded runs and a ROC-AUC of approximately 0.751. The F1-score target
of greater than 0.70 has not yet been achieved.

Results can vary with preprocessing, data splitting, model parameters,
and decision threshold. Evaluate the final model on an untouched test
set. Do not tune the threshold on the test set.

## Evaluation Metrics

Because the target classes are imbalanced, accuracy alone is not
sufficient.

-   **Precision:** Of appointments predicted as no-shows, how many were
    actual no-shows?
-   **Recall:** Of actual no-shows, how many did the model identify?
-   **F1-score:** Harmonic mean of precision and recall.
-   **ROC-AUC:** Measures how well the model ranks positive cases above
    negative cases across thresholds.
-   **Confusion matrix:** Shows true positives, true negatives, false
    positives, and false negatives.

Project targets:

-   F1-score greater than `0.70`
-   ROC-AUC greater than `0.75`

These are goals, not currently achieved results.

## Dataset

The project uses `Medical_appointment_data.csv` as the raw appointment
dataset.

Potential input features include patient demographics, appointment
characteristics, health indicators, reminder status, and environmental
factors, depending on which columns are available and appropriate at
prediction time.

The `no_show` column is the target and must not be included in the input
features.

## Example Project Structure

``` text
project/
├── 01_eda_and_data_cleaning.ipynb
├── 02_no_show_classification.ipynb
├── Medical_appointment_data.csv
├── clean/
│   └── appointment_cleaned.csv
├── models/
│   └── best_no_show_model.pkl
├── app.py
├── requirements.txt
└── README.md
```

Adjust the file names and folders to match the actual repository. Only
include generated files that exist in your project.

## Installation

Create and activate a virtual environment if desired, then install the
required packages:

``` bash
pip install pandas numpy scikit-learn xgboost joblib streamlit matplotlib seaborn
```

If the repository contains a `requirements.txt` file, you can instead
run:

``` bash
pip install -r requirements.txt
```

## Run the Streamlit App

If `app.py` is in the project root and the required model and data files
are available:

``` bash
streamlit run app.py
```

The terminal will show the local URL for the dashboard, commonly
`http://localhost:8501`.

## Important Implementation Notes

-   Fit preprocessing steps on training data only, then apply the fitted
    transformations to validation and test data.
-   Use a stratified split when appropriate to preserve class
    proportions. If deployment involves predicting future appointments,
    consider a time-based split to better reflect real-world use.
-   Tune thresholds using validation data, not the final test set.
-   Check for data leakage: only use information available at the time
    the prediction would be made.
-   Keep the target encoding consistent: `0 = attended`, `1 = no-show`.
-   Save the feature names and preprocessing pipeline alongside the
    model so inference uses the same inputs as training.

## Limitations and Responsible Use

Predictions are estimates, not certainties. A high predicted risk does
not mean a patient will definitely miss an appointment. Use scores to
support proportionate reminders and operational planning, and review
model performance across relevant patient groups. Protect patient
information and follow applicable privacy requirements.

## Author

**Prasant Singh**

-   GitHub: https://github.com/bUNYrAbbito
-   LinkedIn: https://www.linkedin.com/in/prasant-singh-5383b4206/
