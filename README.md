<<<<<<< HEAD
# Medical Appointment No-Show Prediction & Demand Forecasting

This project builds a healthcare operations dashboard that combines two predictive tasks:

- patient-level no-show risk prediction
- daily clinic appointment demand forecasting

It is designed for the University of Vale do Itajaí Center of Specialization in Physical and Intellectual Rehabilitation (CER), located in southern Brazil.

The application is implemented as a Streamlit dashboard and uses cleaned appointment data, trained ML models, and business-focused analytics to support planning decisions for scheduling, staffing, and reminder strategy.

## Project overview

Healthcare providers often face two operational challenges:

- some patients do not attend scheduled visits
- total appointment demand varies by day and season

This project helps address both issues by:

1. predicting whether a given patient is likely to miss an appointment
2. forecasting the number of appointments expected on future dates
3. summarizing no-show and attendance trends for operational decision-making

## Key features

### 1. No-show risk prediction

- predicts attendance risk for an individual patient
- uses patient and appointment information such as age, specialty, shift, reminder status, and health indicators
- outputs a probability score and a risk recommendation

### 2. Demand forecasting

- aggregates appointments into daily counts
- generates a short-horizon forecast for future dates
- helps estimate staffing and daily resource needs

### 3. Business analytics dashboard

- no-show rate by weekday
- no-show rate by appointment hour
- specialty breakdown
- overall clinic metrics and operational insights

## Tech stack

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- Joblib
- Streamlit
- Matplotlib
- Seaborn

## Repository structure

```text
M/
├── 01_eda_and_data_cleaning.ipynb
├── 02_no_show_classification.ipynb
├── 03_demand_forecasting.ipynb
├── app.py
├── Medical_appointment_data.csv
├── clean/
│   └── appointment_cleaned.csv
├── models/
│   ├── best_no_show_model.pkl
│   └── best_demand_forecasting_model.pkl
├── README.md
└── .gitignore
```

## Data and model artifacts

- `Medical_appointment_data.csv` — raw appointment dataset
- `clean/appointment_cleaned.csv` — processed data used by the dashboard
- `models/best_no_show_model.pkl` — trained no-show classification model and metadata
- `models/best_demand_forecasting_model.pkl` — trained demand forecasting model

## How the app works

The dashboard consists of three tabs:

### No-show prediction

Users enter patient and appointment details such as:

- gender
- age
- specialty
- disability type
- appointment date and time
- shift
- rain and heat conditions
- SMS reminder status

The model outputs a probability score and classifies the case as low or high risk.

### Demand forecasting

The app aggregates daily appointment counts and creates a future demand forecast based on recent historical patterns. It then visualizes the forecasted patient volume for a selected future horizon.

### Business insights

This section summarizes clinic operations including:

- attendance trends by day of the week
- no-show patterns by time of day
- variation across specialties
- key operational recommendations

## Setup

### 1. Clone the repository

```bash
git clone <your-repository-url>
cd M
```

### 2. Create a virtual environment

Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

macOS/Linux:

```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install pandas numpy scikit-learn xgboost joblib streamlit matplotlib seaborn
```

If a `requirements.txt` file is added later, you can use:
=======
# Medical Appointment Prediction System

A comprehensive machine learning project that predicts patient no-shows and forecasts appointment demand using medical appointment data.

## 📋 Project Overview

This project consists of an end-to-end ML pipeline with:
- **Exploratory Data Analysis (EDA)**: Understanding data patterns and distributions
- **Data Preprocessing & Feature Engineering**: Preparing data for model training
- **No-Show Prediction Model**: Classifying patients likely to miss appointments
- **Demand Forecasting Model**: Predicting appointment volume
- **Interactive Web UI**: Streamlit-based web application for predictions

## 📁 Project Structure

```
ml/
├── README.md                                          # Project documentation
├── requirements.txt                                   # Python dependencies
├── Medical_appointment_data.csv                       # Raw appointment data
├── clean_data.csv                                     # Processed data
├── cols.txt                                           # Feature column names
│
├── notebooks/                                         # Jupyter notebooks
│   ├── 01_EDA.ipynb                                  # Exploratory Data Analysis
│   ├── 02_Preprocessing + Feature Engineering.ipynb  # Data preprocessing & features
│   ├── 03_No-Show Model Training.ipynb               # Model training & evaluation
│   └── 04_Demand Forecasting.ipynb                   # Demand prediction model
│
├── models/                                            # Trained ML models
│   ├── no_show_model.pkl                             # No-show prediction model
│   └── demand_forecast_model.pkl                     # Demand forecasting model
│
└── app/                                               # Streamlit web application
    └── app.py                                        # UI application code

```

## 📚 Notebooks

### 01_EDA.ipynb - Exploratory Data Analysis
Analyzes the medical appointment dataset to understand:
- Data distribution and summary statistics
- Missing values and data quality
- Correlations between features
- Patient demographics and appointment patterns
- No-show rate analysis

**Output**: Data insights for preprocessing decisions

---

### 02_Preprocessing + Feature Engineering.ipynb
Transforms raw data into ML-ready format:
- **Data Cleaning**: Handling missing values, duplicates, outliers
- **Feature Engineering**: 
  - Extract temporal features (day, month, weekday)
  - Patient health indicators standardization
  - SMS notification impact analysis
  - Feature scaling and normalization
- **Data Splitting**: Train/test/validation splits
- **Results**: `clean_data.csv` with engineered features

---

### 03_No-Show Model Training.ipynb
Builds and evaluates the no-show prediction model:
- **Target Variable**: `No_show` (1=will miss, 0=will attend)
- **Input Features**:
  - `age`: Patient age
  - `SMS_received`: Whether SMS reminder was sent (0/1)
  - `Hipertension`: Patient hypertension status (0/1)
  - `Diabetes`: Patient diabetes status (0/1)
  - `day`: Appointment day of month
  - `month`: Appointment month
  - `weekday`: Appointment day of week (0=Monday, 6=Sunday)

- **Models Trained**:
  - Scikit-learn classifiers
  - XGBoost
  - LightGBM
  
- **Model Evaluation**:
  - Accuracy, Precision, Recall, F1-Score
  - Confusion Matrix
  - ROC-AUC curves
  
- **Output**: `models/no_show_model.pkl` (best performing model)

---

### 04_Demand Forecasting.ipynb
Develops appointment demand prediction model:
- **Target Variable**: Number of expected appointments
- **Input Features**:
  - `day`: Day of month
  - `month`: Month
  - `weekday`: Day of week

- **Approach**: Time-series/regression forecasting
- **Models**: Linear regression, ensemble methods
- **Output**: `models/demand_forecast_model.pkl`

---

## 🤖 Models

### no_show_model.pkl
**Purpose**: Predicts probability of patient no-show  
**Model Type**: Classification (Binary)  
**Features**: 7 features (age, SMS, health conditions, temporal features)  
**Output**: 0 (will attend) or 1 (will miss appointment)  

### demand_forecast_model.pkl
**Purpose**: Forecasts expected appointment volume  
**Model Type**: Regression  
**Features**: 3 features (day, month, weekday)  
**Output**: Integer (expected number of appointments)

---

## 🎨 Web Application (Streamlit)

File: `app/app.py`

### Features

#### Tab 1: No-Show Prediction
Predicts whether a patient will miss their appointment.

**Input Parameters**:
- Age (1-100 years)
- SMS Received (Yes/No)
- Hypertension (Yes/No)
- Diabetes (Yes/No)
- Appointment Day (1-31)
- Appointment Month (1-12)
- Appointment Weekday (0=Monday, 6=Sunday)

**Output**:
- 🔴 **High Risk**: Patient may MISS appointment
- 🟢 **Low Risk**: Patient likely to ATTEND

#### Tab 2: Demand Forecast
Predicts appointment demand for a specific date.

**Input Parameters**:
- Day (1-31)
- Month (1-12)
- Weekday (0=Monday, 6=Sunday)

**Output**:
- Expected number of appointments for the selected date

---

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.7+
- pip or conda

### 1. Install Dependencies
>>>>>>> e847a0bb8613dbdad42a0dd99149144f2705a268

```bash
pip install -r requirements.txt
```

<<<<<<< HEAD
## Run the application

From the project root:

```bash
streamlit run app.py
```

Then open the local URL shown in the terminal, usually:

```text
http://localhost:8501
```

## Notes

- The dashboard expects the cleaned data file at `clean/appointment_cleaned.csv`.
- The trained model files must exist under the `models/` folder.
- If these files are missing, retrain the models from the notebook pipeline or regenerate them in your environment.

## Project intent

This project is intended to support better healthcare scheduling and resource planning through data-driven decision support. The predictions should be reviewed in context and used responsibly alongside operational judgment.

## Author

Prasant Singh

- GitHub: https://github.com/bUNYrAbbito
- LinkedIn: https://www.linkedin.com/in/prasant-singh-5383b4206/
=======
**Required Packages**:
```
pandas          - Data manipulation
numpy           - Numerical computing
scikit-learn    - Machine learning
xgboost         - Gradient boosting
lightgbm        - Light gradient boosting
streamlit       - Web UI framework
matplotlib      - Plotting
seaborn         - Statistical visualization
joblib          - Model serialization
```

### 2. Run the Streamlit App

```bash
streamlit run app/app.py
```

The app will launch at `http://localhost:8501`

---

## 📊 Data Dictionary

### Medical_appointment_data.csv
Raw input dataset with medical appointment records

### clean_data.csv
Processed dataset after preprocessing and feature engineering

**Key Features**:
- `age`: Patient age in years
- `SMS_received`: Binary (0/1) - whether SMS reminder sent
- `Hipertension`: Binary (0/1) - patient hypertension status
- `Diabetes`: Binary (0/1) - patient diabetes status
- `day`: Calendar day (1-31)
- `month`: Calendar month (1-12)
- `weekday`: Day of week (0-6, 0=Monday)
- `No_show`: Target variable (0=attended, 1=no-show)

---

## 🚀 Usage Workflow

```
1. Raw Data
   ↓
2. Run 01_EDA.ipynb
   (Explore patterns)
   ↓
3. Run 02_Preprocessing + Feature Engineering.ipynb
   (Clean & engineer features) → clean_data.csv
   ↓
4. Run 03_No-Show Model Training.ipynb
   (Train classifier) → no_show_model.pkl
   ↓
5. Run 04_Demand Forecasting.ipynb
   (Train regressor) → demand_forecast_model.pkl
   ↓
6. Run Streamlit App
   streamlit run app/app.py
   ↓
7. Make Predictions via Web UI
```

---

## 🔍 Model Performance Metrics

The models are evaluated using:
- **Classification (No-Show)**:
  - Accuracy
  - Precision
  - Recall
  - F1-Score
  - ROC-AUC

- **Regression (Demand)**:
  - Mean Absolute Error (MAE)
  - Root Mean Squared Error (RMSE)
  - R² Score

---

## 📝 File Descriptions

| File | Purpose |
|------|---------|
| `Medical_appointment_data.csv` | Original dataset |
| `clean_data.csv` | Preprocessed data for modeling |
| `cols.txt` | Feature column names |
| `requirements.txt` | Python package dependencies |
| `01_EDA.ipynb` | Data exploration notebook |
| `02_Preprocessing + Feature Engineering.ipynb` | Data preparation notebook |
| `03_No-Show Model Training.ipynb` | Model development notebook |
| `04_Demand Forecasting.ipynb` | Forecasting model notebook |
| `no_show_model.pkl` | Trained classification model |
| `demand_forecast_model.pkl` | Trained regression model |
| `app.py` | Streamlit application |

---

## ⚙️ Running Individual Notebooks

Each notebook is self-contained and can be run independently after dependencies are installed:

```bash
# Navigate to notebooks directory
cd notebooks

# Open with Jupyter Lab
jupyter lab

# Or use VS Code with appropriate notebook extension
```

---

## 🎯 Key Insights

- **No-Show Prediction**: Identifies at-risk patients to reduce appointment wastage
- **Demand Forecasting**: Optimizes clinic resources based on expected appointment volume
- **Feature Importance**: Health conditions and temporal factors significantly impact predictions
- **Decision Support**: Enables proactive patient management strategies

---

## 💡 Future Enhancements

- [ ] Add more patient features (location, appointment type, etc.)
- [ ] Implement time-series cross-validation
- [ ] Deploy app to cloud platform (AWS, GCP, Azure)
- [ ] Add model interpretability (SHAP, LIME)
- [ ] Implement A/B testing for SMS campaigns
- [ ] Create prediction confidence intervals
- [ ] Build data pipeline for continuous model retraining

---

## 📞 Support

For issues or questions about the project, refer to the individual notebook documentation.

---

## 📄 License

This project is for educational and research purposes.

---

**Last Updated**: February 2026  
**Python Version**: 3.7+  
**Framework**: Streamlit, Scikit-learn, XGBoost, LightGBM
>>>>>>> e847a0bb8613dbdad42a0dd99149144f2705a268
