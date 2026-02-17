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

```bash
pip install -r requirements.txt
```

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
