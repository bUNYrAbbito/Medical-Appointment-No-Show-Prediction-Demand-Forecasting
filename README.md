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

```bash
pip install -r requirements.txt
```

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
