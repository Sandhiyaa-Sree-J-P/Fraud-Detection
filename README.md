# 🔍 Financial Fraud Detection System
### End-to-End Machine Learning Pipeline | 95% Model Accuracy | 6.3M+ Transactions Analyzed

---

## 🚀 Project Overview

A production-ready **fraud detection system** built on a real-world financial transaction dataset of **6,362,620 records**. This project covers the complete data science lifecycle — from raw data ingestion and exploratory analysis to feature engineering, machine learning model training, pipeline serialization, and a **live interactive web application** for real-time fraud prediction.

> 💡 Built to simulate a real fintech fraud intelligence system — the kind that flags suspicious transactions before they cause financial damage.

---

## 📊 Dataset at a Glance

| Metric | Value |
|---|---|
| Total Transactions | **6,362,620** |
| Features | **11 columns** |
| Fraudulent Transactions | **8,213** |
| Fraud Rate | **0.13%** (highly imbalanced) |
| Transaction Types | PAYMENT, TRANSFER, CASH_OUT, DEBIT, CASH_IN |
| Max Transaction Amount | **$92,445,516** |
| Median Transaction Amount | **$74,871** |

> ⚠️ The extreme **class imbalance (99.87% vs 0.13%)** is a core challenge addressed via `class_weight="balanced"` in the model.

---

## 🛠️ Tech Stack & Skills Demonstrated

### Languages & Libraries
- **Python** — core language
- **Pandas & NumPy** — data manipulation and numerical computation
- **Matplotlib & Seaborn** — data visualization and EDA
- **Scikit-learn** — machine learning, pipelines, preprocessing, evaluation
- **Joblib** — model serialization
- **Streamlit** — interactive web application deployment

### Data Science Skills
- ✅ **Exploratory Data Analysis (EDA)** — distribution analysis, fraud-by-type breakdown, time-series fraud patterns
- ✅ **Data Cleaning** — null value detection, irrelevant column removal (`step`, `nameOrig`, `nameDest`, `isFlaggedFraud`)
- ✅ **Feature Engineering** — created `balanceDiffOrig` and `balanceDiffDest` to capture account drain behavior
- ✅ **Statistical Analysis** — correlation matrix, log-scale distribution, outlier detection via boxplots
- ✅ **Handling Class Imbalance** — stratified train-test split + balanced class weights
- ✅ **ML Pipeline Design** — end-to-end `sklearn.Pipeline` with `ColumnTransformer`
- ✅ **Model Evaluation** — classification report, confusion matrix, accuracy score
- ✅ **Model Serialization** — `.pkl` export with `joblib` for production deployment
- ✅ **App Deployment** — real-time prediction UI with Streamlit

---

## 🧠 Machine Learning Pipeline

```
Raw Data (6.3M rows)
        ↓
  Data Cleaning & EDA
        ↓
  Feature Engineering
  (balanceDiffOrig, balanceDiffDest)
        ↓
  ColumnTransformer
  ├── StandardScaler  → [amount, balances]
  └── OneHotEncoder   → [transaction type]
        ↓
  Logistic Regression
  (class_weight="balanced", max_iter=1000)
        ↓
  Trained Pipeline → fraud_detection_pipeline.pkl
        ↓
  Streamlit Web App
```

---

## 📈 Model Performance

| Metric | Score |
|---|---|
| **Overall Accuracy** | **~95%** |
| Train/Test Split | 70% / 30% |
| Split Strategy | Stratified (preserves fraud ratio) |
| Algorithm | Logistic Regression |

- Evaluated using **Precision, Recall, F1-Score** via `classification_report`
- Confusion matrix generated to inspect True/False Positive rates
- Class imbalance handled explicitly to avoid the accuracy paradox

---

## 🔬 Key EDA Findings

- **Fraud is exclusive to TRANSFER and CASH_OUT** transaction types — PAYMENT, DEBIT, and CASH_IN transactions have zero fraud in this dataset
- **Account draining pattern detected**: fraudulent senders consistently show `oldbalanceOrg > 0` and `newbalanceOrig = 0` — a strong fraud signal captured via engineered features
- **Fraud peaks at specific time steps** — temporal analysis of fraud frequency over the 30-step simulation window revealed clustered fraud activity
- **Correlation analysis** confirmed `amount` and balance difference features are most correlated with `isFraud`
- Over **1M+ transactions** showed a zero-balance-after-transfer pattern, highlighting systemic fraud behavior

---

## 🌐 Streamlit Web App

A fully functional prediction app where users can input transaction details and get an instant fraud verdict.

**Input Fields:**
- Transaction Type (PAYMENT / TRANSFER / CASH_OUT / DEPOSIT)
- Transaction Amount
- Sender's Old & New Balance
- Receiver's Old & New Balance

**Output:**
- 🔴 `"This transaction can be fraud"` — if model predicts fraud
- 🟢 `"This transaction looks like it is not a fraud"` — if model predicts legitimate

---

## 📁 Project Structure

```
fraud-detection/
├── AIML_Dataset.csv              # Raw dataset (6.3M transactions)
├── Analysis_Model.ipynb          # Full EDA + model training notebook
├── fraud_detection_pipeline.pkl  # Serialized ML pipeline (production-ready)
└── Fraud_Detection.py            # Streamlit web application
```

---

## ▶️ How to Run

**1. Install dependencies**
```bash
pip install streamlit pandas scikit-learn joblib
```

**2. Launch the app**
```bash
streamlit run Fraud_Detection.py
```

**3. Or explore the notebook**
```bash
jupyter notebook Analysis_Model.ipynb
```

---

## 💼 Skills This Project Demonstrates

| Domain | Skill |
|---|---|
| Data Engineering | Large-scale CSV handling (6.3M rows), missing value analysis, column-level cleaning |
| EDA | Univariate, bivariate, and time-series analysis with Matplotlib & Seaborn |
| Feature Engineering | Domain-driven feature creation from balance differentials |
| Machine Learning | Scikit-learn Pipelines, ColumnTransformer, Logistic Regression |
| Model Evaluation | Confusion Matrix, Classification Report, Stratified Splits |
| Class Imbalance | Balanced class weights, stratified sampling |
| MLOps (Basic) | Pipeline serialization with joblib for reusable inference |
| App Development | End-to-end Streamlit UI for non-technical users |
| Communication | Documented notebook with visualizations for stakeholder storytelling |

---

*Built with Python · Scikit-learn · Streamlit · Pandas · Seaborn*
