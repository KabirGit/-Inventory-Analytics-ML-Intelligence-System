# 📊 Inventory Analytics & ML Intelligence System

An end-to-end data analytics + machine learning project built on a unified transactional dataset.

This project demonstrates:
- 📈 Business Intelligence (Dashboarding)
- 🤖 Machine Learning (Prediction + Classification)
- 🗄️ Data Engineering (Ingestion + SQL Modeling)

---

## 📖 Overview

**Domain:** Manufacturing / Inventory Management  
**Objective:**
- Optimize inventory decisions  
- Detect risky invoices  
- Predict freight costs  
- Improve vendor performance  

---

# 🚀 Project Components

## 🔹 1. 📊 Vendor Performance Dashboard (Power BI)

A business intelligence solution to analyze:
- Profit margins  
- Inventory turnover  
- Unsold stock  
- Vendor efficiency  

### Key Features:
- Multi-million row data processing  
- KPI-driven dashboard  
- Interactive filtering & drill-down  
- Business insights using statistical analysis  

---

## 🔹 2. 🤖 Machine Learning System

Two ML models built on the same dataset:

### 🚩 Invoice Flagging (Classification)
Predicts whether an invoice is **risky or safe**

**Features:**
- invoice_quantity  
- invoice_dollars  
- total_quantity  
- total_dollars  
- average_receiving_delay  

---

### 🚚 Freight Cost Prediction (Regression)
Predicts freight cost based on:
- Dollars  

---

## 🏗️ Project Structure


my SALES ANALYSIS/
│
├── Dashboard/
│ ├── images/
│ └── Vendor PerformanceDashboard.pbix
│
├── InvoiceFlagging/
│ └── models/
│ └── predict_flag_invoice.pkl
│
├── FreightCostPrediction/
│ └── models/
│ └── predict_freight_cost_model.pkl
│
├── Inference/
│ ├── predict_flagged_invoice.py
│ └── predict_freight.py
│
├── models/
│ └── scaler.pkl
│
├── app.py # Streamlit App
├── requirements.txt
└── README.md


---

# 📥 Dataset

⚠️ **Data is NOT included in the repository**

👉 Download dataset from:

(Add your dataset link here)


After downloading:
- Place files inside a `data/` folder
- Ensure database (`inventory.db`) is created using ingestion script

---

# ⚠️ IMPORTANT: Absolute Paths

This project uses **absolute file paths**.

Before running:
- Update paths in:
  - `app.py`
  - `predict_flagged_invoice.py`
  - `predict_freight.py`

Example:
```python
MODEL_PATH = r"C:\Users\Kabir\Desktop\my SALES ANALYSIS\InvoiceFlagging\models\predict_flag_invoice.pkl"
⚙️ Setup Instructions
1. Clone Repository
git clone https://github.com/your-username/your-repo.git
cd your-repo
2. Create Virtual Environment
python -m venv venv
venv\Scripts\activate
3. Install Dependencies
pip install -r requirements.txt
4. Run ML Application
streamlit run app.py
5. Run Data Ingestion (Optional)
python Ingestion/ingestion_db.py
6. Open Dashboard

Open:

Dashboard/Vendor PerformanceDashboard.pbix

in Power BI Desktop

🧠 Machine Learning Pipeline
Data Processing

SQL aggregation (CTE queries)

Feature engineering

Label creation using business rules

Models Used

Logistic Regression (classification)

Random Forest (Optuna tuned)

Regression model for freight cost

Key Concepts Applied

Class imbalance handling (class_weight)

Feature scaling (StandardScaler)

Cross-validation & hyperparameter tuning

📊 Streamlit Application

Features:

Sidebar navigation

Invoice risk prediction

Freight cost estimation

Real-time inference

📌 Key Learnings

Debugging model collapse (single-class prediction)

Importance of train-test consistency

End-to-end ML pipeline design

Integrating ML with UI (Streamlit)

Data-driven business insights

🔮 Future Improvements

Replace absolute paths with config-based system

Deploy using FastAPI + Streamlit

Add batch prediction pipeline

Use XGBoost / LightGBM

Model monitoring & logging

👤 Author

Kabir
