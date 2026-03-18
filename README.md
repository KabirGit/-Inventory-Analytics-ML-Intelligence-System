# 📊 Inventory Analytics & ML Intelligence System


An end-to-end data analytics and machine learning system designed to **analyze vendor performance, detect risky invoices, and predict freight costs**. This project combines **data engineering, statistical analysis, machine learning, and dashboarding** to simulate a real-world business intelligence workflow.

> 📌 This is a personal portfolio project intended for demonstration and learning purposes.

---

## 📖 Overview

This project addresses multiple business challenges in supply chain, finance, and vendor management through a unified pipeline:

### 🎯 Objectives

* 🚚 Predict freight costs for better budgeting and planning
* 🚩 Detect potentially risky or anomalous invoices
* 📦 Identify inventory inefficiencies and unsold stock
* 📊 Analyze vendor performance and profitability
* 📉 Support data-driven decision-making through dashboards

---

## 🚀 Key Features

### 1. 📥 Data Ingestion & Processing

* Loads multi-source transactional data into a SQL database
* Handles large datasets (inventory, purchases, sales, invoices)
* Maintains structured data pipelines for scalability

### 2. 🔄 Data Transformation & Feature Engineering

* SQL + Python-based transformations
* Aggregated vendor-level and invoice-level features
* Creation of derived metrics such as:

  * Total quantity & dollars
  * Average receiving delay
  * Invoice vs purchase discrepancies

### 3. 🤖 Machine Learning Models

#### 🚚 Freight Cost Prediction (Regression)

* Linear Regression
* Decision Tree Regressor
* Random Forest Regressor

#### 🚩 Invoice Risk Flagging (Classification)

* Logistic Regression
* Random Forest Classifier (with hyperparameter tuning using Optuna)

---

### 4. 📊 Model Evaluation

* **Regression Metrics:**

  * MAE (Mean Absolute Error)
  * MSE (Mean Squared Error)
  * R² Score

* **Classification Metrics:**

  * Accuracy, Precision, Recall
  * Classification Report

---

### 5. ⚙️ Model Selection & Persistence

* Automatically selects best-performing model
* Saves models and scalers using `joblib`
* Ensures reproducibility and reuse

---

### 6. 📈 Interactive Applications & Dashboard

#### 🌐 Streamlit Application

* Real-time predictions:

  * Freight cost estimation
  * Invoice risk classification

#### 📊 Power BI Dashboard

* Visualizes key business insights:

  * Vendor performance
  * Profit margins
  * Stock-to-sales ratios
  * Inventory inefficiencies
  * Reorder insights

* Includes:

  * KPI cards
  * Interactive charts
  * Drill-down analysis

---

## 🛠️ Tech Stack

### 🔹 Programming & Libraries

* Python 3.x
* pandas, numpy
* scikit-learn
* optuna
* joblib

### 🔹 Data & Storage

* SQLite / PostgreSQL
* SQL (for transformations & aggregations)

### 🔹 Analytics & Visualization

* Jupyter Notebook (EDA & statistical analysis)
* matplotlib, seaborn
* Power BI (dashboarding)

### 🔹 Deployment Interface

* Streamlit

---

## ⚙️ Setup & Installation

### 1. Clone the Repository

```bash
git clone <your-repo-link>
cd <your-repo-name>
```

---

### 2. Create Virtual Environment

```bash
python -m venv venv
```

Activate:

* Windows:

```bash
venv\Scripts\activate
```

* macOS/Linux:

```bash
source venv/bin/activate
```

---

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ How to Run

### 1. Train ML Models

```bash
python train.py
```

---

### 2. Run Streamlit Application

```bash
streamlit run app.py
```

---

### 3. View Power BI Dashboard

* Open **Power BI Desktop**
* Load the `.pbix` file from the Dashboard folder

---

## ⚠️ Important Note

* Ensure you **update all file paths (models, scaler, database)** to **absolute paths** before running the project.
* This is critical for:

  * Model loading
  * Streamlit app execution
  * Cross-environment compatibility

---

## 📌 Notes

* Designed as a **portfolio-level end-to-end data project**
* Demonstrates integration of:

  * Data Engineering
  * Machine Learning
  * Business Intelligence
* Not intended for production deployment

---

## 🔄 Workflow / Pipeline

This project follows a structured **end-to-end data pipeline**, integrating data engineering, machine learning, and business intelligence.

---

### 1. 📥 Data Ingestion

* Raw CSV files are ingested into a SQL database
* Ensures centralized and query-efficient storage
* Logging mechanisms provide traceability of ETL operations

---

### 2. 🔄 Data Transformation & Feature Engineering

* SQL-based aggregations combined with Python processing

* Creation of derived features such as:

  * Total quantity & total dollars
  * Invoice vs purchase discrepancies
  * Average receiving delay

* Clean datasets prepared for:

  * Machine Learning models
  * Dashboard consumption

---

### 3. 🤖 Machine Learning Pipeline

#### 🚚 Freight Cost Prediction

* **Input:** Dollars
* **Output:** Predicted freight cost
* Uses regression models to estimate logistics cost

---

#### 🚩 Invoice Risk Flagging

* **Inputs:**

  * invoice_quantity
  * invoice_dollars
  * total_quantity
  * total_dollars
  * average_receiving_delay

* **Output:**

  * `0 → Safe`
  * `1 → Risky`

* Identifies anomalies and potential inconsistencies in invoices

---

### 4. 📊 Model Training & Evaluation

* Train-test split applied to datasets
* Multiple models trained and benchmarked

**Evaluation Metrics:**

* Regression → MAE, MSE, R²

* Classification → Accuracy, Precision, Recall

* Best-performing model is selected automatically

---

### 5. ⚙️ Model Persistence & Deployment

* Models and scalers saved using `joblib`
* Ensures reusability without retraining
* Integrated into Streamlit for real-time inference

---

### 6. 📈 Business Intelligence Layer

* Processed data visualized in Power BI

* Provides insights into:

  * Vendor performance
  * Profitability trends
  * Inventory inefficiencies
  * Stock-to-sales behavior

* Enables data-driven decision-making

---

## 📊 Dashboard Preview

![Dashboard Preview](Dashboard/images/Dashboard_preview.png)

---

## 📌 Usage

### 👉 Freight Prediction

1. Launch the Streamlit app
2. Input the **Dollars value**
3. View predicted freight cost instantly

---

### 👉 Invoice Risk Detection

1. Provide invoice-related inputs
2. Run prediction
3. System classifies invoice as:

   * ✅ Safe
   * ⚠️ Risky

---

## ⚠️ Best Practices

* Ensure **feature order consistency** during inference
* Always use the **same scaler used during training**
* Keep preprocessing and model pipelines aligned
* Prefer configuration files over hardcoded values for scalability

---

## 🚀 Future Improvements

* 🔧 Replace absolute paths with environment-based configuration (`.env` or config files)
* ☁️ Deploy models using cloud services (AWS / GCP / Azure)
* 🔄 Implement real-time data pipelines using streaming tools (Kafka, Airflow)
* 📦 Containerize the application using Docker
* 🧠 Enhance models with advanced techniques (XGBoost, LightGBM)
* 📊 Integrate dashboard with live database instead of static extracts

---

## 🤝 Contributing

Contributions are welcome!

If you'd like to improve this project:

* Fork the repository
* Create a feature branch
* Submit a pull request

---

## 📬 Contact

For queries, feedback, or collaboration:

* GitHub: https://github.com/KabirGit

---

## 📌 Disclaimer

This project is created for **educational and portfolio purposes only**.
It does not represent production-grade deployment or real business data.

---
