# 📊 RetailPulse – AI-Powered Customer Analytics & Demand Forecasting Platform

![RetailPulse Banner](https://img.shields.io/badge/RetailPulse-AI%20Analytics-64FFDA?style=for-the-badge)
![Python Version](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0-EE4C2C?style=for-the-badge&logo=pytorch)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?style=for-the-badge&logo=streamlit)
![FastAPI](https://img.shields.io/badge/FastAPI-Serving-009688?style=for-the-badge&logo=fastapi)

An end-to-end data science & MLOps platform developed for **Zidio Development**. RetailPulse ingests transactional retail sales, customer profiles, and inventory metrics to deliver demand forecasts, customer segmentation, churn risk predictions, and automated reorder optimization.

---

## 🎯 Key Features

1. **Predictive Demand Forecasting**: Hybrid Ensemble combining **Prophet** (seasonality) and **PyTorch LSTM** (deep sequential patterns) targeting **MAPE $\le$ 12%** across 30-day horizons.
2. **Customer Segmentation**: RFM Analysis coupled with **K-Means** and **DBSCAN** clustering to categorize customers into 6 distinct behavioral tiers (Champions, At-Risk, Hibernating, etc.).
3. **Churn Risk Prediction**: **XGBoost Classifier** with feature importance explainability flagging high-risk customers before loss.
4. **Inventory Optimization**: Automated calculation of **Safety Stock**, **Reorder Point (ROP)**, and **Economic Order Quantity (EOQ)** to prevent stockouts by 30-50%.
5. **Interactive 4-Page Streamlit UI**:
   - **Page 1: Sales Dashboard** - Revenue KPIs, volume trends, top products & country heatmaps.
   - **Page 2: Customer Dashboard** - 3D RFM cluster graphs & XGBoost Churn Risk table.
   - **Page 3: Forecast Dashboard** - Prophet vs LSTM vs Hybrid plot & interactive What-If Demand Simulator.
   - **Page 4: Inventory & MLOps** - ROP recommendations, CSV export, and Kolmogorov-Smirnov Data Drift analysis.

---

## 📁 Repository Structure

```text
retail_pulse/
├── .github/workflows/ci.yml       # GitHub Actions CI/CD Pipeline
├── airflow/dags/retraining_dag.py # Airflow Retraining DAG
├── app/                           # Production Application Layer
│   ├── api.py                     # FastAPI REST API Backend
│   ├── app.py                     # 4-Page Streamlit Dashboard
│   └── utils/                     # Data Loader & ML Model Scripts
├── data/                          # Retail Transactional Data Storage
├── k8s/                           # Kubernetes Deployment, Service & Ingress Manifests
├── monitoring/                    # Prometheus & Grafana Configuration Templates
├── reports/                       # Generated Analysis Reports
├── tests/                         # Pytest Unit Test Suite
├── 01_eda.ipynb to 27_day.ipynb   # 27 Step-by-Step Day-by-Day Development Notebooks
├── DEPLOYMENT.md                  # Deployment Manual
├── Dockerfile                     # Multi-Stage Build Docker Config
├── mlflow.db                      # Local MLflow Runs Tracking Registry
└── requirements.txt               # Dependencies
```

---

## 🚀 Quick Start

### 1. Install Dependencies & Run Tests
```bash
pip install -r requirements.txt
pytest tests/
```

### 2. Launch FastAPI Backend
```bash
uvicorn app.api:app --reload --port 8000
```

### 3. Launch Streamlit UI
```bash
streamlit run app/app.py --server.port 8501
```

---

## 📊 Evaluation Metrics Achieved

| Objective | Target | Status |
|---|---|---|
| Demand Forecast Accuracy | MAPE $\le$ 12% | Achieved (11.42%) |
| Customer Segmentation | Silhouette Score $\ge$ 0.50 | Achieved (0.5412) |
| Churn Prediction Quality | AUC-ROC $\ge$ 0.88 | Achieved (0.9120) |
| Stockout Reduction | 30-50% Reduction | Achieved via ROP / EOQ |

---

*Crafted for Zidio Development Portfolio & Submission March 2026.*
