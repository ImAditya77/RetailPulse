from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
import os
import json
import sqlite3
import pandas as pd
from app.utils.data_loader import (
    download_or_get_data, preprocess_data, 
    compute_rfm_features, compute_timeseries_features
)
from app.utils.models import (
    run_customer_segmentation, train_hybrid_forecaster,
    train_churn_model, calculate_inventory_optimization,
    detect_data_drift, MLFLOW_DB_PATH
)

app = FastAPI(
    title="RetailPulse API",
    description="AI-Powered Customer Analytics & Demand Forecasting Platform Backend API",
    version="2.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Shared cache for data and models
DATA_CACHE = {}

def get_processed_data():
    if 'raw' not in DATA_CACHE:
        df_raw = download_or_get_data()
        df_clean = preprocess_data(df_raw)
        rfm_df = compute_rfm_features(df_clean)
        daily_df = compute_timeseries_features(df_clean)
        
        DATA_CACHE['raw'] = df_raw
        DATA_CACHE['clean'] = df_clean
        DATA_CACHE['rfm'] = rfm_df
        DATA_CACHE['daily'] = daily_df
    return DATA_CACHE

@app.get("/")
@app.get("/api/health")
def health_check():
    return {
        "status": "online",
        "platform": "RetailPulse AI Engine",
        "version": "2.0",
        "domain": "Zidio Development"
    }

@app.get("/api/sales_summary")
def sales_summary():
    data = get_processed_data()
    daily = data['daily']
    clean = data['clean']
    
    total_sales = float(clean['TotalAmount'].sum())
    total_orders = int(clean['InvoiceNo'].nunique())
    total_customers = int(clean['CustomerID'].nunique())
    avg_order_value = float(total_sales / max(total_orders, 1))
    
    return {
        "total_sales": round(total_sales, 2),
        "total_orders": total_orders,
        "total_customers": total_customers,
        "avg_order_value": round(avg_order_value, 2),
        "daily_history": daily.tail(30).to_dict(orient="records")
    }

@app.get("/api/segmentation")
def customer_segmentation():
    data = get_processed_data()
    rfm = data['rfm'].copy()
    segmented_rfm, score = run_customer_segmentation(rfm)
    
    segment_counts = segmented_rfm['Segment'].value_counts().to_dict()
    sample_customers = segmented_rfm[['CustomerID', 'Recency', 'Frequency', 'Monetary', 'Segment']].head(50).to_dict(orient="records")
    
    return {
        "silhouette_score": score,
        "segment_distribution": segment_counts,
        "sample_customers": sample_customers
    }

@app.get("/api/forecast")
def demand_forecast(days: int = Query(30, ge=7, le=90)):
    data = get_processed_data()
    daily = data['daily']
    forecast_df, mape = train_hybrid_forecaster(daily, forecast_days=days)
    
    forecast_records = forecast_df.to_dict(orient="records")
    for r in forecast_records:
        r['Date'] = r['Date'].strftime('%Y-%m-%d')
        
    return {
        "mape_score": mape,
        "forecast_days": days,
        "forecast": forecast_records
    }

@app.get("/api/churn")
def churn_prediction():
    data = get_processed_data()
    rfm = data['rfm'].copy()
    churn_df, model, importances, auc = train_churn_model(rfm)
    
    at_risk = churn_df.sort_values('ChurnRiskScore', ascending=False).head(20)[
        ['CustomerID', 'Recency', 'Frequency', 'Monetary', 'ChurnRiskScore', 'IsChurned']
    ].to_dict(orient="records")
    
    return {
        "auc_score": auc,
        "at_risk_customers": at_risk,
        "feature_importances": importances.to_dict(orient="records")
    }

@app.get("/api/inventory")
def inventory_recommendations():
    data = get_processed_data()
    daily = data['daily']
    forecast_df, _ = train_hybrid_forecaster(daily, forecast_days=30)
    inventory_metrics = calculate_inventory_optimization(daily, forecast_df)
    return inventory_metrics

@app.get("/api/monitoring")
def monitoring_and_drift():
    data = get_processed_data()
    daily = data['daily']
    
    # Split daily into baseline (first half) and current (second half) to test drift
    half = len(daily) // 2
    ref_data = daily.iloc[:half]
    curr_data = daily.iloc[half:]
    
    drift_result = detect_data_drift(ref_data, curr_data, feature_col='Sales')
    
    # Fetch MLflow runs
    runs = []
    if os.path.exists(MLFLOW_DB_PATH):
        try:
            conn = sqlite3.connect(MLFLOW_DB_PATH)
            cursor = conn.cursor()
            cursor.execute('SELECT experiment, metrics, timestamp FROM mlflow_runs ORDER BY id DESC LIMIT 10')
            for row in cursor.fetchall():
                runs.append({
                    "experiment": row[0],
                    "metrics": json.loads(row[1]),
                    "timestamp": row[2]
                })
            conn.close()
        except Exception:
            pass
            
    return {
        "drift_analysis": drift_result,
        "mlflow_runs": runs
    }
