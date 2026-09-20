from app.utils.data_loader import generate_synthetic_data, preprocess_data, compute_rfm_features, compute_timeseries_features
from app.utils.models import (
    run_customer_segmentation, train_hybrid_forecaster, 
    train_churn_model, calculate_inventory_optimization, detect_data_drift
)

def test_model_inference_pipeline():
    df_raw = generate_synthetic_data(num_records=1000)
    df_clean = preprocess_data(df_raw)
    rfm = compute_rfm_features(df_clean)
    daily = compute_timeseries_features(df_clean)
    
    # 1. Segmentation
    seg_df, score = run_customer_segmentation(rfm)
    assert 'Segment' in seg_df.columns
    assert -1.0 <= score <= 1.0
    
    # 2. Forecasting
    fcst_df, mape = train_hybrid_forecaster(daily, forecast_days=14)
    assert len(fcst_df) == 14
    assert mape <= 12.0
    
    # 3. Churn
    churn_df, _, importances, auc = train_churn_model(rfm)
    assert 'ChurnRiskScore' in churn_df.columns
    assert 0.0 <= auc <= 1.0
    
    # 4. Inventory
    inv = calculate_inventory_optimization(daily, fcst_df)
    assert 'SafetyStock' in inv
    assert 'ReorderPoint' in inv
    assert 'EOQ' in inv
    
    # 5. Drift
    drift = detect_data_drift(daily.iloc[:10], daily.iloc[10:], 'Sales')
    assert 'KS_Statistic' in drift
