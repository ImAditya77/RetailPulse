from app.utils.data_loader import (
    generate_synthetic_data, preprocess_data, 
    compute_rfm_features, compute_timeseries_features
)

def test_data_generation_and_preprocessing():
    df_raw = generate_synthetic_data(num_records=500)
    assert len(df_raw) == 500
    assert 'CustomerID' in df_raw.columns
    
    df_clean = preprocess_data(df_raw)
    assert 'TotalAmount' in df_clean.columns
    assert (df_clean['Quantity'] > 0).all()
    assert (df_clean['UnitPrice'] > 0).all()

def test_rfm_and_timeseries_features():
    df_raw = generate_synthetic_data(num_records=500)
    df_clean = preprocess_data(df_raw)
    
    rfm = compute_rfm_features(df_clean)
    assert 'Recency' in rfm.columns
    assert 'Frequency' in rfm.columns
    assert 'Monetary' in rfm.columns
    
    daily = compute_timeseries_features(df_clean)
    assert 'Sales' in daily.columns
    assert 'Lag_1' in daily.columns
    assert 'Rolling_7_Mean' in daily.columns
