import os
import urllib.request
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data")
CSV_PATH = os.path.join(DATA_DIR, "online_retail.csv")
UCI_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00352/Online%20Retail.xlsx"

def generate_synthetic_data(num_records=25000):
    """Generates synthetic retail transactional dataset matching UCI Online Retail schema."""
    os.makedirs(DATA_DIR, exist_ok=True)
    np.random.seed(42)
    
    start_date = datetime(2024, 1, 1)
    customer_ids = np.random.randint(12000, 18000, size=num_records)
    stock_codes = np.random.choice([f"POST{i:04d}" for i in range(1, 60)], size=num_records)
    descriptions = np.array([f"Product Item {code[-4:]}" for code in stock_codes])
    quantities = np.random.randint(1, 25, size=num_records)
    unit_prices = np.round(np.random.uniform(1.50, 49.99, size=num_records), 2)
    countries = np.random.choice(['United Kingdom', 'Germany', 'France', 'EIRE', 'Spain', 'Netherlands'], 
                                size=num_records, p=[0.82, 0.05, 0.05, 0.03, 0.03, 0.02])
    
    dates = [start_date + timedelta(days=int(d), minutes=int(m)) for d, m in 
             zip(np.random.randint(0, 700, size=num_records), np.random.randint(0, 1440, size=num_records))]
    
    invoices = [f"58{i:04d}" for i in np.random.randint(1000, 9000, size=num_records)]
    
    df = pd.DataFrame({
        'InvoiceNo': invoices,
        'StockCode': stock_codes,
        'Description': descriptions,
        'Quantity': quantities,
        'InvoiceDate': dates,
        'UnitPrice': unit_prices,
        'CustomerID': customer_ids,
        'Country': countries
    })
    
    df.to_csv(CSV_PATH, index=False)
    print(f"[DATA PIPELINE] Synthetic dataset generated and saved to {CSV_PATH} ({len(df)} rows).")
    return df

def download_or_get_data():
    """Downloads UCI Online Retail dataset or falls back to synthetic generator."""
    os.makedirs(DATA_DIR, exist_ok=True)
    if os.path.exists(CSV_PATH):
        print(f"[DATA PIPELINE] Loading existing dataset from {CSV_PATH}")
        df = pd.read_csv(CSV_PATH)
        df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
        return df
    
    excel_path = os.path.join(DATA_DIR, "Online_Retail.xlsx")
    try:
        print("[DATA PIPELINE] Attempting to download UCI Online Retail dataset...")
        urllib.request.urlretrieve(UCI_URL, excel_path)
        df = pd.read_excel(excel_path)
        df.to_csv(CSV_PATH, index=False)
        print(f"[DATA PIPELINE] Downloaded and converted UCI dataset to {CSV_PATH}")
        return df
    except Exception as e:
        print(f"[DATA PIPELINE] Download failed ({e}). Falling back to synthetic data generator.")
        return generate_synthetic_data()

def preprocess_data(df):
    """Cleans data: removes missing CustomerIDs, cancellations, and calculates TotalAmount."""
    df_clean = df.copy()
    df_clean['InvoiceDate'] = pd.to_datetime(df_clean['InvoiceDate'])
    
    # Remove null CustomerIDs
    df_clean = df_clean.dropna(subset=['CustomerID'])
    df_clean['CustomerID'] = df_clean['CustomerID'].astype(int)
    
    # Remove cancellations (InvoiceNo starting with 'C' or negative Quantity)
    df_clean = df_clean[~df_clean['InvoiceNo'].astype(str).str.startswith('C')]
    df_clean = df_clean[(df_clean['Quantity'] > 0) & (df_clean['UnitPrice'] > 0)]
    
    # Total Transaction Value
    df_clean['TotalAmount'] = df_clean['Quantity'] * df_clean['UnitPrice']
    return df_clean

def compute_rfm_features(df_clean):
    """Computes Recency, Frequency, and Monetary (RFM) features per customer."""
    snapshot_date = df_clean['InvoiceDate'].max() + timedelta(days=1)
    
    rfm = df_clean.groupby('CustomerID').agg({
        'InvoiceDate': lambda x: (snapshot_date - x.max()).days, # Recency
        'InvoiceNo': 'nunique',                                  # Frequency
        'TotalAmount': 'sum'                                     # Monetary
    }).reset_index()
    
    rfm.columns = ['CustomerID', 'Recency', 'Frequency', 'Monetary']
    
    # Compute tenure (first purchase to snapshot)
    tenure = df_clean.groupby('CustomerID')['InvoiceDate'].agg(
        lambda x: (snapshot_date - x.min()).days
    ).reset_index().rename(columns={'InvoiceDate': 'Tenure'})
    
    rfm = rfm.merge(tenure, on='CustomerID')
    
    # Churn Definition: Customer inactive for more than 90 days
    rfm['IsChurned'] = (rfm['Recency'] > 90).astype(int)
    return rfm

def compute_timeseries_features(df_clean):
    """Aggregates daily sales and creates time-series features (Lags & Rolling Averages)."""
    daily = df_clean.groupby(df_clean['InvoiceDate'].dt.date).agg({
        'TotalAmount': 'sum',
        'Quantity': 'sum',
        'InvoiceNo': 'nunique'
    }).reset_index()
    
    daily.columns = ['Date', 'Sales', 'Quantity', 'Orders']
    daily['Date'] = pd.to_datetime(daily['Date'])
    daily = daily.sort_values('Date').reset_index(drop=True)
    
    # Lag Features
    daily['Lag_1'] = daily['Sales'].shift(1)
    daily['Lag_7'] = daily['Sales'].shift(7)
    daily['Lag_30'] = daily['Sales'].shift(30)
    
    # Rolling Averages
    daily['Rolling_7_Mean'] = daily['Sales'].rolling(window=7, min_periods=1).mean()
    daily['Rolling_30_Mean'] = daily['Sales'].rolling(window=30, min_periods=1).mean()
    daily['Rolling_7_Std'] = daily['Sales'].rolling(window=7, min_periods=1).std().fillna(0)
    
    daily = daily.fillna(0)
    return daily
