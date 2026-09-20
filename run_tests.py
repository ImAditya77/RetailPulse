import sys
import os

# Set python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tests.test_pipeline import test_data_generation_and_preprocessing, test_rfm_and_timeseries_features
from tests.test_models import test_model_inference_pipeline

if __name__ == "__main__":
    print("=== RUNNING RETAILPULSE UNIT TESTS ===")
    
    print("\n[1/3] Testing Data Pipeline Preprocessing...")
    test_data_generation_and_preprocessing()
    print("   -> PASSED!")
    
    print("\n[2/3] Testing RFM & Time-Series Lags Calculation...")
    test_rfm_and_timeseries_features()
    print("   -> PASSED!")
    
    print("\n[3/3] Testing ML Models (Segmentation, Forecast, Churn, Inventory, Drift)...")
    test_model_inference_pipeline()
    print("   -> PASSED!")
    
    print("\n==========================================")
    print("✨ ALL UNIT TESTS PASSED SUCCESSFULLY! ✨")
    print("==========================================")
