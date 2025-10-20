import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import joblib
sys.path.insert(0, os.path.abspath('.'))

from src.utils import preprocess

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODEL_DIR = os.path.join(BASE_DIR, 'models')

def load_latest_data(sensor_type='battery_bank', n_hours=24):
    """Load the most recent n_hours of data for prediction"""
    file_map = {
        'battery_bank': 'battery_bank.csv',
        'solar_inverter': 'solar_inverter.csv',
        'transformer': 'transformer.csv'
    }
    
    file_path = os.path.join(DATA_DIR, file_map[sensor_type])
    df = pd.read_csv(file_path, parse_dates=['time'])
    
    # Get most recent data
    latest_data = df.nlargest(n_hours, 'time').reset_index(drop=True)
    return latest_data

def make_predictions(df, sensor_type='battery_bank'):
    """Make predictions using both models if available"""
    
    # Feature columns for each sensor type
    feature_cols = {
        'battery_bank': ['voltage', 'current', 'temperature', 'SoC'],
        'solar_inverter': ['input_power', 'output_power', 'efficiency', 'temperature', 'frequency'],
        'transformer': ['temperature', 'vibration', 'load_current', 'partial_discharge']
    }
    
    numeric_cols = feature_cols[sensor_type]
    
    # Preprocess data
    scaler_path = os.path.join(MODEL_DIR, f'scaler_{sensor_type}.pkl')
    if not os.path.exists(scaler_path):
        # If no scaler exists, fit a new one
        _, X_pre, _ = preprocess(df, numeric_cols, fit_scaler=True, scaler_path=scaler_path)
    else:
        _, X_pre, _ = preprocess(df, numeric_cols, scaler_path=scaler_path)
    
    results = []
    
    # 1. Isolation Forest predictions
    iso_path = os.path.join(MODEL_DIR, f'iso_model_{sensor_type}.pkl')
    
    # Always train a fresh model on the latest data
    from sklearn.ensemble import IsolationForest
    iso_model = IsolationForest(n_estimators=200, contamination=0.02, random_state=42)
    iso_model.fit(X_pre)
    joblib.dump(iso_model, iso_path)
    
    # Make predictions
    scores = -iso_model.decision_function(X_pre)  # higher = more anomalous
    is_anomaly = (scores > np.percentile(scores, 98)).astype(int)
    
    for idx, row in df.iterrows():
        results.append({
            'time': row['time'],
            'prediction_type': 'Isolation Forest',
            'is_anomaly': bool(is_anomaly[idx]),
            'anomaly_score': float(scores[idx]),
            'confidence': f"{(1 - (scores[idx] / scores.max())) * 100:.1f}%"
        })    # 2. Random Forest predictions (if available)
    rf_path = os.path.join(MODEL_DIR, 'rf_model.pkl')
    if os.path.exists(rf_path):
        rf_model = joblib.load(rf_path)
        X_rf = X_pre[numeric_cols]  # Use only the main numeric columns
        proba = rf_model.predict_proba(X_rf)
        predictions = rf_model.predict(X_rf)
        
        for idx, row in df.iterrows():
            results.append({
                'time': row['time'],
                'prediction_type': 'Random Forest',
                'is_anomaly': bool(predictions[idx]),
                'anomaly_score': float(proba[idx][1]),  # probability of anomaly
                'confidence': f"{max(proba[idx]) * 100:.1f}%"
            })
    
    return pd.DataFrame(results)

if __name__ == "__main__":
    # Make predictions for each sensor type
    for sensor in ['battery_bank', 'solar_inverter', 'transformer']:
        print(f"\n=== Predictions for {sensor} ===")
        
        # Load recent data
        data = load_latest_data(sensor)
        print(f"Loaded {len(data)} recent readings")
        
        # Make predictions
        predictions = make_predictions(data, sensor)
        if len(predictions) == 0:
            print("No models found for prediction")
            continue
            
        # Show results
        print("\nLatest predictions:")
        latest = predictions.sort_values('time', ascending=False).head()
        
        # Group by prediction type
        for pred_type in latest['prediction_type'].unique():
            print(f"\n{pred_type} Results:")
            type_preds = latest[latest['prediction_type'] == pred_type]
            for _, row in type_preds.iterrows():
                print(f"Time: {row['time']}")
                print(f"Anomaly: {'Yes' if row['is_anomaly'] else 'No'}")
                print(f"Score: {row['anomaly_score']:.3f}")
                print(f"Confidence: {row['confidence']}")
                print("---")