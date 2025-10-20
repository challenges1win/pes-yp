# src/train_model.py
import os
import joblib
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score
from src.utils import preprocess

MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')
os.makedirs(MODEL_DIR, exist_ok=True)

def train_isolation_forest(df, numeric_cols, iso_path=None):
    _, X_pre, _ = preprocess(df, numeric_cols, fit_scaler=True, scaler_path=os.path.join(MODEL_DIR, 'scaler.pkl'))
    iso = IsolationForest(n_estimators=200, contamination=0.02, random_state=42)
    iso.fit(X_pre)
    if iso_path is None:
        iso_path = os.path.join(MODEL_DIR, 'iso_model.pkl')
    joblib.dump(iso, iso_path)
    print(f"Isolation Forest saved to {iso_path}")
    return iso

def train_supervised(label_csv, all_sensor_csv, numeric_cols, rf_path=None):
    """
    Train a supervised classifier from labeled data file (labels supplied by technicians)
    label_csv: csv with columns -> time, sensor, index, label (1 anomaly, 0 normal), and feature snapshot columns optionally
    all_sensor_csv: path to source sensor csv (used to rebuild features if needed)
    """
    if not os.path.exists(label_csv):
        print("No labeled data found.")
        return None
    labels = pd.read_csv(label_csv, parse_dates=['time'])
    if labels.empty:
        print("Empty labeled dataset.")
        return None

    # Join labels with original sensor data to get feature columns
    df_full = pd.read_csv(all_sensor_csv, parse_dates=['time'])
    df_full = df_full.reset_index().rename(columns={'index': 'orig_index'})
    # merge on time (careful in real setup: use unique id)
    merged = pd.merge(labels, df_full, on='time', how='left', suffixes=('','_orig'))
    merged = merged.dropna(subset=['voltage'], how='all')  # adjust column check per sensor
    # Remove columns we don't want to use as features
    cols_to_drop = ['iso_score', 'orig_index', 'label', 'time', 'sensor', 'comment', 'created_at']
    feature_cols = [col for col in merged.columns if col not in cols_to_drop]
    
    # Preprocess merged to get feature matrix
    df_merged, X_pre, scaler = preprocess(merged, numeric_cols, scaler_path=os.path.join(MODEL_DIR, 'scaler.pkl'))
    
    # Only use the numeric columns we specified
    X_pre = X_pre[numeric_cols]
    y = merged['label']
    
    X_train, X_test, y_train, y_test = train_test_split(X_pre, y, test_size=0.2, random_state=42, stratify=y)
    clf = RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced')
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    print("Supervised classifier performance:")
    print(classification_report(y_test, preds))
    if rf_path is None:
        rf_path = os.path.join(MODEL_DIR, 'rf_model.pkl')
    joblib.dump(clf, rf_path)
    print(f"RandomForest saved to {rf_path}")
    return clf

if __name__ == "__main__":
    # Example usage: train iso on each sensor file separately (call from shell)
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--sensor', type=str, choices=['battery', 'solar', 'transformer'], required=True)
    args = parser.parse_args()
    if args.sensor == 'battery':
        csv = os.path.join(os.path.dirname(__file__), '..', 'data', 'battery_bank.csv')
        df = pd.read_csv(csv, parse_dates=['time'])
        numeric_cols = ['voltage', 'current', 'temperature', 'SoC']
        train_isolation_forest(df, numeric_cols)
    elif args.sensor == 'solar':
        csv = os.path.join(os.path.dirname(__file__), '..', 'data', 'solar_inverter.csv')
        df = pd.read_csv(csv, parse_dates=['time'])
        numeric_cols = ['input_power', 'output_power', 'efficiency', 'temperature', 'frequency']
        train_isolation_forest(df, numeric_cols)
    elif args.sensor == 'transformer':
        csv = os.path.join(os.path.dirname(__file__), '..', 'data', 'transformer.csv')
        df = pd.read_csv(csv, parse_dates=['time'])
        numeric_cols = ['temperature', 'vibration', 'load_current', 'partial_discharge']
        train_isolation_forest(df, numeric_cols)
