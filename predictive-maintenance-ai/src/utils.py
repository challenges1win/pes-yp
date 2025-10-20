# src/utils.py
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
import joblib

def add_time_features(df):
    df = df.copy()
    df['hour'] = df['time'].dt.hour
    df['dayofweek'] = df['time'].dt.dayofweek
    return df

def rolling_features(df, cols, windows=[3, 6, 24]):
    df = df.copy()
    for w in windows:
        for c in cols:
            df[f'{c}_ma_{w}'] = df[c].rolling(window=w, min_periods=1).mean()
            df[f'{c}_std_{w}'] = df[c].rolling(window=w, min_periods=1).std().fillna(0)
    return df

def preprocess(df, numeric_cols, scaler_path=None, fit_scaler=False):
    """
    - df: DataFrame with 'time' parsed as datetime
    - numeric_cols: list of columns to scale
    """
    df = df.copy()
    df = add_time_features(df)
    df = rolling_features(df, numeric_cols)
    feature_cols = [c for c in df.columns if c not in ['time', 'anomaly', 'fault_code', 'label']]
    # choose numeric features only
    X = df[feature_cols].select_dtypes(include=[np.number]).fillna(0)
    if fit_scaler:
        scaler = StandardScaler()
        Xs = scaler.fit_transform(X)
        if scaler_path:
            joblib.dump(scaler, scaler_path)
    else:
        scaler = None
        if scaler_path and os.path.exists(scaler_path):
            scaler = joblib.load(scaler_path)
            Xs = scaler.transform(X)
        else:
            Xs = X.values
    X_pre = pd.DataFrame(Xs, columns=X.columns, index=df.index)
    return df, X_pre, scaler

def append_label(store_path, row):
    """
    Append a technician label to CSV.
    row: dict-like to append
    """
    df = pd.DataFrame([row])
    if not os.path.exists(store_path):
        df.to_csv(store_path, index=False)
    else:
        df.to_csv(store_path, mode='a', header=False, index=False)

def analyze_latest(df, sensor_type):
    """Return a small analysis summary for the most recent reading.
    Returns: dict with keys: 'latest' (Series), 'concerns' (list of str), 'health' (str)
    """
    df = df.copy()
    if 'time' in df.columns:
        df = df.sort_values('time')
    latest = df.iloc[-1]
    concerns = []

    if sensor_type == 'battery_bank':
        recent = df.tail(6)
        voltage_trend = recent['voltage'].diff().mean()
        temp_trend = recent['temperature'].diff().mean()
        if abs(voltage_trend) > 0.5:
            concerns.append(f"Voltage changing rapidly ({voltage_trend:+.2f} V/hour)")
        if abs(temp_trend) > 2:
            concerns.append(f"Temperature changing rapidly ({temp_trend:+.2f} °C/hour)")
        if latest.get('SoC', 100) < 50:
            concerns.append(f"Low State of Charge ({latest.get('SoC'):.1f}%)")

    elif sensor_type == 'solar_inverter':
        recent = df.tail(6)
        eff_trend = recent['efficiency'].diff().mean() if 'efficiency' in recent.columns else 0
        if latest.get('efficiency', 100) < 90:
            concerns.append(f"Low efficiency ({latest.get('efficiency'):.1f}%)")
        if abs(eff_trend) > 1:
            concerns.append(f"Efficiency changing rapidly ({eff_trend:+.2f} %/hour)")

    elif sensor_type == 'transformer':
        recent = df.tail(6)
        temp_trend = recent['temperature'].diff().mean() if 'temperature' in recent.columns else 0
        vib_trend = recent['vibration'].diff().mean() if 'vibration' in recent.columns else 0
        if latest.get('temperature', 0) > 80:
            concerns.append(f"High temperature ({latest.get('temperature'):.1f} °C)")
        if latest.get('vibration', 0) > 0.5:
            concerns.append(f"High vibration ({latest.get('vibration'):.3f})")
        if abs(temp_trend) > 2:
            concerns.append(f"Temperature changing rapidly ({temp_trend:+.2f} °C/hour)")

    # health classification
    if len(concerns) == 0:
        health = 'Normal'
    elif len(concerns) <= 2:
        health = 'Minor Issues'
    else:
        health = 'Multiple Issues'

    return {'latest': latest, 'concerns': concerns, 'health': health}
