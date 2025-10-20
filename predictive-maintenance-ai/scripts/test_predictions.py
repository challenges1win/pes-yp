import os
import sys
sys.path.insert(0, os.path.abspath('.'))

import pandas as pd
import numpy as np
from src.train_model import train_isolation_forest
from src.utils import preprocess

# Load Battery Bank data (it has good anomaly patterns)
print("Loading data...")
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
df = pd.read_csv(os.path.join(DATA_DIR, 'battery_bank.csv'), parse_dates=['time'])

# Train Isolation Forest
print("\nTraining Isolation Forest...")
numeric_cols = ['voltage', 'current', 'temperature', 'SoC']
iso_model = train_isolation_forest(df, numeric_cols)

# Make predictions
print("\nMaking predictions...")
_, X_pre, _ = preprocess(df, numeric_cols)
scores = -iso_model.decision_function(X_pre)  # higher = more anomalous
df['anomaly_score'] = scores
df['is_anomaly'] = (scores > np.percentile(scores, 98)).astype(int)  # top 2% are anomalies

# Show some detected anomalies
print("\nTop 5 detected anomalies:")
anomalies = df[df['is_anomaly'] == 1].sort_values('anomaly_score', ascending=False).head()
print(anomalies[['time', 'voltage', 'current', 'temperature', 'SoC', 'anomaly_score']].to_string())

# Basic stats
print(f"\nTotal records: {len(df)}")
print(f"Anomalies detected: {df['is_anomaly'].sum()} ({(df['is_anomaly'].mean()*100):.1f}%)")

# Show what makes them anomalous (compare to normal ranges)
print("\nTypical ranges vs anomaly ranges:")
normal = df[df['is_anomaly'] == 0]
anomaly = df[df['is_anomaly'] == 1]

for col in numeric_cols:
    print(f"\n{col}:")
    print(f"  Normal:  {normal[col].mean():.2f} ± {normal[col].std():.2f}")
    print(f"  Anomaly: {anomaly[col].mean():.2f} ± {anomaly[col].std():.2f}")