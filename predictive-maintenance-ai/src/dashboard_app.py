# src/dashboard_app.py
import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
import joblib
from datetime import datetime
import plotly.express as px
# Ensure the project root is on sys.path so `from src...` imports work when Streamlit runs
BASE_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.dirname(BASE_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.utils import preprocess, append_label
from src.utils import analyze_latest
from src.predictive_analysis import predict_issues

DATA_DIR = os.path.join(BASE_DIR, '..', 'data')
MODEL_DIR = os.path.join(BASE_DIR, '..', 'models')
os.makedirs(MODEL_DIR, exist_ok=True)

st.set_page_config(page_title="Predictive Maintenance Dashboard", layout="wide")
st.title("⚙️ Predictive Maintenance Dashboard — Self-Learning")

@st.cache_data
def load_sensor(sensor_name):
    path_map = {
        'Battery Bank': os.path.join(DATA_DIR, 'battery_bank.csv'),
        'Solar Inverter': os.path.join(DATA_DIR, 'solar_inverter.csv'),
        'Transformer': os.path.join(DATA_DIR, 'transformer.csv')
    }
    path = path_map[sensor_name]
    df = pd.read_csv(path, parse_dates=['time'])
    return df, path

def load_iso_model(sensor_name):
    iso_path = os.path.join(MODEL_DIR, f'iso_{sensor_name.lower().replace(" ", "_")}.pkl')
    if os.path.exists(iso_path):
        return joblib.load(iso_path)
    return None

def compute_iso_scores(df, numeric_cols, iso_model):
    dfp, X_pre, _ = preprocess(df, numeric_cols, scaler_path=os.path.join(MODEL_DIR, 'scaler.pkl'))
    if iso_model:
        scores = -iso_model.decision_function(X_pre)  # higher = more anomalous
        df['iso_score'] = scores
        df['iso_anomaly'] = (scores > np.percentile(scores, 98)).astype(int)
    else:
        df['iso_score'] = np.nan
        df['iso_anomaly'] = 0
    return df

def display_summary(df):
    st.markdown("### 📊 Summary")
    cols = df.select_dtypes(include=[np.number]).columns.tolist()
    st.dataframe(df[cols].describe())

# --- Sidebar ---
st.sidebar.header("Controls")
sensor_choice = st.sidebar.selectbox("Sensor System", ['Battery Bank','Solar Inverter','Transformer'])
run_iso = st.sidebar.button("Run/Reload IsolationForest model")
show_anomalies = st.sidebar.checkbox("Highlight Anomalies", value=True)
show_predictions = st.sidebar.checkbox("Show Latest Predictions", value=True)

if show_predictions:
    st.sidebar.markdown("### 🔍 Latest Predictions")
    # Import prediction function
    sys.path.insert(0, os.path.dirname(BASE_DIR))
    from scripts.make_predictions import make_predictions, load_latest_data
    
    for sensor_type in ['battery_bank', 'solar_inverter', 'transformer']:
        data = load_latest_data(sensor_type, n_hours=24)
        predictions = make_predictions(data, sensor_type)
        if len(predictions) > 0:
            latest = predictions.sort_values('time', ascending=False).iloc[0]
            st.sidebar.markdown(f"**{sensor_type.title()}**:")
            if latest['is_anomaly']:
                st.sidebar.error(f"⚠️ Anomaly detected! (Score: {latest['anomaly_score']:.3f})")
            else:
                st.sidebar.success(f"✅ Normal operation (Score: {latest['anomaly_score']:.3f})")
            st.sidebar.caption(f"Confidence: {latest['confidence']}")

df, path = load_sensor(sensor_choice)

# Fit or load models
iso_model = load_iso_model(sensor_choice)
numeric_cols_map = {
    'Battery Bank': ['voltage', 'current', 'temperature', 'SoC'],
    'Solar Inverter': ['input_power','output_power','efficiency','temperature','frequency'],
    'Transformer': ['temperature','vibration','load_current','partial_discharge']
}
numeric_cols = numeric_cols_map[sensor_choice]

if run_iso or iso_model is None:
    st.sidebar.info("Training Isolation Forest (this may take a few seconds)...")
    # call training script programmatically
    import subprocess, sys
    # Run training as a module so Python can resolve package imports (child process sys.path)
    module_name = 'src.train_model'
    arg = sensor_choice.split()[0].lower()
    try:
        subprocess.run([sys.executable, '-m', module_name, '--sensor', arg], cwd=PROJECT_ROOT, check=True)
        iso_model = load_iso_model(sensor_choice)
        st.sidebar.success("Isolation Forest trained and loaded.")
    except Exception as e:
        st.sidebar.error(f"Training failed: {e}")

# compute iso scores
df = compute_iso_scores(df, numeric_cols, iso_model)

# Main view
st.header(f"{sensor_choice} Overview")

# Show predictions if enabled
if show_predictions:
    st.markdown("### 🤖 AI Predictions")
    pred_cols = st.columns(3)
    
    # Load predictions for current sensor
    sensor_type = sensor_choice.lower().replace(" ", "_")
    data = load_latest_data(sensor_type, n_hours=24)
    predictions = make_predictions(data, sensor_type)
    
    if len(predictions) > 0:
        latest = predictions.sort_values('time', ascending=False).iloc[0]
        
        # Anomaly Status
        with pred_cols[0]:
            st.metric("Status", 
                     "⚠️ Anomaly" if latest['is_anomaly'] else "✅ Normal",
                     f"Score: {latest['anomaly_score']:.3f}")
        
        # Confidence
        with pred_cols[1]:
            st.metric("Confidence", latest['confidence'])
        
        # Time
        with pred_cols[2]:
            st.metric("Prediction Time", latest['time'].strftime("%H:%M:%S"))
        
        # Show historical predictions
        st.markdown("#### Recent Predictions")
        hist_preds = predictions.sort_values('time', ascending=False).head()
        st.dataframe(hist_preds[['time', 'prediction_type', 'is_anomaly', 'anomaly_score', 'confidence']])

cols = st.columns((2,1))
with cols[0]:
    if sensor_choice == 'Battery Bank':
        st.subheader("Voltage")
        fig = px.line(df, x='time', y='voltage', title='Voltage Over Time', labels={'voltage':'Voltage (V)'})
        if show_anomalies and 'iso_anomaly' in df.columns:
            anomalies = df[df['iso_anomaly']==1]
            fig.add_scatter(x=anomalies['time'], y=anomalies['voltage'], mode='markers', marker={'color':'red','size':8}, name='Anomaly')
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Temperature")
        fig2 = px.line(df, x='time', y='temperature', title='Temperature Over Time')
        if show_anomalies and 'iso_anomaly' in df.columns:
            anomalies = df[df['iso_anomaly']==1]
            fig2.add_scatter(x=anomalies['time'], y=anomalies['temperature'], mode='markers', marker={'color':'red','size':8}, name='Anomaly')
        st.plotly_chart(fig2, use_container_width=True)
    elif sensor_choice == 'Solar Inverter':
        st.subheader("Input vs Output Power")
        fig = px.line(df, x='time', y=['input_power','output_power'], title='Power Over Time')
        if show_anomalies and 'iso_anomaly' in df.columns:
            anomalies = df[df['iso_anomaly']==1]
            fig.add_scatter(x=anomalies['time'], y=anomalies['output_power'], mode='markers', marker={'color':'red','size':8}, name='Anomaly')
        st.plotly_chart(fig, use_container_width=True)
        st.subheader("Efficiency")
        st.plotly_chart(px.line(df, x='time', y='efficiency', title='Efficiency Over Time'), use_container_width=True)
    else:
        st.subheader("Temperature")
        fig = px.line(df, x='time', y='temperature', title='Temperature Over Time')
        if show_anomalies and 'iso_anomaly' in df.columns:
            anomalies = df[df['iso_anomaly']==1]
            fig.add_scatter(x=anomalies['time'], y=anomalies['temperature'], mode='markers', marker={'color':'red','size':8}, name='Anomaly')
        st.plotly_chart(fig, use_container_width=True)

with cols[1]:
    st.markdown("### System Health")
    total = len(df)
    anomalies_count = int(df['iso_anomaly'].sum()) if 'iso_anomaly' in df.columns else 0
    st.metric("Total Records", total)
    st.metric("Detected Anomalies (ISO)", anomalies_count)
    if 'fault_code' in df.columns:
        faults = int(df['fault_code'].sum())
        st.metric("Fault Codes", faults)

    # System analysis (human-readable concerns)
    st.markdown("### Current Analysis")
    sensor_key = sensor_choice.lower().replace(' ', '_')
    analysis = analyze_latest(df, sensor_key)
    st.write(f"**Health:** {analysis['health']}")
    if len(analysis['concerns']) == 0:
        st.write("No immediate concerns detected.")
    else:
        for c in analysis['concerns']:
            st.write(f"- {c}")
            
    # Predictive Analysis
    st.markdown("### 🔮 Predictive Analysis")
    predictions = predict_issues(df, sensor_key)
    
    if predictions['predictions']:
        # Show time until most urgent issue
        hours_to_critical = predictions['urgent_hours']
        if hours_to_critical < float('inf'):
            if hours_to_critical < 24:
                st.error(f"⚠️ Urgent: Action needed within {hours_to_critical:.1f} hours")
            elif hours_to_critical < 72:
                st.warning(f"⚠️ Plan maintenance within {hours_to_critical/24:.1f} days")
            else:
                st.info(f"📅 Schedule check within {hours_to_critical/24:.1f} days")
        
        # Show predicted issues
        st.write("**Potential Issues:**")
        for pred in predictions['predictions']:
            confidence_color = (
                "🟢" if pred['confidence'] >= 0.8 else
                "🟡" if pred['confidence'] >= 0.6 else
                "🔴"
            )
            with st.expander(f"{confidence_color} {pred['component']}: {pred['issue']} ({pred['severity']})"):
                st.write(f"**Evidence:** {pred['evidence']}")
                if pred['hours_to_critical'] < float('inf'):
                    st.write(f"**Time to critical:** {pred['hours_to_critical']:.1f} hours")
                else:
                    st.write("**Time to critical:** No immediate risk")
                st.write(f"**Confidence Score:** {pred['confidence']*100:.1f}%")
                st.write(f"**Recommended Action:** {pred['maintenance_action']}")
        
        # Maintenance Schedule
        if predictions['maintenance_schedule']:
            st.markdown("### 📅 Maintenance Schedule")
            schedule_df = pd.DataFrame(predictions['maintenance_schedule'])
            
            # Calculate scheduled time based on deadline hours
            now = pd.Timestamp.now()
            schedule_df['scheduled_date'] = schedule_df.apply(
                lambda x: now + pd.Timedelta(hours=float(x['deadline_hours'])), axis=1
            ).dt.strftime('%Y-%m-%d %H:%M')
            
            # Sort by deadline
            schedule_df = schedule_df.sort_values('deadline_hours')
            
            # Color-code urgency
            def color_urgency(val):
                if val == 'Immediate':
                    return 'background-color: #ffcccc'
                elif val == 'Soon':
                    return 'background-color: #ffffcc'
                return ''
            
            st.write("Recommended maintenance timeline:")
            display_cols = ['component', 'action', 'urgency', 'scheduled_date', 'confidence']
            display_df = schedule_df[display_cols]
            styled_df = display_df.style.apply(lambda x: [''] * len(x) if x.name != 'urgency' else [color_urgency(v) for v in x])
            st.dataframe(styled_df)
    else:
        st.success("✅ No potential issues detected")

# Technician review panel
st.markdown("---")
st.subheader("🛠️ Technician Review & Labeling")

# show top anomalies for review (most anomalous iso_score)
if 'iso_score' in df.columns:
    top = df.sort_values('iso_score', ascending=False).head(20).reset_index(drop=True)
else:
    top = df.sample(20)

# show table for review and labeling
top['time_str'] = top['time'].dt.strftime('%Y-%m-%d %H:%M:%S')
st.write("Select an event row then add a label (1 = True Anomaly, 0 = False Alarm).")
selected_idx = st.selectbox("Select row index to label", top.index.tolist())
row = top.loc[selected_idx]

st.write("### Event snapshot")
st.table(row.to_frame().T)

with st.form("label_form"):
    label = st.radio("Label this event:", options=[1,0], index=0, format_func=lambda x: "Anomaly" if x==1 else "Normal")
    comment = st.text_area("Optional comment / technician notes", value="")
    submitted = st.form_submit_button("Submit Label")
    if submitted:
        # Save label to CSV for this sensor
        label_store = os.path.join(DATA_DIR, f'labels_{sensor_choice.lower().replace(" ", "_")}.csv')
        label_row = {
            'time': row['time'].strftime('%Y-%m-%d %H:%M:%S'),
            'sensor': sensor_choice,
            'iso_score': float(row.get('iso_score', np.nan)),
            'label': int(label),
            'comment': comment,
            'created_at': datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')
        }
        append_label(label_store, label_row)
        st.success("Label saved. It will be used for future supervised retraining.")

st.markdown("---")
st.subheader("Retraining (Self-Learning)")
st.write("When enough labeled examples are collected, a supervised model can be trained automatically to reduce false positives.")

label_store = os.path.join(DATA_DIR, f'labels_{sensor_choice.lower().replace(" ", "_")}.csv')
if os.path.exists(label_store):
    labeled = pd.read_csv(label_store, parse_dates=['time'])
    st.write(f"Collected labels: {len(labeled)}")
    if len(labeled) >= 30:  # threshold to start supervised training
        if st.button("Train supervised model from labels (RandomForest)"):
            # call training function
            import subprocess, sys
            train_script = os.path.join(BASE_DIR, 'train_model.py')
            # We will call train_supervised via a small wrapper in this repo or run as a module
            try:
                # call train_model.train_supervised programmatically
                from src.train_model import train_supervised
                # use the sensor csv path as the second arg
                clf = train_supervised(label_store, path, numeric_cols)
                if clf is not None:
                    st.success("Supervised model trained and saved to models/rf_model.pkl")
            except Exception as e:
                st.error(f"Training failed: {e}")
    else:
        st.info("Collect at least 30 labels to begin supervised training.")

st.markdown("---")
display_summary(df)
st.sidebar.caption("© 2025 Predictive Maintenance AI Team — Self-Learning Demo")
