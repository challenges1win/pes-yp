import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

def estimate_time_to_critical(series, critical_threshold, window=24):
    """
    Estimate hours until a metric hits a critical threshold based on recent trend.
    Returns: tuple of (hours, confidence_score)
    """
    recent = series.tail(window)
    if len(recent) < 2:
        return float('inf'), 0.0
    
    # Fit linear trend to recent data
    X = np.arange(len(recent)).reshape(-1, 1)
    y = recent.values.reshape(-1, 1)
    model = LinearRegression()
    model.fit(X, y)
    
    # Calculate R² score for confidence
    from sklearn.metrics import r2_score
    y_pred = model.predict(X)
    confidence = max(0.0, min(1.0, float(r2_score(y, y_pred))))
    
    # Adjust confidence based on data variability
    std_dev = recent.std()
    mean_val = recent.mean()
    if mean_val != 0:
        cv = abs(std_dev / mean_val)  # Coefficient of variation
        if cv > 0.5:  # High variability
            confidence *= 0.7  # Reduce confidence
    
    # If trend is improving or flat, return inf
    if model.coef_[0] <= 0 and critical_threshold > recent.iloc[-1]:
        return float('inf'), confidence
    if model.coef_[0] >= 0 and critical_threshold < recent.iloc[-1]:
        return float('inf'), confidence
    
    # Calculate hours until threshold crossed
    current_val = recent.iloc[-1]
    if model.coef_[0] != 0:
        hours = abs(float(critical_threshold - current_val) / float(model.coef_[0]))
        # Further adjust confidence based on prediction length
        if hours > 168:  # If prediction is more than 1 week out
            confidence *= 0.8  # Reduce confidence for long-term predictions
        return max(0, float(hours)), confidence  # Convert to Python float
    return float('inf'), 0.0

def predict_issues(df, sensor_type):
    """
    Analyze sensor data to predict potential issues and estimate time to critical thresholds.
    Returns: dict with 'predictions' (list of issue predictions), 'urgent_hours' (hours until most urgent issue),
    'maintenance_schedule' (list of recommended maintenance tasks) and confidence scores
    """
    predictions = []
    urgent_hours = float('inf')
    maintenance_schedule = []
    
    # Get most recent readings and trends
    recent = df.tail(24)  # Last 24 hours
    latest = recent.iloc[-1]
    
    if sensor_type == 'battery_bank':
        # Voltage analysis
        voltage_trend = recent['voltage'].diff().mean()
        voltage_std = recent['voltage'].std()
        if voltage_std > 0.5:
            hours, confidence = estimate_time_to_critical(recent['voltage'], 52.0)  # Upper threshold
            predictions.append({
                'component': 'Battery Voltage',
                'issue': 'High voltage instability detected',
                'evidence': f'Voltage variation {voltage_std:.2f}V (normal < 0.5V)',
                'severity': 'High' if voltage_std > 1.0 else 'Medium',
                'hours_to_critical': hours,
                'confidence': confidence,
                'maintenance_action': 'Check battery connections and load distribution'
            })
        
        # Temperature trend
        if latest['temperature'] > 35:
            hours, confidence = estimate_time_to_critical(recent['temperature'], 45.0)  # Critical temp
            predictions.append({
                'component': 'Battery Temperature',
                'issue': 'Rising temperature trend',
                'evidence': f'Current: {latest["temperature"]:.1f}°C, Trend: {recent["temperature"].diff().mean():+.2f}°C/hour',
                'severity': 'High' if latest['temperature'] > 40 else 'Medium',
                'hours_to_critical': hours,
                'confidence': confidence,
                'maintenance_action': 'Check ventilation and cooling systems'
            })
        
        # State of Charge (SoC)
        soc_trend = recent['SoC'].diff().mean() * 24  # Daily rate
        if latest['SoC'] < 90 and soc_trend < 0:
            hours, confidence = estimate_time_to_critical(recent['SoC'], 50.0)  # Critical SoC
            predictions.append({
                'component': 'Battery Capacity',
                'issue': 'Declining charge capacity',
                'evidence': f'SoC: {latest["SoC"]:.1f}%, Trend: {soc_trend:+.1f}%/day',
                'severity': 'High' if latest['SoC'] < 60 else 'Medium',
                'hours_to_critical': hours,
                'confidence': confidence,
                'maintenance_action': 'Check charging system and battery health'
            })
    
    elif sensor_type == 'solar_inverter':
        # Efficiency analysis
        if latest['efficiency'] < 95:
            eff_trend = recent['efficiency'].diff().mean() * 24  # Daily rate
            hours, confidence = estimate_time_to_critical(recent['efficiency'], 85.0)
            predictions.append({
                'component': 'Inverter Efficiency',
                'issue': 'Degrading conversion efficiency',
                'evidence': f'Current: {latest["efficiency"]:.1f}%, Trend: {eff_trend:+.1f}%/day',
                'severity': 'High' if latest['efficiency'] < 90 else 'Medium',
                'hours_to_critical': hours,
                'confidence': confidence,
                'maintenance_action': 'Clean inverter and check MPPT settings'
            })
        
        # Temperature monitoring
        if latest['temperature'] > 50:
            hours, confidence = estimate_time_to_critical(recent['temperature'], 70.0)
            predictions.append({
                'component': 'Inverter Temperature',
                'issue': 'High operating temperature',
                'evidence': f'Current: {latest["temperature"]:.1f}°C, Threshold: 70°C',
                'severity': 'High' if latest['temperature'] > 60 else 'Medium',
                'hours_to_critical': hours,
                'confidence': confidence,
                'maintenance_action': 'Check cooling system and clean heat sinks'
            })
        
        # Power output vs input
        power_ratio = latest['output_power'] / latest['input_power'] if latest['input_power'] > 0 else 0
        if power_ratio < 0.9:
            # Calculate confidence based on recent ratio stability
            ratio_std = recent['output_power'].div(recent['input_power']).std()
            confidence = max(0.0, min(1.0, 1.0 - ratio_std))
            predictions.append({
                'component': 'Power Conversion',
                'issue': 'Poor power conversion ratio',
                'evidence': f'Conversion ratio: {power_ratio:.2f} (should be > 0.9)',
                'severity': 'High' if power_ratio < 0.8 else 'Medium',
                'hours_to_critical': 24 if power_ratio < 0.8 else 72,
                'confidence': confidence,
                'maintenance_action': 'Check inverter efficiency and connections'
            })
    
    elif sensor_type == 'transformer':
        # Temperature analysis
        if latest['temperature'] > 65:
            hours, confidence = estimate_time_to_critical(recent['temperature'], 85.0)
            predictions.append({
                'component': 'Transformer Temperature',
                'issue': 'High operating temperature',
                'evidence': f'Current: {latest["temperature"]:.1f}°C, Threshold: 85°C',
                'severity': 'High' if latest['temperature'] > 75 else 'Medium',
                'hours_to_critical': hours,
                'confidence': confidence,
                'maintenance_action': 'Check cooling system and oil levels'
            })
        
        # Vibration analysis
        if latest['vibration'] > 0.4:
            hours, confidence = estimate_time_to_critical(recent['vibration'], 0.8)
            predictions.append({
                'component': 'Transformer Vibration',
                'issue': 'Excessive vibration',
                'evidence': f'Current: {latest["vibration"]:.3f}, Threshold: 0.8',
                'severity': 'High' if latest['vibration'] > 0.6 else 'Medium',
                'hours_to_critical': hours,
                'confidence': confidence,
                'maintenance_action': 'Check mounting and core condition'
            })
        
        # Partial discharge trending
        pd_trend = recent['partial_discharge'].diff().mean() * 24
        if pd_trend > 0.1 or latest['partial_discharge'] > 5.0:
            hours, confidence = estimate_time_to_critical(recent['partial_discharge'], 8.0)
            predictions.append({
                'component': 'Insulation',
                'issue': 'Increasing partial discharge',
                'evidence': f'Current: {latest["partial_discharge"]:.1f}, Trend: {pd_trend:+.2f}/day',
                'severity': 'High' if latest['partial_discharge'] > 6.0 else 'Medium',
                'hours_to_critical': hours,
                'confidence': confidence,
                'maintenance_action': 'Inspect insulation and check for partial discharge sources'
            })
    
    # Sort predictions by urgency (hours to critical)
    predictions.sort(key=lambda x: x['hours_to_critical'])
    
    # Get most urgent timeline
    if predictions:
        urgent_hours = min(p['hours_to_critical'] for p in predictions)
    
    # Generate maintenance schedule from predictions
    maintenance_schedule = []
    for p in predictions:
        maintenance_schedule.append({
            'component': p['component'],
            'action': p['maintenance_action'],
            'urgency': 'Immediate' if p['hours_to_critical'] < 24 else 'Soon' if p['hours_to_critical'] < 72 else 'Planned',
            'deadline_hours': p['hours_to_critical'],
            'confidence': p['confidence']
        })
    
    return {
        'predictions': predictions,
        'urgent_hours': urgent_hours,
        'maintenance_schedule': maintenance_schedule,
        'overall_health': 'Critical' if urgent_hours < 24 else 'Warning' if urgent_hours < 72 else 'Good',
        'confidence_threshold': 0.7  # Minimum confidence for high-priority alerts
    }