import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime
sys.path.insert(0, os.path.abspath('.'))

# Load the data
data_dir = os.path.join('data')
sensors = {
    'battery_bank': pd.read_csv(os.path.join(data_dir, 'battery_bank.csv'), parse_dates=['time']),
    'solar_inverter': pd.read_csv(os.path.join(data_dir, 'solar_inverter.csv'), parse_dates=['time']),
    'transformer': pd.read_csv(os.path.join(data_dir, 'transformer.csv'), parse_dates=['time'])
}

print("=== Current System Status Analysis ===")
print(f"Analysis Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

# Analyze each sensor
for sensor_name, df in sensors.items():
    print(f"\n{sensor_name.upper()} ANALYSIS:")
    latest = df.iloc[-1]  # Get most recent reading
    
    if sensor_name == 'battery_bank':
        print(f"Latest Reading Time: {latest['time']}")
        print(f"Voltage: {latest['voltage']:.2f}V")
        print(f"Current: {latest['current']:.2f}A")
        print(f"Temperature: {latest['temperature']:.2f}°C")
        print(f"State of Charge: {latest['SoC']:.1f}%")
        
        # Analyze trends
        recent = df.tail(6)  # Last 6 hours
        voltage_trend = recent['voltage'].diff().mean()
        temp_trend = recent['temperature'].diff().mean()
        
        # Report concerns
        concerns = []
        if abs(voltage_trend) > 0.5:
            concerns.append(f"Voltage changing rapidly ({voltage_trend:+.2f}V/hour)")
        if abs(temp_trend) > 2:
            concerns.append(f"Temperature changing rapidly ({temp_trend:+.2f}°C/hour)")
        if latest['SoC'] < 50:
            concerns.append(f"Low State of Charge ({latest['SoC']:.1f}%)")
        
        if concerns:
            print("\nConcerns Detected:")
            for c in concerns:
                print(f"- {c}")
            
    elif sensor_name == 'solar_inverter':
        print(f"Latest Reading Time: {latest['time']}")
        print(f"Input Power: {latest['input_power']:.2f}W")
        print(f"Output Power: {latest['output_power']:.2f}W")
        print(f"Efficiency: {latest['efficiency']:.1f}%")
        print(f"Temperature: {latest['temperature']:.2f}°C")
        
        # Calculate efficiency trend
        recent = df.tail(6)
        eff_trend = recent['efficiency'].diff().mean()
        
        # Report concerns
        concerns = []
        if latest['efficiency'] < 90:
            concerns.append(f"Low efficiency ({latest['efficiency']:.1f}%)")
        if abs(eff_trend) > 1:
            concerns.append(f"Efficiency changing rapidly ({eff_trend:+.2f}%/hour)")
        
        if concerns:
            print("\nConcerns Detected:")
            for c in concerns:
                print(f"- {c}")
                
    else:  # transformer
        print(f"Latest Reading Time: {latest['time']}")
        print(f"Temperature: {latest['temperature']:.2f}°C")
        print(f"Vibration: {latest['vibration']:.3f}")
        print(f"Load Current: {latest['load_current']:.2f}A")
        print(f"Partial Discharge: {latest['partial_discharge']:.2f}")
        
        # Analyze trends
        recent = df.tail(6)
        temp_trend = recent['temperature'].diff().mean()
        vib_trend = recent['vibration'].diff().mean()
        
        # Report concerns
        concerns = []
        if latest['temperature'] > 80:
            concerns.append(f"High temperature ({latest['temperature']:.1f}°C)")
        if latest['vibration'] > 0.5:
            concerns.append(f"High vibration ({latest['vibration']:.3f})")
        if abs(temp_trend) > 2:
            concerns.append(f"Temperature changing rapidly ({temp_trend:+.2f}°C/hour)")
            
        if concerns:
            print("\nConcerns Detected:")
            for c in concerns:
                print(f"- {c}")
    
    print("\nHealth Assessment:", end=" ")
    if len(concerns) == 0:
        print("✅ Normal Operation")
    elif len(concerns) <= 2:
        print("⚠️ Minor Issues Detected")
    else:
        print("🚨 Multiple Issues Detected")