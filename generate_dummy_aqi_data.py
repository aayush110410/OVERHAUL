import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Generate dates for 2024
dates = pd.date_range(start="2024-01-01", end="2024-12-31", freq="D")

# 1. Generate Synthetic AQI Data
# Winter (Nov-Jan): High AQI (250-450)
# Summer (Apr-Jun): Moderate (150-250)
# Monsoon (Jul-Sep): Low (50-100)
aqi_values = []
for d in dates:
    m = d.month
    base = 100
    if m in [11, 12, 1]: # Winter
        base = 350
    elif m in [10, 2]: # Transition
        base = 250
    elif m in [4, 5, 6]: # Summer
        base = 180
    elif m in [7, 8, 9]: # Monsoon
        base = 70
    
    noise = np.random.normal(0, 30)
    aqi_values.append(max(20, base + noise))

df_aqi = pd.DataFrame({
    "Date": dates,
    "AQI": aqi_values
})
df_aqi.to_excel("data/noida_aqi_2024.xlsx", index=False)
print("Generated data/noida_aqi_2024.xlsx")

# 2. Generate Synthetic TomTom Traffic Data
# Traffic correlates somewhat with AQI (winter inversions trap pollutants, but traffic is constant-ish)
# But let's add some variation.
avg_speeds = []
jam_lengths = []
jam_counts = []

for d in dates:
    dow = d.dayofweek
    is_weekend = dow >= 5
    
    # Weekdays have more traffic (lower speed, higher jams)
    if is_weekend:
        speed = np.random.normal(35, 5)
        jam_len = np.random.normal(2, 1)
        jam_cnt = np.random.randint(0, 5)
    else:
        speed = np.random.normal(25, 5)
        jam_len = np.random.normal(8, 3)
        jam_cnt = np.random.randint(5, 15)
        
    avg_speeds.append(max(5, speed))
    jam_lengths.append(max(0, jam_len))
    jam_counts.append(max(0, jam_cnt))

df_traffic = pd.DataFrame({
    "timestamp": dates,
    "avg_speed": avg_speeds,
    "jam_length_km": jam_lengths,
    "jam_count": jam_counts
})
df_traffic.to_csv("data/noida_tomtom_daily.csv", index=False)
print("Generated data/noida_tomtom_daily.csv")
