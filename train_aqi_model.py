import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# ----------------------------
# LOAD AQI XLS (your data)
# ----------------------------
try:
    aqi = pd.read_excel("data/noida_aqi_2024.xlsx")
except FileNotFoundError:
    print("Error: data/noida_aqi_2024.xlsx not found. Please ensure data exists.")
    exit(1)

aqi.columns = [c.strip().lower() for c in aqi.columns]
# Handle potential column name variations
if "date" in aqi.columns:
    aqi = aqi.rename(columns={"date": "timestamp"})
elif "timestamp" not in aqi.columns:
    # Fallback if no date column found, assuming first column is date
    aqi = aqi.rename(columns={aqi.columns[0]: "timestamp"})

aqi["timestamp"] = pd.to_datetime(aqi["timestamp"])
aqi = aqi.sort_values("timestamp")

# Ensure daily resolution
aqi = aqi.set_index("timestamp").resample("D").mean().reset_index()

# ----------------------------
# LOAD TRAFFIC CSV (TomTom)
# ----------------------------
try:
    traffic = pd.read_csv("data/noida_tomtom_daily.csv")
except FileNotFoundError:
    print("Error: data/noida_tomtom_daily.csv not found.")
    exit(1)

traffic["timestamp"] = pd.to_datetime(traffic["timestamp"])

# Merge AQI + Traffic
df = pd.merge(aqi, traffic, on="timestamp", how="inner")

# ----------------------------
# TIME FEATURES
# ----------------------------
df["doy"] = df["timestamp"].dt.dayofyear
df["dow"] = df["timestamp"].dt.dayofweek

df["doy_sin"] = np.sin(2 * np.pi * df["doy"] / 365)
df["doy_cos"] = np.cos(2 * np.pi * df["doy"] / 365)

# Festival feature – customize
df["is_diwali_week"] = df["timestamp"].between("2024-10-28", "2024-11-04")

# Season
def season(d):
    m = d.month
    if m in (12,1,2): return "winter"
    if m in (3,4): return "pre_summer"
    if m in (5,6): return "summer"
    if m in (7,8,9): return "monsoon"
    return "post_monsoon"

df["season"] = df["timestamp"].apply(season)
df = pd.get_dummies(df, columns=["season"], drop_first=True)

# ----------------------------
# FEATURES & TARGET
# ----------------------------
feature_cols = [
    "avg_speed",
    "jam_length_km",
    "jam_count",
    "doy_sin",
    "doy_cos",
    "dow",
    "is_diwali_week"
]

# add season one-hots
feature_cols += [c for c in df.columns if c.startswith("season_")]

# Identify target column
target_col = "aqi"
if "aqi" not in df.columns:
    if "pm25" in df.columns:
        target_col = "pm25"
    elif "pm2.5" in df.columns:
        target_col = "pm2.5"

print(f"Training on target: {target_col}")

X = df[feature_cols]
y = df[target_col]

# ----------------------------
# TRAIN / TEST SPLIT
# ----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)

# ----------------------------
# MODEL
# ----------------------------
model = RandomForestRegressor(
    n_estimators=500,
    max_depth=6,
    random_state=42
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("\n===== MODEL PERFORMANCE =====")
print("MAE:", mean_absolute_error(y_test, y_pred))
print("R2:", r2_score(y_test, y_pred))
print("==============================\n")

# ----------------------------
# SAVE MODEL
# ----------------------------
import os
os.makedirs("models", exist_ok=True)

joblib.dump(model, "models/noida_aqi_rf.pkl")
joblib.dump(feature_cols, "models/noida_feature_cols.pkl")

print("Model saved successfully!")
