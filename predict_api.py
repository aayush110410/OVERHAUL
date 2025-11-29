from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import joblib
import datetime as dt
import os

app = FastAPI()

# Load model
model_path = "models/noida_aqi_rf.pkl"
cols_path = "models/noida_feature_cols.pkl"

if os.path.exists(model_path) and os.path.exists(cols_path):
    model = joblib.load(model_path)
    feature_cols = joblib.load(cols_path)
else:
    print("Warning: Model files not found. API will fail on predict.")
    model = None
    feature_cols = []

class AQIRequest(BaseModel):
    date: str
    avg_speed: float
    jam_length_km: float
    jam_count: float
    is_diwali_week: bool = False

# Helper: season flags
def season_flags(d):
    m = d.month
    return {
        "season_winter": int(m in (12,1,2)),
        "season_pre_summer": int(m in (3,4)),
        "season_summer": int(m in (5,6)),
        "season_monsoon": int(m in (7,8,9)),
        "season_post_monsoon": int(m in (10,11)),
    }

@app.post("/predict_aqi")
def predict(req: AQIRequest):
    if model is None:
        return {"error": "Model not loaded"}
        
    d = dt.datetime.fromisoformat(req.date)
    doy = d.timetuple().tm_yday
    dow = d.weekday()

    row = {
        "avg_speed": req.avg_speed,
        "jam_length_km": req.jam_length_km,
        "jam_count": req.jam_count,
        "doy_sin": np.sin(2*np.pi*doy/365),
        "doy_cos": np.cos(2*np.pi*doy/365),
        "dow": dow,
        "is_diwali_week": int(req.is_diwali_week),
        **season_flags(d)
    }

    # align columns
    x = np.array([[row.get(col, 0.0) for col in feature_cols]])
    pred = float(model.predict(x)[0])

    return {
        "predicted_aqi": pred,
        "inputs_used": row
    }
