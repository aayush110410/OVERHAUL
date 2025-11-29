"""AQI Agent: Supervised prediction of Air Quality based on Traffic State.

This agent wraps the trained Random Forest model to provide grounded,
non-hallucinated AQI predictions. It includes input validation (supervision)
to ensure the model is not queried with out-of-distribution data.
"""
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

class AQIAgent:
    def __init__(self, model_dir: str = "models"):
        self.model_dir = Path(model_dir)
        self.model_path = self.model_dir / "noida_aqi_rf.pkl"
        self.cols_path = self.model_dir / "noida_feature_cols.pkl"
        self.model = None
        self.feature_cols = []
        self.load_model()

    def load_model(self):
        """Load the trained Random Forest model and feature columns."""
        if self.model_path.exists() and self.cols_path.exists():
            try:
                self.model = joblib.load(self.model_path)
                self.feature_cols = joblib.load(self.cols_path)
                print(f"AQIAgent: Loaded supervised model from {self.model_path}")
            except Exception as e:
                print(f"AQIAgent: Failed to load model: {e}")
        else:
            print(f"AQIAgent: Model files not found in {self.model_dir}")

    def _get_season_flags(self, d: datetime) -> Dict[str, int]:
        """Derive season one-hot flags from date."""
        m = d.month
        return {
            "season_winter": int(m in (12, 1, 2)),
            "season_pre_summer": int(m in (3, 4)),
            "season_summer": int(m in (5, 6)),
            "season_monsoon": int(m in (7, 8, 9)),
            "season_post_monsoon": int(m in (10, 11)),
        }

    def supervise_inputs(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and clamp inputs to realistic bounds (Supervision Layer)."""
        # Clamp speed (0 to 120 km/h)
        inputs["avg_speed"] = max(0.0, min(120.0, float(inputs.get("avg_speed", 30.0))))
        
        # Clamp jam length (0 to 50 km - unlikely to be higher in a city corridor)
        inputs["jam_length_km"] = max(0.0, min(50.0, float(inputs.get("jam_length_km", 0.0))))
        
        # Clamp jam count (0 to 50)
        inputs["jam_count"] = max(0, min(50, int(inputs.get("jam_count", 0))))
        
        return inputs

    def predict(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Predict AQI based on traffic context.
        
        Args:
            context: Dict containing:
                - date (str or datetime)
                - avg_speed (float)
                - jam_length_km (float)
                - jam_count (int)
                - is_diwali_week (bool)
                
        Returns:
            Dict with 'predicted_aqi', 'confidence', 'source'
        """
        if not self.model:
            return {
                "predicted_aqi": 150.0,
                "confidence": "low",
                "source": "fallback_heuristic",
                "reason": "Model not loaded"
            }

        # 1. Supervision: Validate Inputs
        clean_inputs = self.supervise_inputs(context.copy())
        
        # 2. Feature Engineering
        try:
            date_val = clean_inputs.get("date", datetime.now())
            if isinstance(date_val, str):
                d = datetime.fromisoformat(date_val)
            else:
                d = date_val
                
            doy = d.timetuple().tm_yday
            dow = d.weekday()
            
            row = {
                "avg_speed": clean_inputs["avg_speed"],
                "jam_length_km": clean_inputs["jam_length_km"],
                "jam_count": clean_inputs["jam_count"],
                "doy_sin": np.sin(2 * np.pi * doy / 365),
                "doy_cos": np.cos(2 * np.pi * doy / 365),
                "dow": dow,
                "is_diwali_week": int(clean_inputs.get("is_diwali_week", False)),
                **self._get_season_flags(d)
            }
            
            # Align with model columns
            # Note: feature_cols might contain extra season columns or different order
            # We construct the vector carefully
            vector = []
            for col in self.feature_cols:
                val = row.get(col, 0.0)
                vector.append(val)
            
            X = np.array([vector])
            
            # 3. Prediction
            pred = float(self.model.predict(X)[0])
            
            # 4. Post-Prediction Supervision (Sanity Check)
            # AQI shouldn't be negative or impossibly high (unless apocalypse)
            pred = max(10.0, min(999.0, pred))
            
            return {
                "predicted_aqi": round(pred, 2),
                "confidence": "high",
                "source": "supervised_rf_model",
                "inputs": clean_inputs
            }
            
        except Exception as e:
            print(f"AQIAgent: Prediction error: {e}")
            return {
                "predicted_aqi": 150.0,
                "confidence": "low",
                "source": "error_fallback",
                "error": str(e)
            }
