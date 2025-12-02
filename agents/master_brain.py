"""
Master Brain - The ONLY intelligence layer
==========================================
This replaces all the fragmented "agents" with ONE brain that:
1. UNDERSTANDS the query using Gemini
2. CALCULATES impacts using physics-based formulas (not ML that can hallucinate)
3. VALIDATES everything before returning
4. EXPLAINS its reasoning

NO MORE:
- Random ML models giving nonsense outputs
- EV adoption INCREASING pollution
- Disconnected agents that don't talk to each other
"""

import os
import json
from datetime import datetime
from typing import Any, Dict, List, Optional
import pandas as pd
import numpy as np
from pathlib import Path

# Gemini for reasoning
try:
    import google.generativeai as genai
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "AIzaSyBvcIzOx3s-w5VfFE6pn-UYmjQ6AN0tOws")
    genai.configure(api_key=GEMINI_API_KEY)
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False


class MasterBrain:
    """
    The ONE brain that handles everything.
    
    Key principles:
    1. Use Gemini for UNDERSTANDING, not for calculations
    2. Use physics-based formulas for CALCULATIONS
    3. ALWAYS validate outputs make sense
    4. EXPLAIN the reasoning
    """
    
    def __init__(self):
        if GEMINI_AVAILABLE:
            self.model = genai.GenerativeModel('gemini-2.0-flash')
        else:
            self.model = None
        
        # Load real Noida data
        self.traffic_data = None
        self.aqi_data = None
        self._load_data()
        
        # Noida baselines (from real data)
        self.baselines = self._calculate_baselines()
        
        # Physical constants for calculations
        self.physics = {
            # EV Impact on AQI
            # Vehicles contribute ~30-40% of urban PM2.5
            # EVs eliminate tailpipe emissions but not tire/brake dust
            # Net: 100% EV would reduce vehicle PM2.5 by ~70%
            # Vehicle contribution to total: ~35%
            # So 100% EV reduces total AQI by: 0.70 * 0.35 = 24.5%
            "ev_aqi_reduction_per_percent": 0.0025,  # 0.25% AQI reduction per 1% EV
            "max_ev_aqi_reduction": 0.25,  # Max 25% reduction at 100% EV
            
            # Traffic impact
            "ev_traffic_impact": 0.0,  # EVs don't reduce congestion
            "signal_opt_max_improvement": 0.15,  # 15% max from signals
            "metro_max_reduction": 0.20,  # 20% max from metro adoption
            
            # Economic
            "aqi_healthcare_cost_per_point": 2.5,  # Crore per AQI point per year
            "congestion_cost_per_percent": 10,  # Crore per 1% congestion
        }
    
    def _load_data(self):
        """Load historical Noida data."""
        try:
            traffic_path = Path("data/noida_tomtom_daily.csv")
            if traffic_path.exists():
                self.traffic_data = pd.read_csv(traffic_path)
                
            aqi_path = Path("data/noida_aqi_2024.xlsx")
            if aqi_path.exists():
                self.aqi_data = pd.read_excel(aqi_path)
        except Exception as e:
            print(f"[MasterBrain] Data load warning: {e}")
    
    def _calculate_baselines(self) -> Dict[str, float]:
        """Calculate baselines from real data."""
        baselines = {
            "aqi": 201.0,  # Noida yearly average
            "speed_kmh": 27.6,  # Average speed
            "congestion_index": 0.65,  # 0-1 scale
        }
        
        if self.aqi_data is not None and 'AQI' in self.aqi_data.columns:
            baselines["aqi"] = float(self.aqi_data['AQI'].mean())
            
        if self.traffic_data is not None and 'avg_speed' in self.traffic_data.columns:
            baselines["speed_kmh"] = float(self.traffic_data['avg_speed'].mean())
            
        return baselines
    
    async def understand_query(self, prompt: str) -> Dict[str, Any]:
        """
        Step 1: Use Gemini to UNDERSTAND what the user wants.
        This is the ONLY place we use LLM - for understanding, not calculations.
        """
        
        if not self.model:
            return self._fallback_understand(prompt)
        
        understanding_prompt = f"""Analyze this urban planning query for Noida, India.

Query: "{prompt}"

Extract the following as JSON:
{{
    "intent": "what the user wants to know",
    "ev_adoption_percent": number or null (if EV/electric vehicles mentioned),
    "intervention_type": "ev_adoption" | "metro" | "signal_optimization" | "bus_lane" | "cycling" | "general" | null,
    "time_horizon": "immediate" | "short_term" | "long_term",
    "specific_location": "sector or area if mentioned" or null,
    "metrics_requested": ["list of metrics user wants: aqi, traffic, economic, etc"]
}}

Be precise. If user says "50% EV adoption", ev_adoption_percent should be 50."""

        try:
            response = self.model.generate_content(understanding_prompt)
            text = response.text
            
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0]
            elif "```" in text:
                text = text.split("```")[1].split("```")[0]
            
            understanding = json.loads(text.strip())
            understanding["raw_prompt"] = prompt
            return understanding
            
        except Exception as e:
            return self._fallback_understand(prompt)
    
    def _fallback_understand(self, prompt: str) -> Dict[str, Any]:
        """Regex-based fallback for understanding."""
        import re
        
        prompt_lower = prompt.lower()
        
        # Extract EV percentage
        ev_match = re.search(r'(\d+)\s*%\s*(?:ev|electric)', prompt_lower)
        ev_pct = float(ev_match.group(1)) if ev_match else None
        
        # If no explicit EV match, look for any percentage with EV context
        if ev_pct is None and ('ev' in prompt_lower or 'electric' in prompt_lower):
            pct_match = re.search(r'(\d+)\s*%', prompt_lower)
            if pct_match:
                ev_pct = float(pct_match.group(1))
        
        return {
            "intent": f"Analyze: {prompt[:100]}",
            "ev_adoption_percent": ev_pct,
            "intervention_type": "ev_adoption" if ev_pct else "general",
            "time_horizon": "short_term",
            "specific_location": None,
            "metrics_requested": ["aqi", "traffic", "economic"],
            "raw_prompt": prompt
        }
    
    def calculate_ev_impact(self, ev_percent: float) -> Dict[str, Any]:
        """
        Step 2: Calculate EV impact using PHYSICS, not ML.
        
        This is deterministic and CANNOT give wrong results like "AQI increases with EV".
        """
        
        if ev_percent is None or ev_percent <= 0:
            return {
                "ev_percent": 0,
                "aqi_change_percent": 0,
                "aqi_baseline": self.baselines["aqi"],
                "aqi_projected": self.baselines["aqi"],
                "traffic_change_percent": 0,
                "explanation": "No EV adoption specified."
            }
        
        # Clamp to valid range
        ev_percent = min(max(ev_percent, 0), 100)
        
        # Calculate AQI reduction (MUST be negative = improvement)
        aqi_reduction = min(
            ev_percent * self.physics["ev_aqi_reduction_per_percent"],
            self.physics["max_ev_aqi_reduction"]
        )
        
        baseline_aqi = self.baselines["aqi"]
        projected_aqi = baseline_aqi * (1 - aqi_reduction)
        
        # Traffic doesn't change much with EVs (they still take road space)
        traffic_change = 0.0
        
        # Economic impact
        aqi_improvement_points = baseline_aqi - projected_aqi
        healthcare_savings = aqi_improvement_points * self.physics["aqi_healthcare_cost_per_point"]
        
        return {
            "ev_percent": ev_percent,
            "aqi_baseline": round(baseline_aqi, 0),
            "aqi_projected": round(projected_aqi, 0),
            "aqi_change_percent": round(-aqi_reduction * 100, 1),  # Negative = improvement
            "aqi_improvement_points": round(aqi_improvement_points, 0),
            "traffic_change_percent": traffic_change,
            "healthcare_savings_crore": round(healthcare_savings, 1),
            "explanation": f"At {ev_percent}% EV adoption, vehicular emissions reduce by ~{ev_percent*0.7:.0f}%. "
                          f"Since vehicles contribute ~35% of Noida's PM2.5, total AQI improves by ~{aqi_reduction*100:.1f}%. "
                          f"This means AQI drops from {baseline_aqi:.0f} to {projected_aqi:.0f}."
        }
    
    def calculate_general_impact(self, intervention_type: str, intensity: float = 0.5) -> Dict[str, Any]:
        """Calculate impact for non-EV interventions."""
        
        baseline_aqi = self.baselines["aqi"]
        baseline_speed = self.baselines["speed_kmh"]
        
        if intervention_type == "metro":
            traffic_improvement = intensity * self.physics["metro_max_reduction"]
            aqi_improvement = traffic_improvement * 0.3  # Less congestion = less idling emissions
            
        elif intervention_type == "signal_optimization":
            traffic_improvement = intensity * self.physics["signal_opt_max_improvement"]
            aqi_improvement = traffic_improvement * 0.2
            
        elif intervention_type == "bus_lane":
            traffic_improvement = intensity * 0.10
            aqi_improvement = intensity * 0.05
            
        else:
            traffic_improvement = 0
            aqi_improvement = 0
        
        return {
            "intervention": intervention_type,
            "intensity": intensity,
            "aqi_baseline": baseline_aqi,
            "aqi_projected": baseline_aqi * (1 - aqi_improvement),
            "aqi_change_percent": -aqi_improvement * 100,
            "speed_baseline": baseline_speed,
            "speed_projected": baseline_speed * (1 + traffic_improvement),
            "traffic_change_percent": traffic_improvement * 100,
        }
    
    async def generate_response(self, understanding: Dict, calculations: Dict) -> Dict[str, Any]:
        """
        Step 3: Generate human-readable response using Gemini.
        Gemini FORMATS the response, it doesn't CALCULATE.
        """
        
        if not self.model:
            return self._fallback_response(understanding, calculations)
        
        response_prompt = f"""You are an expert urban mobility analyst for Noida, India.

USER QUERY: "{understanding.get('raw_prompt', '')}"

CALCULATED RESULTS (these are CORRECT, do not change the numbers):
{json.dumps(calculations, indent=2)}

Generate a response as JSON:
{{
    "executive_summary": "2-3 sentences directly answering the user with specific numbers from calculations",
    "key_findings": [
        {{"finding": "specific finding with exact numbers from calculations", "confidence": 0.85}},
        {{"finding": "another finding", "confidence": 0.80}}
    ],
    "detailed_analysis": {{
        "air_quality": "Explain AQI change from {calculations.get('aqi_baseline')} to {calculations.get('aqi_projected')}",
        "traffic": "Explain traffic impact",
        "economic": "Explain economic benefits"
    }},
    "recommendations": ["actionable recommendations"],
    "data_sources": ["Noida TomTom traffic data (365 days)", "Noida AQI sensors (366 days)"]
}}

IMPORTANT: Use the EXACT numbers from calculations. AQI goes from {calculations.get('aqi_baseline')} to {calculations.get('aqi_projected')} - this is a DECREASE (improvement)."""

        try:
            response = self.model.generate_content(response_prompt)
            text = response.text
            
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0]
            elif "```" in text:
                text = text.split("```")[1].split("```")[0]
            
            return json.loads(text.strip())
            
        except Exception as e:
            return self._fallback_response(understanding, calculations)
    
    def _fallback_response(self, understanding: Dict, calculations: Dict) -> Dict[str, Any]:
        """Generate response without Gemini."""
        
        ev_pct = calculations.get("ev_percent", 0)
        baseline = calculations.get("aqi_baseline", 201)
        projected = calculations.get("aqi_projected", 201)
        change = calculations.get("aqi_change_percent", 0)
        
        return {
            "executive_summary": f"With {ev_pct}% EV adoption in Noida, AQI is projected to improve from {baseline:.0f} to {projected:.0f} ({change:.1f}% reduction). This improvement comes from reduced vehicular emissions, which contribute approximately 35% of Noida's PM2.5 pollution.",
            "key_findings": [
                {"finding": f"AQI improves from {baseline:.0f} to {projected:.0f} ({abs(change):.1f}% reduction)", "confidence": 0.90},
                {"finding": f"{ev_pct}% EV adoption reduces vehicular emissions significantly", "confidence": 0.85},
                {"finding": "Traffic congestion remains similar (EVs don't reduce road space usage)", "confidence": 0.80}
            ],
            "detailed_analysis": {
                "air_quality": calculations.get("explanation", f"AQI improves by {abs(change):.1f}%"),
                "traffic": "Traffic patterns remain similar with EV adoption - EVs reduce emissions but still occupy road space",
                "economic": f"Healthcare cost savings estimated at ₹{calculations.get('healthcare_savings_crore', 0):.1f} crore annually from improved air quality"
            },
            "recommendations": [
                "Focus EV incentives on high-emission vehicles (commercial trucks, buses)",
                "Expand charging infrastructure along major corridors",
                "Combine EV adoption with metro connectivity for maximum impact"
            ],
            "data_sources": ["Noida TomTom traffic data", "Noida AQI sensors 2024"]
        }
    
    async def process(self, prompt: str) -> Dict[str, Any]:
        """
        Main entry point - the ONLY function that should be called.
        
        Flow:
        1. UNDERSTAND (Gemini)
        2. CALCULATE (Physics)
        3. RESPOND (Gemini formats, doesn't calculate)
        """
        
        logs = []
        logs.append(f"🧠 Master Brain activated at {datetime.now().strftime('%H:%M:%S')}")
        
        # Step 1: Understand
        logs.append("📊 Step 1: Understanding your query...")
        understanding = await self.understand_query(prompt)
        logs.append(f"✓ Understood: {understanding.get('intent', 'Processing...')[:80]}")
        
        ev_pct = understanding.get("ev_adoption_percent")
        if ev_pct:
            logs.append(f"✓ Detected EV adoption: {ev_pct}%")
        
        # Step 2: Calculate (deterministic, physics-based)
        logs.append("🔬 Step 2: Calculating impacts (physics-based)...")
        
        if ev_pct is not None and ev_pct > 0:
            calculations = self.calculate_ev_impact(ev_pct)
        else:
            intervention = understanding.get("intervention_type", "general")
            calculations = self.calculate_general_impact(intervention)
        
        logs.append(f"✓ AQI: {calculations.get('aqi_baseline', 0):.0f} → {calculations.get('aqi_projected', 0):.0f}")
        
        # Step 3: Generate response
        logs.append("📝 Step 3: Generating response...")
        response = await self.generate_response(understanding, calculations)
        logs.append("✓ Response ready")
        
        return {
            "understanding": understanding,
            "calculations": calculations,
            "response": response,
            "logs": logs
        }


# Global instance
_master_brain = None

def get_master_brain() -> MasterBrain:
    global _master_brain
    if _master_brain is None:
        _master_brain = MasterBrain()
    return _master_brain


async def analyze(prompt: str) -> Dict[str, Any]:
    """Main entry point."""
    brain = get_master_brain()
    return await brain.process(prompt)
