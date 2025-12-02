"""
LDRAGo Brain - The Intelligent Orchestrator
============================================
This is the core AI brain of OVERHAUL that uses Gemini to:
1. THINK - Understand what the user is asking
2. PLAN - Decide which data sources and models to use
3. COMMAND - Execute the plan by calling appropriate agents
4. SYNTHESIZE - Combine results into actionable insights

Unlike the old system that just ran a fixed pipeline, LDRAGo Brain
actually reasons about each prompt and makes intelligent decisions.
"""

import os
import json
import asyncio
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
import google.generativeai as genai

# Configure Gemini
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)


class LDRAgoBrain:
    """
    The intelligent orchestrator that thinks like a human expert.
    
    Core Principles:
    - Every prompt deserves careful analysis before action
    - The system should understand WHAT the user wants, WHY they want it
    - Resources should be fetched based on need, not a fixed pipeline
    - Responses should be grounded in real data, not generic fluff
    """
    
    def __init__(self, data_context: Dict[str, Any] = None):
        # Use Gemini 3 Pro for maximum intelligence
        self.model = genai.GenerativeModel('gemini-3-pro-preview')
        self.data_context = data_context or {}
        self.conversation_history = []
        
        # Available capabilities the brain can use
        self.capabilities = {
            "traffic_simulation": "Run SUMO-based traffic flow simulation",
            "aqi_prediction": "Predict air quality based on traffic and weather",
            "live_traffic": "Fetch real-time traffic data from TomTom/OSRM",
            "live_aqi": "Fetch real-time AQI from OpenAQ sensors",
            "historical_analysis": "Analyze historical trends from Noida data",
            "event_impact": "Assess impact of events/festivals on traffic",
            "ev_adoption": "Model electric vehicle adoption scenarios",
            "infrastructure": "Suggest infrastructure improvements",
            "economic_impact": "Calculate economic effects of interventions",
            "route_optimization": "Find optimal routes considering multiple factors",
        }
        
        # Noida-specific knowledge
        self.noida_knowledge = {
            "sectors": ["Sector 62", "Sector 63", "Sector 78", "Sector 18", "Sector 137"],
            "key_roads": ["Noida Expressway", "DND Flyway", "NH24", "Film City Road"],
            "metro_stations": ["Noida City Centre", "Botanical Garden", "Sector 62"],
            "traffic_hotspots": ["DND Toll", "Sector 18 Market", "Sector 62 IT Hub"],
            "festivals": {
                "diwali": {"months": [10, 11], "impact": "severe", "description": "40-60% traffic increase"},
                "holi": {"months": [3], "impact": "moderate", "description": "20-30% traffic decrease day-of"},
                "republic_day": {"months": [1], "impact": "moderate", "description": "Route diversions near Delhi"},
                "durga_puja": {"months": [10], "impact": "high", "description": "30-40% increase in Sector 18"},
                "new_year": {"months": [1, 12], "impact": "high", "description": "Night traffic surge"},
            },
            "weekly_patterns": {
                "monday": "High morning rush, moderate evening",
                "tuesday": "Normal traffic day",
                "wednesday": "Normal traffic day", 
                "thursday": "Normal traffic day",
                "friday": "Moderate morning, high evening rush",
                "saturday": "Low morning, high afternoon (shopping)",
                "sunday": "Low traffic overall, evening returns",
            }
        }
    
    async def think(self, prompt: str) -> Dict[str, Any]:
        """
        Phase 1: THINK - Deeply analyze what the user is asking.
        
        This is where the magic happens. Instead of keyword matching,
        we use Gemini to truly understand the prompt.
        """
        
        thinking_prompt = f"""You are LDRAGo, an expert urban mobility analyst for Noida, India.

CONTEXT:
- Current date: {datetime.now().strftime("%Y-%m-%d")}
- Day of week: {datetime.now().strftime("%A")}
- Available capabilities: {json.dumps(list(self.capabilities.keys()))}
- Noida knowledge: Key sectors are 62, 63, 78 (IT hubs), Sector 18 (commercial)

USER PROMPT: "{prompt}"

Analyze this prompt and respond with a JSON object containing:
{{
    "understood_intent": "What the user actually wants (1-2 sentences)",
    "key_entities": ["List of important entities mentioned (locations, percentages, etc)"],
    "required_data": ["List of data sources needed from: traffic_live, aqi_live, historical_traffic, historical_aqi, events_calendar, economic_data"],
    "required_capabilities": ["List of capabilities to use from the available ones"],
    "time_scope": "immediate | short_term (days) | medium_term (months) | long_term (years)",
    "confidence": 0.0 to 1.0,
    "clarifying_questions": ["Questions to ask if confidence < 0.7"],
    "noida_specific_context": "Any relevant Noida-specific knowledge to apply"
}}

Think step by step. What is the user really asking for?"""

        try:
            response = await asyncio.to_thread(
                self.model.generate_content, thinking_prompt
            )
            
            # Parse JSON from response
            text = response.text
            # Extract JSON from markdown code blocks if present
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0]
            elif "```" in text:
                text = text.split("```")[1].split("```")[0]
            
            thought = json.loads(text.strip())
            thought["raw_prompt"] = prompt
            thought["thinking_timestamp"] = datetime.now().isoformat()
            
            return thought
            
        except Exception as e:
            # Fallback to basic analysis if Gemini fails
            return {
                "understood_intent": f"Analyze: {prompt[:100]}",
                "key_entities": [],
                "required_data": ["traffic_live", "aqi_live"],
                "required_capabilities": ["traffic_simulation"],
                "time_scope": "immediate",
                "confidence": 0.5,
                "clarifying_questions": [],
                "noida_specific_context": "General Noida corridor analysis",
                "error": str(e)
            }
    
    async def plan(self, thought: Dict[str, Any], available_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Phase 2: PLAN - Create an execution strategy based on thinking.
        
        This decides the order of operations and what to fetch/compute.
        """
        
        planning_prompt = f"""You are LDRAGo planning the execution of an analysis task.

THOUGHT ANALYSIS:
{json.dumps(thought, indent=2)}

AVAILABLE DATA:
- Historical traffic data: 365 days of TomTom data (avg_speed, jam_length, jam_count)
- Historical AQI data: 366 days of Noida AQI measurements
- Live traffic API: Real-time route from OSRM
- Live AQI API: Real-time PM2.5 from sensors
- Simulation engine: SUMO-based traffic flow model
- ML Models: Travel time predictor, AQI predictor, congestion predictor

Create an execution plan as a JSON object:
{{
    "plan_summary": "Brief description of the plan",
    "steps": [
        {{
            "step_id": 1,
            "action": "fetch_live_traffic | fetch_live_aqi | run_simulation | query_historical | analyze_events | predict_impact",
            "description": "What this step does",
            "inputs": {{}},
            "depends_on": []  // step_ids this depends on
        }}
    ],
    "expected_outputs": ["List of outputs the user will receive"],
    "estimated_confidence": 0.0 to 1.0
}}

Create an efficient plan that gets the user what they need."""

        try:
            response = await asyncio.to_thread(
                self.model.generate_content, planning_prompt
            )
            
            text = response.text
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0]
            elif "```" in text:
                text = text.split("```")[1].split("```")[0]
            
            plan = json.loads(text.strip())
            plan["thought"] = thought
            plan["planning_timestamp"] = datetime.now().isoformat()
            
            return plan
            
        except Exception as e:
            # Fallback to basic plan
            return {
                "plan_summary": "Execute standard analysis pipeline",
                "steps": [
                    {"step_id": 1, "action": "fetch_live_traffic", "description": "Get current traffic state", "inputs": {}, "depends_on": []},
                    {"step_id": 2, "action": "fetch_live_aqi", "description": "Get current air quality", "inputs": {}, "depends_on": []},
                    {"step_id": 3, "action": "run_simulation", "description": "Run traffic simulation", "inputs": {}, "depends_on": [1]},
                    {"step_id": 4, "action": "predict_impact", "description": "Calculate impacts", "inputs": {}, "depends_on": [2, 3]},
                ],
                "expected_outputs": ["Traffic analysis", "AQI impact", "Recommendations"],
                "estimated_confidence": 0.6,
                "error": str(e)
            }
    
    async def synthesize(
        self, 
        prompt: str,
        thought: Dict[str, Any], 
        plan: Dict[str, Any], 
        execution_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Phase 4: SYNTHESIZE - Combine all results into a coherent response.
        
        This is where we generate the final insight that the user sees.
        """
        
        synthesis_prompt = f"""You are LDRAGo synthesizing analysis results for the user.

ORIGINAL PROMPT: "{prompt}"

UNDERSTANDING:
{json.dumps(thought, indent=2)}

EXECUTION RESULTS:
{json.dumps(execution_results, indent=2)}

Generate a comprehensive response as JSON:
{{
    "executive_summary": "2-3 sentence summary answering the user's question directly",
    "key_findings": [
        {{"finding": "...", "confidence": 0.0-1.0, "data_source": "..."}}
    ],
    "detailed_analysis": {{
        "traffic": "Traffic-specific insights",
        "air_quality": "AQI-specific insights if relevant",
        "economic": "Economic impact if relevant",
        "recommendations": ["Actionable recommendations"]
    }},
    "data_transparency": {{
        "sources_used": ["List of data sources"],
        "limitations": ["Any limitations or caveats"],
        "confidence_level": "high | medium | low"
    }},
    "follow_up_suggestions": ["Questions the user might want to ask next"]
}}

Be specific, use numbers from the data, and be honest about uncertainties."""

        try:
            response = await asyncio.to_thread(
                self.model.generate_content, synthesis_prompt
            )
            
            text = response.text
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0]
            elif "```" in text:
                text = text.split("```")[1].split("```")[0]
            
            synthesis = json.loads(text.strip())
            synthesis["synthesis_timestamp"] = datetime.now().isoformat()
            
            return synthesis
            
        except Exception as e:
            # Generate basic synthesis from results
            return {
                "executive_summary": f"Analysis of: {thought.get('understood_intent', prompt[:100])}",
                "key_findings": [
                    {"finding": "Analysis completed with available data", "confidence": 0.6, "data_source": "mixed"}
                ],
                "detailed_analysis": {
                    "traffic": execution_results.get("traffic_summary", "Traffic data processed"),
                    "air_quality": execution_results.get("aqi_summary", "AQI data processed"),
                    "economic": "Economic impact requires further analysis",
                    "recommendations": ["Continue monitoring traffic patterns", "Consider EV adoption incentives"]
                },
                "data_transparency": {
                    "sources_used": ["live_traffic", "live_aqi", "historical_data"],
                    "limitations": ["Real-time data may have delays"],
                    "confidence_level": "medium"
                },
                "follow_up_suggestions": ["What specific interventions would you like to explore?"],
                "error": str(e)
            }
    
    async def generate_narrative(self, synthesis: Dict[str, Any], mode: str = "concise") -> str:
        """
        Generate human-readable narrative from synthesis.
        """
        
        narrative_prompt = f"""Convert this analysis into a {'brief 2-3 sentence' if mode == 'concise' else 'detailed paragraph'} narrative for a city planner:

{json.dumps(synthesis, indent=2)}

Write naturally, like an expert briefing a colleague. Use specific numbers. No JSON, just plain text."""

        try:
            response = await asyncio.to_thread(
                self.model.generate_content, narrative_prompt
            )
            return response.text.strip()
        except Exception as e:
            # Fallback narrative
            summary = synthesis.get("executive_summary", "Analysis complete.")
            findings = synthesis.get("key_findings", [])
            if findings:
                summary += f" Key finding: {findings[0].get('finding', '')}"
            return summary
    
    async def process(self, prompt: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Main entry point - orchestrates the full THINK -> PLAN -> EXECUTE -> SYNTHESIZE pipeline.
        """
        
        logs = []
        logs.append(f"🧠 LDRAGo Brain activated at {datetime.now().strftime('%H:%M:%S')}")
        
        # Phase 1: THINK
        logs.append("📊 Phase 1: Analyzing your prompt...")
        thought = await self.think(prompt)
        logs.append(f"✓ Understood: {thought.get('understood_intent', 'Processing...')}")
        
        # Phase 2: PLAN
        logs.append("📋 Phase 2: Creating execution plan...")
        plan = await self.plan(thought, context or {})
        logs.append(f"✓ Plan: {plan.get('plan_summary', 'Standard analysis')}")
        
        # Phase 3: EXECUTE (results passed in from caller)
        logs.append("⚡ Phase 3: Executing analysis...")
        
        # Phase 4: SYNTHESIZE (will be called after execution)
        
        return {
            "thought": thought,
            "plan": plan,
            "logs": logs
        }


class NoidaPatternLearner:
    """
    Learns patterns from historical Noida data.
    This gives the system actual knowledge about local traffic/AQI patterns.
    """
    
    def __init__(self, traffic_data_path: str = None, aqi_data_path: str = None):
        self.traffic_data = None
        self.aqi_data = None
        self.patterns = {}
        
        if traffic_data_path:
            self._load_traffic_data(traffic_data_path)
        if aqi_data_path:
            self._load_aqi_data(aqi_data_path)
    
    def _load_traffic_data(self, path: str):
        """Load and process TomTom traffic data."""
        try:
            import pandas as pd
            self.traffic_data = pd.read_csv(path, parse_dates=['timestamp'])
            self._learn_traffic_patterns()
        except Exception as e:
            print(f"Failed to load traffic data: {e}")
    
    def _load_aqi_data(self, path: str):
        """Load and process AQI data."""
        try:
            import pandas as pd
            self.aqi_data = pd.read_excel(path, parse_dates=['Date'])
            self._learn_aqi_patterns()
        except Exception as e:
            print(f"Failed to load AQI data: {e}")
    
    def _learn_traffic_patterns(self):
        """Extract patterns from traffic data."""
        if self.traffic_data is None:
            return
        
        df = self.traffic_data.copy()
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['month'] = df['timestamp'].dt.month
        df['is_weekend'] = df['day_of_week'].isin([5, 6])
        
        self.patterns['traffic'] = {
            'avg_speed_overall': float(df['avg_speed'].mean()),
            'avg_speed_weekday': float(df[~df['is_weekend']]['avg_speed'].mean()),
            'avg_speed_weekend': float(df[df['is_weekend']]['avg_speed'].mean()),
            'avg_jam_length': float(df['jam_length_km'].mean()),
            'worst_jam_day': df.loc[df['jam_length_km'].idxmax(), 'timestamp'].strftime('%Y-%m-%d'),
            'best_traffic_day': df.loc[df['avg_speed'].idxmax(), 'timestamp'].strftime('%Y-%m-%d'),
            'monthly_avg_speed': df.groupby('month')['avg_speed'].mean().to_dict(),
            'daily_avg_speed': df.groupby('day_of_week')['avg_speed'].mean().to_dict(),
        }
    
    def _learn_aqi_patterns(self):
        """Extract patterns from AQI data."""
        if self.aqi_data is None:
            return
        
        df = self.aqi_data.copy()
        df['month'] = df['Date'].dt.month
        df['day_of_week'] = df['Date'].dt.dayofweek
        
        self.patterns['aqi'] = {
            'avg_aqi': float(df['AQI'].mean()),
            'max_aqi': float(df['AQI'].max()),
            'min_aqi': float(df['AQI'].min()),
            'worst_aqi_day': df.loc[df['AQI'].idxmax(), 'Date'].strftime('%Y-%m-%d'),
            'best_aqi_day': df.loc[df['AQI'].idxmin(), 'Date'].strftime('%Y-%m-%d'),
            'monthly_avg_aqi': df.groupby('month')['AQI'].mean().to_dict(),
            'winter_avg': float(df[df['month'].isin([11, 12, 1, 2])]['AQI'].mean()),
            'summer_avg': float(df[df['month'].isin([4, 5, 6])]['AQI'].mean()),
            'monsoon_avg': float(df[df['month'].isin([7, 8, 9])]['AQI'].mean()),
        }
    
    def get_context_for_date(self, date: datetime = None) -> Dict[str, Any]:
        """Get relevant patterns for a specific date."""
        if date is None:
            date = datetime.now()
        
        context = {
            'date': date.strftime('%Y-%m-%d'),
            'day_of_week': date.strftime('%A'),
            'month': date.month,
            'is_weekend': date.weekday() >= 5,
        }
        
        if self.patterns.get('traffic'):
            tp = self.patterns['traffic']
            dow = date.weekday()
            context['expected_avg_speed'] = tp['daily_avg_speed'].get(dow, tp['avg_speed_overall'])
            context['traffic_baseline'] = tp['avg_speed_weekend'] if context['is_weekend'] else tp['avg_speed_weekday']
        
        if self.patterns.get('aqi'):
            ap = self.patterns['aqi']
            month = date.month
            context['expected_aqi'] = ap['monthly_avg_aqi'].get(month, ap['avg_aqi'])
            if month in [11, 12, 1, 2]:
                context['aqi_season'] = 'winter (poor air quality typical)'
                context['seasonal_aqi_avg'] = ap['winter_avg']
            elif month in [7, 8, 9]:
                context['aqi_season'] = 'monsoon (better air quality)'
                context['seasonal_aqi_avg'] = ap['monsoon_avg']
            else:
                context['aqi_season'] = 'summer/transition'
                context['seasonal_aqi_avg'] = ap['summer_avg']
        
        return context
    
    def get_all_patterns(self) -> Dict[str, Any]:
        """Return all learned patterns."""
        return self.patterns


# Initialize global instances
ldrago_brain = LDRAgoBrain()
pattern_learner = NoidaPatternLearner(
    traffic_data_path="data/noida_tomtom_daily.csv",
    aqi_data_path="data/noida_aqi_2024.xlsx"
)
