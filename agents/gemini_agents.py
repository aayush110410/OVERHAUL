"""
Gemini-powered specialist agents for OVERHAUL.

Each agent uses Gemini 3 Pro with Google Search grounding to:
1. Search the internet for real-time data
2. Analyze their specific domain
3. Return structured insights

Coverage Area: Delhi NCR (National Capital Region)
- Delhi: All districts (Central, North, South, East, West, New Delhi, etc.)
- Noida: Sectors 1-150+, Greater Noida, Noida Extension
- Ghaziabad: Indirapuram, Vaishali, Kaushambi, Raj Nagar, Crossing Republik

Agents:
- TrafficAgent: Real-time traffic data, congestion patterns, incidents
- WeatherAgent: Current weather, forecasts, impact on traffic
- AQIAgent: Air quality data, pollution trends, health advisories
- PolicyAgent: Government policies, regulations, economic impacts
- BehaviorAgent: Commuter patterns, adoption trends, social factors
"""

from __future__ import annotations
import os
import json
import httpx
from typing import Any, Dict, List, Optional
from datetime import datetime

# Gemini API configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
# Using gemini-2.5-flash for free tier quota (gemini-3-pro has no free tier)
# gemini-2.5-flash supports Google Search grounding and is very fast
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
GEMINI_ENDPOINT = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"

# Delhi NCR coverage areas
DELHI_NCR_AREAS = {
    "delhi": [
        "Central Delhi", "North Delhi", "South Delhi", "East Delhi", "West Delhi",
        "New Delhi", "Shahdara", "Dwarka", "Rohini", "Pitampura", "Karol Bagh",
        "Connaught Place", "Chandni Chowk", "Lajpat Nagar", "Saket", "Vasant Kunj",
        "Nehru Place", "ITO", "India Gate", "AIIMS", "JNU", "IGI Airport"
    ],
    "noida": [
        "Sector 18", "Sector 62", "Sector 63", "Sector 15", "Sector 16",
        "Sector 125", "Sector 137", "Sector 142", "Sector 143", "Sector 144",
        "Greater Noida", "Noida Expressway", "DND Flyway", "Botanical Garden",
        "Noida City Centre", "Noida Electronic City", "Film City", "Sector 78"
    ],
    "ghaziabad": [
        "Indirapuram", "Vaishali", "Kaushambi", "Raj Nagar Extension",
        "Crossing Republik", "Vasundhara", "Mohan Nagar", "Sahibabad",
        "NH24", "NH9", "Ghaziabad Junction", "Dilshad Garden Border"
    ]
}

# Key corridors in Delhi NCR
KEY_CORRIDORS = [
    "DND Flyway (Delhi-Noida)", "Noida-Greater Noida Expressway",
    "NH24 (Delhi-Ghaziabad)", "Ring Road Delhi", "Outer Ring Road",
    "Yamuna Expressway", "Eastern Peripheral Expressway",
    "Meerut Expressway", "Delhi-Gurgaon Expressway"
]


async def call_gemini(
    prompt: str,
    system: str = "",
    enable_search: bool = True,
    temperature: float = 0.7,
    max_tokens: int = 2048,
    max_retries: int = 3,
) -> str:
    """Call Gemini API with optional Google Search grounding and retry logic."""
    
    api_key = GEMINI_API_KEY or os.getenv("GEMINI_API_KEY", "")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not set")
    
    # Build the request
    contents = []
    if system:
        contents.append({
            "role": "user",
            "parts": [{"text": f"System instructions: {system}"}]
        })
        contents.append({
            "role": "model", 
            "parts": [{"text": "Understood. I will follow these instructions."}]
        })
    
    contents.append({
        "role": "user",
        "parts": [{"text": prompt}]
    })
    
    payload: Dict[str, Any] = {
        "contents": contents,
        "generationConfig": {
            "temperature": temperature,
            "maxOutputTokens": max_tokens,
        }
    }
    
    # Enable Google Search grounding for real-time data
    if enable_search:
        payload["tools"] = [{
            "googleSearch": {}
        }]
    
    # Retry logic with exponential backoff for rate limits
    import asyncio
    last_error = None
    
    for attempt in range(max_retries):
        try:
            async with httpx.AsyncClient(timeout=90.0) as client:
                resp = await client.post(
                    GEMINI_ENDPOINT,
                    params={"key": api_key},
                    json=payload,
                )
                
                if resp.status_code == 429:
                    # Rate limited - wait and retry
                    wait_time = (2 ** attempt) * 5  # 5s, 10s, 20s
                    print(f"⏳ Gemini rate limited, waiting {wait_time}s (attempt {attempt + 1}/{max_retries})")
                    await asyncio.sleep(wait_time)
                    continue
                
                if resp.status_code != 200:
                    error = resp.text[:500]
                    raise RuntimeError(f"Gemini API error {resp.status_code}: {error}")
                
                data = resp.json()
                
                # Extract text from response
                candidates = data.get("candidates", [])
                if not candidates:
                    raise RuntimeError("Gemini returned no candidates")
                
                content = candidates[0].get("content", {})
                parts = content.get("parts", [])
                
                text_parts = [p.get("text", "") for p in parts if "text" in p]
                return "\n".join(text_parts).strip()
                
        except Exception as e:
            last_error = e
            if attempt < max_retries - 1:
                wait_time = (2 ** attempt) * 2
                await asyncio.sleep(wait_time)
                continue
            raise
    
    # If we exhausted retries due to rate limits
    raise RuntimeError(f"Gemini API failed after {max_retries} retries: {last_error}")


class BaseAgent:
    """Base class for specialist agents."""
    
    name: str = "BaseAgent"
    domain: str = "general"
    
    def __init__(self):
        self.search_enabled = True
    
    async def analyze(self, query: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Override in subclasses."""
        raise NotImplementedError


class TrafficAgent(BaseAgent):
    """Real-time traffic analysis agent for Delhi NCR."""
    
    name = "TrafficAgent"
    domain = "traffic"
    
    async def analyze(self, query: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        context = context or {}
        location = context.get("location", "Delhi NCR")
        
        # Extract specific area if mentioned, otherwise cover all NCR
        area_focus = self._detect_area(query)
        
        system = f"""You are a traffic analysis expert for Delhi NCR (Delhi, Noida, Ghaziabad).

COVERAGE AREAS:
- DELHI: Connaught Place, ITO, Ring Road, Outer Ring Road, Dwarka, Rohini, Pitampura, Saket, South Extension, Nehru Place, Lajpat Nagar, Karol Bagh, Chandni Chowk, Shahdara, IGI Airport
- NOIDA: Sectors 15-18 (Film City), Sectors 62-63 (IT Hub), Sector 125-144 (Expressway), Greater Noida, DND Flyway, Botanical Garden Metro
- GHAZIABAD: Indirapuram, Vaishali, Kaushambi, Raj Nagar, Crossing Republik, Vasundhara, NH24, Mohan Nagar

KEY CORRIDORS: DND Flyway, NH24, Noida Expressway, Ring Road, Eastern Peripheral Expressway, Yamuna Expressway

Your job is to:
1. Search for CURRENT real-time traffic conditions across Delhi NCR
2. Find recent traffic incidents, road closures, construction in ALL THREE cities
3. Get historical traffic patterns for this time of day/week
4. Analyze congestion hotspots: ITO, CP, DND toll, NH24, Sector 18, Indirapuram
5. Check metro connectivity and alternatives
6. Reference any trained CSV data about corridors if available

Always cite your sources. Be specific with numbers, locations and times."""

        prompt = f"""Analyze traffic across Delhi NCR for: {query}

FOCUS AREA: {area_focus}

Current context:
- Time: {datetime.now().strftime('%Y-%m-%d %H:%M %A')}
- Live speed data: {context.get('live_speed', 'Search for current data')}

Search the internet for ALL of these:

DELHI:
1. Current traffic on Ring Road, ITO, CP, AIIMS area
2. IGI Airport traffic and T1/T2/T3 access
3. South Delhi (Saket, GK, Nehru Place) congestion
4. North Delhi (Rohini, Pitampura) traffic

NOIDA:
1. DND Flyway traffic and toll plaza congestion
2. Sector 18 market area traffic
3. Noida Expressway traffic (Sectors 62-144)
4. Greater Noida link road status

GHAZIABAD:
1. NH24 traffic from Delhi border
2. Indirapuram internal roads
3. Vaishali/Kaushambi metro connectivity
4. Raj Nagar Extension access

ALSO SEARCH:
- Any accidents or incidents TODAY in Delhi NCR
- Road closures or diversions
- VIP movements affecting traffic
- Weather impact on traffic
- Comparison with last week same time

Return comprehensive traffic analysis covering ALL THREE cities."""

        try:
            response = await call_gemini(prompt, system, enable_search=True)
            return {
                "agent": self.name,
                "domain": self.domain,
                "analysis": response,
                "coverage": ["Delhi", "Noida", "Ghaziabad"],
                "focus_area": area_focus,
                "timestamp": datetime.now().isoformat(),
                "search_used": True,
                "status": "success"
            }
        except Exception as e:
            return {
                "agent": self.name,
                "domain": self.domain,
                "error": str(e),
                "status": "error"
            }
    
    def _detect_area(self, query: str) -> str:
        """Detect which area the query focuses on."""
        query_lower = query.lower()
        if any(x in query_lower for x in ["noida", "sector", "greater noida", "dnd"]):
            return "Noida (with Delhi/Ghaziabad context)"
        elif any(x in query_lower for x in ["ghaziabad", "indirapuram", "vaishali", "nh24", "kaushambi"]):
            return "Ghaziabad (with Delhi/Noida context)"
        elif any(x in query_lower for x in ["delhi", "cp", "ito", "ring road", "dwarka", "rohini"]):
            return "Delhi (with Noida/Ghaziabad context)"
        return "All Delhi NCR (Delhi + Noida + Ghaziabad)"


class AQIAgent(BaseAgent):
    """Air quality and pollution analysis agent for Delhi NCR."""
    
    name = "AQIAgent"
    domain = "air_quality"
    
    async def analyze(self, query: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        context = context or {}
        location = context.get("location", "Delhi NCR")
        
        system = """You are an air quality expert for Delhi NCR (Delhi, Noida, Ghaziabad).

KEY MONITORING STATIONS:
- DELHI: ITO, Anand Vihar, RK Puram, Punjabi Bagh, DTU, IGI Airport, Lodhi Road
- NOIDA: Sector 62, Sector 125, Greater Noida
- GHAZIABAD: Vasundhara, Indirapuram, Loni

Your job is to:
1. Search for TODAY's AQI readings across ALL stations in Delhi NCR
2. Get PM2.5, PM10, NO2, O3, SO2, CO levels
3. Find last week's and last month's AQI trends
4. Search for news about pollution in Delhi NCR
5. Find CPCB/DPCC advisories and GRAP measures
6. Compare with cities like Beijing that solved AQI problems
7. Get health advisories for current AQI levels
8. Check stubble burning contribution if relevant

Always cite CPCB, SAFAR, or IQAir sources with specific numbers and dates."""

        prompt = f"""Analyze air quality across Delhi NCR for: {query}

Current context:
- Time: {datetime.now().strftime('%Y-%m-%d %H:%M %A')}
- Live PM2.5: {context.get('pm25', 'Search for current data')} µg/m³

Search the internet for ALL of these:

DELHI AQI DATA:
1. Current AQI at ITO (usually worst in Delhi)
2. Current AQI at Anand Vihar (industrial area)
3. Current AQI at RK Puram (South Delhi)
4. Current AQI at IGI Airport
5. Delhi overall AQI average today

NOIDA AQI DATA:
1. Current AQI at Sector 62 monitoring station
2. Current AQI at Sector 125
3. Greater Noida AQI levels
4. Noida overall average

GHAZIABAD AQI DATA:
1. Current AQI at Vasundhara station
2. Current AQI at Indirapuram
3. Current AQI at Loni (industrial)
4. Ghaziabad overall average

ALSO SEARCH:
1. AQI trend over last 7 days for NCR
2. Comparison: Delhi vs Noida vs Ghaziabad today
3. Which city/area has best air quality right now
4. Which city/area has worst air quality right now
5. GRAP stage currently in effect (if any)
6. Stubble burning contribution to current AQI
7. Government advisories for outdoor activities
8. How Beijing reduced pollution - applicable lessons

HEALTH IMPACT:
- Sensitive groups advisory
- Outdoor exercise recommendations
- Mask recommendations

Return comprehensive air quality analysis covering ALL THREE cities with specific numbers."""

        try:
            response = await call_gemini(prompt, system, enable_search=True)
            return {
                "agent": self.name,
                "domain": self.domain,
                "analysis": response,
                "coverage": ["Delhi", "Noida", "Ghaziabad"],
                "monitoring_stations": [
                    "ITO Delhi", "Anand Vihar", "RK Puram", 
                    "Sector 62 Noida", "Vasundhara Ghaziabad"
                ],
                "timestamp": datetime.now().isoformat(),
                "search_used": True,
                "status": "success"
            }
        except Exception as e:
            return {
                "agent": self.name,
                "domain": self.domain,
                "error": str(e),
                "status": "error"
            }


class WeatherAgent(BaseAgent):
    """Weather analysis agent for Delhi NCR."""
    
    name = "WeatherAgent"
    domain = "weather"
    
    async def analyze(self, query: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        context = context or {}
        location = context.get("location", "Delhi NCR")
        
        system = """You are a weather expert for Delhi NCR (Delhi, Noida, Ghaziabad).

Your job is to:
1. Get current weather in Delhi, Noida, and Ghaziabad
2. Analyze how weather affects traffic (rain, fog, heat)
3. Check temperature inversions affecting AQI
4. Provide 3-day forecast for all three cities
5. Identify weather warnings or advisories

Key weather stations: Safdarjung (Delhi), Palam (Delhi), Noida Met, Ghaziabad"""

        prompt = f"""Weather analysis for Delhi NCR: {query}

Time: {datetime.now().strftime('%Y-%m-%d %H:%M %A')}

Search for ALL of these:

CURRENT WEATHER:
1. Delhi current temperature, humidity, wind
2. Noida current temperature, humidity, wind
3. Ghaziabad current temperature, humidity, wind
4. Visibility in all three cities
5. Any fog or smog conditions

FORECAST:
1. 3-day forecast for Delhi
2. 3-day forecast for Noida
3. 3-day forecast for Ghaziabad
4. Any rain expected

TRAFFIC/AQI IMPACT:
1. Is there fog affecting visibility on highways?
2. Is there temperature inversion trapping pollutants?
3. Wind speed - will it help disperse pollution?
4. Rain expected to reduce dust?

WARNINGS:
1. IMD weather warnings for NCR
2. Coldwave/heatwave advisories
3. Thunderstorm alerts

Return weather analysis for all three cities."""

        try:
            response = await call_gemini(prompt, system, enable_search=True)
            return {
                "agent": self.name,
                "domain": self.domain,
                "analysis": response,
                "coverage": ["Delhi", "Noida", "Ghaziabad"],
                "timestamp": datetime.now().isoformat(),
                "status": "success"
            }
        except Exception as e:
            return {
                "agent": self.name,
                "domain": self.domain,
                "error": str(e),
                "status": "error"
            }


class PolicyAgent(BaseAgent):
    """Policy and economics analysis agent for Delhi NCR."""
    
    name = "PolicyAgent"
    domain = "policy_economics"
    
    async def analyze(self, query: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        context = context or {}
        location = context.get("location", "Delhi NCR")
        
        system = """You are a policy and economics expert for Delhi NCR (Delhi, Noida, Ghaziabad).

KEY AUTHORITIES:
- Delhi: Delhi Government, DPCC, DTC, DMRC, PWD Delhi
- Noida: Noida Authority, GNIDA (Greater Noida), UP Pollution Control Board
- Ghaziabad: Ghaziabad Development Authority, GDA, UP State Govt

Your job is to:
1. Find current policies on traffic, pollution, EVs in all three cities
2. Analyze GRAP (Graded Response Action Plan) measures
3. Check odd-even schemes or vehicle restrictions
4. Find EV subsidies (Delhi vs UP policies differ!)
5. Analyze infrastructure projects affecting traffic"""

        prompt = f"""Policy analysis for Delhi NCR: {query}

Time: {datetime.now().strftime('%Y-%m-%d %H:%M')}

Search for ALL of these:

DELHI POLICIES:
1. Delhi EV policy 2020 subsidies (currently active?)
2. Odd-even scheme status
3. DPCC pollution control measures
4. Delhi Metro expansion plans
5. Bus lane policies
6. Congestion pricing proposals

NOIDA/UP POLICIES:
1. UP EV policy subsidies (different from Delhi!)
2. Noida Authority traffic regulations
3. Greater Noida land use policies
4. Noida-Greater Noida Metro updates
5. Expressway toll policies

GHAZIABAD POLICIES:
1. Ghaziabad traffic management
2. NH24 expansion status
3. Rapid Rail (RRTS) updates
4. Industrial area regulations

COMMON NCR POLICIES:
1. Current GRAP stage (1/2/3/4) - what's restricted?
2. Supreme Court directives on pollution
3. Construction ban status
4. Diesel vehicle restrictions
5. BS-VI fuel norms enforcement

ECONOMIC ANALYSIS:
1. Economic impact of traffic congestion in NCR
2. Healthcare costs due to pollution
3. Productivity loss estimates
4. Cost-benefit of EV adoption

Return policy analysis covering all three cities and their different regulations."""

        try:
            response = await call_gemini(prompt, system, enable_search=True)
            return {
                "agent": self.name,
                "domain": self.domain,
                "analysis": response,
                "coverage": ["Delhi", "Noida (UP)", "Ghaziabad (UP)"],
                "key_policies": ["GRAP", "EV Policy", "Odd-Even", "RRTS"],
                "timestamp": datetime.now().isoformat(),
                "status": "success"
            }
        except Exception as e:
            return {
                "agent": self.name,
                "domain": self.domain,
                "error": str(e),
                "status": "error"
            }


class BehaviorAgent(BaseAgent):
    """Commuter behavior analysis agent for Delhi NCR."""
    
    name = "BehaviorAgent"
    domain = "behavior"
    
    async def analyze(self, query: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        context = context or {}
        location = context.get("location", "Delhi NCR")
        
        system = """You are a commuter behavior expert for Delhi NCR (Delhi, Noida, Ghaziabad).

KEY COMMUTE PATTERNS:
- Noida→Delhi: Huge morning rush via DND, Kalindi Kunj
- Ghaziabad→Delhi: NH24, Anand Vihar ISBT, Vaishali Metro
- Inter-sector (Noida): Sector 62/63 IT hub peak traffic
- Delhi internal: Ring Road, ITO bottleneck, CP parking

Your job is to:
1. Analyze commuter patterns specific to each city
2. Find data on mode choice (metro/car/bike/bus/auto)
3. Track EV adoption rates in Delhi vs Noida vs Ghaziabad
4. Identify behavioral changes post-COVID
5. Analyze work-from-home impact on traffic"""

        prompt = f"""Commuter behavior analysis for Delhi NCR: {query}

Time: {datetime.now().strftime('%Y-%m-%d %H:%M %A')}

Search for ALL of these:

COMMUTE PATTERNS - NOIDA:
1. How many people commute from Noida to Delhi daily?
2. Peak hours for Noida→Delhi (DND) traffic
3. Sector 62/63 IT companies - work patterns
4. Noida Metro ridership trends
5. E-rickshaw usage in Noida sectors

COMMUTE PATTERNS - GHAZIABAD:
1. Ghaziabad→Delhi commuter volume
2. NH24 traffic patterns by hour
3. Vaishali/Kaushambi Metro usage
4. Indirapuram residential commute behavior
5. RRTS expected impact on commuters

COMMUTE PATTERNS - DELHI:
1. Internal Delhi commute patterns
2. South Delhi→Gurgaon vs South Delhi→Noida
3. Metro ridership statistics (latest)
4. DTC bus usage trends
5. Two-wheeler vs car preference

MODE CHOICE:
1. Metro ridership growth 2023-2024
2. Private car ownership trends in NCR
3. Electric vehicle adoption rates:
   - Delhi EV registrations
   - Noida/UP EV registrations
   - Ghaziabad EV registrations
4. Bike/cycle commuting trends
5. Cab/auto vs own vehicle preference

BEHAVIORAL CHANGES:
1. Work-from-home percentage still active
2. Hybrid work impact on peak hours
3. Weekend vs weekday traffic difference
4. Shopping/leisure travel patterns
5. School timing impact on traffic

Return comprehensive behavior analysis for all three cities."""

        try:
            response = await call_gemini(prompt, system, enable_search=True)
            return {
                "agent": self.name,
                "domain": self.domain,
                "analysis": response,
                "coverage": ["Delhi", "Noida", "Ghaziabad"],
                "commute_corridors": ["Noida→Delhi", "Ghaziabad→Delhi", "Delhi Internal"],
                "timestamp": datetime.now().isoformat(),
                "status": "success"
            }
        except Exception as e:
            return {
                "agent": self.name,
                "domain": self.domain,
                "error": str(e),
                "status": "error"
            }


# Initialize all agents
traffic_agent = TrafficAgent()
aqi_agent = AQIAgent()
weather_agent = WeatherAgent()
policy_agent = PolicyAgent()
behavior_agent = BehaviorAgent()

ALL_AGENTS = [traffic_agent, aqi_agent, weather_agent, policy_agent, behavior_agent]


async def run_all_agents(query: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
    """Run all specialist agents in parallel."""
    import asyncio
    
    context = context or {}
    
    # Run all agents concurrently
    tasks = [agent.analyze(query, context) for agent in ALL_AGENTS]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    agent_outputs = {}
    for agent, result in zip(ALL_AGENTS, results):
        if isinstance(result, Exception):
            agent_outputs[agent.domain] = {
                "agent": agent.name,
                "error": str(result),
                "status": "error"
            }
        else:
            agent_outputs[agent.domain] = result
    
    return {
        "query": query,
        "timestamp": datetime.now().isoformat(),
        "agents": agent_outputs
    }
