"""
LDRAGo Hybrid Orchestrator - Azure GPT-5-mini + Gemini 2.5 Flash

This orchestrator:
1. Loads local CSV data (AQI, Traffic for Delhi NCR)
2. Gets analysis from Azure GPT-5-mini (fast)
3. Optionally runs Gemini agents for internet search (slower)

Flow:
User Query → Local Data → Azure Analysis → (Optional) Gemini Enhancement → Response
"""

from __future__ import annotations
import os
import json
import asyncio
from typing import Any, Dict, List, Optional, Callable
from datetime import datetime

from agents.gemini_agents import (
    call_gemini, 
    run_all_agents,
    GEMINI_MODEL
)
from agents.ncr_data_loader import (
    get_ncr_summary,
    format_ncr_data_for_prompt,
    get_aqi_category,
)
from azure_ai.openai.chat import azure_openai_chat_text
from azure_ai.config import load_azure_config, azure_openai_enabled


ORCHESTRATOR_MODEL = "gemini-2.5-flash"  # Using 2.5 flash for free tier quota


async def azure_initial_analysis(query: str, context: Dict[str, Any], ncr_data: str = "") -> str:
    """Get initial analysis from Azure GPT-5-mini (FAST - primary response)."""
    try:
        cfg = load_azure_config()
        if not azure_openai_enabled(cfg):
            return "[Azure not configured]"
        
        # Include local NCR data for grounding
        simple_context = f"""Query: {query}

{ncr_data}

Additional Live Context:
- Traffic Speed: {context.get('live_speed', 'N/A')} km/h
- PM2.5: {context.get('pm25', 'N/A')} µg/m³
- Location: {context.get('location', 'Delhi NCR')}
- Time: {datetime.now().strftime('%Y-%m-%d %H:%M %A')}"""
        
        system = """You are OVERHAUL's expert traffic and environmental analyst for Delhi NCR (Delhi, Noida, Ghaziabad).

Your job is to provide **detailed, well-explained, easy-to-read** analysis. Write like you're explaining to someone who wants to understand the full picture.

## WRITING STYLE:
- Use simple, clear language (no jargon)
- Explain the "why" behind every number and recommendation
- Use proper spacing between sections for readability
- Include specific numbers and percentages
- Make it engaging and informative

## RESPONSE FORMAT (Use exactly this structure):

## 
Urban Mobility & Environmental Intelligence Report

⸻

1. Executive Response

(Direct answer to the user’s prompt)
A concise, decision-ready response addressing the user’s exact question in 2–4 sentences.
This section:
	•	Synthesizes traffic + air quality (if relevant)
	•	Mentions only the cities requested
	•	States impact and urgency level

This section alone should satisfy an impatient user.

⸻

2. Analysis Scope & Context

Cities Covered:
	•	{City A}
	•	{City B (if applicable)}
	•	{City C (if applicable)}

Time Horizon:
	•	Current conditions
	•	Short-term projection (next 6–12 hours)

Analytical Domains:
	•	Urban traffic flow
	•	Ambient air quality
	•	Cross-domain interaction (if relevant to the prompt)

⸻

3. Key Findings Snapshot

(Visual, scannable, zero explanation)

Indicator	{City A}	{City B}	{City C}
Traffic Flow	🔴 / 🟠 / 🟢	🔴 / 🟠 / 🟢	🔴 / 🟠 / 🟢
Avg Speed	X km/h	X km/h	X km/h
Congestion Level	X%	X%	X%
AQI Category	X	X	X
Risk Level	Low / Moderate / High	Low / Moderate / High	Low / Moderate / High

Rows appear only if data is relevant to the user’s query.

⸻

4. City-Level Diagnostics

(Repeatable block — instantiated only for queried cities)

{City Name}

Observed Metrics
	•	Average vehicular speed: X km/h
	•	Congestion index: X%
	•	AQI / PM2.5 level: X

Critical Stress Zones
	•	{Area / Corridor 1}
	•	{Area / Corridor 2}
	•	{Area / Corridor 3 (if applicable)}

System Behavior
	•	Flow efficiency: Below / Near / Above optimal
	•	Congestion driver: Structural / Demand-driven / Incident-driven
	•	Pollution accumulation tendency: Low / Medium / High

⸻

5. Comparative Analysis

(Rendered only if ≥2 cities are requested)

Relative Performance
	•	{City A} exhibits higher congestion intensity compared to {City B}
	•	{City C} shows elevated pollution sensitivity under similar traffic loads

Key Differentiators
	•	Infrastructure capacity
	•	Vehicle mix (private vs commercial)
	•	Traffic signal density
	•	Urban ventilation (open vs enclosed corridors)

⸻

6. Cross-System Interaction Analysis

(Traffic ↔ Air Quality)

Observed Coupling Effects
	•	Low-speed traffic increases per-km particulate emissions
	•	Congestion clusters correlate with localized AQI spikes
	•	Peak traffic windows act as pollution multipliers

Inference
Traffic congestion under current conditions is a primary amplifier of environmental and health risk rather than a secondary factor.

⸻

7. Impact Assessment

Mobility Impact
	•	Increased commute time volatility
	•	Reduced predictability during peak windows
	•	Elevated energy and fuel inefficiency

Health & Environmental Impact
	•	Short-term respiratory stress likely
	•	Disproportionate impact on sensitive populations
	•	Outdoor exposure cost elevated near traffic corridors

⸻

8. Evidence-Based Recommendations

(Ranked by impact)

Immediate Actions
	1.	Avoid peak congestion corridors where possible
	2.	Shift travel timing by ±30–60 minutes
	3.	Prefer mass transit or low-exposure routes

Exposure Mitigation
	4.	Reduce outdoor activity in high-AQI microzones
	5.	Use cabin / indoor filtration where available

Planning Guidance
	6.	Add buffer time for critical commutes
	7.	Remote or hybrid work advised under high-risk conditions

⸻

9. Short-Term Outlook

(6–12 Hour Horizon)
	•	Traffic Trend: Improving / Stable / Deteriorating
	•	Air Quality Trend: Improving / Stable / Deteriorating
	•	Primary Drivers: Traffic volume, weather dispersion, urban activity cycles

⸻

10. Bottom-Line Insight

(One sentence, high authority)

Current conditions represent a compound urban stress scenario, where small behavioral adjustments can significantly reduce travel inefficiency and health exposure.

---

## 📈 Data Sources & Confidence

- Traffic data: Source and freshness
- AQI data: Source and last updated
- Confidence level: High/Medium/Low and why

Be thorough, educational, and helpful. Users should finish reading and feel they truly understand the situation."""
        
        response = await azure_openai_chat_text(
            prompt=simple_context,
            system=system,
            cfg=cfg,
            max_output_tokens=12000,  # Increased for detailed responses
        )
        return response
    except Exception as e:
        return f"[Azure error: {str(e)[:100]}]"


async def gemini_final_synthesis(
    query: str,
    agent_results: Dict[str, Any],
    azure_response: str,
    context: Dict[str, Any],
) -> str:
    """
    Gemini 3 Pro synthesizes all inputs into final response.
    
    This is the master orchestrator that:
    1. Collects all agent outputs
    2. Incorporates Azure's analysis
    3. Cross-checks for consistency
    4. Removes ambiguity
    5. Produces the final comprehensive answer
    """
    
    system = """You are LDRAGo, the master AI orchestrator for OVERHAUL - an urban mobility analysis platform for Delhi NCR.

COVERAGE AREA: Delhi NCR (National Capital Region)
- DELHI: All districts, Ring Road, ITO, CP, Dwarka, Rohini, South Delhi, IGI Airport
- NOIDA: Sectors 1-150+, Greater Noida, Noida Expressway, DND Flyway
- GHAZIABAD: Indirapuram, Vaishali, Kaushambi, Raj Nagar, NH24, RRTS corridor

Your role is to:
1. Synthesize inputs from multiple specialist AI agents covering ALL THREE cities
2. Cross-check the Azure GPT-5-mini analysis for accuracy
3. Ensure the response covers Delhi, Noida, AND Ghaziabad appropriately
4. Remove any ambiguity or contradictions between sources
5. Produce a clear, comprehensive, actionable final response

Guidelines:
- Cover all three cities (Delhi, Noida, Ghaziabad) in your response
- Use specific numbers and cite sources when available
- If agents disagree, note the discrepancy and give your best judgment
- Note differences between cities (e.g., Delhi EV policy vs UP policy)
- Structure the response clearly with sections
- Be concise but thorough
- End with actionable recommendations for each city"""

    # Build the synthesis prompt
    agent_summaries = []
    for domain, result in agent_results.get("agents", {}).items():
        if result.get("status") == "success":
            analysis = result.get("analysis", "")[:2000]  # Truncate for context
            agent_summaries.append(f"### {domain.upper()} AGENT:\n{analysis}\n")
        else:
            agent_summaries.append(f"### {domain.upper()} AGENT:\n[Error: {result.get('error', 'Unknown')}]\n")
    
    prompt = f"""# SYNTHESIS TASK

## USER QUERY:
{query}

## SPECIALIST AGENT OUTPUTS:
{chr(10).join(agent_summaries)}

## AZURE GPT-5-MINI ANALYSIS:
{azure_response}

## CONTEXT:
- Location: {context.get('location', 'Noida, India')}
- Current Time: {datetime.now().strftime('%Y-%m-%d %H:%M')}
- Live Traffic Speed: {context.get('live_speed', 'N/A')} km/h
- Live PM2.5: {context.get('pm25', 'N/A')} µg/m³

---

Now synthesize all the above into a final, comprehensive response for the user.

Structure your response as:
1. **Executive Summary** (2-3 sentences answering the core query)
2. **Current Situation** (traffic, AQI, weather conditions)
3. **Key Insights** (from agent analyses)
4. **Recommendations** (actionable steps)
5. **Data Sources** (brief mention of where data came from)

Be specific, use numbers, and ensure consistency across all inputs."""

    try:
        response = await call_gemini(
            prompt=prompt,
            system=system,
            enable_search=False,  # Don't search again, synthesize existing data
            temperature=0.5,
            max_tokens=3000,
        )
        return response
    except Exception as e:
        # Fallback to Azure response if Gemini fails
        return f"[Synthesis note: Gemini unavailable, using Azure response]\n\n{azure_response}"


async def ldrago_orchestrate(
    query: str,
    context: Dict[str, Any] = None,
    run_agents: bool = True,
) -> Dict[str, Any]:
    """
    Main LDRAGo orchestration function.
    
    Steps:
    1. Run all specialist agents in parallel (Gemini 3 Pro + Search)
    2. Run the user prompt and self analyse through searching on the internet
    3. Get Azure GPT-5-mini initial analysis
    4. Gemini 3 Pro synthesizes everything, analyses the outputs , recheks for consistency, removes ambiguity
       and produces final comprehensive response
    5. Return final response
    """
    context = context or {}
    logs = []
    start_time = datetime.now()
    
    logs.append(f"🚀 LDRAGo Orchestrator started at {start_time.strftime('%H:%M:%S')}")
    
    # Step 1: Run specialist agents (parallel)
    agent_results = {}
    if run_agents:
        logs.append("📡 Running specialist agents (Gemini 3 Pro + Google Search)...")
        try:
            agent_results = await run_all_agents(query, context)
            successful = sum(1 for a in agent_results.get("agents", {}).values() if a.get("status") == "success")
            logs.append(f"✓ {successful}/5 agents completed successfully")
        except Exception as e:
            logs.append(f"⚠ Agent error: {str(e)[:80]}")
            agent_results = {"agents": {}}
    
    # Step 2: Azure initial analysis (parallel with agents if possible)
    logs.append("🔷 Getting Azure GPT-5-mini analysis...")
    azure_response = await azure_initial_analysis(query, context)
    if azure_response.startswith("[Azure"):
        logs.append("⚠ Azure analysis unavailable")
    else:
        logs.append("✓ Azure analysis complete")
    
    # Step 3: Gemini final synthesis
    logs.append("🧠 Gemini 3 Pro synthesizing final response...")
    try:
        final_response = await gemini_final_synthesis(
            query=query,
            agent_results=agent_results,
            azure_response=azure_response,
            context=context,
        )
        logs.append("✓ Final synthesis complete")
    except Exception as e:
        logs.append(f"⚠ Synthesis error: {str(e)[:80]}")
        final_response = azure_response  # Fallback
    
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    logs.append(f"✅ LDRAGo completed in {duration:.1f}s")
    
    return {
        "query": query,
        "response": final_response,
        "agent_results": agent_results,
        "azure_response": azure_response,
        "logs": logs,
        "duration_seconds": duration,
        "timestamp": end_time.isoformat(),
        "models_used": {
            "agents": GEMINI_MODEL,
            "azure": "gpt-5-mini",
            "orchestrator": ORCHESTRATOR_MODEL,
        }
    }


async def ldrago_fast(
    query: str,
    context: Dict[str, Any] = None,
    progress_callback: Callable[[str, int], None] = None,
) -> Dict[str, Any]:
    """
    FAST LDRAGo - Uses local CSV data + Azure only.
    Much faster than full orchestration (5-10 seconds vs 60+ seconds).
    
    Steps:
    1. Load local NCR data from CSV (instant)
    2. Get Azure GPT-5-mini analysis (5-10 seconds)
    3. Return formatted response
    """
    context = context or {}
    logs = []
    start_time = datetime.now()
    
    def report_progress(msg: str, pct: int):
        logs.append(msg)
        if progress_callback:
            progress_callback(msg, pct)
    
    report_progress("🚀 LDRAGo Fast Mode started", 0)
    
    # Step 1: Load local NCR data (instant)
    report_progress("📊 Loading NCR data from CSV...", 10)
    try:
        ncr_data = format_ncr_data_for_prompt()
        ncr_summary = get_ncr_summary()
        report_progress("✓ NCR data loaded", 20)
    except Exception as e:
        ncr_data = ""
        ncr_summary = {}
        report_progress(f"⚠ NCR data unavailable: {str(e)[:50]}", 20)
    
    # Step 2: Azure analysis with local data (fast)
    report_progress("🔷 Analyzing with Azure GPT-5-mini...", 30)
    try:
        azure_response = await azure_initial_analysis(query, context, ncr_data)
        report_progress("✓ Azure analysis complete", 90)
    except Exception as e:
        azure_response = f"Analysis error: {str(e)[:100]}"
        report_progress(f"⚠ Azure error: {str(e)[:50]}", 90)
    
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    report_progress(f"✅ Complete in {duration:.1f}s", 100)
    
    return {
        "query": query,
        "response": azure_response,
        "ncr_data": ncr_summary,
        "logs": logs,
        "duration_seconds": duration,
        "timestamp": end_time.isoformat(),
        "mode": "fast",
        "models_used": {
            "primary": "gpt-5-mini",
            "data_source": "Local CSV (NCR_AQI_2024_2025, delhi_ncr_traffic)",
        }
    }


# Quick mode - just Azure + Gemini synthesis (no agent searches)
async def ldrago_quick(query: str, context: Dict[str, Any] = None) -> str:
    """Quick mode - Azure + Gemini without full agent search."""
    context = context or {}
    
    # Load local data
    ncr_data = format_ncr_data_for_prompt()
    
    # Just get Azure response with local data
    azure_response = await azure_initial_analysis(query, context, ncr_data)
    
    return azure_response
