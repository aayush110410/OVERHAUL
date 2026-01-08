import os
import json
from typing import Dict, Any

import requests

SYSTEM_PROMPT = """
You are an urban infrastructure planning AI.
You receive factual inputs about roads and constraints.
You must NOT guess dimensions.
You must only decide planning parameters.


Output STRICT JSON ONLY.
No markdown. No explanations.

Schema:
{
  "flyover_lanes": number,
  "pillar_spacing_m": number,
  "alignment": "parallel",
  "design_reasoning": string,
  "traffic_assumption": {
    "speed_before": number,
    "speed_after": number,
    "congestion_reduction_percent": number
  }
}

Gemini must NEVER output raw widths or heights.
"""

GEMINI_ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent"


def call_gemini(facts: Dict[str, Any]) -> Dict[str, Any]:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        # Deterministic fallback when key is absent.
        return {
            "flyover_lanes": min(2, max(1, facts.get("lanes", 2))),
            "pillar_spacing_m": 30,
            "alignment": "parallel",
            "design_reasoning": "Fallback plan: conservative two-lane flyover with 30m pillars.",
            "traffic_assumption": {
                "speed_before": 20,
                "speed_after": 35,
                "congestion_reduction_percent": 25,
            },
        }

    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [
                    {"text": SYSTEM_PROMPT},
                    {"text": json.dumps(facts)},
                ],
            }
        ]
    }
    resp = requests.post(
        GEMINI_ENDPOINT,
        params={"key": api_key},
        json=payload,
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()
    text = data.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "{}")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Safety fallback if model disobeys.
        return {
            "flyover_lanes": min(2, max(1, facts.get("lanes", 2))),
            "pillar_spacing_m": 30,
            "alignment": "parallel",
            "design_reasoning": "Fallback after invalid JSON.",
            "traffic_assumption": {
                "speed_before": 20,
                "speed_after": 35,
                "congestion_reduction_percent": 25,
            },
        }
