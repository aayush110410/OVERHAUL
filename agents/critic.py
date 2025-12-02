"""Critic agent for intelligent safety & feasibility review using Gemini."""
from __future__ import annotations

import os
import json
from typing import Any, Dict, List
from datetime import datetime

# Try to import Gemini for intelligent analysis
try:
    import google.generativeai as genai
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "AIzaSyBvcIzOx3s-w5VfFE6pn-UYmjQ6AN0tOws")
    genai.configure(api_key=GEMINI_API_KEY)
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False


class NoidaCritic:
    """
    Intelligent critic that reviews proposals for:
    - Feasibility in Noida's context
    - Potential biases or blind spots
    - Implementation challenges
    - Counterfactual scenarios
    - Equity and accessibility concerns
    """
    
    # Noida-specific constraints and considerations
    NOIDA_CONSTRAINTS = {
        "governance": {
            "authority": "Noida Authority / YEIDA",
            "complexity": "Multi-stakeholder (UP govt, Central, Private)",
            "approval_time": "6-24 months for major projects"
        },
        "infrastructure": {
            "road_capacity": "Many roads already at capacity",
            "metro_expansion": "Aqua line operational, more phases planned",
            "parking": "Severe shortage in commercial areas"
        },
        "demographics": {
            "population_growth": "5-7% annually",
            "income_inequality": "Significant gap between sectors",
            "commuter_inflow": "Large daily inflow from Delhi, Ghaziabad"
        },
        "environmental": {
            "aqi_status": "Severe to Very Poor most of winter",
            "green_cover": "Below national average",
            "waste_management": "Improving but challenged"
        }
    }
    
    # Common pitfalls in urban planning proposals
    COMMON_PITFALLS = [
        "Overestimating behavior change rates",
        "Ignoring last-mile connectivity",
        "Underestimating implementation timeline",
        "Not accounting for induced demand",
        "Ignoring informal sector workers",
        "Assuming uniform adoption across income groups",
        "Overlooking maintenance costs",
        "Not considering seasonal variations (monsoon, pollution season)"
    ]
    
    def __init__(self):
        self.review_history = []
    
    def _basic_feasibility_check(self, artifacts: Dict[str, Any]) -> Dict[str, Any]:
        """Rule-based feasibility check."""
        
        issues = []
        warnings = []
        suggestions = []
        
        impact_report = artifacts.get("impact_report", {})
        intervention = artifacts.get("intervention", {})
        
        # Check for overly optimistic projections
        traffic_impact = impact_report.get("traffic_impact", {})
        if traffic_impact.get("congestion_reduction", 0) > 0.4:
            issues.append("Congestion reduction >40% is highly optimistic for Noida")
            suggestions.append("Consider phased implementation with realistic targets")
            
        aqi_impact = impact_report.get("aqi_impact", {})
        if aqi_impact.get("aqi_reduction", 0) > 0.2:
            warnings.append("AQI improvements depend heavily on regional emissions, not just local traffic")
            
        economic_impact = impact_report.get("economic_impact", {})
        if economic_impact.get("total_annual_benefit_crore", 0) > 500:
            warnings.append("Economic projections above 500 crores should be independently verified")
        
        # Check intervention type for known challenges
        intervention_type = intervention.get("type", "").lower()
        
        if "ev" in intervention_type:
            issues.append("EV charging infrastructure is limited in Noida")
            suggestions.append("Prioritize charging stations at IT parks and residential complexes")
            
        if "metro" in intervention_type:
            warnings.append("Metro stations need good last-mile connectivity to be effective")
            suggestions.append("Consider e-rickshaw/bike-sharing integration")
            
        if "signal" in intervention_type or "timing" in intervention_type:
            suggestions.append("AI-based adaptive signals work best when connected in corridors")
            
        # Equity check
        if not any("low income" in str(v).lower() or "equity" in str(v).lower() 
                   for v in artifacts.values()):
            warnings.append("Proposal doesn't explicitly address impact on lower-income commuters")
        
        return {
            "issues": issues,
            "warnings": warnings,
            "suggestions": suggestions,
            "basic_approval": len(issues) == 0
        }
    
    def _generate_counterfactuals(self, artifacts: Dict[str, Any]) -> List[Dict[str, str]]:
        """Generate counterfactual scenarios to test assumptions."""
        
        counterfactuals = []
        intervention = artifacts.get("intervention", {})
        intervention_type = intervention.get("type", "general")
        
        # Universal counterfactuals
        counterfactuals.append({
            "scenario": "What if population grows 2x faster than projected?",
            "implication": "Infrastructure may be overwhelmed sooner",
            "test": "Run simulation with 10% higher demand"
        })
        
        counterfactuals.append({
            "scenario": "What if fuel prices drop significantly?",
            "implication": "Modal shift incentives may not work",
            "test": "Analyze elasticity of transport choice to fuel prices"
        })
        
        # Type-specific counterfactuals
        if "ev" in intervention_type.lower():
            counterfactuals.append({
                "scenario": "What if battery technology doesn't improve?",
                "implication": "Range anxiety persists, slower adoption",
                "test": "Model adoption with current battery specs"
            })
            
        if "metro" in intervention_type.lower():
            counterfactuals.append({
                "scenario": "What if last-mile connectivity remains poor?",
                "implication": "Metro ridership lower than projected",
                "test": "Survey current metro users about barriers"
            })
        
        if "signal" in intervention_type.lower():
            counterfactuals.append({
                "scenario": "What if there's a major construction project nearby?",
                "implication": "Traffic patterns disrupted, optimization ineffective",
                "test": "Include construction impact in baseline"
            })
        
        return counterfactuals
    
    async def review_with_gemini(self, artifacts: Dict[str, Any]) -> Dict[str, Any]:
        """Use Gemini for intelligent, nuanced review."""
        
        if not GEMINI_AVAILABLE:
            return self.review(artifacts)
        
        context = {
            "proposal": artifacts,
            "noida_constraints": self.NOIDA_CONSTRAINTS,
            "common_pitfalls": self.COMMON_PITFALLS
        }
        
        review_prompt = f"""You are a critical reviewer of urban mobility proposals for Noida, India.
Your job is to find problems, biases, and blind spots - not to approve everything.

PROPOSAL TO REVIEW:
{json.dumps(context, indent=2, default=str)[:4000]}

Provide a thorough critical review covering:

1. FEASIBILITY: Is this actually achievable in Noida's context?
2. ASSUMPTIONS: What questionable assumptions does this make?
3. EQUITY: Who benefits? Who might be harmed or left out?
4. IMPLEMENTATION: What are the practical challenges?
5. COUNTERFACTUALS: What scenarios could make this fail?
6. RECOMMENDATIONS: How could this proposal be improved?

Respond with a JSON object:
{{
    "approval": true/false,
    "confidence": "low" | "medium" | "high",
    "critical_issues": ["list of serious problems"],
    "warnings": ["list of concerns"],
    "blind_spots": ["things the proposal missed"],
    "equity_assessment": "who wins and who loses",
    "counterfactual_risks": ["scenarios that could derail this"],
    "recommendations": ["specific improvements"],
    "overall_verdict": "1-2 sentence summary"
}}

Be critical but constructive. Real urban planning fails when reviews are too lenient."""

        try:
            model = genai.GenerativeModel('gemini-2.0-flash')
            response = model.generate_content(review_prompt)
            text = response.text
            
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0]
            elif "```" in text:
                text = text.split("```")[1].split("```")[0]
            
            review_result = json.loads(text.strip())
            
            # Add metadata
            review_result["reviewer"] = "gemini-critic"
            review_result["reviewed_at"] = datetime.now().isoformat()
            
            return review_result
            
        except Exception as e:
            # Fall back to rule-based review
            result = self.review(artifacts)
            result["gemini_error"] = str(e)
            return result
    
    def review(self, artifacts: Dict[str, Any]) -> Dict[str, Any]:
        """Rule-based review - used as fallback or complement to Gemini."""
        
        basic_check = self._basic_feasibility_check(artifacts)
        counterfactuals = self._generate_counterfactuals(artifacts)
        
        result = {
            "approval": basic_check["basic_approval"],
            "confidence": "medium",
            "issues": basic_check["issues"],
            "warnings": basic_check["warnings"],
            "suggestions": basic_check["suggestions"],
            "counterfactuals": counterfactuals,
            "noida_constraints_considered": list(self.NOIDA_CONSTRAINTS.keys()),
            "reviewer": "rule-based-critic",
            "reviewed_at": datetime.now().isoformat()
        }
        
        # Store in history for learning
        self.review_history.append(result)
        
        return result


# Global instance
_critic = None

def get_critic() -> NoidaCritic:
    global _critic
    if _critic is None:
        _critic = NoidaCritic()
    return _critic


def review(artifacts: Dict[str, Any]) -> Dict[str, Any]:
    """Main entry point - maintains backward compatibility."""
    return get_critic().review(artifacts)
