from __future__ import annotations

import re
from typing import Dict, List

import httpx

from .config import Demo4Settings
from .providers import WorldFeedFusionService
from .schemas import AgentTrace, OrchestrationResponse, SimulationIntervention, SimulationRequest
from .simulation import NetworkTrafficSimulation


class LDRAGOOrchestrator:
    def __init__(
        self,
        settings: Demo4Settings,
        providers: WorldFeedFusionService,
        simulation: NetworkTrafficSimulation,
    ) -> None:
        self.settings = settings
        self.providers = providers
        self.simulation = simulation

    async def orchestrate(self, request: SimulationRequest):
        focus_name = request.focus_name or self._extract_location_phrase(request.prompt)
        snapshot = await self.providers.build_snapshot(request.focus, focus_name)
        selected_models = self._select_models(request.prompt)
        interventions = request.interventions or self._infer_interventions(request.prompt, snapshot.focus)
        simulation_request = SimulationRequest(
            prompt=request.prompt,
            focus=request.focus,
            focus_name=focus_name or snapshot.focus.get("name"),
            years=request.years,
            interventions=interventions,
        )
        report = self.simulation.simulate(
            simulation_request,
            snapshot.layers["traffic"],
            snapshot.focus,
        )

        agent_trace = await self._run_agents(
            prompt=request.prompt,
            focus_name=snapshot.focus.get("name", "Unknown"),
            selected_models=selected_models,
            summary=report.summary,
        )

        return OrchestrationResponse(
            prompt=request.prompt,
            parsed_location=snapshot.focus,
            selected_models=selected_models,
            agent_trace=agent_trace,
            simulation=report,
            visualization_commands={
                "camera": {
                    "focus": snapshot.focus,
                    "mode": "cinematic",
                    "timeline": [step.label for step in report.timeline],
                },
                "layers": {
                    "traffic": True,
                    "simulation": True,
                    "intelligence": True,
                    "buildings": snapshot.focus.get("zoom_city", False),
                },
                "hud": {
                    "reportSummary": report.summary,
                    "showTimeline": True,
                },
            },
        )

    def _extract_location_phrase(self, prompt: str) -> str | None:
        patterns = [
            r"(?:near|around|at|in)\s+([A-Z][A-Za-z0-9 .,'-]{2,})",
            r"(?:between)\s+([A-Z][A-Za-z0-9 .,'-]{2,})\s+(?:and)\s+([A-Z][A-Za-z0-9 .,'-]{2,})",
        ]
        for pattern in patterns:
            match = re.search(pattern, prompt)
            if match:
                if len(match.groups()) == 2:
                    return f"{match.group(1)} and {match.group(2)}"
                return match.group(1).strip()
        return None

    def _select_models(self, prompt: str) -> List[str]:
        lowered = prompt.lower()
        models = ["traffic_flow"]
        if any(keyword in lowered for keyword in ("pollution", "emission", "air quality")):
            models.append("pollution_impact")
        if any(keyword in lowered for keyword in ("road", "lane", "flyover", "closure", "bridge")):
            models.append("infrastructure_change")
        if "travel time" in lowered or "reroute" in lowered:
            models.append("travel_time_estimation")
        return list(dict.fromkeys(models))

    def _infer_interventions(
        self,
        prompt: str,
        focus: Dict[str, float],
    ) -> List[SimulationIntervention]:
        lowered = prompt.lower()
        centre = [
            focus["longitude"] - 0.01,
            focus["latitude"] - 0.005,
            focus["longitude"] + 0.012,
            focus["latitude"] + 0.006,
        ]
        coordinates = [
            {"longitude": centre[0], "latitude": centre[1], "altitude": 0},
            {"longitude": centre[2], "latitude": centre[3], "altitude": 0},
        ]
        interventions: List[SimulationIntervention] = []

        if "closure" in lowered or "close" in lowered:
            interventions.append(
                SimulationIntervention(
                    kind="road_closure",
                    name="Temporary closure",
                    coordinates=coordinates,
                    notes="Prompt requested a closure or removal of traffic from a corridor.",
                )
            )
        if "flyover" in lowered or "bridge" in lowered:
            interventions.append(
                SimulationIntervention(
                    kind="flyover",
                    name="New flyover",
                    coordinates=coordinates,
                    lane_delta=2,
                    speed_delta=18,
                    notes="Prompt suggested an elevated bypass corridor.",
                )
            )
        if "lane" in lowered or "expand" in lowered or "widen" in lowered:
            interventions.append(
                SimulationIntervention(
                    kind="lane_expansion",
                    name="Lane expansion",
                    coordinates=coordinates,
                    lane_delta=1,
                    capacity_delta=350,
                    notes="Prompt suggested widening or lane expansion.",
                )
            )
        if not interventions:
            interventions.append(
                SimulationIntervention(
                    kind="signal_optimization",
                    name="Adaptive signal timing",
                    coordinates=coordinates,
                    speed_delta=8,
                    notes="Default congestion relief intervention inferred from prompt.",
                )
            )
        return interventions

    async def _run_agents(
        self,
        prompt: str,
        focus_name: str,
        selected_models: List[str],
        summary: str,
    ) -> List[AgentTrace]:
        heuristics = self._heuristic_agent_trace(prompt, focus_name, selected_models, summary)
        if not self.settings.enable_external_llms:
            return heuristics

        qwen = await self._call_openrouter_model(
            model="qwen/qwen3-32b",
            system_prompt="You are the prompt parser and geospatial planner for OVERHAUL.",
            user_prompt=f"Prompt: {prompt}\nFocus: {focus_name}\nModels: {selected_models}",
        )
        llama = await self._call_openrouter_model(
            model="meta-llama/llama-3.3-70b-instruct",
            system_prompt="You are the infrastructure simulation critic for OVERHAUL.",
            user_prompt=f"Prompt: {prompt}\nCurrent summary: {summary}",
        )
        gemini = await self._call_gemini_model(
            prompt=(
                "Summarize the best way to visualize a geospatial traffic simulation in one sentence.\n"
                f"Prompt: {prompt}\nSummary: {summary}"
            )
        )

        merged = []
        for fallback in heuristics:
            if fallback.agent == "qwen" and qwen:
                merged.append(AgentTrace(agent="qwen", model="qwen/qwen3-32b", summary=qwen, confidence=0.73))
            elif fallback.agent == "llama" and llama:
                merged.append(
                    AgentTrace(agent="llama", model="meta-llama/llama-3.3-70b-instruct", summary=llama, confidence=0.7)
                )
            elif fallback.agent == "gemini" and gemini:
                merged.append(AgentTrace(agent="gemini", model="gemini-2.0-flash", summary=gemini, confidence=0.69))
            else:
                merged.append(fallback)
        return merged

    def _heuristic_agent_trace(
        self,
        prompt: str,
        focus_name: str,
        selected_models: List[str],
        summary: str,
    ) -> List[AgentTrace]:
        return [
            AgentTrace(
                agent="qwen",
                model="heuristic-qwen-planner",
                summary=f"Parsed the prompt around {focus_name} and selected models: {', '.join(selected_models)}.",
                confidence=0.64,
            ),
            AgentTrace(
                agent="llama",
                model="heuristic-llama-critic",
                summary=(
                    "Recommended infrastructure-first mitigation because the request implies a corridor-level "
                    f"bottleneck. {summary}"
                ),
                confidence=0.61,
            ),
            AgentTrace(
                agent="gemini",
                model="heuristic-gemini-visualizer",
                summary=(
                    "Suggested a before/after timeline with traffic particles, heatmap shifts, and infrastructure "
                    "geometry fade-in to keep the result legible."
                ),
                confidence=0.6,
            ),
        ]

    async def _call_openrouter_model(self, model: str, system_prompt: str, user_prompt: str) -> str | None:
        if not self.settings.openrouter_api_key:
            return None
        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                response = await client.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.settings.openrouter_api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": model,
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt},
                        ],
                        "max_tokens": 180,
                    },
                )
                response.raise_for_status()
                payload = response.json()
                return payload["choices"][0]["message"]["content"].strip()
        except Exception:
            return None

    async def _call_gemini_model(self, prompt: str) -> str | None:
        if not self.settings.gemini_api_key:
            return None
        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                response = await client.post(
                    "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent",
                    params={"key": self.settings.gemini_api_key},
                    json={"contents": [{"parts": [{"text": prompt}]}]},
                )
                response.raise_for_status()
                payload = response.json()
                return (
                    payload.get("candidates", [{}])[0]
                    .get("content", {})
                    .get("parts", [{}])[0]
                    .get("text", "")
                    .strip()
                ) or None
        except Exception:
            return None
