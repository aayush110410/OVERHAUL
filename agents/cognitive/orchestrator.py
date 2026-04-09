"""LDRAGO v2 Orchestrator — Multi-agent cognitive pipeline.

Logic Driven Reasoning and Adaptive Governance Orchestrator (v2)

Pipeline:
  ┌─────────┐   ┌──────────┐   ┌──────────┐   ┌────────────┐
  │ Parser  │──▶│ Location │──▶│ Planner  │──▶│ Researcher │
  │ (Qwen)  │   │ Resolver │   │(Heuristic)│  │ (Data APIs)│
  └─────────┘   └──────────┘   └──────────┘   └─────┬──────┘
                                                      │
                                              ┌───────▼───────┐
                                              │  SIM ENGINES  │
                                              │  (Parallel)   │
                                              └───────┬───────┘
                                                      │
                            ┌─────────────────▼────────────────┐
                            │          PARALLEL                 │
                            │  ┌──────────┐  ┌───────────┐    │
                            │  │ Reasoner │  │  Critic   │    │
                            │  │ (Llama)  │  │ (GPT-OSS) │    │
                            │  └──────────┘  └───────────┘    │
                            └─────────────────┬────────────────┘
                                              │
                                      ┌───────▼───────┐
                                      │ Synthesizer  │
                                      │  (Gemini)    │
                                      └──────┬───────┘
                                             │
                                      ┌──────▼───────┐
                                      │  Viz Output  │
                                      │  (GeoJSON)   │
                                      └──────────────┘

Model routing:
  Parser     → Qwen 3 4B        (fast, <1s)
  Location   → Nominatim/local  (geocoding, <0.5s)
  Planner    → Heuristic         (instant, 0ms)
  Researcher → Data APIs         (parallel fetch, ~1s)
  Engines    → Python compute    (parallel, <2s)
  Reasoner   → Llama 3.3 70B    (deep, ~5s)
  Critic     → GPT-OSS-120B     (validation, ~5s)
  Synthesizer→ Gemini 3.1 Pro   (synthesis, ~5s)
  Viz Output → Python compute    (instant, <0.1s)

Total latency: ~8-12s (Reasoner + Critic run in parallel)
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, Callable, Dict, Optional

from agents.cognitive.roles import AgentContext, AgentOutput, AgentRole
from agents.cognitive.parser_agent import parse_intent
from agents.cognitive.location_resolver import resolve_locations
from agents.cognitive.planner_agent import plan_execution
from agents.cognitive.researcher_agent import gather_research
from agents.cognitive.reasoner_agent import deep_reasoning
from agents.cognitive.critic_agent import critique_results
from agents.cognitive.synthesizer_agent import synthesize_response
from agents.cognitive.viz_output import build_viz_output, build_temporal_viz


class LDRAGOv2:
    """Multi-agent cognitive orchestrator.

    Usage:
        orchestrator = LDRAGOv2()
        result = await orchestrator.run("What if Delhi implements congestion pricing?")
        # result includes "viz_data" for globe rendering
    """

    def __init__(self):
        self._agents = {
            AgentRole.PARSER: parse_intent,
            AgentRole.PLANNER: plan_execution,
            AgentRole.RESEARCHER: gather_research,
            AgentRole.REASONER: deep_reasoning,
            AgentRole.CRITIC: critique_results,
            AgentRole.SYNTHESIZER: synthesize_response,
        }

    async def run(
        self,
        query: str,
        *,
        city: str = "delhi",
        metadata: Optional[Dict[str, Any]] = None,
        progress_callback: Optional[Callable[[str, int], None]] = None,
        skip_engines: bool = False,
    ) -> Dict[str, Any]:
        """Execute the full LDRAGO v2 cognitive pipeline.

        Returns a dict with:
          - query, response, agent_trace, engine_results,
            critique, duration_seconds, models_used, viz_data
        """
        t0 = time.time()
        ctx = AgentContext(query=query, city=city, metadata=metadata or {})

        def _progress(stage: str, pct: int):
            if progress_callback:
                progress_callback(stage, pct)

        # ── Phase 1: Parse + Locate + Plan (sequential, fast) ──
        _progress("Parsing query...", 5)
        await self._agents[AgentRole.PARSER](ctx)

        _progress("Resolving locations...", 8)
        await resolve_locations(ctx)

        _progress("Planning execution...", 10)
        await self._agents[AgentRole.PLANNER](ctx)

        # ── Phase 2: Research + Engines (parallel) ──
        _progress("Gathering data...", 20)
        research_coro = self._agents[AgentRole.RESEARCHER](ctx)

        if skip_engines:
            await research_coro
        else:
            engine_coro = self._run_engines(ctx)
            await asyncio.gather(research_coro, engine_coro)

        _progress("Engines complete", 50)

        # ── Phase 3: Reason + Critique (parallel) ──
        _progress("Analyzing...", 60)
        reason_coro = self._agents[AgentRole.REASONER](ctx)
        critic_coro = self._agents[AgentRole.CRITIC](ctx)
        await asyncio.gather(reason_coro, critic_coro)

        _progress("Cross-validating...", 80)

        # ── Phase 4: Synthesize ──
        _progress("Synthesizing...", 90)
        await self._agents[AgentRole.SYNTHESIZER](ctx)

        # ── Phase 5: Build visualization output ──
        _progress("Building visualization...", 95)
        viz_data = self._build_viz(ctx)

        _progress("Complete", 100)

        duration = time.time() - t0

        return {
            "query": query,
            "response": ctx.synthesis,
            "city": city,
            "parsed_intent": ctx.parsed_intent,
            "execution_plan": {
                "strategy": ctx.execution_plan.get("strategy", "standard"),
                "engines_used": [t["engine"] for t in ctx.execution_plan.get("engine_tasks", [])],
            },
            "engine_results": {k: v for k, v in ctx.engine_results.items() if k != "raw"},
            "critique": ctx.critique,
            "agent_trace": ctx.agent_logs,
            "errors": ctx.errors,
            "duration_seconds": round(duration, 2),
            "models_used": self._models_summary(ctx),
            "pipeline": "ldrago_v2",
            "viz_data": viz_data,
            "locations": ctx.parsed_intent.get("resolved_locations", []),
        }

    async def run_fast(
        self,
        query: str,
        *,
        city: str = "delhi",
        progress_callback: Optional[Callable[[str, int], None]] = None,
    ) -> Dict[str, Any]:
        """Fast mode — skip Critic, use lightweight parsing.

        Latency: ~5-8s (Parser + Location + Research/Engines parallel + Reasoner + Synthesizer)
        """
        t0 = time.time()
        ctx = AgentContext(query=query, city=city)

        def _progress(stage: str, pct: int):
            if progress_callback:
                progress_callback(stage, pct)

        _progress("Parsing...", 10)
        await self._agents[AgentRole.PARSER](ctx)
        await resolve_locations(ctx)
        await self._agents[AgentRole.PLANNER](ctx)

        _progress("Running simulation...", 30)
        await asyncio.gather(
            self._agents[AgentRole.RESEARCHER](ctx),
            self._run_engines(ctx),
        )

        _progress("Analyzing...", 60)
        await self._agents[AgentRole.REASONER](ctx)

        _progress("Synthesizing...", 85)
        await self._agents[AgentRole.SYNTHESIZER](ctx)

        # Build viz output
        viz_data = self._build_viz(ctx)

        _progress("Done", 100)
        duration = time.time() - t0

        return {
            "query": query,
            "response": ctx.synthesis,
            "city": city,
            "engine_results": {k: v for k, v in ctx.engine_results.items() if k != "raw"},
            "agent_trace": ctx.agent_logs,
            "errors": ctx.errors,
            "duration_seconds": round(duration, 2),
            "models_used": self._models_summary(ctx),
            "pipeline": "ldrago_v2_fast",
            "viz_data": viz_data,
            "locations": ctx.parsed_intent.get("resolved_locations", []),
        }

    def _should_use_agent_sim(self, ctx: AgentContext) -> bool:
        """Decide whether to use agent-based simulation for this query.

        Uses agent sim for complex what-if scenarios where emergent
        behavior matters. Skips it for simple data lookups.
        """
        intent = ctx.parsed_intent
        if not intent:
            return False
        # Use agent sim when interventions are detected
        interventions = intent.get("interventions", [])
        query_type = intent.get("query_type", "")
        # Complex scenarios with interventions benefit from agent simulation
        if len(interventions) >= 1 and query_type in ("what_if", "scenario", "simulation", "compare"):
            return True
        # Explicit request
        if "agent" in ctx.query.lower() or "simulate" in ctx.query.lower():
            return True
        return False

    async def _run_engines(self, ctx: AgentContext):
        """Run simulation engines based on the execution plan.

        Automatically selects agent-based simulation for complex
        what-if scenarios, falling back to standard engines otherwise.
        """
        try:
            from engines import get_registry
            from data_integration.bridge import (
                build_scenario_from_prompt,
                ncr_data_to_engine_input,
                format_engine_results_for_chat,
            )

            registry = get_registry()
            scenario = build_scenario_from_prompt(ctx.query, ctx.city)

            # Build engine input data from research
            ncr = ctx.research_data.get("ncr_data", {}).get("data", {})
            data = ncr_data_to_engine_input(ncr) if ncr else {}

            # Check if agent-based simulation should be used
            use_agent_sim = self._should_use_agent_sim(ctx)
            if use_agent_sim:
                try:
                    from engines.agent_simulation.config import get_agent_sim_config
                    if get_agent_sim_config().enabled:
                        # Run agent-based simulation for TRANSPORT capability
                        # Other engines still run normally via registry
                        engines_to_run = [
                            e["name"] for e in registry.list_engines()
                            if e["name"] != "TransportEngine"
                        ]
                        # Run agent sim + other engines in parallel
                        results = await registry.run_scenario(
                            scenario, data, engines=engines_to_run
                        )
                        ctx.log("engines", f"Agent-based simulation + {len(results)} engines", 0)
                    else:
                        results = await registry.run_scenario(scenario, data)
                        ctx.log("engines", f"Ran {len(results)} engines", 0)
                except ImportError:
                    results = await registry.run_scenario(scenario, data)
                    ctx.log("engines", f"Ran {len(results)} engines (agent sim unavailable)", 0)
            else:
                # Standard engine execution
                results = await registry.run_scenario(scenario, data)
                ctx.log("engines", f"Ran {len(results)} engines", 0)

            formatted = format_engine_results_for_chat(results)
            # Keep raw SimulationResult objects for viz generation
            formatted["raw"] = results
            ctx.engine_results = formatted

        except Exception as e:
            ctx.errors.append(f"Engine execution failed: {e}")
            ctx.engine_results = {"error": str(e)}

    async def run_temporal(
        self,
        query: str,
        *,
        city: str = "delhi",
        steps: int = 4,
        step_days: int = 90,
        progress_callback: Optional[Callable[[str, int], None]] = None,
    ) -> Dict[str, Any]:
        """Temporal simulation: Parse → Locate → Plan → Engines over time → Viz.

        Runs the full parse + location pipeline, then executes temporal
        simulation with cross-engine feedback loops, producing timeline
        animation data for the globe.
        """
        t0 = time.time()
        ctx = AgentContext(query=query, city=city)

        def _progress(stage: str, pct: int):
            if progress_callback:
                progress_callback(stage, pct)

        _progress("Parsing...", 5)
        await self._agents[AgentRole.PARSER](ctx)
        await resolve_locations(ctx)
        await self._agents[AgentRole.PLANNER](ctx)

        _progress("Running temporal simulation...", 20)
        try:
            from engines import get_registry
            from data_integration.bridge import build_scenario_from_prompt, ncr_data_to_engine_input

            registry = get_registry()
            scenario = build_scenario_from_prompt(ctx.query, ctx.city)
            data = ncr_data_to_engine_input({})

            temporal_result = await registry.run_temporal(
                scenario=scenario,
                data=data,
                steps=steps,
                step_days=step_days,
            )
        except Exception as e:
            ctx.errors.append(f"Temporal simulation failed: {e}")
            temporal_result = {"steps": [], "timeline_days": 0, "trend": {}}

        _progress("Building visualization...", 90)
        locations = ctx.parsed_intent.get("resolved_locations", [])
        center = ctx.parsed_intent.get("center_point", {"lat": 28.6139, "lon": 77.2090})
        viz_data = build_temporal_viz(temporal_result, locations, center)

        _progress("Done", 100)
        duration = time.time() - t0

        return {
            "query": query,
            "city": city,
            "parsed_intent": ctx.parsed_intent,
            "temporal_result": {
                "steps": temporal_result.get("steps", []),
                "timeline_days": temporal_result.get("timeline_days", 0),
                "trends": temporal_result.get("trend", {}),
            },
            "viz_data": viz_data,
            "locations": locations,
            "agent_trace": ctx.agent_logs,
            "errors": ctx.errors,
            "duration_seconds": round(duration, 2),
            "pipeline": "ldrago_v2_temporal",
        }

    def _build_viz(self, ctx: AgentContext) -> Dict[str, Any]:
        """Build visualization data from current context."""
        locations = ctx.parsed_intent.get("resolved_locations", [])
        center = ctx.parsed_intent.get("center_point", {"lat": 28.6139, "lon": 77.2090})
        return build_viz_output(ctx.to_dict(), ctx.engine_results, locations, center)

    def _models_summary(self, ctx: AgentContext) -> Dict[str, str]:
        """Extract which models were used from agent logs."""
        models = {}
        for log in ctx.agent_logs:
            role = log.get("role", "")
            if role == "parser":
                models["parser"] = "qwen/qwen3-4b:free"
            elif role == "location_resolver":
                models["location"] = "nominatim/ncr_landmarks"
            elif role == "reasoner":
                models["reasoner"] = "meta-llama/llama-3.3-70b-instruct:free"
            elif role == "critic":
                models["critic"] = "openai/gpt-oss-120b:free"
            elif role == "synthesizer":
                models["synthesizer"] = "gemini-3.1-pro-preview"
        models["engines"] = "python-compute"
        models["viz"] = "geospatial-generator"
        return models
