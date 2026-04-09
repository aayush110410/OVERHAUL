"""
Agent-Based Simulation Engine
==============================

Multi-agent urban simulation powered by swarm intelligence.
Individual commuter, freight, policy, and environment agents
interact on the NCR road graph to produce emergent traffic,
pollution, and behavioral patterns.

Architecture:
  config.py       → MiroFish/OASIS/Zep configuration
  agents.py       → Agent type definitions (Commuter, Freight, Policy, Environment)
  graph_bridge.py → NCR road graph ↔ knowledge graph bridge
  swarm.py        → Swarm orchestrator (timestep loop, agent coordination)
  engine.py       → AgentSimulationEngine (SimulationEngine interface)
"""

from engines.agent_simulation.engine import AgentSimulationEngine

__all__ = ["AgentSimulationEngine"]
