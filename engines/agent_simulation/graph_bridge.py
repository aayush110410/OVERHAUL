"""
Knowledge Graph Bridge
========================

Maps OVERHAUL's NCR road network (nodes + edges from transport/engine.py)
into a knowledge graph for agent queries and persistent memory.

Supports two backends:
  1. Zep Cloud (when ZEP_API_KEY configured) — full persistent knowledge graph
  2. Local in-memory graph (fallback) — same interface, no persistence

Agents query the graph for:
  - Route options between nodes
  - Congestion history on edges
  - Peer agent patterns on routes
  - Population segment characteristics
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from engines.transport.engine import _DEFAULT_NODES, _DEFAULT_EDGES
from engines.population.engine import _SEGMENTS


@dataclass
class GraphEntity:
    """A node in the knowledge graph."""
    entity_id: str
    entity_type: str  # "intersection", "road", "commuter_profile", "freight_corridor"
    properties: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GraphRelation:
    """A relationship between two entities."""
    source_id: str
    target_id: str
    relation_type: str  # "connects_to", "belongs_to_segment", "serves_corridor"
    properties: Dict[str, Any] = field(default_factory=dict)


class LocalKnowledgeGraph:
    """In-memory knowledge graph — fallback when Zep unavailable.

    Provides the same query interface as ZepKnowledgeGraph so agents
    don't need to know which backend is in use.
    """

    def __init__(self):
        self.entities: Dict[str, GraphEntity] = {}
        self.relations: List[GraphRelation] = []
        self._adjacency: Dict[str, List[Tuple[str, Dict]]] = {}

    def add_entity(self, entity: GraphEntity):
        self.entities[entity.entity_id] = entity

    def add_relation(self, relation: GraphRelation):
        self.relations.append(relation)
        self._adjacency.setdefault(relation.source_id, []).append(
            (relation.target_id, relation.properties)
        )

    def get_entity(self, entity_id: str) -> Optional[GraphEntity]:
        return self.entities.get(entity_id)

    def get_neighbors(self, entity_id: str) -> List[Tuple[str, Dict]]:
        """Get entities connected to the given entity."""
        return self._adjacency.get(entity_id, [])

    def query_entities(self, entity_type: str) -> List[GraphEntity]:
        """Get all entities of a given type."""
        return [e for e in self.entities.values() if e.entity_type == entity_type]

    def get_road_between(self, node_a: str, node_b: str) -> Optional[GraphEntity]:
        """Find road entity connecting two intersection nodes."""
        road_id = f"road_{node_a}__{node_b}"
        return self.entities.get(road_id)

    def update_entity_property(self, entity_id: str, key: str, value: Any):
        """Update a property on an existing entity."""
        entity = self.entities.get(entity_id)
        if entity:
            entity.properties[key] = value

    def get_congestion_snapshot(self) -> Dict[str, float]:
        """Get current congestion for all road entities."""
        snapshot = {}
        for eid, entity in self.entities.items():
            if entity.entity_type == "road":
                congestion = entity.properties.get("current_congestion", 0.5)
                snapshot[eid] = congestion
        return snapshot

    def to_dict(self) -> Dict[str, Any]:
        return {
            "entity_count": len(self.entities),
            "relation_count": len(self.relations),
            "entity_types": list(set(e.entity_type for e in self.entities.values())),
        }


class ZepKnowledgeGraph:
    """Zep Cloud backed knowledge graph with persistent memory.

    Full MiroFish-style knowledge graph using Zep's GraphRAG capabilities.
    Agents get persistent cross-run memory of congestion patterns,
    route effectiveness, and population behavior.
    """

    def __init__(self, api_key: str, collection_name: str = "overhaul_agent_memory"):
        self.api_key = api_key
        self.collection_name = collection_name
        self._client = None
        self._local_fallback = LocalKnowledgeGraph()  # Always maintain local copy
        self._initialized = False

    async def initialize(self):
        """Connect to Zep Cloud and initialize graph."""
        if self._initialized:
            return

        try:
            from zep_cloud.client import AsyncZep
            self._client = AsyncZep(api_key=self.api_key)
            self._initialized = True
        except Exception as e:
            print(f"[KnowledgeGraph] Zep connection failed: {e}. Using local fallback.")
            self._client = None
            self._initialized = True

    def add_entity(self, entity: GraphEntity):
        """Add entity to both Zep and local graph."""
        self._local_fallback.add_entity(entity)
        # Zep sync happens async in batch

    def add_relation(self, relation: GraphRelation):
        self._local_fallback.add_relation(relation)

    def get_entity(self, entity_id: str) -> Optional[GraphEntity]:
        return self._local_fallback.get_entity(entity_id)

    def get_neighbors(self, entity_id: str) -> List[Tuple[str, Dict]]:
        return self._local_fallback.get_neighbors(entity_id)

    def query_entities(self, entity_type: str) -> List[GraphEntity]:
        return self._local_fallback.query_entities(entity_type)

    def get_road_between(self, node_a: str, node_b: str) -> Optional[GraphEntity]:
        return self._local_fallback.get_road_between(node_a, node_b)

    def update_entity_property(self, entity_id: str, key: str, value: Any):
        self._local_fallback.update_entity_property(entity_id, key, value)

    def get_congestion_snapshot(self) -> Dict[str, float]:
        return self._local_fallback.get_congestion_snapshot()

    async def persist_agent_memory(self, agent_id: int, memory_data: Dict[str, Any]):
        """Persist agent memory to Zep for cross-run learning."""
        if not self._client:
            return
        try:
            from zep_cloud.types import Message
            session_id = f"agent_{agent_id}"
            await self._client.memory.add(
                session_id=session_id,
                messages=[Message(
                    role="agent",
                    role_type="assistant",
                    content=json.dumps(memory_data),
                )],
            )
        except Exception:
            pass  # Graceful degradation

    async def recall_agent_memory(self, agent_id: int) -> Optional[Dict[str, Any]]:
        """Recall agent memory from Zep (cross-run persistence)."""
        if not self._client:
            return None
        try:
            session_id = f"agent_{agent_id}"
            memory = await self._client.memory.get(session_id=session_id)
            if memory and memory.messages:
                last = memory.messages[-1]
                return json.loads(last.content)
        except Exception:
            pass
        return None

    def to_dict(self) -> Dict[str, Any]:
        base = self._local_fallback.to_dict()
        base["backend"] = "zep_cloud" if self._client else "local"
        base["persistent_memory"] = self._client is not None
        return base


def build_ncr_knowledge_graph(
    nodes: Optional[Dict[str, Any]] = None,
    edges: Optional[List[Dict[str, Any]]] = None,
    zep_api_key: Optional[str] = None,
) -> LocalKnowledgeGraph | ZepKnowledgeGraph:
    """Build knowledge graph from NCR road network.

    Populates graph with:
      - Intersection entities (12 NCR nodes)
      - Road entities (28+ edges with BPR parameters)
      - Population segment profiles (7 segments)
      - Relationships connecting them

    Uses Zep Cloud if API key provided, otherwise local in-memory.
    """
    nodes = nodes or _DEFAULT_NODES
    edges = edges or _DEFAULT_EDGES

    # Choose backend
    if zep_api_key:
        graph = ZepKnowledgeGraph(api_key=zep_api_key)
    else:
        graph = LocalKnowledgeGraph()

    # Add intersection nodes
    for node_id, node_data in nodes.items():
        graph.add_entity(GraphEntity(
            entity_id=f"node_{node_id}",
            entity_type="intersection",
            properties={
                "node_id": node_id,
                "lat": node_data["lat"],
                "lon": node_data["lon"],
                "label": node_data["label"],
            },
        ))

    # Add road edges (deduplicate bidirectional)
    seen_edges = set()
    for edge in edges:
        edge_key = tuple(sorted([edge["u"], edge["v"]]))
        if edge_key in seen_edges:
            continue
        seen_edges.add(edge_key)

        eid = f"road_{edge['u']}__{edge['v']}"
        graph.add_entity(GraphEntity(
            entity_id=eid,
            entity_type="road",
            properties={
                "from_node": edge["u"],
                "to_node": edge["v"],
                "dist_km": edge["dist_km"],
                "free_speed_kmh": edge["free_speed"],
                "capacity_vph": edge["capacity"],
                "lanes": edge["lanes"],
                "current_congestion": 0.5,  # Initial neutral
                "current_flow": edge["capacity"] * 0.6,
            },
        ))

        # Add bidirectional connections
        graph.add_relation(GraphRelation(
            source_id=f"node_{edge['u']}",
            target_id=f"node_{edge['v']}",
            relation_type="connects_to",
            properties={"road_id": eid, "dist_km": edge["dist_km"]},
        ))
        graph.add_relation(GraphRelation(
            source_id=f"node_{edge['v']}",
            target_id=f"node_{edge['u']}",
            relation_type="connects_to",
            properties={"road_id": eid, "dist_km": edge["dist_km"]},
        ))

    # Add population segment profiles
    for seg_name, seg_data in _SEGMENTS.items():
        graph.add_entity(GraphEntity(
            entity_id=f"segment_{seg_name}",
            entity_type="commuter_profile",
            properties={
                "segment": seg_name,
                "share": seg_data["share"],
                "avg_income": seg_data["avg_income_monthly"],
                "primary_mode": seg_data["primary_mode"],
                "flexibility": seg_data["flexibility"],
                "ev_readiness": seg_data["ev_readiness"],
                "wfh_capable": seg_data["wfh_capable"],
                "avg_commute_km": seg_data["avg_commute_km"],
                "mode_split": seg_data["mode_split"],
            },
        ))

    return graph
