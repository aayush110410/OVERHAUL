from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Iterable, List

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from .schemas import LayerSnapshot, SimulationReport, WorldSnapshot


@dataclass
class RepositoryHealth:
    backend: str
    ready: bool
    postgis_enabled: bool
    detail: str = ""


@dataclass
class InMemoryStore:
    snapshots: List[WorldSnapshot] = field(default_factory=list)
    simulations: List[SimulationReport] = field(default_factory=list)


class FeedRepository:
    def __init__(self, database_url: str | None):
        self.database_url = database_url
        self.engine: Engine | None = None
        self.health = RepositoryHealth(backend="memory", ready=True, postgis_enabled=False)
        self.memory = InMemoryStore()

    def start(self) -> None:
        if not self.database_url:
            return

        try:
            self.engine = create_engine(self.database_url, future=True, pool_pre_ping=True)
            with self.engine.begin() as conn:
                if self.database_url.startswith("postgresql"):
                    conn.execute(text("CREATE EXTENSION IF NOT EXISTS postgis"))
                    conn.execute(
                        text(
                            """
                            CREATE TABLE IF NOT EXISTS demo4_feed_features (
                                id BIGSERIAL PRIMARY KEY,
                                layer_id TEXT NOT NULL,
                                item_id TEXT NOT NULL,
                                captured_at TIMESTAMPTZ NOT NULL,
                                geom GEOMETRY(GEOMETRYZ, 4326),
                                payload JSONB NOT NULL
                            )
                            """
                        )
                    )
                    conn.execute(
                        text(
                            """
                            CREATE INDEX IF NOT EXISTS demo4_feed_features_gix
                            ON demo4_feed_features
                            USING GIST (geom)
                            """
                        )
                    )
                    conn.execute(
                        text(
                            """
                            CREATE TABLE IF NOT EXISTS demo4_simulation_runs (
                                id BIGSERIAL PRIMARY KEY,
                                created_at TIMESTAMPTZ NOT NULL,
                                summary TEXT NOT NULL,
                                payload JSONB NOT NULL
                            )
                            """
                        )
                    )
                    self.health = RepositoryHealth(
                        backend="postgresql",
                        ready=True,
                        postgis_enabled=True,
                        detail="PostGIS persistence enabled",
                    )
                else:
                    conn.execute(
                        text(
                            """
                            CREATE TABLE IF NOT EXISTS demo4_feed_features (
                                id INTEGER PRIMARY KEY AUTOINCREMENT,
                                layer_id TEXT NOT NULL,
                                item_id TEXT NOT NULL,
                                captured_at TEXT NOT NULL,
                                geom_wkt TEXT,
                                payload TEXT NOT NULL
                            )
                            """
                        )
                    )
                    conn.execute(
                        text(
                            """
                            CREATE TABLE IF NOT EXISTS demo4_simulation_runs (
                                id INTEGER PRIMARY KEY AUTOINCREMENT,
                                created_at TEXT NOT NULL,
                                summary TEXT NOT NULL,
                                payload TEXT NOT NULL
                            )
                            """
                        )
                    )
                    self.health = RepositoryHealth(
                        backend="sqlite",
                        ready=True,
                        postgis_enabled=False,
                        detail="SQLite fallback persistence enabled",
                    )
        except Exception as exc:
            self.engine = None
            self.health = RepositoryHealth(
                backend="memory",
                ready=True,
                postgis_enabled=False,
                detail=f"Fell back to memory store: {exc}",
            )

    def get_health(self) -> Dict[str, Any]:
        return {
            "backend": self.health.backend,
            "ready": self.health.ready,
            "postgis_enabled": self.health.postgis_enabled,
            "detail": self.health.detail,
            "cached_snapshots": len(self.memory.snapshots),
            "cached_simulations": len(self.memory.simulations),
        }

    def persist_snapshot(self, snapshot: WorldSnapshot) -> None:
        self.memory.snapshots.append(snapshot)
        self.memory.snapshots = self.memory.snapshots[-5:]

        if not self.engine:
            return

        with self.engine.begin() as conn:
            for layer in snapshot.layers.values():
                self._persist_layer(conn, layer, snapshot.timestamp)

    def persist_simulation(self, report: SimulationReport) -> None:
        self.memory.simulations.append(report)
        self.memory.simulations = self.memory.simulations[-10:]

        if not self.engine:
            return

        payload = report.model_dump(mode="json")
        with self.engine.begin() as conn:
            if self.health.backend == "postgresql":
                conn.execute(
                    text(
                        """
                        INSERT INTO demo4_simulation_runs (created_at, summary, payload)
                        VALUES (:created_at, :summary, CAST(:payload AS JSONB))
                        """
                    ),
                    {
                        "created_at": datetime.utcnow(),
                        "summary": report.summary,
                        "payload": json.dumps(payload),
                    },
                )
            else:
                conn.execute(
                    text(
                        """
                        INSERT INTO demo4_simulation_runs (created_at, summary, payload)
                        VALUES (:created_at, :summary, :payload)
                        """
                    ),
                    {
                        "created_at": datetime.utcnow().isoformat(),
                        "summary": report.summary,
                        "payload": json.dumps(payload),
                    },
                )

    def latest_snapshot(self) -> WorldSnapshot | None:
        return self.memory.snapshots[-1] if self.memory.snapshots else None

    def _persist_layer(self, conn: Any, layer: LayerSnapshot, captured_at: datetime) -> None:
        payloads = list(self._iter_feature_payloads(layer))
        if not payloads:
            return

        if self.health.backend == "postgresql":
            for row in payloads:
                if row["geom_wkt"]:
                    conn.execute(
                        text(
                            """
                            INSERT INTO demo4_feed_features (layer_id, item_id, captured_at, geom, payload)
                            VALUES (
                                :layer_id,
                                :item_id,
                                :captured_at,
                                ST_GeomFromText(:geom_wkt, 4326),
                                CAST(:payload AS JSONB)
                            )
                            """
                        ),
                        {
                            "layer_id": row["layer_id"],
                            "item_id": row["item_id"],
                            "captured_at": captured_at,
                            "geom_wkt": row["geom_wkt"],
                            "payload": row["payload"],
                        },
                    )
                else:
                    conn.execute(
                        text(
                            """
                            INSERT INTO demo4_feed_features (layer_id, item_id, captured_at, geom, payload)
                            VALUES (:layer_id, :item_id, :captured_at, NULL, CAST(:payload AS JSONB))
                            """
                        ),
                        {
                            "layer_id": row["layer_id"],
                            "item_id": row["item_id"],
                            "captured_at": captured_at,
                            "payload": row["payload"],
                        },
                    )
            return

        for row in payloads:
            conn.execute(
                text(
                    """
                    INSERT INTO demo4_feed_features (layer_id, item_id, captured_at, geom_wkt, payload)
                    VALUES (:layer_id, :item_id, :captured_at, :geom_wkt, :payload)
                    """
                ),
                {
                    "layer_id": row["layer_id"],
                    "item_id": row["item_id"],
                    "captured_at": captured_at.isoformat(),
                    "geom_wkt": row["geom_wkt"],
                    "payload": row["payload"],
                },
            )

    def _iter_feature_payloads(self, layer: LayerSnapshot) -> Iterable[Dict[str, Any]]:
        for item in layer.items:
            geom_wkt = None
            if item.position:
                geom_wkt = (
                    f"POINT Z ({item.position.longitude} {item.position.latitude} {item.position.altitude})"
                )
            elif item.path and item.path.coordinates:
                coords = ", ".join(
                    f"{point.longitude} {point.latitude} {point.altitude}" for point in item.path.coordinates
                )
                geom_wkt = f"LINESTRING Z ({coords})"

            yield {
                "layer_id": layer.layer_id,
                "item_id": item.id,
                "geom_wkt": geom_wkt,
                "payload": json.dumps(item.model_dump(mode="json")),
            }
