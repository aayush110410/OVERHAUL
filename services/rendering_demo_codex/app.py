from __future__ import annotations

import asyncio
import os
from contextlib import asynccontextmanager
from typing import Any, Dict, List

from fastapi import FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from .config import Demo4Settings, get_settings
from .ldrago import LDRAGOOrchestrator
from .providers import WorldFeedFusionService
from .repository import FeedRepository
from .schemas import BootstrapResponse, GeoPoint, SimulationRequest
from .simulation import NetworkTrafficSimulation


class WebSocketHub:
    def __init__(self) -> None:
        self._connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket) -> None:
        await websocket.accept()
        self._connections.append(websocket)

    def disconnect(self, websocket: WebSocket) -> None:
        if websocket in self._connections:
            self._connections.remove(websocket)

    async def broadcast(self, payload: Dict[str, Any]) -> None:
        disconnected: List[WebSocket] = []
        for connection in self._connections:
            try:
                await connection.send_json(payload)
            except Exception:
                disconnected.append(connection)
        for connection in disconnected:
            self.disconnect(connection)


settings = get_settings()
repository = FeedRepository(settings.database_url)
simulation = NetworkTrafficSimulation()
providers = WorldFeedFusionService(settings, repository)
orchestrator = LDRAGOOrchestrator(settings, providers, simulation)
hub = WebSocketHub()
stream_task: asyncio.Task | None = None


async def _stream_loop() -> None:
    while True:
        snapshot = await providers.build_snapshot()
        await hub.broadcast({"type": "world_snapshot", "payload": snapshot.model_dump(mode="json")})
        await asyncio.sleep(settings.stream_interval_seconds)


@asynccontextmanager
async def lifespan(_: FastAPI):
    global stream_task
    repository.start()
    await providers.build_snapshot()
    stream_task = asyncio.create_task(_stream_loop())
    try:
        yield
    finally:
        if stream_task:
            stream_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await stream_task


import contextlib  # noqa: E402  pylint: disable=wrong-import-position


app = FastAPI(title=settings.service_name, version="1.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _focus_from_query(lat: float | None, lon: float | None, altitude: float | None) -> GeoPoint | None:
    if lat is None or lon is None:
        return None
    return GeoPoint(longitude=lon, latitude=lat, altitude=altitude or 12000.0)


@app.get("/health")
async def health() -> Dict[str, Any]:
    latest = repository.latest_snapshot()
    return {
        "service": settings.service_name,
        "status": "ok",
        "target_fps": 60,
        "stream_interval_seconds": settings.stream_interval_seconds,
        "repository": repository.get_health(),
        "latest_sequence": latest.sequence if latest else 0,
        "available_layers": [layer["id"] for layer in providers.get_available_layers()],
    }


@app.get("/api/demo4/bootstrap", response_model=BootstrapResponse)
async def bootstrap(
    focus_name: str | None = Query(None),
    lat: float | None = Query(None),
    lon: float | None = Query(None),
    altitude: float | None = Query(None),
) -> BootstrapResponse:
    snapshot = await providers.build_snapshot(
        focus=_focus_from_query(lat, lon, altitude),
        focus_name=focus_name,
    )
    return BootstrapResponse(
        snapshot=snapshot,
        available_layers=providers.get_available_layers(),
        presets=providers.get_presets(),
        service_status={
            "repository": repository.get_health(),
            "target_fps": 60,
            "stream_interval_seconds": settings.stream_interval_seconds,
        },
    )


@app.get("/api/demo4/layers")
async def layer_status() -> Dict[str, Any]:
    latest = repository.latest_snapshot()
    if not latest:
        raise HTTPException(status_code=503, detail="No snapshot available yet.")
    return {
        "sequence": latest.sequence,
        "timestamp": latest.timestamp,
        "layers": {layer_id: snapshot.stats.model_dump(mode="json") for layer_id, snapshot in latest.layers.items()},
    }


@app.post("/api/demo4/simulate")
async def simulate_endpoint(request: SimulationRequest):
    snapshot = await providers.build_snapshot(request.focus, request.focus_name)
    report = simulation.simulate(request, snapshot.layers["traffic"], snapshot.focus)
    repository.persist_simulation(report)
    return report


@app.post("/api/demo4/orchestrate")
async def orchestrate_endpoint(request: SimulationRequest):
    response = await orchestrator.orchestrate(request)
    repository.persist_simulation(response.simulation)
    return response


@app.websocket("/ws/demo4/stream")
async def stream_endpoint(websocket: WebSocket):
    await hub.connect(websocket)
    latest = repository.latest_snapshot()
    if latest:
        await websocket.send_json({"type": "world_snapshot", "payload": latest.model_dump(mode="json")})
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        hub.disconnect(websocket)


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("DEMO4_PORT", "8014"))
    uvicorn.run("services.rendering_demo_codex.app:app", host="0.0.0.0", port=port, reload=False)
