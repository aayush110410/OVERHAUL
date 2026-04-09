from __future__ import annotations

import asyncio
import math
import random
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import httpx

from .config import Demo4Settings
from .repository import FeedRepository
from .schemas import (
    DatasetRecord,
    GeoLine,
    GeoPoint,
    LayerSnapshot,
    LayerStats,
    PerformanceBudget,
    SimulationReport,
    TimelineState,
    WorldSnapshot,
)

try:
    from sgp4.api import Satrec, jday  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    Satrec = None
    jday = None


GLOBAL_WEATHER_CITIES: Sequence[Tuple[str, float, float]] = (
    ("New Delhi", 28.6139, 77.2090),
    ("London", 51.5072, -0.1276),
    ("New York", 40.7128, -74.0060),
    ("Singapore", 1.3521, 103.8198),
    ("Tokyo", 35.6762, 139.6503),
    ("Sydney", -33.8688, 151.2093),
    ("Dubai", 25.2048, 55.2708),
    ("Sao Paulo", -23.5505, -46.6333),
    ("Nairobi", -1.2921, 36.8219),
    ("Johannesburg", -26.2041, 28.0473),
    ("Los Angeles", 34.0522, -118.2437),
    ("Berlin", 52.5200, 13.4050),
)


class WorldFeedFusionService:
    def __init__(self, settings: Demo4Settings, repository: FeedRepository):
        self.settings = settings
        self.repository = repository
        self.sequence = 0
        self._transport_cache: Dict[str, Dict[str, Any]] = {}

    async def build_snapshot(self, focus: GeoPoint | None = None, focus_name: str | None = None) -> WorldSnapshot:
        focus_context = await self.resolve_focus(focus, focus_name)
        async with httpx.AsyncClient(timeout=10.0, headers={"User-Agent": self.settings.user_agent}) as client:
            satellites, flights, earthquakes, weather, transport = await asyncio.gather(
                self._with_timeout(self.fetch_satellites(client), 6.0, self.fallback_satellite_layer),
                self._with_timeout(
                    self.fetch_flights(client, focus_context),
                    6.0,
                    lambda: self.fallback_flight_layer(focus_context),
                ),
                self._with_timeout(self.fetch_earthquakes(client), 5.0, self.fallback_earthquake_layer),
                self._with_timeout(self.fetch_weather(client), 7.0, self.fallback_weather_layer),
                self._with_timeout(
                    self.fetch_transport_bundle(client, focus_context),
                    7.0,
                    lambda: {
                        "roads": self._fallback_roads(focus_context),
                        "buildings": self._fallback_buildings(focus_context),
                    },
                ),
            )

        traffic = self.build_traffic_layer(transport["roads"])
        buildings = self.build_buildings_layer(transport["buildings"])
        intelligence = self.build_intelligence_layer(earthquakes)
        simulation = self.build_simulation_layer(traffic, focus_context)

        self.sequence += 1
        snapshot = WorldSnapshot(
            sequence=self.sequence,
            focus=focus_context,
            budget=PerformanceBudget(
                target_fps=60,
                preferred_resolution_scale=1.0,
                max_particles=self.settings.traffic_particle_budget,
                max_instanced_objects=max(
                    self.settings.satellite_limit,
                    self.settings.flight_limit,
                    self.settings.building_limit,
                ),
            ),
            layers={
                "satellites": satellites,
                "flights": flights,
                "traffic": traffic,
                "weather": weather,
                "simulation": simulation,
                "buildings": buildings,
                "intelligence": intelligence,
            },
        )
        self.repository.persist_snapshot(snapshot)
        return snapshot

    async def _with_timeout(self, coro, timeout: float, fallback_factory):
        try:
            return await asyncio.wait_for(coro, timeout=timeout)
        except Exception:
            fallback = fallback_factory()
            if asyncio.iscoroutine(fallback):
                return await fallback
            return fallback

    async def resolve_focus(self, focus: GeoPoint | None, focus_name: str | None) -> Dict[str, Any]:
        if focus:
            return {
                "name": focus_name or "Custom focus",
                "longitude": focus.longitude,
                "latitude": focus.latitude,
                "altitude": focus.altitude or 12000.0,
                "zoom_city": True,
            }
        if focus_name and focus_name.strip().lower() in {
            "tower bridge",
            "tower bridge, london",
            "tower bridge london",
        }:
            return {
                "name": "Tower Bridge, London",
                "longitude": -0.0754,
                "latitude": 51.5055,
                "altitude": 14000.0,
                "zoom_city": True,
            }
        if focus_name:
            resolved = await self.geocode_name(focus_name)
            if resolved:
                return resolved
        return {
            "name": "Tower Bridge, London",
            "longitude": -0.0754,
            "latitude": 51.5055,
            "altitude": 14000.0,
            "zoom_city": True,
        }

    async def geocode_name(self, name: str) -> Dict[str, Any] | None:
        try:
            async with httpx.AsyncClient(timeout=10.0, headers={"User-Agent": self.settings.user_agent}) as client:
                response = await client.get(
                    "https://nominatim.openstreetmap.org/search",
                    params={"q": name, "format": "jsonv2", "limit": 1},
                )
                response.raise_for_status()
                payload = response.json()
                if not payload:
                    return None
                result = payload[0]
                return {
                    "name": result.get("display_name", name),
                    "longitude": float(result["lon"]),
                    "latitude": float(result["lat"]),
                    "altitude": 12000.0,
                    "zoom_city": True,
                }
        except Exception:
            return None

    async def fetch_satellites(self, client: httpx.AsyncClient) -> LayerSnapshot:
        source = "celestrak"
        records: List[DatasetRecord] = []
        try:
            response = await client.get(
                "https://celestrak.org/NORAD/elements/gp.php",
                params={"GROUP": "active", "FORMAT": "json"},
            )
            response.raise_for_status()
            payload = response.json()
            now = datetime.now(timezone.utc)
            for raw in payload[: self.settings.satellite_limit]:
                position = self._propagate_satellite(raw, now)
                if not position:
                    continue
                records.append(
                    DatasetRecord(
                        id=str(raw.get("OBJECT_ID") or raw.get("NORAD_CAT_ID") or len(records)),
                        position=position,
                        properties={
                            "name": raw.get("OBJECT_NAME", "Unknown"),
                            "norad_id": raw.get("NORAD_CAT_ID"),
                            "classification": raw.get("OBJECT_TYPE", "satellite"),
                        },
                    )
                )
        except Exception:
            source = "demo-fallback"
            records = self._fallback_satellites()

        return LayerSnapshot(
            layer_id="satellites",
            label="Satellite layer",
            kind="points",
            stats=LayerStats(
                object_count=len(records),
                update_frequency_hz=1 / max(self.settings.stream_interval_seconds, 1),
                source=source,
            ),
            items=records,
            meta={"maxRecommended": self.settings.satellite_limit},
        )

    def fallback_satellite_layer(self) -> LayerSnapshot:
        records = self._fallback_satellites()
        return LayerSnapshot(
            layer_id="satellites",
            label="Satellite layer",
            kind="points",
            stats=LayerStats(
                object_count=len(records),
                update_frequency_hz=1 / max(self.settings.stream_interval_seconds, 1),
                source="demo-fallback",
            ),
            items=records,
            meta={"maxRecommended": self.settings.satellite_limit},
        )

    async def fetch_flights(self, client: httpx.AsyncClient, focus: Dict[str, Any]) -> LayerSnapshot:
        source = "opensky"
        records: List[DatasetRecord] = []
        try:
            bbox = {
                "lamin": max(-85.0, focus["latitude"] - 18),
                "lomin": max(-180.0, focus["longitude"] - 25),
                "lamax": min(85.0, focus["latitude"] + 18),
                "lomax": min(180.0, focus["longitude"] + 25),
            }
            response = await client.get("https://opensky-network.org/api/states/all", params=bbox)
            response.raise_for_status()
            payload = response.json()
            states = payload.get("states") or []
            for state in states[: self.settings.flight_limit]:
                lon = state[5]
                lat = state[6]
                if lon is None or lat is None:
                    continue
                records.append(
                    DatasetRecord(
                        id=str(state[0]),
                        position=GeoPoint(
                            longitude=float(lon),
                            latitude=float(lat),
                            altitude=float(state[13] or state[7] or 0.0),
                        ),
                        properties={
                            "callsign": (state[1] or "").strip(),
                            "country": state[2],
                            "velocity_mps": float(state[9] or 0.0),
                            "heading_deg": float(state[10] or 0.0),
                            "vertical_rate_mps": float(state[11] or 0.0),
                        },
                    )
                )
        except Exception:
            source = "demo-fallback"
            records = self._fallback_flights(focus)

        return LayerSnapshot(
            layer_id="flights",
            label="Flight layer",
            kind="points",
            stats=LayerStats(
                object_count=len(records),
                update_frequency_hz=1 / max(self.settings.stream_interval_seconds, 1),
                source=source,
            ),
            items=records,
            meta={"maxRecommended": self.settings.flight_limit},
        )

    def fallback_flight_layer(self, focus: Dict[str, Any]) -> LayerSnapshot:
        records = self._fallback_flights(focus)
        return LayerSnapshot(
            layer_id="flights",
            label="Flight layer",
            kind="points",
            stats=LayerStats(
                object_count=len(records),
                update_frequency_hz=1 / max(self.settings.stream_interval_seconds, 1),
                source="demo-fallback",
            ),
            items=records,
            meta={"maxRecommended": self.settings.flight_limit},
        )

    async def fetch_earthquakes(self, client: httpx.AsyncClient) -> LayerSnapshot:
        source = "usgs"
        items: List[DatasetRecord] = []
        try:
            response = await client.get(
                "https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/all_day.geojson"
            )
            response.raise_for_status()
            payload = response.json()
            for feature in payload.get("features", [])[: 200]:
                coords = feature.get("geometry", {}).get("coordinates", [0, 0, 0])
                props = feature.get("properties", {})
                items.append(
                    DatasetRecord(
                        id=str(feature.get("id")),
                        position=GeoPoint(
                            longitude=float(coords[0]),
                            latitude=float(coords[1]),
                            altitude=float(coords[2] or 0.0) * 1000,
                        ),
                        properties={
                            "place": props.get("place"),
                            "mag": float(props.get("mag") or 0.0),
                            "sig": int(props.get("sig") or 0),
                            "status": props.get("status"),
                        },
                    )
                )
        except Exception:
            source = "demo-fallback"
            items = self._fallback_earthquakes()

        return LayerSnapshot(
            layer_id="intelligence",
            label="Intelligence layer",
            kind="points",
            stats=LayerStats(
                object_count=len(items),
                update_frequency_hz=1 / max(self.settings.stream_interval_seconds, 1),
                source=source,
            ),
            items=items,
            meta={"category": "earthquakes"},
        )

    def fallback_earthquake_layer(self) -> LayerSnapshot:
        items = self._fallback_earthquakes()
        return LayerSnapshot(
            layer_id="intelligence",
            label="Intelligence layer",
            kind="points",
            stats=LayerStats(
                object_count=len(items),
                update_frequency_hz=1 / max(self.settings.stream_interval_seconds, 1),
                source="demo-fallback",
            ),
            items=items,
            meta={"category": "earthquakes"},
        )

    async def fetch_weather(self, client: httpx.AsyncClient) -> LayerSnapshot:
        source = "open-meteo"
        points: List[DatasetRecord] = []

        async def fetch_city(city: Tuple[str, float, float]) -> DatasetRecord | None:
            name, lat, lon = city
            try:
                response = await client.get(
                    "https://api.open-meteo.com/v1/forecast",
                    params={
                        "latitude": lat,
                        "longitude": lon,
                        "current": "temperature_2m,cloud_cover,wind_speed_10m,wind_direction_10m,precipitation",
                    },
                )
                response.raise_for_status()
                current = response.json().get("current", {})
                return DatasetRecord(
                    id=name.lower().replace(" ", "-"),
                    position=GeoPoint(longitude=lon, latitude=lat, altitude=1200),
                    properties={
                        "city": name,
                        "temperature_c": float(current.get("temperature_2m") or 0.0),
                        "cloud_cover": float(current.get("cloud_cover") or 0.0),
                        "wind_speed": float(current.get("wind_speed_10m") or 0.0),
                        "wind_direction": float(current.get("wind_direction_10m") or 0.0),
                        "precipitation": float(current.get("precipitation") or 0.0),
                    },
                )
            except Exception:
                return None

        tasks = [fetch_city(city) for city in GLOBAL_WEATHER_CITIES[: self.settings.weather_city_limit]]
        results = await asyncio.gather(*tasks)
        for item in results:
            if item:
                points.append(item)

        if not points:
            source = "demo-fallback"
            points = self._fallback_weather()

        return LayerSnapshot(
            layer_id="weather",
            label="Weather layer",
            kind="grid",
            stats=LayerStats(
                object_count=len(points),
                update_frequency_hz=1 / max(self.settings.stream_interval_seconds, 1),
                source=source,
            ),
            items=points,
            meta={"visualization": "city-cells"},
        )

    def fallback_weather_layer(self) -> LayerSnapshot:
        points = self._fallback_weather()
        return LayerSnapshot(
            layer_id="weather",
            label="Weather layer",
            kind="grid",
            stats=LayerStats(
                object_count=len(points),
                update_frequency_hz=1 / max(self.settings.stream_interval_seconds, 1),
                source="demo-fallback",
            ),
            items=points,
            meta={"visualization": "city-cells"},
        )

    async def fetch_transport_bundle(
        self,
        client: httpx.AsyncClient,
        focus: Dict[str, Any],
    ) -> Dict[str, List[DatasetRecord]]:
        cache_key = f"{round(focus['latitude'], 2)}:{round(focus['longitude'], 2)}"
        if cache_key in self._transport_cache:
            return self._transport_cache[cache_key]

        source = "openstreetmap"
        node_map: Dict[int, Tuple[float, float]] = {}
        roads: List[DatasetRecord] = []
        buildings: List[DatasetRecord] = []

        try:
            overpass_query = f"""
                [out:json][timeout:25];
                (
                  way["highway"](around:{self.settings.transport_radius_m},{focus["latitude"]},{focus["longitude"]});
                  way["building"](around:{self.settings.transport_radius_m},{focus["latitude"]},{focus["longitude"]});
                );
                (._;>;);
                out body;
            """
            response = await client.post(
                "https://overpass-api.de/api/interpreter",
                content=overpass_query.strip(),
            )
            response.raise_for_status()
            payload = response.json()
            elements = payload.get("elements", [])

            for element in elements:
                if element.get("type") == "node":
                    node_map[element["id"]] = (float(element["lon"]), float(element["lat"]))

            for element in elements:
                if element.get("type") != "way":
                    continue
                tags = element.get("tags", {})
                coords = [node_map[node_id] for node_id in element.get("nodes", []) if node_id in node_map]
                if len(coords) < 2:
                    continue
                simplified_coords = self._simplify_coords(coords, max_points=10)
                if "highway" in tags and len(roads) < 600:
                    roads.append(
                        DatasetRecord(
                            id=f"road-{element['id']}",
                            path=GeoLine(
                                coordinates=[
                                    GeoPoint(longitude=lon, latitude=lat, altitude=20) for lon, lat in simplified_coords
                                ]
                            ),
                            properties={
                                "name": tags.get("name", tags.get("highway", "road")),
                                "road_class": tags.get("highway", "road"),
                                "lanes": int(str(tags.get("lanes", "2")).split(";")[0] or 2),
                                "maxspeed": self._safe_float(tags.get("maxspeed"), fallback=50.0),
                                "source": source,
                            },
                        )
                    )
                elif "building" in tags and len(buildings) < self.settings.building_limit:
                    centroid = self._centroid(coords)
                    height = self._safe_float(tags.get("height"), fallback=0.0)
                    if height <= 0:
                        height = self._safe_float(tags.get("building:levels"), fallback=4.0) * 3.4
                    width, depth = self._bbox_size(coords)
                    buildings.append(
                        DatasetRecord(
                            id=f"building-{element['id']}",
                            position=GeoPoint(longitude=centroid[0], latitude=centroid[1], altitude=height / 2),
                            properties={
                                "height_m": height,
                                "width_m": width,
                                "depth_m": depth,
                                "name": tags.get("name", "Building"),
                            },
                        )
                    )
        except Exception:
            roads = self._fallback_roads(focus)
            buildings = self._fallback_buildings(focus)

        bundle = {"roads": roads, "buildings": buildings}
        self._transport_cache[cache_key] = bundle
        if len(self._transport_cache) > 6:
            oldest = next(iter(self._transport_cache))
            del self._transport_cache[oldest]
        return bundle

    def build_traffic_layer(self, roads: List[DatasetRecord]) -> LayerSnapshot:
        time_factor = self._time_of_day_factor()
        items: List[DatasetRecord] = []
        for road in roads:
            road_class = road.properties.get("road_class", "road")
            class_factor = {
                "motorway": 0.92,
                "trunk": 0.84,
                "primary": 0.78,
                "secondary": 0.7,
                "tertiary": 0.62,
            }.get(str(road_class), 0.56)
            jitter = 0.07 * math.sin(len(items) * 0.37 + time_factor * 10)
            congestion = min(0.98, max(0.08, class_factor * time_factor + jitter))
            speed_limit = float(road.properties.get("maxspeed") or 45.0)
            lane_capacity = max(1.0, float(road.properties.get("lanes") or 2)) * 850.0

            items.append(
                DatasetRecord(
                    id=road.id,
                    path=road.path,
                    properties={
                        **road.properties,
                        "congestion": congestion,
                        "density": congestion * lane_capacity,
                        "speed_kph": max(8.0, speed_limit * (1.0 - congestion * 0.7)),
                        "lane_capacity": lane_capacity,
                    },
                )
            )

        return LayerSnapshot(
            layer_id="traffic",
            label="Traffic layer",
            kind="lines",
            stats=LayerStats(
                object_count=len(items),
                update_frequency_hz=1 / max(self.settings.stream_interval_seconds, 1),
                source="openstreetmap+derived",
            ),
            items=items,
            meta={
                "particleBudget": self.settings.traffic_particle_budget,
                "supportsGpuParticles": True,
            },
        )

    def build_buildings_layer(self, buildings: List[DatasetRecord]) -> LayerSnapshot:
        return LayerSnapshot(
            layer_id="buildings",
            label="3D building extrusion",
            kind="points",
            stats=LayerStats(
                object_count=len(buildings),
                update_frequency_hz=0.02,
                source="openstreetmap",
            ),
            items=buildings,
            meta={"maxRecommended": self.settings.building_limit},
        )

    def build_intelligence_layer(self, earthquakes: LayerSnapshot) -> LayerSnapshot:
        return LayerSnapshot(
            layer_id="intelligence",
            label="Intelligence analysis",
            kind="points",
            stats=earthquakes.stats,
            items=earthquakes.items,
            meta={
                "analysis": {
                    "highestMagnitude": max(
                        [float(item.properties.get("mag") or 0.0) for item in earthquakes.items] or [0.0]
                    ),
                    "alertCount": sum(
                        1 for item in earthquakes.items if float(item.properties.get("mag") or 0.0) >= 4.5
                    ),
                }
            },
        )

    def build_simulation_layer(self, traffic: LayerSnapshot, focus: Dict[str, Any]) -> LayerSnapshot:
        latest_report = self.repository.memory.simulations[-1] if self.repository.memory.simulations else None
        if latest_report and latest_report.timeline:
            current = latest_report.timeline[0]
            items = current.heatmap + current.flows + current.infrastructure
            meta = {
                "summary": latest_report.summary,
                "timeline": [step.model_dump(mode="json") for step in latest_report.timeline],
                "recommendations": latest_report.recommendations,
            }
            return LayerSnapshot(
                layer_id="simulation",
                label="Simulation layer",
                kind="hybrid",
                stats=LayerStats(
                    object_count=len(items),
                    update_frequency_hz=0.25,
                    source="networkx",
                ),
                items=items,
                meta=meta,
            )

        traffic_items = sorted(
            traffic.items,
            key=lambda item: float(item.properties.get("congestion") or 0.0),
            reverse=True,
        )[: 160]
        heatmap = []
        for item in traffic_items:
            if not item.path or not item.path.coordinates:
                continue
            midpoint = item.path.coordinates[len(item.path.coordinates) // 2]
            heatmap.append(
                DatasetRecord(
                    id=f"heat-{item.id}",
                    position=GeoPoint(
                        longitude=midpoint.longitude,
                        latitude=midpoint.latitude,
                        altitude=180,
                    ),
                    properties={
                        "intensity": float(item.properties.get("congestion") or 0.0),
                        "kind": "heat",
                        "focusName": focus["name"],
                    },
                )
            )
        meta = {
            "summary": f"Live congestion hotspot model centered on {focus['name']}.",
            "timeline": [
                TimelineState(
                    label="Current state",
                    year_offset=0,
                    metrics={
                        "avg_speed_kph": 31.2,
                        "travel_time_index": 1.42,
                        "congestion_pct": 58.0,
                    },
                    heatmap=heatmap,
                ).model_dump(mode="json")
            ],
            "recommendations": [
                "Prioritize lane balancing on the three most congested corridors.",
                "Keep particle density adaptive to maintain the 60 FPS target.",
            ],
        }
        return LayerSnapshot(
            layer_id="simulation",
            label="Simulation layer",
            kind="hybrid",
            stats=LayerStats(
                object_count=len(heatmap),
                update_frequency_hz=0.25,
                source="derived",
            ),
            items=heatmap,
            meta=meta,
        )

    def get_available_layers(self) -> List[Dict[str, Any]]:
        return [
            {"id": "satellites", "name": "Satellite layer", "type": "satellite"},
            {"id": "flights", "name": "Flight layer", "type": "flight"},
            {"id": "traffic", "name": "Traffic layer", "type": "traffic"},
            {"id": "weather", "name": "Weather layer", "type": "weather"},
            {"id": "simulation", "name": "Simulation layer", "type": "simulation"},
            {"id": "buildings", "name": "3D building extrusion", "type": "building"},
            {"id": "intelligence", "name": "Intelligence layer", "type": "intelligence"},
        ]

    def get_presets(self) -> List[Dict[str, Any]]:
        return [
            {"id": "standard", "name": "Standard mode"},
            {"id": "night", "name": "Night operations mode"},
            {"id": "thermal", "name": "Thermal vision mode"},
            {"id": "satellite", "name": "Satellite intelligence mode"},
        ]

    def _propagate_satellite(self, raw: Dict[str, Any], now: datetime) -> GeoPoint | None:
        if Satrec and raw.get("TLE_LINE1") and raw.get("TLE_LINE2"):
            try:
                satellite = Satrec.twoline2rv(raw["TLE_LINE1"], raw["TLE_LINE2"])
                jd, fr = jday(
                    now.year,
                    now.month,
                    now.day,
                    now.hour,
                    now.minute,
                    now.second + now.microsecond / 1_000_000,
                )
                error, position, _velocity = satellite.sgp4(jd, fr)
                if error == 0:
                    lon = math.degrees(math.atan2(position[1], position[0]))
                    hyp = math.sqrt(position[0] ** 2 + position[1] ** 2)
                    lat = math.degrees(math.atan2(position[2], hyp))
                    altitude = max(100000.0, (math.sqrt(sum(coord * coord for coord in position)) - 6378.137) * 1000)
                    return GeoPoint(longitude=lon, latitude=lat, altitude=altitude)
            except Exception:
                pass

        mean_motion = float(raw.get("MEAN_MOTION") or 15.0)
        inclination = math.radians(float(raw.get("INCLINATION") or 53.0))
        raan = math.radians(float(raw.get("RA_OF_ASC_NODE") or 0.0))
        mean_anomaly = math.radians(float(raw.get("MEAN_ANOMALY") or 0.0))
        epoch_seconds = now.timestamp()
        angle = mean_anomaly + epoch_seconds * (mean_motion / 86400.0) * math.tau
        altitude = 450000 + max(0.0, (16.5 - mean_motion) * 80000)
        radius = 6371000 + altitude
        x = radius * math.cos(angle)
        y = radius * math.sin(angle) * math.cos(inclination)
        z = radius * math.sin(angle) * math.sin(inclination)
        rx = x * math.cos(raan) - y * math.sin(raan)
        ry = x * math.sin(raan) + y * math.cos(raan)
        lon = math.degrees(math.atan2(ry, rx))
        lat = math.degrees(math.atan2(z, math.sqrt(rx * rx + ry * ry)))
        return GeoPoint(longitude=lon, latitude=lat, altitude=altitude)

    def _fallback_satellites(self) -> List[DatasetRecord]:
        satellites: List[DatasetRecord] = []
        for index in range(self.settings.satellite_limit):
            lon = ((index * 137.5) % 360) - 180
            lat = math.sin(index * 0.37) * 70
            altitude = 420000 + (index % 8) * 150000
            satellites.append(
                DatasetRecord(
                    id=f"fallback-sat-{index}",
                    position=GeoPoint(longitude=lon, latitude=lat, altitude=altitude),
                    properties={"name": f"Fallback SAT {index}", "classification": "demo"},
                )
            )
        return satellites

    def _fallback_flights(self, focus: Dict[str, Any]) -> List[DatasetRecord]:
        flights = []
        for index in range(self.settings.flight_limit):
            lon = focus["longitude"] + math.sin(index * 0.11) * 18
            lat = focus["latitude"] + math.cos(index * 0.07) * 12
            flights.append(
                DatasetRecord(
                    id=f"fallback-flight-{index}",
                    position=GeoPoint(longitude=lon, latitude=lat, altitude=8500 + (index % 12) * 800),
                    properties={
                        "callsign": f"OH{1000 + index}",
                        "country": "Demo",
                        "velocity_mps": 240 + (index % 12) * 4,
                        "heading_deg": (index * 13) % 360,
                    },
                )
            )
        return flights

    def _fallback_earthquakes(self) -> List[DatasetRecord]:
        hotspots = [
            ("ring-of-fire-a", 142.36, 38.3, 5.4),
            ("ring-of-fire-b", -72.95, -36.12, 4.8),
            ("anatolia", 37.04, 37.17, 4.5),
            ("himalaya", 86.93, 27.98, 4.2),
        ]
        items = []
        for name, lon, lat, mag in hotspots:
            items.append(
                DatasetRecord(
                    id=name,
                    position=GeoPoint(longitude=lon, latitude=lat, altitude=1200),
                    properties={"place": name.replace("-", " ").title(), "mag": mag, "sig": int(mag * 100)},
                )
            )
        return items

    def _fallback_weather(self) -> List[DatasetRecord]:
        items: List[DatasetRecord] = []
        for city, lat, lon in GLOBAL_WEATHER_CITIES[: self.settings.weather_city_limit]:
            items.append(
                DatasetRecord(
                    id=city.lower().replace(" ", "-"),
                    position=GeoPoint(longitude=lon, latitude=lat, altitude=1200),
                    properties={
                        "city": city,
                        "temperature_c": 18 + math.sin(lat) * 12,
                        "cloud_cover": abs(math.cos(lon)) * 80,
                        "wind_speed": 6 + abs(math.sin(lat * lon)) * 18,
                        "wind_direction": (lat * lon) % 360,
                        "precipitation": abs(math.sin(lat + lon)) * 3,
                    },
                )
            )
        return items

    def _fallback_roads(self, focus: Dict[str, Any]) -> List[DatasetRecord]:
        roads: List[DatasetRecord] = []
        lon0 = focus["longitude"]
        lat0 = focus["latitude"]
        step = 0.01
        for xi in range(-12, 13):
            coords = []
            for yi in range(-12, 13):
                coords.append(
                    GeoPoint(longitude=lon0 + xi * step, latitude=lat0 + yi * step, altitude=20)
                )
            roads.append(
                DatasetRecord(
                    id=f"grid-v-{xi}",
                    path=GeoLine(coordinates=coords),
                    properties={"road_class": "primary" if xi % 3 == 0 else "secondary", "lanes": 3, "maxspeed": 55},
                )
            )
        for yi in range(-12, 13):
            coords = []
            for xi in range(-12, 13):
                coords.append(
                    GeoPoint(longitude=lon0 + xi * step, latitude=lat0 + yi * step, altitude=20)
                )
            roads.append(
                DatasetRecord(
                    id=f"grid-h-{yi}",
                    path=GeoLine(coordinates=coords),
                    properties={"road_class": "primary" if yi % 4 == 0 else "tertiary", "lanes": 2, "maxspeed": 45},
                )
            )
        return roads

    def _fallback_buildings(self, focus: Dict[str, Any]) -> List[DatasetRecord]:
        buildings: List[DatasetRecord] = []
        random.seed(42)
        for index in range(self.settings.building_limit):
            lon = focus["longitude"] + (random.random() - 0.5) * 0.2
            lat = focus["latitude"] + (random.random() - 0.5) * 0.16
            height = 12 + random.random() * 120
            buildings.append(
                DatasetRecord(
                    id=f"fallback-building-{index}",
                    position=GeoPoint(longitude=lon, latitude=lat, altitude=height / 2),
                    properties={"height_m": height, "width_m": 24, "depth_m": 18, "name": f"Building {index}"},
                )
            )
        return buildings

    def _time_of_day_factor(self) -> float:
        hour = datetime.utcnow().hour
        if 7 <= hour <= 10:
            return 0.94
        if 16 <= hour <= 20:
            return 0.97
        if 11 <= hour <= 15:
            return 0.68
        return 0.45

    def _centroid(self, coords: Sequence[Tuple[float, float]]) -> Tuple[float, float]:
        lon = sum(pt[0] for pt in coords) / len(coords)
        lat = sum(pt[1] for pt in coords) / len(coords)
        return lon, lat

    def _safe_float(self, value: Any, fallback: float) -> float:
        if value is None:
            return fallback
        if isinstance(value, (int, float)):
            return float(value)
        text = str(value).split(" ")[0].split(";")[0]
        try:
            return float(text)
        except ValueError:
            return fallback

    def _simplify_coords(
        self,
        coords: Sequence[Tuple[float, float]],
        max_points: int,
    ) -> List[Tuple[float, float]]:
        if len(coords) <= max_points:
            return list(coords)
        stride = max(1, len(coords) // (max_points - 1))
        simplified = list(coords[::stride])
        if simplified[-1] != coords[-1]:
            simplified.append(coords[-1])
        return simplified[:max_points]

    def _bbox_size(self, coords: Sequence[Tuple[float, float]]) -> Tuple[float, float]:
        if not coords:
            return 20.0, 20.0
        lons = [coord[0] for coord in coords]
        lats = [coord[1] for coord in coords]
        width = max(12.0, (max(lons) - min(lons)) * 111_320 * math.cos(math.radians(sum(lats) / len(lats))))
        depth = max(12.0, (max(lats) - min(lats)) * 110_540)
        return width, depth
