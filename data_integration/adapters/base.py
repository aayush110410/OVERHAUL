"""
Data Adapter System — Base Classes & Registry
===============================================

Provides the abstract adapter interface and a central registry
that discovers, initialises and queries adapters by data domain.

Adapter hierarchy:
  DataAdapter (abstract)
    ├─ CSVAdapter        (local CSV / Excel files)
    ├─ LiveAPIAdapter    (external REST APIs: Open-Meteo, OSRM, TomTom, Nominatim)
    ├─ JSONAdapter       (local JSON files)
    └─ DatabaseAdapter   (SQLite / Supabase)

Every adapter exposes:
  - metadata:  name, source_type, domains, refresh policy
  - fetch():   returns normalised DataResult
  - health():  connectivity / freshness check
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Type


# ── Enums ─────────────────────────────────────────────────────────

class SourceType(str, Enum):
    CSV = "csv"
    JSON = "json"
    API = "api"
    DATABASE = "database"
    COMPUTED = "computed"


class DataDomain(str, Enum):
    AQI = "aqi"
    TRAFFIC = "traffic"
    WEATHER = "weather"
    ROUTING = "routing"
    GEOCODING = "geocoding"
    VALIDATION = "validation"
    HISTORICAL = "historical"


class RefreshPolicy(str, Enum):
    STATIC = "static"            # loaded once, never refreshed
    TTL = "ttl"                  # re-fetch after TTL expires
    PER_REQUEST = "per_request"  # fresh every call
    ON_DEMAND = "on_demand"      # caller triggers refresh


# ── Data result envelope ──────────────────────────────────────────

@dataclass
class DataResult:
    """Normalised wrapper returned by every adapter."""
    data: Any                               # the payload (dict, list, DataFrame, …)
    source: str = ""                        # human label e.g. "Open-Meteo AQI"
    source_type: SourceType = SourceType.CSV
    domain: DataDomain = DataDomain.AQI
    timestamp: float = 0.0                  # epoch when data was fetched
    cached: bool = False                    # True if served from cache
    stale: bool = False                     # True if past TTL but returned anyway
    error: Optional[str] = None             # non-None ⇒ failed fetch
    meta: Dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.error is None


# ── Abstract adapter ──────────────────────────────────────────────

class DataAdapter(ABC):
    """
    Base class for all data adapters.

    Subclasses MUST implement:
      - name        (property)
      - source_type (property)
      - domains     (property)
      - fetch(**kw) (async)

    Optionally override:
      - health()
      - close()
    """

    # ── Cache layer ───────────────────────────────────────────────

    def __init__(self, ttl_seconds: float = 0):
        self._ttl = ttl_seconds
        self._cache: Dict[str, DataResult] = {}

    @property
    @abstractmethod
    def name(self) -> str: ...

    @property
    @abstractmethod
    def source_type(self) -> SourceType: ...

    @property
    @abstractmethod
    def domains(self) -> List[DataDomain]: ...

    @property
    def refresh_policy(self) -> RefreshPolicy:
        if self._ttl <= 0:
            return RefreshPolicy.STATIC
        return RefreshPolicy.TTL

    # ── Core lifecycle ────────────────────────────────────────────

    @abstractmethod
    async def fetch(self, **kwargs) -> DataResult:
        """Return normalised data.  Subclasses implement the real work."""
        ...

    async def get(self, *, force: bool = False, **kwargs) -> DataResult:
        """
        Public entry-point.  Handles TTL caching around ``fetch()``.
        Pass ``force=True`` to bypass cache.
        """
        cache_key = self._make_cache_key(**kwargs)
        now = time.time()

        if not force and cache_key in self._cache:
            cached = self._cache[cache_key]
            age = now - cached.timestamp
            if self._ttl <= 0 or age < self._ttl:
                cached.cached = True
                return cached
            # stale but still return while we try to refresh
            cached.stale = True

        try:
            result = await self.fetch(**kwargs)
            result.timestamp = now
            self._cache[cache_key] = result
            return result
        except Exception as exc:
            # If we have stale data, return it with the error noted
            if cache_key in self._cache:
                stale = self._cache[cache_key]
                stale.stale = True
                stale.meta["last_error"] = str(exc)[:200]
                return stale
            return DataResult(
                data=None,
                source=self.name,
                source_type=self.source_type,
                error=str(exc)[:300],
                timestamp=now,
            )

    async def health(self) -> Dict[str, Any]:
        """Quick connectivity / freshness check."""
        return {
            "adapter": self.name,
            "source_type": self.source_type.value,
            "domains": [d.value for d in self.domains],
            "cache_entries": len(self._cache),
            "status": "ok",
        }

    async def close(self) -> None:
        """Release resources (HTTP clients, DB connections, etc.)."""
        pass

    def invalidate(self, **kwargs) -> None:
        """Drop a specific cached entry."""
        key = self._make_cache_key(**kwargs)
        self._cache.pop(key, None)

    def invalidate_all(self) -> None:
        """Drop the entire cache for this adapter."""
        self._cache.clear()

    # ── Internals ─────────────────────────────────────────────────

    @staticmethod
    def _make_cache_key(**kwargs) -> str:
        parts = sorted(f"{k}={v}" for k, v in kwargs.items() if v is not None)
        return "|".join(parts) or "__default__"


# ══════════════════════════════════════════════════════════════════
#  Adapter Registry
# ══════════════════════════════════════════════════════════════════

class AdapterRegistry:
    """
    Central registry of all active data adapters.

    Usage::

        reg = AdapterRegistry()
        reg.register(OpenMeteoAdapter())
        reg.register(NCRAqiCSVAdapter())

        # query by domain
        results = await reg.query(DataDomain.AQI, city="Delhi")
        # → list of DataResult from every adapter that serves AQI

        # query by name
        result = await reg.query_one("open_meteo_aqi", lat=28.62, lon=77.22)
    """

    def __init__(self):
        self._adapters: Dict[str, DataAdapter] = {}
        self._domain_index: Dict[DataDomain, List[str]] = {}

    def register(self, adapter: DataAdapter) -> None:
        self._adapters[adapter.name] = adapter
        for domain in adapter.domains:
            self._domain_index.setdefault(domain, []).append(adapter.name)

    def get(self, name: str) -> Optional[DataAdapter]:
        return self._adapters.get(name)

    def list_adapters(self) -> List[Dict[str, Any]]:
        return [
            {
                "name": a.name,
                "source_type": a.source_type.value,
                "domains": [d.value for d in a.domains],
                "refresh_policy": a.refresh_policy.value,
                "cache_entries": len(a._cache),
            }
            for a in self._adapters.values()
        ]

    def adapters_for_domain(self, domain: DataDomain) -> List[DataAdapter]:
        names = self._domain_index.get(domain, [])
        return [self._adapters[n] for n in names if n in self._adapters]

    async def query(self, domain: DataDomain, **kwargs) -> List[DataResult]:
        """Query all adapters registered for a domain."""
        adapters = self.adapters_for_domain(domain)
        results: List[DataResult] = []
        for adapter in adapters:
            result = await adapter.get(**kwargs)
            result.domain = domain
            results.append(result)
        return results

    async def query_one(self, adapter_name: str, **kwargs) -> DataResult:
        """Query a specific adapter by name."""
        adapter = self._adapters.get(adapter_name)
        if adapter is None:
            return DataResult(
                data=None,
                error=f"Adapter '{adapter_name}' not registered",
            )
        return await adapter.get(**kwargs)

    async def health(self) -> Dict[str, Any]:
        """Aggregate health from all adapters."""
        statuses = {}
        for name, adapter in self._adapters.items():
            statuses[name] = await adapter.health()
        all_ok = all(s.get("status") == "ok" for s in statuses.values())
        return {
            "status": "ok" if all_ok else "degraded",
            "adapters": statuses,
            "total": len(self._adapters),
        }

    async def close_all(self) -> None:
        for adapter in self._adapters.values():
            await adapter.close()


# ── Module-level singleton ────────────────────────────────────────

_registry: Optional[AdapterRegistry] = None


def get_adapter_registry() -> AdapterRegistry:
    """Return the singleton adapter registry (lazy-init, no adapters registered yet)."""
    global _registry
    if _registry is None:
        _registry = AdapterRegistry()
    return _registry
