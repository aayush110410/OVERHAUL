"""
Service Client — HTTP inter-service communication
===================================================

Lightweight async HTTP client for calling other microservices.
Handles service discovery, retries, and circuit breaking.
"""

from __future__ import annotations

import asyncio
import os
import time
from typing import Any, Dict, Optional

import httpx

from services.shared.contracts import ServiceName

# ── Service URL resolution ───────────────────────────────────────

# Default ports for local development
_DEFAULT_PORTS = {
    ServiceName.GATEWAY: 8000,
    ServiceName.SIMULATION: 8001,
    ServiceName.LLM: 8002,
    ServiceName.DATA: 8003,
    ServiceName.VALIDATION: 8004,
    ServiceName.TRAFFIC_GOD: 8005,
}


def get_service_url(service: ServiceName) -> str:
    """Resolve service URL from env vars or default to localhost."""
    env_key = f"{service.value.upper()}_SERVICE_URL"
    url = os.getenv(env_key)
    if url:
        return url.rstrip("/")
    port = _DEFAULT_PORTS.get(service, 8000)
    return f"http://localhost:{port}"


# ── Circuit breaker ──────────────────────────────────────────────

class _CircuitState:
    def __init__(self, failure_threshold: int = 5, recovery_timeout: float = 30.0):
        self.failures = 0
        self.threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.last_failure: float = 0
        self.open = False

    def record_failure(self):
        self.failures += 1
        self.last_failure = time.time()
        if self.failures >= self.threshold:
            self.open = True

    def record_success(self):
        self.failures = 0
        self.open = False

    def allow_request(self) -> bool:
        if not self.open:
            return True
        if time.time() - self.last_failure > self.recovery_timeout:
            self.open = False  # half-open: allow one probe
            return True
        return False


_circuits: Dict[str, _CircuitState] = {}


def _get_circuit(service: ServiceName) -> _CircuitState:
    if service.value not in _circuits:
        _circuits[service.value] = _CircuitState()
    return _circuits[service.value]


# ── Service client ───────────────────────────────────────────────

class ServiceClient:
    """Async HTTP client for inter-service calls with circuit breaking."""

    def __init__(self, timeout: float = 30.0):
        self._timeout = timeout

    async def call(
        self,
        service: ServiceName,
        method: str,
        path: str,
        data: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None,
        retries: int = 2,
    ) -> Dict[str, Any]:
        """Make an HTTP call to another service.

        Args:
            service: Target service
            method: HTTP method (GET, POST, etc.)
            path: URL path (e.g., "/simulate")
            data: JSON body for POST/PUT
            params: Query parameters for GET
            retries: Number of retry attempts on failure

        Returns:
            Parsed JSON response dict

        Raises:
            ServiceUnavailableError: If the service is down or circuit is open
        """
        circuit = _get_circuit(service)
        if not circuit.allow_request():
            raise ServiceUnavailableError(
                f"{service.value} service circuit breaker open"
            )

        url = f"{get_service_url(service)}{path}"

        for attempt in range(retries + 1):
            try:
                async with httpx.AsyncClient(timeout=self._timeout) as client:
                    resp = await client.request(
                        method=method.upper(),
                        url=url,
                        json=data,
                        params=params,
                    )
                    resp.raise_for_status()
                    circuit.record_success()
                    return resp.json()

            except (httpx.ConnectError, httpx.ConnectTimeout) as exc:
                circuit.record_failure()
                if attempt == retries:
                    raise ServiceUnavailableError(
                        f"{service.value} service unreachable at {url}: {exc}"
                    ) from exc
                await asyncio.sleep(0.5 * (attempt + 1))

            except httpx.HTTPStatusError as exc:
                if exc.response.status_code >= 500 and attempt < retries:
                    await asyncio.sleep(0.5 * (attempt + 1))
                    continue
                circuit.record_failure()
                raise ServiceError(
                    f"{service.value} returned {exc.response.status_code}: "
                    f"{exc.response.text[:200]}"
                ) from exc

        raise ServiceUnavailableError(f"{service.value} exhausted retries")

    # ── Convenience methods ──

    async def get(self, service: ServiceName, path: str, **kwargs) -> Dict[str, Any]:
        return await self.call(service, "GET", path, **kwargs)

    async def post(self, service: ServiceName, path: str, data: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        return await self.call(service, "POST", path, data=data, **kwargs)


class ServiceError(Exception):
    """Non-retryable service error (e.g., 4xx)."""


class ServiceUnavailableError(Exception):
    """Service is down or circuit breaker is open."""


# Singleton client
_client: Optional[ServiceClient] = None


def get_client() -> ServiceClient:
    global _client
    if _client is None:
        _client = ServiceClient()
    return _client
