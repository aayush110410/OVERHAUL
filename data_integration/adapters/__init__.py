"""Data adapter system — unified data access layer."""

from data_integration.adapters.base import (
    AdapterRegistry,
    DataAdapter,
    DataDomain,
    DataResult,
    RefreshPolicy,
    SourceType,
    get_adapter_registry,
)

__all__ = [
    "AdapterRegistry",
    "DataAdapter",
    "DataDomain",
    "DataResult",
    "RefreshPolicy",
    "SourceType",
    "get_adapter_registry",
]
