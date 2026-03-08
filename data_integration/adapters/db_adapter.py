"""
Database Adapter
=================

Adapter for the validation store (SQLite local / Supabase production).
Wraps validation_store.py behind the DataAdapter interface.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from data_integration.adapters.base import (
    DataAdapter,
    DataDomain,
    DataResult,
    SourceType,
    RefreshPolicy,
)


class ValidationDBAdapter(DataAdapter):
    """
    Adapter for customer validation / survey data.

    Kwargs accepted by fetch():
      action:   "stats" | "list" | "create" | "approve"
      page:     page number (for list)
      page_size: items per page
      data:     dict payload (for create)
      entry_id: str (for approve)
    """

    def __init__(self):
        super().__init__(ttl_seconds=0)  # no caching — always fresh from DB

    @property
    def name(self) -> str:
        return "validation_db"

    @property
    def source_type(self) -> SourceType:
        return SourceType.DATABASE

    @property
    def domains(self) -> List[DataDomain]:
        return [DataDomain.VALIDATION]

    @property
    def refresh_policy(self) -> RefreshPolicy:
        return RefreshPolicy.PER_REQUEST

    async def fetch(self, **kwargs) -> DataResult:
        # Lazy import to avoid circular deps and heavy startup cost
        from validation_store import (
            create_entry,
            get_stats,
            list_entries,
            approve_entry,
        )

        action = kwargs.get("action", "stats")

        if action == "stats":
            result = await get_stats()
            return DataResult(
                data=result,
                source="ValidationDB",
                source_type=SourceType.DATABASE,
                domain=DataDomain.VALIDATION,
            )

        if action == "list":
            page = kwargs.get("page", 1)
            page_size = kwargs.get("page_size", 20)
            approved_only = kwargs.get("approved_only", True)
            result = await list_entries(
                page=page, page_size=page_size, approved_only=approved_only,
            )
            return DataResult(
                data=result,
                source="ValidationDB",
                source_type=SourceType.DATABASE,
                domain=DataDomain.VALIDATION,
            )

        if action == "create":
            payload = kwargs.get("data", {})
            result = await create_entry(payload)
            return DataResult(
                data=result,
                source="ValidationDB",
                source_type=SourceType.DATABASE,
                domain=DataDomain.VALIDATION,
            )

        if action == "approve":
            entry_id = kwargs.get("entry_id", "")
            result = await approve_entry(entry_id)
            return DataResult(
                data=result,
                source="ValidationDB",
                source_type=SourceType.DATABASE,
                domain=DataDomain.VALIDATION,
            )

        return DataResult(
            data=None,
            source=self.name,
            error=f"Unknown action: {action}",
        )

    async def health(self) -> Dict[str, Any]:
        base = await super().health()
        try:
            from validation_store import get_stats
            stats = await get_stats()
            base["entries"] = stats
        except Exception as e:
            base["status"] = "degraded"
            base["error"] = str(e)[:100]
        return base
