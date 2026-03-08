"""
Validation Microservice
========================

Standalone FastAPI service owning customer validation / survey
data persistence. Supports dual backends:
  - SQLite (local dev)
  - Supabase PostgreSQL (production)

Port: 8004 (default)

Endpoints:
  GET  /health
  GET  /stats
  GET  /entries
  POST /entries
  POST /entries/{entry_id}/approve
"""

from __future__ import annotations

import os
import sys
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException, Header, Query
from pydantic import BaseModel, Field

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from validation_store import (  # noqa: E402
    init_validation_db,
    create_entry,
    list_entries,
    get_stats,
    approve_entry,
)

app = FastAPI(title="OVERHAUL Validation Service", version="1.0.0")

_ADMIN_TOKEN = os.getenv("VALIDATION_ADMIN_TOKEN", "")


@app.on_event("startup")
async def _startup():
    await init_validation_db()


# ── Health ────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    try:
        stats = await get_stats()
        return {"service": "validation", "status": "ok", "entries": stats}
    except Exception as e:
        return {"service": "validation", "status": "degraded", "error": str(e)[:100]}


# ── CRUD ──────────────────────────────────────────────────────────

@app.get("/stats")
async def stats_endpoint():
    return await get_stats()


@app.get("/entries")
async def list_entries_endpoint(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    approved_only: bool = Query(True),
):
    return await list_entries(page=page, page_size=page_size, approved_only=approved_only)


class CreateEntryRequest(BaseModel):
    user_name: str = Field(..., min_length=1, max_length=200)
    email: str = Field("", max_length=200)
    role: str = Field("", max_length=100)
    organization: str = Field("", max_length=200)
    problem_relevance: int = Field(3, ge=1, le=5)
    has_experience: bool = False
    tools_shortcoming: str = Field("", max_length=2000)
    usage_contexts: str = Field("", max_length=2000)
    custom_feedback: str = Field("", max_length=5000)
    city: str = Field("delhi", max_length=100)


@app.post("/entries")
async def create_entry_endpoint(req: CreateEntryRequest):
    try:
        result = await create_entry(req.model_dump())
        return result
    except Exception as e:
        raise HTTPException(400, str(e)[:200])


@app.post("/entries/{entry_id}/approve")
async def approve_entry_endpoint(
    entry_id: str,
    authorization: str = Header(""),
):
    if not _ADMIN_TOKEN or authorization != f"Bearer {_ADMIN_TOKEN}":
        raise HTTPException(403, "Admin token required")
    try:
        result = await approve_entry(entry_id)
        return result
    except Exception as e:
        raise HTTPException(400, str(e)[:200])


# ── Run standalone ────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("VALIDATION_PORT", "8004"))
    uvicorn.run(app, host="0.0.0.0", port=port)
