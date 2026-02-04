"""Customer validation persistence with Supabase (PostgreSQL) support.

Supports both:
- SQLite (local development, default)
- Supabase PostgreSQL (production, when SUPABASE_URL is set)

Set environment variables:
- SUPABASE_URL: Your Supabase project URL (e.g., https://xxx.supabase.co)
- SUPABASE_ANON_KEY: Your Supabase anon/public key
- SUPABASE_SERVICE_KEY: (Optional) Service role key for admin operations
"""

from __future__ import annotations

import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, cast

# Check if we're using Supabase
SUPABASE_URL = os.getenv("SUPABASE_URL", "").strip()
SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY", "").strip()
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY", "").strip()
USE_SUPABASE = bool(SUPABASE_URL and SUPABASE_ANON_KEY)

supabase = None
if USE_SUPABASE:
    try:
        from supabase import create_client, Client
        supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY or SUPABASE_ANON_KEY)
        print(f"✓ Connected to Supabase: {SUPABASE_URL[:40]}...")
    except ImportError:
        print("⚠ supabase-py not installed. Run: pip install supabase")
        USE_SUPABASE = False
    except Exception as e:
        print(f"⚠ Supabase connection failed: {e}")
        USE_SUPABASE = False

# Fallback to SQLite for local dev
if not USE_SUPABASE:
    from sqlalchemy import Boolean, DateTime, Integer, String, Text, func, select
    from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker, create_async_engine
    from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

_EMAIL_RE = re.compile(r"^[^\s@]+@[^\s@]+\.[^\s@]+$")


# =============================================================================
# SQLITE FALLBACK (Local Development)
# =============================================================================

def _default_db_url() -> str:
    data_dir = Path(__file__).parent / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    return f"sqlite+aiosqlite:///{(data_dir / 'validation.db').as_posix()}"


def get_validation_db_url() -> str:
    return os.getenv("OVERHAUL_VALIDATION_DB_URL", _default_db_url()).strip()


if not USE_SUPABASE:
    class Base(DeclarativeBase):
        pass

    class ValidationEntry(Base):
        __tablename__ = "validation_entries"

        id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
        created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)

        name: Mapped[str] = mapped_column(String(100), nullable=False)
        email: Mapped[str] = mapped_column(String(254), nullable=False)
        role: Mapped[Optional[str]] = mapped_column(String(80), nullable=True)
        organization: Mapped[Optional[str]] = mapped_column(String(120), nullable=True)

        # New survey fields
        problem_relevance: Mapped[int] = mapped_column(Integer, nullable=False, default=5)
        has_experience: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
        tools_shortcoming: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
        usage_contexts: Mapped[Optional[str]] = mapped_column(Text, nullable=True)  # JSON array as string
        learning_tool_value: Mapped[Optional[str]] = mapped_column(String(10), nullable=True)
        interesting_aspect: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
        suggested_improvement: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
        interest_in_trying: Mapped[Optional[str]] = mapped_column(String(10), nullable=True)

        # Legacy fields
        rating: Mapped[int] = mapped_column(Integer, nullable=False, default=5)
        feedback: Mapped[str] = mapped_column(Text, nullable=False, default="")
        need_validation: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
        intended_use: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

        location_query: Mapped[Optional[str]] = mapped_column(String(200), nullable=True)
        location_label: Mapped[Optional[str]] = mapped_column(String(300), nullable=True)
        location_lat: Mapped[Optional[str]] = mapped_column(String(40), nullable=True)
        location_lon: Mapped[Optional[str]] = mapped_column(String(40), nullable=True)

        is_approved: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
        moderation_reason: Mapped[Optional[str]] = mapped_column(String(200), nullable=True)


_engine: Optional[Any] = None
_sessionmaker: Optional[Any] = None


def get_engine():
    global _engine
    if _engine is None and not USE_SUPABASE:
        _engine = create_async_engine(
            get_validation_db_url(),
            echo=False,
            pool_pre_ping=True,
        )
    return _engine


def get_sessionmaker():
    global _sessionmaker
    if _sessionmaker is None and not USE_SUPABASE:
        _sessionmaker = async_sessionmaker(get_engine(), expire_on_commit=False)
    return _sessionmaker


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def validate_and_normalize_email(email: Optional[str]) -> str:
    value = (email or "").strip()
    if not value or len(value) > 254 or not _EMAIL_RE.match(value):
        raise ValueError("Invalid email")
    return value


def anonymize_email(email: str) -> str:
    value = (email or "").strip()
    if "@" not in value:
        return "***"
    local, domain = value.split("@", 1)
    if not local:
        return f"***@{domain}"
    head = local[0]
    return f"{head}{'*' * 3}@{domain}"


def moderation_heuristic(text_blob: str) -> Tuple[bool, Optional[str]]:
    text = (text_blob or "").strip().lower()
    if not text:
        return True, None  # Empty is OK for optional fields
    if "http://" in text or "https://" in text or "www." in text:
        return False, "contains_link"
    return True, None


def sanitize_text(value: Optional[str], *, max_len: int) -> Optional[str]:
    if value is None:
        return None
    cleaned = " ".join(str(value).replace("\u0000", " ").split())
    cleaned = cleaned.strip()
    if not cleaned:
        return None
    return cleaned[:max_len]


# =============================================================================
# DATABASE INITIALIZATION
# =============================================================================

async def init_validation_db() -> None:
    if USE_SUPABASE:
        # Supabase tables are created via SQL migration
        print("✓ Using Supabase for validation storage")
        return

    engine = get_engine()
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

        # Lightweight SQLite migration: add new columns if the DB already exists.
        try:
            rows = await conn.exec_driver_sql("PRAGMA table_info(validation_entries)")
            existing_cols = {r[1] for r in rows.fetchall()}
            
            new_columns = [
                ("problem_relevance", "INTEGER DEFAULT 5"),
                ("has_experience", "BOOLEAN DEFAULT 0"),
                ("tools_shortcoming", "TEXT"),
                ("usage_contexts", "TEXT"),
                ("learning_tool_value", "VARCHAR(10)"),
                ("interesting_aspect", "VARCHAR(100)"),
                ("suggested_improvement", "TEXT"),
                ("interest_in_trying", "VARCHAR(10)"),
                ("need_validation", "TEXT"),
                ("intended_use", "TEXT"),
            ]
            
            for col_name, col_type in new_columns:
                if col_name not in existing_cols:
                    await conn.exec_driver_sql(f"ALTER TABLE validation_entries ADD COLUMN {col_name} {col_type}")
        except Exception:
            pass

    print("✓ SQLite validation database initialized")


# =============================================================================
# CREATE ENTRY
# =============================================================================

async def create_entry(*, payload: Dict[str, Any]) -> Dict[str, Any]:
    name = sanitize_text(payload.get("name"), max_len=100) or "Anonymous"
    email = validate_and_normalize_email(cast(Optional[str], payload.get("email")))
    role = sanitize_text(payload.get("role"), max_len=80)
    organization = sanitize_text(payload.get("organization"), max_len=120)

    problem_relevance = int(payload.get("problem_relevance", 5) or 5)
    if problem_relevance < 1 or problem_relevance > 5:
        problem_relevance = 5

    has_experience = payload.get("has_experience", False) in [True, "true", "yes", "Yes", 1, "1"]
    tools_shortcoming = sanitize_text(payload.get("tools_shortcoming"), max_len=2000)
    
    usage_contexts = payload.get("usage_contexts", [])
    if isinstance(usage_contexts, list):
        usage_contexts_json = json.dumps(usage_contexts)
    else:
        usage_contexts_json = "[]"

    learning_tool_value = sanitize_text(payload.get("learning_tool_value"), max_len=10)
    interesting_aspect = sanitize_text(payload.get("interesting_aspect"), max_len=100)
    suggested_improvement = sanitize_text(payload.get("suggested_improvement"), max_len=2000)
    interest_in_trying = sanitize_text(payload.get("interest_in_trying"), max_len=10)

    # Moderation on text fields
    approved = True
    reason = None
    for text in [tools_shortcoming, suggested_improvement]:
        if text:
            ok, r = moderation_heuristic(text)
            if not ok:
                approved = False
                reason = r
                break

    if USE_SUPABASE and supabase:
        data = {
            "name": name,
            "email": email,
            "role": role,
            "organization": organization,
            "problem_relevance": problem_relevance,
            "has_experience": has_experience,
            "tools_shortcoming": tools_shortcoming,
            "usage_contexts": usage_contexts if isinstance(usage_contexts, list) else [],
            "learning_tool_value": learning_tool_value,
            "interesting_aspect": interesting_aspect,
            "suggested_improvement": suggested_improvement,
            "interest_in_trying": interest_in_trying,
            "rating": problem_relevance,
            "feedback": f"Survey response from {name}",
            "is_approved": approved,
            "moderation_reason": reason,
        }
        
        result = supabase.table("validation_entries").insert(data).execute()
        
        if result.data and len(result.data) > 0:
            entry = cast(Dict[str, Any], result.data[0])
            return {
                "id": entry.get("id"),
                "created_at": entry.get("created_at"),
                "is_approved": entry.get("is_approved", True),
                "display_email": anonymize_email(email),
            }
        else:
            raise ValueError("Failed to create entry in Supabase")

    else:
        # SQLite fallback
        Session = get_sessionmaker()
        
        sqlite_entry = ValidationEntry(
            name=name,
            email=email,
            role=role,
            organization=organization,
            problem_relevance=problem_relevance,
            has_experience=has_experience,
            tools_shortcoming=tools_shortcoming,
            usage_contexts=usage_contexts_json,
            learning_tool_value=learning_tool_value,
            interesting_aspect=interesting_aspect,
            suggested_improvement=suggested_improvement,
            interest_in_trying=interest_in_trying,
            rating=problem_relevance,
            feedback=f"Survey response from {name}",
            is_approved=approved,
            moderation_reason=reason,
        )

        async with Session() as session:
            session.add(sqlite_entry)
            await session.commit()
            await session.refresh(sqlite_entry)

        return {
            "id": sqlite_entry.id,
            "created_at": sqlite_entry.created_at.isoformat() if sqlite_entry.created_at else datetime.utcnow().isoformat() + "Z",
            "is_approved": bool(sqlite_entry.is_approved),
            "display_email": anonymize_email(sqlite_entry.email),
        }


# =============================================================================
# LIST ENTRIES
# =============================================================================

async def list_entries(*, page: int, page_size: int, approved_only: bool = True) -> Dict[str, Any]:
    page = max(1, int(page))
    page_size = max(1, min(int(page_size), 50))

    if USE_SUPABASE and supabase:
        query = supabase.table("validation_entries").select("*")
        
        if approved_only:
            query = query.eq("is_approved", True)
        
        query = query.order("created_at", desc=True).range((page - 1) * page_size, page * page_size - 1)
        
        result = query.execute()
        rows = result.data if result.data else []

        items = []
        for r in rows:
            row = cast(Dict[str, Any], r)
            items.append({
                "id": row.get("id"),
                "created_at": row.get("created_at"),
                "name": row.get("name"),
                "role": row.get("role"),
                "organization": row.get("organization"),
                "problem_relevance": row.get("problem_relevance"),
                "has_experience": row.get("has_experience"),
                "tools_shortcoming": row.get("tools_shortcoming"),
                "usage_contexts": row.get("usage_contexts", []),
                "learning_tool_value": row.get("learning_tool_value"),
                "interesting_aspect": row.get("interesting_aspect"),
                "suggested_improvement": row.get("suggested_improvement"),
                "interest_in_trying": row.get("interest_in_trying"),
                "display_email": anonymize_email(cast(str, row.get("email", ""))),
            })

        return {"page": page, "page_size": page_size, "items": items}

    else:
        # SQLite fallback
        Session = get_sessionmaker()
        async with Session() as session:
            stmt = select(ValidationEntry)
            if approved_only:
                stmt = stmt.where(ValidationEntry.is_approved.is_(True))
            stmt = stmt.order_by(ValidationEntry.created_at.desc(), ValidationEntry.id.desc())
            stmt = stmt.offset((page - 1) * page_size).limit(page_size)

            rows = (await session.execute(stmt)).scalars().all()

        items = []
        for r in rows:
            contexts = []
            if r.usage_contexts:
                try:
                    contexts = json.loads(r.usage_contexts)
                except:
                    pass
            
            items.append({
                "id": r.id,
                "created_at": r.created_at.isoformat() if r.created_at else None,
                "name": r.name,
                "role": r.role,
                "organization": r.organization,
                "problem_relevance": r.problem_relevance,
                "has_experience": r.has_experience,
                "tools_shortcoming": r.tools_shortcoming,
                "usage_contexts": contexts,
                "learning_tool_value": r.learning_tool_value,
                "interesting_aspect": r.interesting_aspect,
                "suggested_improvement": r.suggested_improvement,
                "interest_in_trying": r.interest_in_trying,
                "display_email": anonymize_email(r.email),
            })

        return {"page": page, "page_size": page_size, "items": items}


# =============================================================================
# GET STATS (For live counter)
# =============================================================================

async def get_stats() -> Dict[str, Any]:
    if USE_SUPABASE and supabase:
        try:
            # Try to use the stats function
            result = supabase.rpc("get_validation_stats").execute()
            if result.data:
                return result.data
        except:
            pass
        
        # Fallback: manual count
        result = supabase.table("validation_entries").select("id", count="exact").eq("is_approved", True).execute()
        return {
            "total_entries": result.count if result.count else 0,
        }

    else:
        # SQLite
        Session = get_sessionmaker()
        async with Session() as session:
            from sqlalchemy import func as sqlfunc
            stmt = select(sqlfunc.count(ValidationEntry.id)).where(ValidationEntry.is_approved.is_(True))
            result = await session.execute(stmt)
            total = result.scalar() or 0
            
            return {
                "total_entries": total,
            }


# =============================================================================
# APPROVE ENTRY (Admin)
# =============================================================================

async def approve_entry(*, entry_id: int) -> bool:
    if USE_SUPABASE and supabase:
        result = supabase.table("validation_entries").update({
            "is_approved": True,
            "moderation_reason": None
        }).eq("id", entry_id).execute()
        
        return bool(result.data)

    else:
        Session = get_sessionmaker()
        async with Session() as session:
            stmt = select(ValidationEntry).where(ValidationEntry.id == int(entry_id))
            entry = (await session.execute(stmt)).scalars().first()
            if entry is None:
                return False
            entry.is_approved = True
            entry.moderation_reason = None
            await session.commit()
            return True


async def set_entry_location(
    *,
    entry_id: int,
    location_label: Optional[str],
    lat: Optional[float],
    lon: Optional[float],
) -> bool:
    if USE_SUPABASE and supabase:
        result = supabase.table("validation_entries").update({
            "location_label": location_label,
            "location_lat": str(lat) if lat else None,
            "location_lon": str(lon) if lon else None,
        }).eq("id", entry_id).execute()
        
        return bool(result.data)

    else:
        Session = get_sessionmaker()
        async with Session() as session:
            stmt = select(ValidationEntry).where(ValidationEntry.id == int(entry_id))
            entry = (await session.execute(stmt)).scalars().first()
            if entry is None:
                return False

            entry.location_label = (location_label or None)
            entry.location_lat = None if lat is None else str(lat)
            entry.location_lon = None if lon is None else str(lon)
            await session.commit()
            return True
