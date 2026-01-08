"""Customer validation persistence (SQLite via SQLAlchemy async).

This module intentionally contains no FastAPI-specific code.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from sqlalchemy import Boolean, DateTime, Integer, String, Text, func, select
from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


_EMAIL_RE = re.compile(r"^[^\s@]+@[^\s@]+\.[^\s@]+$")


def _default_db_url() -> str:
    data_dir = Path(__file__).parent / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    return f"sqlite+aiosqlite:///{(data_dir / 'validation.db').as_posix()}"


def get_validation_db_url() -> str:
    return os.getenv("OVERHAUL_VALIDATION_DB_URL", _default_db_url()).strip()


class Base(DeclarativeBase):
    pass


class ValidationEntry(Base):
    __tablename__ = "validation_entries"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    name: Mapped[str] = mapped_column(String(100), nullable=False)
    role: Mapped[Optional[str]] = mapped_column(String(80), nullable=True)
    organization: Mapped[Optional[str]] = mapped_column(String(120), nullable=True)

    email: Mapped[str] = mapped_column(String(254), nullable=False)

    rating: Mapped[int] = mapped_column(Integer, nullable=False)
    # Backwards-compat: keep feedback as a single combined field.
    feedback: Mapped[str] = mapped_column(Text, nullable=False)

    # New structured fields requested by product UX.
    need_validation: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    intended_use: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    location_query: Mapped[Optional[str]] = mapped_column(String(200), nullable=True)
    location_label: Mapped[Optional[str]] = mapped_column(String(300), nullable=True)
    location_lat: Mapped[Optional[str]] = mapped_column(String(40), nullable=True)
    location_lon: Mapped[Optional[str]] = mapped_column(String(40), nullable=True)

    is_approved: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    moderation_reason: Mapped[Optional[str]] = mapped_column(String(200), nullable=True)


_engine: Optional[AsyncEngine] = None
_sessionmaker: Optional[async_sessionmaker] = None


def get_engine() -> AsyncEngine:
    global _engine
    if _engine is None:
        _engine = create_async_engine(
            get_validation_db_url(),
            echo=False,
            pool_pre_ping=True,
        )
    return _engine


def get_sessionmaker() -> async_sessionmaker:
    global _sessionmaker
    if _sessionmaker is None:
        _sessionmaker = async_sessionmaker(get_engine(), expire_on_commit=False)
    return _sessionmaker


async def init_validation_db() -> None:
    engine = get_engine()
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

        # Lightweight SQLite migration: add new columns if the DB already exists.
        try:
            rows = await conn.exec_driver_sql("PRAGMA table_info(validation_entries)")
            existing_cols = {r[1] for r in rows.fetchall()}
            if "need_validation" not in existing_cols:
                await conn.exec_driver_sql("ALTER TABLE validation_entries ADD COLUMN need_validation TEXT")
            if "intended_use" not in existing_cols:
                await conn.exec_driver_sql("ALTER TABLE validation_entries ADD COLUMN intended_use TEXT")
        except Exception:
            # Best-effort migration; create_all covers fresh DBs.
            pass


def validate_and_normalize_email(email: str) -> str:
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
        return False, "empty"
    if len(text) < 12:
        return False, "too_short"
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


async def create_entry(*, payload: Dict[str, Any]) -> Dict[str, Any]:
    Session = get_sessionmaker()

    name = sanitize_text(payload.get("name"), max_len=100) or "Anonymous"
    role = sanitize_text(payload.get("role"), max_len=80)
    organization = sanitize_text(payload.get("organization"), max_len=120)

    email = validate_and_normalize_email(payload.get("email"))

    rating = int(payload.get("rating", 0) or 0)
    if rating < 1 or rating > 5:
        raise ValueError("Rating must be between 1 and 5")

    need_validation = sanitize_text(payload.get("need_validation"), max_len=2000)
    intended_use = sanitize_text(payload.get("intended_use"), max_len=2000)

    # Backwards compatibility: if older clients still send `feedback`, use it as need_validation.
    if not need_validation:
        need_validation = sanitize_text(payload.get("feedback"), max_len=2000)

    if not need_validation:
        raise ValueError("Need validation is required")
    if not intended_use:
        raise ValueError("How you will use it is required")

    combined_feedback = f"NEED: {need_validation}\n\nUSE: {intended_use}"

    approved, reason = moderation_heuristic(f"{need_validation} {intended_use}")

    entry = ValidationEntry(
        name=name,
        role=role,
        organization=organization,
        email=email,
        rating=rating,
        feedback=combined_feedback,
        need_validation=need_validation,
        intended_use=intended_use,
        location_query=sanitize_text(payload.get("location"), max_len=200),
        is_approved=approved,
        moderation_reason=reason,
    )

    async with Session() as session:
        session.add(entry)
        await session.commit()
        await session.refresh(entry)

    return {
        "id": entry.id,
        "created_at": entry.created_at.isoformat() if entry.created_at else datetime.utcnow().isoformat() + "Z",
        "is_approved": bool(entry.is_approved),
        "display_email": anonymize_email(entry.email),
    }


async def list_entries(*, page: int, page_size: int, approved_only: bool = True) -> Dict[str, Any]:
    page = max(1, int(page))
    page_size = max(1, min(int(page_size), 50))

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
        items.append(
            {
                "id": r.id,
                "created_at": r.created_at.isoformat() if r.created_at else None,
                "name": r.name,
                "role": r.role,
                "organization": r.organization,
                "rating": r.rating,
                "need_validation": r.need_validation,
                "intended_use": r.intended_use,
                "feedback": r.feedback,
                "location_label": r.location_label,
                "display_email": anonymize_email(r.email),
            }
        )

    return {"page": page, "page_size": page_size, "items": items}


async def approve_entry(*, entry_id: int) -> bool:
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
