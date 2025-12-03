"""Cleaner agent: normalizes collected datasets."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

CLEAN_DIR = Path("storage/cleaned")
CLEAN_DIR.mkdir(parents=True, exist_ok=True)


def clean(raw_blobs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    cleaned: List[Dict[str, Any]] = []
    for blob in raw_blobs:
        if "payload" in blob:
            cleaned.append({
                "source": blob["source"],
                "data": blob["payload"],
            })
        else:
            cleaned.append({"source": blob.get("source", "unknown"), "note": blob})
    (CLEAN_DIR / "latest_cleaned.json").write_text(str(cleaned))
    return cleaned
