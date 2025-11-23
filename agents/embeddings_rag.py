"""Embeddings + RAG agent."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

try:
    from sentence_transformers import SentenceTransformer
except ImportError:  # pragma: no cover - optional heavy dep
    SentenceTransformer = None

VECTOR_DB = Path("storage/vector_db") / "documents.json"


def ensure_index(config: Dict[str, Any]) -> None:
    VECTOR_DB.parent.mkdir(parents=True, exist_ok=True)
    if not VECTOR_DB.exists():
        VECTOR_DB.write_text(json.dumps({"documents": []}))


def _load_model() -> Any:
    if SentenceTransformer is None:
        return None
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    return SentenceTransformer(model_name)


MODEL = _load_model()


def upsert_documents(clean_docs: List[Dict[str, Any]]) -> Dict[str, Any]:
    db = json.loads(VECTOR_DB.read_text())
    for doc in clean_docs:
        text = json.dumps(doc)[:2000]
        embedding = MODEL.encode(text).tolist() if MODEL else []
        db["documents"].append({"text": text, "embedding": embedding})
    VECTOR_DB.write_text(json.dumps(db, indent=2))
    return {"count": len(db["documents"])}


def query(question: str, top_k: int = 3) -> List[Dict[str, Any]]:
    db = json.loads(VECTOR_DB.read_text())
    # TODO: implement cosine similarity. For now return first K docs.
    return db["documents"][:top_k]
