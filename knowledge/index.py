from __future__ import annotations

import json
import math
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .pdf_utils import chunk_text, extract_pdf_pages
from .csv_utils import csv_to_documents, summarize_csv


INDEX_PATH = Path("storage/knowledge/index.json")


@dataclass
class KnowledgeChunk:
    id: str
    source_path: str
    page_number: int
    chunk_index: int
    text: str
    embedding: Optional[List[float]] = None


@dataclass
class KnowledgeMatch:
    chunk: KnowledgeChunk
    score: float


def _try_load_embedder() -> Optional[Any]:
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore

        return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    except Exception:
        return None


_EMBEDDER = None


def _embed_texts(texts: Sequence[str]) -> Optional[np.ndarray]:
    global _EMBEDDER
    if _EMBEDDER is None:
        _EMBEDDER = _try_load_embedder()
    if _EMBEDDER is None:
        return None

    vecs = _EMBEDDER.encode(list(texts), normalize_embeddings=True)
    return np.asarray(vecs, dtype=np.float32)


def load_index(path: Path = INDEX_PATH) -> List[KnowledgeChunk]:
    if not path.exists():
        return []
    payload = json.loads(path.read_text())
    docs = payload.get("documents", [])
    chunks: List[KnowledgeChunk] = []
    for d in docs:
        chunks.append(
            KnowledgeChunk(
                id=d["id"],
                source_path=d["source_path"],
                page_number=int(d.get("page_number", 1)),
                chunk_index=int(d.get("chunk_index", 0)),
                text=d.get("text", ""),
                embedding=d.get("embedding"),
            )
        )
    return chunks


def save_index(chunks: Sequence[KnowledgeChunk], path: Path = INDEX_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": 1,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "documents": [asdict(c) for c in chunks],
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))


def ingest_pdf_paths(
    pdf_paths: Iterable[Path],
    *,
    max_chars: int = 1200,
    overlap_chars: int = 150,
    write_to: Path = INDEX_PATH,
) -> Dict[str, Any]:
    pdf_paths = [Path(p) for p in pdf_paths]
    extracted: List[KnowledgeChunk] = []

    for pdf in pdf_paths:
        if not pdf.exists() or pdf.suffix.lower() != ".pdf":
            continue

        pages = extract_pdf_pages(pdf)
        for page in pages:
            for chunk_idx, chunk in enumerate(chunk_text(page.text, max_chars=max_chars, overlap_chars=overlap_chars)):
                extracted.append(
                    KnowledgeChunk(
                        id=str(uuid.uuid4()),
                        source_path=str(pdf.as_posix()),
                        page_number=page.page_number,
                        chunk_index=chunk_idx,
                        text=chunk,
                        embedding=None,
                    )
                )

    embeddings = _embed_texts([c.text for c in extracted])
    if embeddings is not None:
        for c, vec in zip(extracted, embeddings):
            c.embedding = vec.astype(float).tolist()

    save_index(extracted, path=write_to)

    return {
        "pdf_count": len(pdf_paths),
        "chunk_count": len(extracted),
        "embedded": embeddings is not None,
        "index_path": str(write_to.as_posix()),
    }


def ingest_csv_paths(
    csv_paths: Iterable[Path],
    *,
    write_to: Path = INDEX_PATH,
) -> Dict[str, Any]:
    csv_paths = [Path(p) for p in csv_paths]
    extracted: List[KnowledgeChunk] = []

    for csv_path in csv_paths:
        if not csv_path.exists() or csv_path.suffix.lower() != ".csv":
            continue
        summary = summarize_csv(csv_path)
        docs = csv_to_documents(summary)
        for i, doc in enumerate(docs):
            extracted.append(
                KnowledgeChunk(
                    id=str(uuid.uuid4()),
                    source_path=doc.get("source_path", str(csv_path.as_posix())),
                    page_number=1,
                    chunk_index=i,
                    text=doc.get("text", ""),
                    embedding=None,
                )
            )

    embeddings = _embed_texts([c.text for c in extracted])
    if embeddings is not None:
        for c, vec in zip(extracted, embeddings):
            c.embedding = vec.astype(float).tolist()

    save_index(extracted, path=write_to)
    return {
        "csv_count": len(csv_paths),
        "chunk_count": len(extracted),
        "embedded": embeddings is not None,
        "index_path": str(write_to.as_posix()),
    }


def ingest_paths(
    paths: Iterable[Path],
    *,
    max_chars: int = 1200,
    overlap_chars: int = 150,
    write_to: Path = INDEX_PATH,
) -> Dict[str, Any]:
    """Ingest a mixed set of files (PDF + CSV) into a single index."""
    paths = [Path(p) for p in paths]

    pdfs = [p for p in paths if p.suffix.lower() == ".pdf"]
    csvs = [p for p in paths if p.suffix.lower() == ".csv"]

    chunks: List[KnowledgeChunk] = []

    # PDFs
    for pdf in pdfs:
        if not pdf.exists():
            continue
        pages = extract_pdf_pages(pdf)
        for page in pages:
            for chunk_idx, chunk in enumerate(chunk_text(page.text, max_chars=max_chars, overlap_chars=overlap_chars)):
                chunks.append(
                    KnowledgeChunk(
                        id=str(uuid.uuid4()),
                        source_path=str(pdf.as_posix()),
                        page_number=page.page_number,
                        chunk_index=chunk_idx,
                        text=chunk,
                        embedding=None,
                    )
                )

    # CSVs (summary docs)
    for csv_path in csvs:
        if not csv_path.exists():
            continue
        summary = summarize_csv(csv_path)
        docs = csv_to_documents(summary)
        for i, doc in enumerate(docs):
            chunks.append(
                KnowledgeChunk(
                    id=str(uuid.uuid4()),
                    source_path=doc.get("source_path", str(csv_path.as_posix())),
                    page_number=1,
                    chunk_index=i,
                    text=doc.get("text", ""),
                    embedding=None,
                )
            )

    embeddings = _embed_texts([c.text for c in chunks])
    if embeddings is not None:
        for c, vec in zip(chunks, embeddings):
            c.embedding = vec.astype(float).tolist()

    save_index(chunks, path=write_to)
    return {
        "file_count": len(paths),
        "pdf_count": len(pdfs),
        "csv_count": len(csvs),
        "chunk_count": len(chunks),
        "embedded": embeddings is not None,
        "index_path": str(write_to.as_posix()),
    }


def _keyword_score(query: str, text: str) -> float:
    q = [t for t in query.lower().split() if len(t) > 2]
    if not q:
        return 0.0
    t = text.lower()
    hits = sum(1 for tok in q if tok in t)
    return hits / max(1, len(q))


def query_knowledge(
    question: str,
    *,
    top_k: int = 4,
    index_path: Path = INDEX_PATH,
) -> List[KnowledgeMatch]:
    chunks = load_index(index_path)
    if not chunks:
        return []

    query_vec = _embed_texts([question])
    matches: List[Tuple[KnowledgeChunk, float]] = []

    if query_vec is not None:
        q = query_vec[0]
        for c in chunks:
            if not c.embedding:
                continue
            v = np.asarray(c.embedding, dtype=np.float32)
            score = float(np.dot(q, v))
            matches.append((c, score))
    else:
        for c in chunks:
            score = _keyword_score(question, c.text)
            matches.append((c, score))

    matches.sort(key=lambda x: x[1], reverse=True)
    results: List[KnowledgeMatch] = []
    for c, score in matches[: max(0, top_k)]:
        results.append(KnowledgeMatch(chunk=c, score=score))
    return results
