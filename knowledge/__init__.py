"""Local knowledge (PDF) indexing + retrieval.

This module provides a lightweight, offline-friendly RAG layer over workspace documents.
It defaults to local embeddings via sentence-transformers when available.
"""

from .index import (
    KnowledgeChunk,
    KnowledgeMatch,
    ingest_paths,
    ingest_pdf_paths,
    ingest_csv_paths,
    query_knowledge,
    load_index,
    save_index,
)
