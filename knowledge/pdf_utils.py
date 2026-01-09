from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

from pypdf import PdfReader


@dataclass(frozen=True)
class PdfPageText:
    path: Path
    page_number: int  # 1-based
    text: str


def extract_pdf_pages(path: Path) -> List[PdfPageText]:
    reader = PdfReader(str(path))
    pages: List[PdfPageText] = []
    for idx, page in enumerate(reader.pages):
        raw = page.extract_text() or ""
        text = raw.replace("\uf0de", " ").replace("\uf0dd", " ").strip()
        if text:
            pages.append(PdfPageText(path=path, page_number=idx + 1, text=text))
    return pages


def chunk_text(text: str, *, max_chars: int = 1200, overlap_chars: int = 150) -> Iterable[str]:
    """Simple character-based chunker.

    Keeps implementation dependency-free; overlap helps preserve context.
    """
    if max_chars <= 0:
        raise ValueError("max_chars must be > 0")
    if overlap_chars < 0:
        raise ValueError("overlap_chars must be >= 0")

    cleaned = " ".join(text.split())
    if not cleaned:
        return

    start = 0
    n = len(cleaned)
    while start < n:
        end = min(n, start + max_chars)
        yield cleaned[start:end]
        if end >= n:
            break
        start = max(0, end - overlap_chars)
