from __future__ import annotations

import argparse
from pathlib import Path

from .index import INDEX_PATH, ingest_paths


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Ingest PDFs into the local knowledge index")
    p.add_argument("--root", default=".", help="Workspace folder to scan for PDFs/CSVs")
    p.add_argument("--output", default=str(INDEX_PATH), help="Where to write index.json")
    p.add_argument("--max-chars", type=int, default=1200)
    p.add_argument("--overlap-chars", type=int, default=150)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    out = Path(args.output)

    pdfs = sorted([p for p in root.rglob("*.pdf") if p.is_file()])
    csvs = sorted([p for p in root.rglob("*.csv") if p.is_file()])
    report = ingest_paths(pdfs + csvs, max_chars=args.max_chars, overlap_chars=args.overlap_chars, write_to=out)
    print(report)


if __name__ == "__main__":
    main()
