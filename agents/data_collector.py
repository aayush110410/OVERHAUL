"""Data Collector Agent.

Responsibilities:
- Execute scripted web/API queries (OpenAQ, WHO, World Bank, etc.).
- Fallback to polite scraping when APIs unavailable.
- Emit structured JSON artifacts per Agentic prompt-chaining guidance.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List

import httpx

SEARCH_QUERIES = [
    "EV adoption AQI reduction study China 2018 2019 2020",
    "electric vehicle noise impact study urban",
    "Noida vehicle count dataset",
    "Delhi NCR vehicular emissions 2019 dataset",
    "e-rickshaw adoption India study",
    "electric two-wheeler impacts India study",
    "study 'EV penetration' 'AQI' 'China' 'health' 'case study'",
]

RAW_DIR = Path("storage/raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)


async def fetch_openaq(session: httpx.AsyncClient, config: Dict[str, Any]) -> Dict[str, Any]:
    params = {
        "coordinates": f"{config['region']['bbox']['min_lat']},{config['region']['bbox']['min_lon']}",
        "radius": 5000,
        "limit": 100,
    }
    resp = await session.get(config["apis"]["openaq"], params=params, timeout=30)
    resp.raise_for_status()
    return resp.json()


def collect_sources(config: Dict[str, Any], prompt: str) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    timestamp = int(time.time())

    # Async API fetches (OpenAQ as example)
    try:
        import anyio

        async def main() -> None:
            async with httpx.AsyncClient(headers=config["apis"].get("custom_headers")) as session:
                data = await fetch_openaq(session, config)
                results.append({"source": "openaq", "payload": data})

        anyio.run(main)
    except Exception as exc:  # pragma: no cover - network variability
        results.append({"source": "openaq", "error": str(exc)})

    # Keyword search placeholders
    for query in SEARCH_QUERIES:
        # TODO: integrate real search API (SerpAPI, Bing, etc.)
        results.append({
            "source": "web_search",
            "query": query,
            "snippets": [f"Placeholder snippet for '{query}'"],
        })

    raw_path = RAW_DIR / f"collector_{timestamp}.json"
    raw_path.write_text(json.dumps({"prompt": prompt, "results": results}, indent=2))
    return results
