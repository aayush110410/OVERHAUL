import asyncio
import json

import httpx


async def main() -> None:
    payload = {
        "prompt": (
            "Give a very detailed corridor analysis for Sector-78 to Vasundhara. "
            "Include: live travel time & distance, current PM2.5, TomTom speed, "
            "baseline vs scenario table, and 5 quantified recommendations."
        ),
        "mode": "deep",
        "scenario": {
            "title": "test",
            "region": "Sector-78 to Vasundhara",
            "intervention": "prompt_only",
            "horizon_months": 12,
            "parameters": {},
        },
    }

    async with httpx.AsyncClient(timeout=180.0) as c:
        r = await c.post("http://127.0.0.1:8000/chat", json=payload)
        print("status", r.status_code)
        if r.status_code != 200:
            print("body", r.text[:2000])
            return
        data = r.json()

    outputs = data.get("outputs") or {}
    tldr = outputs.get("tldr") or ""
    provider = outputs.get("llm_provider")
    logs = outputs.get("logs") or []

    lower = tldr.lower()
    print("llm_provider", provider)
    print("tldr_chars", len(tldr))
    print("has_sections", all(k in lower for k in ["executive", "baseline", "recommend"]))
    print("tldr_head", tldr[:600].replace("\n", "\\n"))
    if logs:
        tail = logs[-12:]
        print("logs_tail", json.dumps(tail, ensure_ascii=False))


if __name__ == "__main__":
    asyncio.run(main())
