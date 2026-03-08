"""
Traffic God Microservice
=========================

Standalone FastAPI service owning the Traffic God subsystem:
  - Custom Traffic God LLM (local transformer model)
  - Perception pipeline (YOLO-based video analysis)
  - RL training bridge

Port: 8005 (default)

Endpoints:
  GET  /health
  POST /llm            — Traffic God custom LLM inference
  GET  /llm/status     — Model load status
  POST /perception     — Video perception pipeline
"""

from __future__ import annotations

import os
import sys
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

app = FastAPI(title="OVERHAUL Traffic God Service", version="1.0.0")

# ── Lazy-load Traffic God LLM ─────────────────────────────────────

_TRAFFIC_GOD_LLM = None
_TG_AVAILABLE = False


def _get_traffic_god_llm():
    global _TRAFFIC_GOD_LLM, _TG_AVAILABLE
    if _TRAFFIC_GOD_LLM is not None:
        return _TRAFFIC_GOD_LLM
    try:
        from traffic_god_bridge import TrafficGodLLM
        _TRAFFIC_GOD_LLM = TrafficGodLLM()
        _TG_AVAILABLE = True
        return _TRAFFIC_GOD_LLM
    except Exception:
        _TG_AVAILABLE = False
        return None


# ── Lazy-load Traffic God Service ──────────────────────────────────

_TRAFFIC_GOD_SERVICE = None


def _get_traffic_god_service():
    global _TRAFFIC_GOD_SERVICE
    if _TRAFFIC_GOD_SERVICE is not None:
        return _TRAFFIC_GOD_SERVICE
    try:
        from traffic_god_bridge import TrafficGodService
        _TRAFFIC_GOD_SERVICE = TrafficGodService()
        return _TRAFFIC_GOD_SERVICE
    except Exception:
        return None


# ── Health ────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    llm = _get_traffic_god_llm()
    svc = _get_traffic_god_service()
    return {
        "service": "traffic_god",
        "status": "ok" if (llm or svc) else "degraded",
        "llm_loaded": llm is not None,
        "perception_available": svc is not None,
    }


# ── Traffic God LLM ──────────────────────────────────────────────

class TrafficGodLLMRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=5000)
    temperature: float = Field(0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(512, ge=50, le=4096)


@app.post("/llm")
async def traffic_god_llm_chat(req: TrafficGodLLMRequest):
    llm = _get_traffic_god_llm()
    if llm is None:
        raise HTTPException(503, "Traffic God LLM not available (model not loaded)")
    try:
        response = llm.generate(
            req.message,
            temperature=req.temperature,
            max_tokens=req.max_tokens,
        )
        return {
            "response": response,
            "model": "traffic-god-custom",
            "temperature": req.temperature,
        }
    except Exception as e:
        raise HTTPException(500, f"Traffic God LLM error: {str(e)[:200]}")


@app.get("/llm/status")
async def traffic_god_llm_status():
    llm = _get_traffic_god_llm()
    if llm is None:
        return {"loaded": False, "model": None}
    return {
        "loaded": True,
        "model": getattr(llm, "model_name", "traffic-god-custom"),
        "parameters": getattr(llm, "param_count", "unknown"),
    }


# ── Perception pipeline ──────────────────────────────────────────

class PerceptionRequest(BaseModel):
    video_path: str
    output_csv: Optional[str] = None
    dry_run: bool = False


@app.post("/perception")
async def perception_pipeline(req: PerceptionRequest):
    svc = _get_traffic_god_service()
    if svc is None:
        raise HTTPException(503, "Traffic God perception pipeline not available")
    try:
        result = svc.run_perception(
            video_path=req.video_path,
            output_csv=req.output_csv,
            dry_run=req.dry_run,
        )
        return {"status": "ok", "pipeline": "perception", "result": result}
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc))
    except RuntimeError as exc:
        raise HTTPException(503, str(exc))
    except Exception as exc:
        raise HTTPException(500, f"Perception pipeline error: {str(exc)[:200]}")


# ── Run standalone ────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("TRAFFIC_GOD_PORT", "8005"))
    uvicorn.run(app, host="0.0.0.0", port=port)
