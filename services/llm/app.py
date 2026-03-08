"""
LLM Microservice
=================

Standalone FastAPI service owning all LLM interactions:
  - Qwen 3 4B (via OpenRouter)
  - Gemini 3 Pro Preview (via Google AI)
  - LDRAGo orchestration modes (fast, deep, agents)
  - Gemini specialist agents

Port: 8002 (default)

Endpoints:
  GET  /health
  POST /chat           — Full LDRAGo orchestrated chat
  POST /chat/fast      — Fast mode (LLM + local data, no agents)
  POST /chat/raw       — Raw LLM completion (no orchestration)
  GET  /config         — LLM config status (masked keys)
"""

from __future__ import annotations

import os
import sys
import time
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from llm.config import load_llm_config, qwen_enabled, gemini_enabled, get_config_debug_info  # noqa: E402
from llm.chat import llm_chat_text  # noqa: E402

app = FastAPI(title="OVERHAUL LLM Service", version="1.0.0")


# ── Health ────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    cfg = load_llm_config()
    return {
        "service": "llm",
        "status": "ok",
        "qwen_available": qwen_enabled(cfg),
        "gemini_available": gemini_enabled(cfg),
    }


@app.get("/config")
async def config_status():
    return get_config_debug_info()


# ── Raw LLM completion ───────────────────────────────────────────

class RawChatRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=10000)
    system: str = ""
    max_tokens: int = Field(4000, ge=100, le=16000)
    prefer: str = "auto"  # auto | qwen | gemini


@app.post("/chat/raw")
async def raw_chat(req: RawChatRequest):
    """Direct LLM completion without orchestration."""
    cfg = load_llm_config()
    if not qwen_enabled(cfg) and not gemini_enabled(cfg):
        raise HTTPException(503, "No LLM provider configured")

    t0 = time.time()
    response = await llm_chat_text(
        prompt=req.prompt,
        system=req.system or None,
        cfg=cfg,
        max_output_tokens=req.max_tokens,
        prefer=req.prefer if req.prefer != "auto" else None,
    )
    return {
        "response": response,
        "duration_seconds": round(time.time() - t0, 2),
        "model": "qwen3-4b / gemini-3-pro",
    }


# ── LDRAGo Fast chat ─────────────────────────────────────────────

class FastChatRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=5000)
    context: Dict[str, Any] = {}
    ncr_data: str = ""
    engine_results: Dict[str, Any] = {}


@app.post("/chat/fast")
async def fast_chat(req: FastChatRequest):
    """LDRAGo Fast: LLM analysis with pre-supplied NCR data and engine results.
    
    The gateway is responsible for gathering data and engine results
    before calling this endpoint.
    """
    from agents.ldrago_orchestrator import llm_initial_analysis

    t0 = time.time()
    context = dict(req.context)

    response = await llm_initial_analysis(req.prompt, context, req.ncr_data)

    return {
        "response": response,
        "mode": "fast",
        "duration_seconds": round(time.time() - t0, 2),
        "models_used": {"primary": "qwen3-4b / gemini-3-pro"},
    }


# ── Full LDRAGo orchestrated chat ────────────────────────────────

class FullChatRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=5000)
    mode: str = "fast"  # fast | deep | agents
    context: Dict[str, Any] = {}


@app.post("/chat")
async def full_chat(req: FullChatRequest):
    """Full LDRAGo orchestration (agents + synthesis).
    
    - fast:   LLM + local data only (no agent search)
    - deep:   LLM + full agent search + synthesis  
    - agents: Run all specialist agents with Google Search
    """
    t0 = time.time()

    if req.mode == "agents" or req.mode == "deep":
        from agents.ldrago_orchestrator import ldrago_orchestrate
        result = await ldrago_orchestrate(
            query=req.prompt,
            context=req.context,
            run_agents=True,
        )
        return {
            "response": result.get("response", ""),
            "mode": req.mode,
            "agent_results": result.get("agent_results", {}),
            "logs": result.get("logs", []),
            "duration_seconds": round(time.time() - t0, 2),
            "models_used": result.get("models_used", {}),
        }
    else:
        from agents.ldrago_orchestrator import ldrago_fast
        result = await ldrago_fast(query=req.prompt, context=req.context)
        return {
            "response": result.get("response", ""),
            "mode": "fast",
            "engine_results": result.get("engine_results", {}),
            "ncr_data": result.get("ncr_data", {}),
            "logs": result.get("logs", []),
            "duration_seconds": round(time.time() - t0, 2),
            "models_used": result.get("models_used", {}),
        }


# ── Run standalone ────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("LLM_PORT", "8002"))
    uvicorn.run(app, host="0.0.0.0", port=port)
