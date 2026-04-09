#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

echo "========================================"
echo "OVERHAUL Demo 4 Codex"
echo "========================================"

if [ ! -d ".venv" ]; then
  echo "ERROR: .venv not found. Create it first with: python3 -m venv .venv"
  exit 1
fi

if ! command -v npm >/dev/null 2>&1; then
  echo "ERROR: npm not found. Install Node.js first."
  exit 1
fi

source .venv/bin/activate

echo "[1/3] Starting Demo 4 Codex backend on http://127.0.0.1:8014 ..."
python3 -m uvicorn services.rendering_demo_codex.app:app --host 127.0.0.1 --port 8014 &
BACKEND_PID=$!

echo "[2/3] Starting rendering engine on http://127.0.0.1:5175/#demo-4-codex ..."
cd "$ROOT_DIR/rendering-engine"
if [ ! -d "node_modules" ]; then
  npm install
fi
VITE_DEMO4_API_URL=http://127.0.0.1:8014 npm run dev -- --host 127.0.0.1 --port 5175 &
FRONTEND_PID=$!

sleep 3

echo "[3/3] Opening browser..."
TARGET_URL="http://127.0.0.1:5175/#demo-4-codex"
if command -v open >/dev/null 2>&1; then
  open "$TARGET_URL"
else
  echo "Open: $TARGET_URL"
fi

echo ""
echo "Running:"
echo "- Demo 4 backend:  http://127.0.0.1:8014"
echo "- Demo 4 frontend: http://127.0.0.1:5175/#demo-4-codex"
echo ""
echo "Press Ctrl+C to stop"

cleanup() {
  kill "$BACKEND_PID" "$FRONTEND_PID" 2>/dev/null || true
}

trap cleanup INT TERM EXIT
wait
