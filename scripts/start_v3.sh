#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")"

echo "========================================"
echo "OVERHAUL Local Live Demo (v3)"
echo "========================================"

echo "[0/3] Checking prerequisites..."
if [ ! -d ".venv" ]; then
  echo "ERROR: .venv not found. Create it first: python3 -m venv .venv"
  exit 1
fi
if ! command -v npm >/dev/null 2>&1; then
  echo "ERROR: npm not found. Install Node.js first."
  exit 1
fi

source .venv/bin/activate

echo "[1/3] Starting FastAPI backend on http://127.0.0.1:8000 ..."
uvicorn app:app --reload --host 127.0.0.1 --port 8000 &
BACKEND_PID=$!

echo "[2/3] Starting Live Demo site (Vite) ..."
cd landing-react
if [ ! -d "node_modules" ]; then
  npm install
fi
npm run dev -- --host 127.0.0.1 --port 5173 &
FRONTEND_PID=$!

sleep 2

echo "[3/3] Opening browser..."
if command -v open >/dev/null 2>&1; then
  open "http://127.0.0.1:5173"
else
  echo "Open: http://127.0.0.1:5173"
fi

echo ""
echo "========================================"
echo "Running:"
echo "- Backend:  http://127.0.0.1:8000"
echo "- Frontend: http://127.0.0.1:5173"
echo "========================================"

echo "Press Ctrl+C to stop"
trap "kill $BACKEND_PID $FRONTEND_PID 2>/dev/null || true" INT TERM
wait
