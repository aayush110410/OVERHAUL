#!/bin/bash

echo "========================================"
echo "OVERHAUL v3 Startup Script"
echo "========================================"
echo ""

# Check if virtual environment exists
if [ ! -d "myvenv" ]; then
    echo "ERROR: Virtual environment not found!"
    echo "Please create venv first: python -m venv myvenv"
    exit 1
fi

# Activate virtual environment
source myvenv/bin/activate

echo "[1/3] Starting FastAPI backend on port 8000..."
uvicorn app:app --reload --host 0.0.0.0 --port 8000 &
BACKEND_PID=$!

sleep 3

echo "[2/3] Starting HTTP server on port 8080..."
python -m http.server 8080 &
FRONTEND_PID=$!

sleep 2

echo "[3/3] Opening browser..."
if command -v xdg-open > /dev/null; then
    xdg-open http://localhost:8080/index_v3.html
elif command -v open > /dev/null; then
    open http://localhost:8080/index_v3.html
else
    echo "Please open: http://localhost:8080/index_v3.html"
fi

echo ""
echo "========================================"
echo "OVERHAUL v3 is now running!"
echo "========================================"
echo "Backend API: http://localhost:8000"
echo "Frontend UI: http://localhost:8080/index_v3.html"
echo ""
echo "Backend PID: $BACKEND_PID"
echo "Frontend PID: $FRONTEND_PID"
echo ""
echo "Press Ctrl+C to stop both servers"
echo "========================================"

# Trap Ctrl+C and kill both processes
trap "kill $BACKEND_PID $FRONTEND_PID; exit" INT

# Wait for both processes
wait
