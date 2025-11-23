@echo off
echo ========================================
echo OVERHAUL v3 Startup Script
echo ========================================
echo.

REM Check if virtual environment exists
if not exist "myvenv\Scripts\activate.bat" (
    echo ERROR: Virtual environment not found!
    echo Please create venv first: python -m venv myvenv
    pause
    exit /b 1
)

REM Activate virtual environment
call myvenv\Scripts\activate.bat

echo [1/3] Starting FastAPI backend on port 8000...
start "OVERHAUL Backend" cmd /k "myvenv\Scripts\activate.bat && uvicorn app:app --reload --host 0.0.0.0 --port 8000"

timeout /t 3 /nobreak > nul

echo [2/3] Starting HTTP server on port 8080...
start "OVERHAUL Frontend" cmd /k "myvenv\Scripts\activate.bat && python -m http.server 8080"

timeout /t 2 /nobreak > nul

echo [3/3] Opening browser...
start http://localhost:8080/index_v3.html
echo.
echo ========================================
echo OVERHAUL v3 is now running!
echo ========================================
echo Backend API: http://localhost:8000
echo Frontend UI: http://localhost:8080/index_v3.html
echo.
echo Press Ctrl+C in each terminal window to stop
echo ========================================
pause
