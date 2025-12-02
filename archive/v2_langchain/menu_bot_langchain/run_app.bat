@echo off
REM Launch script for Menu Recommendation Streamlit App (Windows)

echo.
echo ========================================
echo   Menu Recommendation Bot
echo ========================================
echo.

REM Check if streamlit is installed
where streamlit >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Streamlit not found!
    echo Please install: pip install streamlit
    pause
    exit /b 1
)

REM Check if app file exists
if not exist "app.py" (
    echo [ERROR] app.py not found!
    pause
    exit /b 1
)

echo [INFO] Starting Streamlit app...
echo [INFO] Press Ctrl+C to stop
echo.

streamlit run app.py --server.port 8502

pause
