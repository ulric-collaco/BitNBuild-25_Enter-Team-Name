@echo off
echo 🚀 Starting Review Radar - Complete Stack
echo ========================================

echo.
echo 📚 Step 1: Starting Main API (port 8000)...
cd /d "d:\Coding\Projects\hackathon type shit\bnb\BitNBuild-25_Enter-Team-Name\backend"
start cmd /k "echo Main API Server && echo ================== && python main.py"

echo.
echo 🕷️ Step 2: Starting Web Scraper API (port 8001)...
timeout /t 2 /nobreak >nul
start cmd /k "echo Web Scraper API && echo ================== && python web_scraper.py"

echo.
echo ⚛️ Step 3: Starting React Frontend (port 3000)...
cd /d "d:\Coding\Projects\hackathon type shit\bnb\BitNBuild-25_Enter-Team-Name"
timeout /t 3 /nobreak >nul
start cmd /k "echo React Frontend && echo =============== && npm start"

echo.
echo ✅ All services are starting up!
echo.
echo 📋 Service URLs:
echo   - Main API:      http://localhost:8000
echo   - Web Scraper:   http://localhost:8001  
echo   - Frontend:      http://localhost:3000
echo.
echo 🎯 USAGE:
echo   1. Wait for all services to start (about 30 seconds)
echo   2. Open http://localhost:3000 in your browser
echo   3. Enter a product URL (e.g., Amazon product page)
echo.
echo Press any key to close this window...
pause >nul