@echo off
cd backend
start cmd /k "python main.py"
start cmd /k "python web_scraper.py"
cd ..
start cmd /k "npm start"