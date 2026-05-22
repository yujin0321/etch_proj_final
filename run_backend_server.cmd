@echo off
cd /d "%~dp0"
"C:\Users\SAMSUNG\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe" "run_backend.py" > "%~dp0backend-server.log" 2> "%~dp0backend-server.err.log"
