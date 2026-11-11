@echo off
setlocal
cd /d "%~dp0"
title Arbitro de Ajedrez Inteligente (ESP32-CAM)

if not exist "%~dp0.venv\Scripts\python.exe" (
    echo [ERROR] No se encontro el entorno virtual en .venv\
    pause
    exit /b 1
)

echo Iniciando interfaz de arbitro de ajedrez...
"%~dp0.venv\Scripts\python.exe" iniciar_interfaz.py %*
if %ERRORLEVEL% neq 0 (
    echo.
    echo [AVISO] La aplicacion se cerro con codigo %ERRORLEVEL%.
    pause
)
