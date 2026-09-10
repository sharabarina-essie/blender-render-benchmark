@echo off
REM Double-clickable bootstrap: downloads Blender builds + models into place.
REM Pass --Force to re-download everything: bootstrap.cmd --Force
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0bootstrap.ps1" %*
echo.
echo Done. Press any key to close.
pause >nul
