@echo off
REM Install requirements dengan konfigurasi untuk koneksi internet lambat
REM Mencegah timeout saat download package besar (torch, etc)

set TIMEOUT=3600
set RETRIES=5

echo Installing requirements...
echo Timeout: %TIMEOUT%s ^| Retries: %RETRIES%
echo.

pip install -r requirements.txt --default-timeout=%TIMEOUT% --retries %RETRIES%

echo.
echo Installation selesai.
pause
