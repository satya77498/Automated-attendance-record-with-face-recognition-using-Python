@echo off
setlocal
cd /d "%~dp0"

if /i "%~1"=="build" goto build_release

if not exist ".venv\Scripts\python.exe" (
    py -3.12 -m venv .venv >nul 2>nul
    if errorlevel 1 (
        python -m venv .venv
        if errorlevel 1 goto setup_failed
    )
)

".venv\Scripts\python.exe" -m pip install --disable-pip-version-check -r requirements.txt
if errorlevel 1 goto setup_failed

".venv\Scripts\python.exe" train.py
if errorlevel 1 goto app_failed
exit /b 0

:setup_failed
echo.
echo Setup failed. Install Python 3.12 and run run.bat again.
pause
exit /b 1

:app_failed
echo.
echo The app exited with an error. Review the message above.
pause
exit /b 1

:build_release
if not exist ".build-venv\Scripts\python.exe" (
    py -3.12 -m venv .build-venv >nul 2>nul
    if errorlevel 1 (
        python -m venv .build-venv
        if errorlevel 1 goto build_failed
    )
)

".build-venv\Scripts\python.exe" -m pip install --disable-pip-version-check --upgrade pip
if errorlevel 1 goto build_failed

".build-venv\Scripts\python.exe" -m pip install --disable-pip-version-check -r requirements.txt
if errorlevel 1 goto build_failed

".build-venv\Scripts\python.exe" setup.py build_exe --build-exe release\FaceAttendance
if errorlevel 1 goto build_failed

echo Standalone bundle created in release\FaceAttendance
exit /b 0

:build_failed
echo.
echo Windows release build failed. Review the message above.
pause
exit /b 1