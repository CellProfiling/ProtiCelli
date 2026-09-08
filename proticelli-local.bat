@echo off
setlocal EnableDelayedExpansion
cd /d "%~dp0"

set "PROTICELLI_PYTHON=%CD%\.venv\Scripts\python.exe"
set "PROTICELLI_BOOTSTRAP="

if exist "%PROTICELLI_PYTHON%" goto :check_env

where py >nul 2>&1
if not errorlevel 1 set "PROTICELLI_BOOTSTRAP=py -3"
if not defined PROTICELLI_BOOTSTRAP (
    where python >nul 2>&1
    if errorlevel 1 goto :no_python
    set "PROTICELLI_BOOTSTRAP=python"
)

%PROTICELLI_BOOTSTRAP% -c "import sys; sys.exit(0 if (3, 10) <= sys.version_info < (3, 14) else 1)" >nul 2>&1
if errorlevel 1 goto :old_python

echo.
echo  ProtiCelli Interactive Gallery - first-time setup
echo  Creating a private Python environment. This can take several minutes.
echo.

%PROTICELLI_BOOTSTRAP% -m venv .venv
if errorlevel 1 goto :setup_failed

"%PROTICELLI_PYTHON%" -m pip install --upgrade pip
if errorlevel 1 goto :setup_failed

"%PROTICELLI_PYTHON%" -m pip install -e ".[web]"
if errorlevel 1 goto :setup_failed

:check_env
rem Revalidates on every launch, not just first-time setup, so an environment
rem built by an older release is not silently reused.
"%PROTICELLI_PYTHON%" -c "import sys; sys.exit(0 if (3, 10) <= sys.version_info < (3, 14) else 1)" >nul 2>&1
if errorlevel 1 goto :stale_env

rem Fails in seconds instead of after the 6.5 GB asset download.
"%PROTICELLI_PYTHON%" -c "import proticelli_web.app" >nul 2>&1
if errorlevel 1 goto :broken_env

if /I not "%PROTICELLI_SKIP_ASSET_DOWNLOAD%"=="1" (
    "%PROTICELLI_PYTHON%" -m proticelli.utils.download --check >nul 2>&1
    if errorlevel 1 (
        echo.
        echo  Downloading ProtiCelli model assets. This happens once and may take several minutes.
        echo  Keep this window open; interrupted downloads are retried safely on the next launch.
        echo.
        "%PROTICELLI_PYTHON%" -m proticelli.utils.download
        if errorlevel 1 goto :asset_failed
    )
)

where nvidia-smi >nul 2>&1
if not errorlevel 1 (
    "%PROTICELLI_PYTHON%" -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" >nul 2>&1
    if errorlevel 1 (
        echo.
        echo  NVIDIA GPU detected, but this Python environment cannot use CUDA.
        echo  Run proticelli-enable-nvidia.bat once to install the CUDA-enabled PyTorch wheel.
        echo.
    )
)

"%PROTICELLI_PYTHON%" -m proticelli_web --mode auto
if errorlevel 1 goto :failed
exit /b 0

:no_python
echo.
echo  Python was not found.
echo  Install Python 3.10 through 3.13 from https://www.python.org/downloads/
echo  During installation, enable "Add Python to PATH".
echo.
pause
exit /b 1

:old_python
echo.
echo  ProtiCelli requires Python 3.10 through 3.13. The Python found on this system is:
%PROTICELLI_BOOTSTRAP% --version
echo.
echo  Install Python 3.10 through 3.13 from https://www.python.org/downloads/
echo  During installation, enable "Add Python to PATH", then run this launcher again.
echo.
pause
exit /b 1

:stale_env
echo.
echo  The private environment in .venv does not use a supported Python version.
echo  ProtiCelli requires Python 3.10 through 3.13.
echo.
echo  Delete the .venv folder in this directory, then run this launcher again.
echo  Downloaded model assets and proticelli_web_data are not affected.
echo.
pause
exit /b 1

:broken_env
echo.
echo  The private environment in .venv exists but cannot load ProtiCelli.
echo  The error is shown below:
echo.
"%PROTICELLI_PYTHON%" -c "import proticelli_web.app"
echo.
echo  Delete the .venv folder in this directory, then run this launcher again.
echo.
pause
exit /b 1

:asset_failed
echo.
echo  Model asset download failed. Check the internet connection and run this launcher again.
echo  Existing complete assets were not changed.
echo.
pause
exit /b 1

:setup_failed
rem A half-built .venv still contains python.exe, so without this the next
rem launch skips setup entirely and runs against a broken environment.
if exist "%CD%\.venv" rmdir /s /q "%CD%\.venv"
echo.
echo  ProtiCelli setup failed. The error is shown above.
echo  The incomplete environment was removed; run this launcher again to retry.
echo.
pause
exit /b 1

:failed
echo.
echo  ProtiCelli failed to launch. The error is shown above.
echo.
pause
exit /b 1
