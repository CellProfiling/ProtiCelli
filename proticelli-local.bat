@echo off
setlocal EnableDelayedExpansion
cd /d "%~dp0"

set "PROTICELLI_PYTHON=%CD%\.venv\Scripts\python.exe"

if not exist "%PROTICELLI_PYTHON%" (
    where py >nul 2>&1
    if not errorlevel 1 (
        set "PROTICELLI_BOOTSTRAP=py -3"
    ) else (
        where python >nul 2>&1
        if errorlevel 1 goto :no_python
        set "PROTICELLI_BOOTSTRAP=python"
    )

    echo.
    echo  ProtiCelli Interactive Gallery - first-time setup
    echo  Creating a private Python environment. This can take several minutes.
    echo.

    !PROTICELLI_BOOTSTRAP! -m venv .venv
    if errorlevel 1 goto :failed

    "%PROTICELLI_PYTHON%" -m pip install --upgrade pip
    if errorlevel 1 goto :failed

    "%PROTICELLI_PYTHON%" -m pip install -e ".[web]"
    if errorlevel 1 goto :failed
)

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
echo  Install Python 3.9 or newer from https://www.python.org/downloads/
echo  During installation, enable "Add Python to PATH".
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

:failed
echo.
echo  ProtiCelli setup or launch failed. The error is shown above.
echo.
pause
exit /b 1
