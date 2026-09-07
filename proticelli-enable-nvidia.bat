@echo off
setlocal
cd /d "%~dp0"

where nvidia-smi >nul 2>&1
if errorlevel 1 goto :no_nvidia

set "PROTICELLI_PYTHON=%CD%\.venv\Scripts\python.exe"
if not exist "%PROTICELLI_PYTHON%" (
    where python >nul 2>&1
    if errorlevel 1 goto :no_python
    set "PROTICELLI_PYTHON=python"
)

echo.
echo  ProtiCelli NVIDIA acceleration setup
echo  This installs the official PyTorch CUDA 12.6 wheels in the same Python environment.
echo  The download is large and can take several minutes.
echo.
choice /M "Continue"
if errorlevel 2 exit /b 0

"%PROTICELLI_PYTHON%" -m pip install --upgrade torch torchvision --index-url https://download.pytorch.org/whl/cu126
if errorlevel 1 goto :failed

"%PROTICELLI_PYTHON%" -c "import torch,sys; print('PyTorch:',torch.__version__); print('CUDA wheel:',torch.version.cuda); print('CUDA available:',torch.cuda.is_available()); print('GPU:',torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'not detected'); sys.exit(0 if torch.cuda.is_available() else 2)"
if errorlevel 1 goto :driver

echo.
echo  CUDA is ready. Restart ProtiCelli Interactive Gallery.
echo.
pause
exit /b 0

:no_nvidia
echo.
echo  No NVIDIA driver was detected. CUDA acceleration requires a supported NVIDIA GPU.
echo  Intel and AMD laptop graphics are not supported by the current ProtiCelli runtime on Windows.
echo.
pause
exit /b 1

:no_python
echo.
echo  No ProtiCelli environment was found. Run proticelli-local.bat first.
echo.
pause
exit /b 1

:driver
echo.
echo  The CUDA wheel was installed, but PyTorch still cannot use the GPU.
echo  Update the NVIDIA driver, restart Windows, then run: proticelli-web --diagnose
echo.
pause
exit /b 1

:failed
echo.
echo  PyTorch CUDA installation failed. Review the error above.
echo.
pause
exit /b 1
