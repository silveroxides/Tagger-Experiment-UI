@echo off
setlocal enabledelayedexpansion

echo.
echo  DINOv3 Tagger - Local Installer
echo  ================================
echo.

:: --- venv directory ---
set /p VENV_DIR=Virtual environment directory (default: .venv): 
if "!VENV_DIR!"=="" set VENV_DIR=.venv

:: --- check Python ---
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: python not found on PATH. Install Python 3.10+ and retry.
    pause & exit /b 1
)
for /f "tokens=2 delims= " %%v in ('python --version 2^>^&1') do set PY_VER=%%v
for /f "tokens=1,2 delims=." %%a in ("!PY_VER!") do (
    set PY_MAJOR=%%a
    set PY_MINOR=%%b
)
if !PY_MAJOR! LSS 3 ( echo ERROR: Python 3.10+ required. & pause & exit /b 1 )
if !PY_MAJOR! EQU 3 if !PY_MINOR! LSS 10 ( echo ERROR: Python 3.10+ required, got !PY_VER!. & pause & exit /b 1 )
echo Python !PY_VER! OK.

:: --- create venv ---
if not exist "!VENV_DIR!\Scripts\activate.bat" (
    echo Creating virtual environment in !VENV_DIR! ...
    python -m venv "!VENV_DIR!"
    if errorlevel 1 ( echo ERROR: venv creation failed. & pause & exit /b 1 )
)
call "!VENV_DIR!\Scripts\activate.bat"

:: --- torch index URL ---
echo.
echo  Select PyTorch build:
echo  [1] CUDA 12.4  (recommended, RTX 30/40 series)
echo  [2] CUDA 12.1
echo  [3] CUDA 11.8  (older GPUs)
echo  [4] CPU only
echo  [5] Enter custom URL
echo.
set /p TORCH_CHOICE=Choice (1-5): 

if "!TORCH_CHOICE!"=="1" set TORCH_URL=https://download.pytorch.org/whl/cu124 & set DEVICE=cuda
if "!TORCH_CHOICE!"=="2" set TORCH_URL=https://download.pytorch.org/whl/cu121 & set DEVICE=cuda
if "!TORCH_CHOICE!"=="3" set TORCH_URL=https://download.pytorch.org/whl/cu118 & set DEVICE=cuda
if "!TORCH_CHOICE!"=="4" set TORCH_URL=https://download.pytorch.org/whl/cpu  & set DEVICE=cpu
if "!TORCH_CHOICE!"=="5" (
    set /p TORCH_URL=Enter extra-index-url: 
    set DEVICE=cuda
)
if "!TORCH_URL!"=="" ( echo Invalid choice. & pause & exit /b 1 )

:: --- install ---
echo.
echo Installing dependencies...
pip install -r requirements_local.txt --extra-index-url !TORCH_URL!
if errorlevel 1 ( echo ERROR: pip install failed. & pause & exit /b 1 )

:: --- model weights ---
echo.
set /p HAS_WEIGHTS=Do you already have tagger_proto.safetensors locally? (y/n): 
if /i "!HAS_WEIGHTS!"=="y" (
    set /p CHECKPOINT_PATH=Path to tagger_proto.safetensors: 
    if not exist "!CHECKPOINT_PATH!" (
        echo ERROR: File not found: !CHECKPOINT_PATH!
        pause & exit /b 1
    )
) else (
    set CHECKPOINT_PATH=tagger_proto.safetensors
    echo Downloading model weights (~4 GB)...
    where curl >nul 2>&1
    if not errorlevel 1 (
        curl -L -o "!CHECKPOINT_PATH!" "https://huggingface.co/lodestones/tagger-experiment/resolve/main/tagger_proto.safetensors"
    ) else (
        powershell -Command "Invoke-WebRequest -Uri 'https://huggingface.co/lodestones/tagger-experiment/resolve/main/tagger_proto.safetensors' -OutFile '!CHECKPOINT_PATH!'"
    )
    if not exist "!CHECKPOINT_PATH!" ( echo ERROR: Download failed. & pause & exit /b 1 )
)

:: --- write run.bat ---
echo @echo off > run.bat
echo call "!VENV_DIR!\Scripts\activate.bat" >> run.bat
echo python server_local.py --checkpoint "!CHECKPOINT_PATH!" --vocab tagger_vocab_with_categories.json --device !DEVICE! --port 7860 >> run.bat

echo.
echo  Done! Run the server with:  run.bat
echo.
pause
