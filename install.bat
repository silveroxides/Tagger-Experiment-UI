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
if !PY_MAJOR! LSS 3 (
    echo ERROR: Python 3.10+ required.
    pause & exit /b 1
)
if !PY_MAJOR! EQU 3 if !PY_MINOR! LSS 10 (
    echo ERROR: Python 3.10+ required, got !PY_VER!.
    pause & exit /b 1
)
echo Python !PY_VER! OK.

:: --- create venv ---
if not exist "!VENV_DIR!\Scripts\activate.bat" (
    echo Creating virtual environment in !VENV_DIR! ...
    python -m venv "!VENV_DIR!"
    if errorlevel 1 (
        echo ERROR: venv creation failed.
        pause & exit /b 1
    )
)
call "!VENV_DIR!\Scripts\activate.bat"

:: --- torch index URL ---
echo.
echo  Select PyTorch build:
echo  [1] CUDA 13.0  (recommended, RTX 40/50 series)
echo  [2] CUDA 12.8
echo  [3] CUDA 12.6
echo  [4] CPU only
echo  [5] Enter custom URL
echo.
set /p TORCH_CHOICE=Choice (1-5): 

set TORCH_URL=
set DEVICE=cuda
if "!TORCH_CHOICE!"=="1" set TORCH_URL=https://download.pytorch.org/whl/cu130
if "!TORCH_CHOICE!"=="2" set TORCH_URL=https://download.pytorch.org/whl/cu128
if "!TORCH_CHOICE!"=="3" set TORCH_URL=https://download.pytorch.org/whl/cu126
if "!TORCH_CHOICE!"=="4" (
    set TORCH_URL=https://download.pytorch.org/whl/cpu
    set DEVICE=cpu
)
if "!TORCH_CHOICE!"=="5" (
    set /p TORCH_URL=Enter extra-index-url: 
)
if "!TORCH_URL!"=="" (
    echo Invalid choice.
    pause & exit /b 1
)

:: --- install ---
echo.
echo Installing torch...
pip install torch torchvision --extra-index-url !TORCH_URL!
if errorlevel 1 (
    echo ERROR: torch install failed.
    pause & exit /b 1
)
echo Installing remaining dependencies...
pip install -r requirements_local.txt
if errorlevel 1 (
    echo ERROR: pip install failed.
    pause & exit /b 1
)

:: --- model weights ---
echo.
set WEIGHTS_DEFAULT=n
if exist tagger_proto.safetensors set WEIGHTS_DEFAULT=y
if "!WEIGHTS_DEFAULT!"=="y" (
    echo  Found tagger_proto.safetensors in current directory.
)
set /p HAS_WEIGHTS=Do you already have tagger_proto.safetensors locally? (y/n, default !WEIGHTS_DEFAULT!): 
if "!HAS_WEIGHTS!"=="" set HAS_WEIGHTS=!WEIGHTS_DEFAULT!
if /i "!HAS_WEIGHTS!"=="y" goto :weights_local

:: --- download weights ---
set CHECKPOINT_PATH=tagger_proto.safetensors
echo Installing huggingface_hub...
pip install -q huggingface_hub
if errorlevel 1 (
    echo ERROR: Could not install huggingface_hub.
    pause & exit /b 1
)
echo Downloading model weights (~4 GB)...
hf download lodestones/tagger-experiment tagger_proto.safetensors --local-dir .
if not exist tagger_proto.safetensors (
    echo ERROR: Download failed.
    pause & exit /b 1
)
goto :weights_done

:weights_local
set /p CHECKPOINT_PATH=Path to tagger_proto.safetensors (default: tagger_proto.safetensors): 
if "!CHECKPOINT_PATH!"=="" set CHECKPOINT_PATH=tagger_proto.safetensors
if not exist "!CHECKPOINT_PATH!" (
    echo ERROR: File not found: !CHECKPOINT_PATH!
    pause & exit /b 1
)

:weights_done

:: --- vocab ---
echo Downloading vocabulary...
hf download lodestones/tagger-experiment tagger_vocab_with_categories.json --local-dir .
if not exist tagger_vocab_with_categories.json (
    echo ERROR: Vocab download failed.
    pause & exit /b 1
)

:: --- GPU selection ---
echo.
echo  Select GPU / device to use:
echo  [1] cuda        (default - let PyTorch pick the first available GPU)
echo  [2] cuda:0      (first GPU, explicit)
echo  [3] cuda:1      (second GPU)
echo  [4] cuda:2      (third GPU)
echo  [5] cuda:3      (fourth GPU)
echo  [6] cpu
echo  [7] Enter custom (e.g. cuda:0, mps, cpu)
echo.
set /p GPU_CHOICE=Choice (1-7, default 1): 
if "!GPU_CHOICE!"=="" set GPU_CHOICE=1

if "!GPU_CHOICE!"=="1" set DEVICE=cuda
if "!GPU_CHOICE!"=="2" set DEVICE=cuda:0
if "!GPU_CHOICE!"=="3" set DEVICE=cuda:1
if "!GPU_CHOICE!"=="4" set DEVICE=cuda:2
if "!GPU_CHOICE!"=="5" set DEVICE=cuda:3
if "!GPU_CHOICE!"=="6" set DEVICE=cpu
if "!GPU_CHOICE!"=="7" set /p DEVICE=Enter device string: 

if "!DEVICE!"=="" (
    echo Invalid choice.
    pause & exit /b 1
)

echo Validating device !DEVICE!...
python -c "import torch; torch.zeros(1).to('!DEVICE!'); print('Device OK')" 2>nul
if not errorlevel 1 goto :gpu_ok
echo.
echo  WARNING: Could not initialise device '!DEVICE!'.
echo  This may mean the GPU is not available or CUDA is not installed correctly.
echo.
set /p CONTINUE_ANYWAY=Continue anyway? (y/n): 
if /i not "!CONTINUE_ANYWAY!"=="y" (
    pause & exit /b 1
)
:gpu_ok

:: --- batch mode ---
echo.
set /p ENABLE_BATCH=Enable batch tagging and batch similarity endpoints by default? (y/n, default n): 
set BATCH_FLAG=
if /i "!ENABLE_BATCH!"=="y" set BATCH_FLAG= --enable-batch

:: --- write run.bat ---
(
    echo @echo off
    echo call "!VENV_DIR!\Scripts\activate.bat"
    echo python server_local.py --checkpoint "!CHECKPOINT_PATH!" --vocab tagger_vocab_with_categories.json --device !DEVICE! --port 7860!BATCH_FLAG!
) > run.bat

echo.
echo  Done! Run the server with:  run.bat
echo.
pause
