#!/usr/bin/env bash
set -e

echo ""
echo " DINOv3 Tagger - Local Installer"
echo " ================================"
echo ""

# --- venv directory ---
read -rp "Virtual environment directory (default: .venv): " VENV_DIR
VENV_DIR="${VENV_DIR:-.venv}"

# --- check Python ---
PYTHON=""
for cmd in python3 python; do
    if command -v "$cmd" &>/dev/null; then
        VER=$("$cmd" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
        MAJOR="${VER%%.*}"
        MINOR="${VER##*.}"
        if [ "$MAJOR" -gt 3 ] || { [ "$MAJOR" -eq 3 ] && [ "$MINOR" -ge 10 ]; }; then
            PYTHON="$cmd"
            break
        fi
    fi
done
if [ -z "$PYTHON" ]; then
    echo "ERROR: Python 3.10+ not found. Install it and retry."
    exit 1
fi
echo "Using $PYTHON ($VER)"

# --- create venv ---
if [ ! -f "$VENV_DIR/bin/activate" ]; then
    echo "Creating virtual environment in $VENV_DIR ..."
    "$PYTHON" -m venv "$VENV_DIR"
fi
# shellcheck disable=SC1090
source "$VENV_DIR/bin/activate"

# --- torch index URL ---
echo ""
echo " Select PyTorch build:"
echo "  [1] CUDA 13.0  (recommended, RTX 40/50 series)"
echo "  [2] CUDA 12.8"
echo "  [3] CUDA 12.6"
echo "  [4] CPU only"
echo "  [5] Enter custom URL"
echo ""
read -rp "Choice (1-5): " TORCH_CHOICE

case "$TORCH_CHOICE" in
    1) TORCH_URL="https://download.pytorch.org/whl/cu130"; DEVICE="cuda" ;;
    2) TORCH_URL="https://download.pytorch.org/whl/cu128"; DEVICE="cuda" ;;
    3) TORCH_URL="https://download.pytorch.org/whl/cu126"; DEVICE="cuda" ;;
    4) TORCH_URL="https://download.pytorch.org/whl/cpu";   DEVICE="cpu"  ;;
    5) read -rp "Enter extra-index-url: " TORCH_URL; DEVICE="cuda" ;;
    *) echo "Invalid choice."; exit 1 ;;
esac

# --- install ---
echo ""
echo "Installing torch..."
if ! pip install torch torchvision --extra-index-url "$TORCH_URL"; then
    echo "ERROR: torch install failed."
    exit 1
fi
echo "Installing remaining dependencies..."
if ! pip install -r requirements_local.txt; then
    echo "ERROR: pip install failed."
    exit 1
fi

# --- model weights ---
echo ""
WEIGHTS_DEFAULT="n"
if [ -f "tagger_proto.safetensors" ]; then
    WEIGHTS_DEFAULT="y"
    echo " Found tagger_proto.safetensors in current directory."
fi
read -rp "Do you already have the model weights (.safetensors) locally? (y/n, default ${WEIGHTS_DEFAULT}): " HAS_WEIGHTS
HAS_WEIGHTS="${HAS_WEIGHTS:-$WEIGHTS_DEFAULT}"
if [[ "$HAS_WEIGHTS" =~ ^[Yy]$ ]]; then
    read -rp "Path to model weights file (default: tagger_proto.safetensors): " CHECKPOINT_PATH
    CHECKPOINT_PATH="${CHECKPOINT_PATH:-tagger_proto.safetensors}"
    # strip surrounding quotes the user might have typed
    CHECKPOINT_PATH="${CHECKPOINT_PATH%\"}"
    CHECKPOINT_PATH="${CHECKPOINT_PATH#\"}"
    CHECKPOINT_PATH="${CHECKPOINT_PATH%\'}"
    CHECKPOINT_PATH="${CHECKPOINT_PATH#\'}"
    if [ ! -f "$CHECKPOINT_PATH" ]; then
        echo "ERROR: File not found: $CHECKPOINT_PATH"
        exit 1
    fi
else
    CHECKPOINT_PATH="tagger_proto.safetensors"
    echo "Installing huggingface_hub..."
    if ! pip install -q huggingface_hub; then
        echo "ERROR: Could not install huggingface_hub."
        exit 1
    fi
    echo "Downloading model weights (~4 GB)..."
    hf download lodestones/tagger-experiment tagger_proto.safetensors --local-dir .
    if [ ! -f "$CHECKPOINT_PATH" ]; then
        echo "ERROR: Download failed."
        exit 1
    fi
fi

# --- vocab ---
echo "Downloading vocabulary..."
hf download lodestones/tagger-experiment tagger_vocab_with_categories.json --local-dir .
if [ ! -f "tagger_vocab_with_categories.json" ]; then
    echo "ERROR: Vocab download failed."
    exit 1
fi

# --- GPU selection ---
echo ""
echo " Select GPU / device to use:"
echo "  [1] cuda        (default - let PyTorch pick the first available GPU)"
echo "  [2] cuda:0      (first GPU, explicit)"
echo "  [3] cuda:1      (second GPU)"
echo "  [4] cuda:2      (third GPU)"
echo "  [5] cuda:3      (fourth GPU)"
echo "  [6] cpu"
echo "  [7] Enter custom (e.g. cuda:0, mps, cpu)"
echo ""
read -rp "Choice (1-7, default 1): " GPU_CHOICE
GPU_CHOICE="${GPU_CHOICE:-1}"

case "$GPU_CHOICE" in
    1) DEVICE="cuda"   ;;
    2) DEVICE="cuda:0" ;;
    3) DEVICE="cuda:1" ;;
    4) DEVICE="cuda:2" ;;
    5) DEVICE="cuda:3" ;;
    6) DEVICE="cpu"    ;;
    7) read -rp "Enter device string: " DEVICE ;;
    *) echo "Invalid choice."; exit 1 ;;
esac

echo "Validating device '$DEVICE'..."
if python -c "import torch; torch.zeros(1).to('$DEVICE'); print('Device OK')" 2>/dev/null; then
    :
else
    echo ""
    echo " WARNING: Could not initialise device '$DEVICE'."
    echo " This may mean the GPU is not available or CUDA is not installed correctly."
    echo ""
    read -rp "Continue anyway? (y/n): " CONTINUE_ANYWAY
    if [[ ! "$CONTINUE_ANYWAY" =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# --- batch mode ---
echo ""
read -rp "Enable batch tagging and batch similarity endpoints by default? (y/n, default n): " ENABLE_BATCH
BATCH_FLAG=""
if [[ "$ENABLE_BATCH" =~ ^[Yy]$ ]]; then BATCH_FLAG=" --enable-batch"; fi

# --- write run.sh ---
cat > run.sh << RUNEOF
#!/usr/bin/env bash
source "${VENV_DIR}/bin/activate"
python server_local.py --checkpoint "${CHECKPOINT_PATH}" --vocab tagger_vocab_with_categories.json --device ${DEVICE} --port 7860${BATCH_FLAG}
RUNEOF
chmod +x run.sh

echo ""
echo " Done! Run the server with:  ./run.sh"
echo ""
