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
        MAJOR="${VER%%.*}"; MINOR="${VER##*.}"
        if [ "$MAJOR" -gt 3 ] || { [ "$MAJOR" -eq 3 ] && [ "$MINOR" -ge 10 ]; }; then
            PYTHON="$cmd"; break
        fi
    fi
done
if [ -z "$PYTHON" ]; then
    echo "ERROR: Python 3.10+ not found. Install it and retry."; exit 1
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
echo "  [1] CUDA 12.4  (recommended, RTX 30/40 series)"
echo "  [2] CUDA 12.1"
echo "  [3] CUDA 11.8  (older GPUs)"
echo "  [4] CPU only"
echo "  [5] Enter custom URL"
echo ""
read -rp "Choice (1-5): " TORCH_CHOICE

case "$TORCH_CHOICE" in
    1) TORCH_URL="https://download.pytorch.org/whl/cu124"; DEVICE="cuda" ;;
    2) TORCH_URL="https://download.pytorch.org/whl/cu121"; DEVICE="cuda" ;;
    3) TORCH_URL="https://download.pytorch.org/whl/cu118"; DEVICE="cuda" ;;
    4) TORCH_URL="https://download.pytorch.org/whl/cpu";   DEVICE="cpu"  ;;
    5) read -rp "Enter extra-index-url: " TORCH_URL; DEVICE="cuda" ;;
    *) echo "Invalid choice."; exit 1 ;;
esac

# --- install ---
echo ""
echo "Installing dependencies..."
pip install -r requirements_local.txt --extra-index-url "$TORCH_URL"

# --- model weights ---
echo ""
read -rp "Do you already have tagger_proto.safetensors locally? (y/n): " HAS_WEIGHTS
if [[ "$HAS_WEIGHTS" =~ ^[Yy]$ ]]; then
    read -rp "Path to tagger_proto.safetensors: " CHECKPOINT_PATH
    if [ ! -f "$CHECKPOINT_PATH" ]; then
        echo "ERROR: File not found: $CHECKPOINT_PATH"; exit 1
    fi
else
    CHECKPOINT_PATH="tagger_proto.safetensors"
    echo "Downloading model weights (~4 GB)..."
    HF_URL="https://huggingface.co/lodestones/tagger-experiment/resolve/main/tagger_proto.safetensors"
    if command -v wget &>/dev/null; then
        wget -q --show-progress -O "$CHECKPOINT_PATH" "$HF_URL"
    elif command -v curl &>/dev/null; then
        curl -L -o "$CHECKPOINT_PATH" "$HF_URL"
    else
        echo "ERROR: Neither wget nor curl found. Install one and retry."; exit 1
    fi
fi

# --- write run.sh ---
cat > run.sh <<EOF
#!/usr/bin/env bash
source "$VENV_DIR/bin/activate"
python server_local.py --checkpoint "$CHECKPOINT_PATH" --vocab tagger_vocab_with_categories.json --device $DEVICE --port 7860
EOF
chmod +x run.sh

echo ""
echo " Done! Run the server with:  ./run.sh"
echo ""
