#!/usr/bin/env bash

# ============================================================
# Configuration — edit these paths before running
# ============================================================
VENV_DIR=".venv"
CHECKPOINT="tagger_proto.safetensors"
VOCAB="tagger_vocab_with_categories.json"
DEVICE="cuda"
PORT="7860"
HOST="0.0.0.0"
# ============================================================

# shellcheck disable=SC1090
source "${VENV_DIR}/bin/activate"
python server_local.py --checkpoint "${CHECKPOINT}" --vocab "${VOCAB}" --device "${DEVICE}" --host "${HOST}" --port "${PORT}"
