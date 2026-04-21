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
# Set to 1 to enable batch tagging and batch similarity endpoints
ENABLE_BATCH=0
# ============================================================

# shellcheck disable=SC1090
source "${VENV_DIR}/bin/activate"

BATCH_FLAG=""
if [ "${ENABLE_BATCH}" = "1" ]; then BATCH_FLAG="--enable-batch"; fi

python server_local.py --checkpoint "${CHECKPOINT}" --vocab "${VOCAB}" --device "${DEVICE}" --host "${HOST}" --port "${PORT}" ${BATCH_FLAG}
