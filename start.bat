@echo off

:: ============================================================
:: Configuration — edit these paths before running
:: ============================================================
set VENV_DIR=.venv
set CHECKPOINT=tagger_proto.safetensors
set VOCAB=tagger_vocab_with_categories.json
set DEVICE=cuda
set PORT=7860
set HOST=0.0.0.0
:: ============================================================

call "%VENV_DIR%\Scripts\activate.bat"
python server_local.py --checkpoint "%CHECKPOINT%" --vocab "%VOCAB%" --device %DEVICE% --host %HOST% --port %PORT%
