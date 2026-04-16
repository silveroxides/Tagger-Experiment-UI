# Lodestone Tagger UI — Local Runner

DINOv3 ViT-H/16+ image tagger with a web UI. Runs fully locally via FastAPI — no Gradio queue, no HuggingFace Spaces required.

---

## Requirements

- Python 3.10+
- A CUDA-capable GPU recommended (CPU works but is slow)
- ~4 GB disk space for model weights

---

## Installation

### Windows

```bat
install.bat
```

### Linux / macOS

```sh
chmod +x install.sh
./install.sh
```

Both scripts will interactively ask for:

1. **Virtual environment directory** — where to create the venv (default: `.venv`)
2. **PyTorch build** — pick your CUDA version or CPU-only:
   - CUDA 13.0 (recommended, RTX 40/50 series)
   - CUDA 12.8
   - CUDA 12.6
   - CPU only
   - Custom index URL
3. **Model weights** — provide a local path to an existing `.safetensors` file, or let the script download from HuggingFace (~4 GB)

After install a `run.bat` / `run.sh` launch script is written automatically.

---

## Starting the server

### Option 1 — generated launch script (after running installer)

```bat
run.bat
```
```sh
./run.sh
```

### Option 2 — manual start scripts

Edit the config block at the top of `start.bat` or `start.sh`, then run:

```bat
start.bat
```
```sh
chmod +x start.sh
./start.sh
```

The config block looks like this:

```bat
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
```

### Option 3 — run directly

```sh
.venv/Scripts/activate        # Windows
source .venv/bin/activate     # Linux/macOS

python server_local.py \
    --checkpoint tagger_proto.safetensors \
    --vocab      tagger_vocab_with_categories.json \
    --device     cuda \
    --port       7860
```

**CLI flags:**

| Flag | Default | Description |
|---|---|---|
| `--checkpoint` | required | Path to `.safetensors` weights file |
| `--vocab` | required | Path to `tagger_vocab_with_categories.json` |
| `--device` | `cuda` | `cuda`, `cuda:0`, `cpu`, etc. |
| `--host` | `0.0.0.0` | Bind address |
| `--port` | `7860` | Port |
| `--max-size` | `1024` | Long-edge pixel cap for inference |

Once running, open `http://localhost:7860` in your browser.

Interactive API docs are available at `http://localhost:7860/docs`.

---

## REST API

All endpoints accept and return JSON. No authentication required.

### Tagging

#### `POST /tag/url`

Tag an image from a URL.

**Query parameters:**

| Param | Default | Description |
|---|---|---|
| `url` | required | HTTP(S) image URL |
| `max_size` | `1024` | Resize long edge to this before inference |
| `floor` | `0.05` | Minimum score threshold (0–1) |

```sh
curl -X POST "http://localhost:7860/tag/url?url=https://example.com/image.jpg&floor=0.1"
```

```python
import requests
r = requests.post("http://localhost:7860/tag/url", params={
    "url": "https://example.com/image.jpg",
    "floor": 0.1,
})
print(r.json())
```

**Response:**
```json
{
  "tags": [
    {"tag": "cat", "score": 0.9821, "category": 1},
    {"tag": "indoors", "score": 0.8743, "category": 1}
  ],
  "categories": [
    {
      "id": 1,
      "name": "general",
      "color": "#4ade80",
      "tags": [{"tag": "cat", "score": 0.9821, "category": 1}]
    }
  ],
  "count": 2
}
```

---

#### `POST /tag/upload`

Tag an uploaded image file.

**Query parameters:** `max_size`, `floor` (same as above)

**Body:** `multipart/form-data` with field `file`

```sh
curl -X POST "http://localhost:7860/tag/upload?floor=0.1" \
     -F "file=@/path/to/image.jpg"
```

```python
import requests
with open("image.jpg", "rb") as f:
    r = requests.post("http://localhost:7860/tag/upload",
                      params={"floor": 0.1},
                      files={"file": f})
print(r.json())
```

---

### PCA Visualisation

Returns two base64-encoded PNG images of the patch-token PCA features:
- `full` — standard false-colour (PC1→R, PC2→G, PC3→B)
- `custom` — user-defined colour gradient mapped across PCA magnitude

#### `POST /pca/url`

**Query parameters:**

| Param | Default | Description |
|---|---|---|
| `url` | required | HTTP(S) image URL |
| `max_size` | `1024` | Resize long edge |
| `colors` | `#0000ff,#00ff00,#ff0000` | Comma-separated hex colour stops for the custom gradient |

```sh
curl -X POST "http://localhost:7860/pca/url?url=https://example.com/image.jpg&colors=%230000ff,%2300ff00,%23ff0000"
```

```python
import requests, base64
from PIL import Image
from io import BytesIO

r = requests.post("http://localhost:7860/pca/url", params={
    "url": "https://example.com/image.jpg",
    "colors": "#0000ff,#00ff00,#ff0000",
})
data = r.json()
Image.open(BytesIO(base64.b64decode(data["full"]))).save("pca_full.png")
Image.open(BytesIO(base64.b64decode(data["custom"]))).save("pca_custom.png")
```

**Response:**
```json
{
  "full":   "<base64 PNG string>",
  "custom": "<base64 PNG string>"
}
```

---

#### `POST /pca/upload`

**Query parameters:** `max_size`, `colors` (same as above)

**Body:** `multipart/form-data` with field `file`

```sh
curl -X POST "http://localhost:7860/pca/upload?colors=%230000ff,%2300ff00,%23ff0000" \
     -F "file=@/path/to/image.jpg"
```

```python
import requests
with open("image.jpg", "rb") as f:
    r = requests.post("http://localhost:7860/pca/upload",
                      params={"colors": "#0000ff,#00ff00,#ff0000"},
                      files={"file": f})
data = r.json()
```

---

### Image Similarity

Computes cosine similarity between two images using the FEATURE_DIM=6400 backbone descriptor (CLS + register tokens). Score is in `[-1, 1]`, higher = more similar.

#### `POST /similarity/url`

**Query parameters:**

| Param | Default | Description |
|---|---|---|
| `url_a` | required | URL of first image |
| `url_b` | required | URL of second image |
| `max_size` | `1024` | Resize long edge |

```sh
curl -X POST "http://localhost:7860/similarity/url?url_a=https://example.com/a.jpg&url_b=https://example.com/b.jpg"
```

```python
import requests
r = requests.post("http://localhost:7860/similarity/url", params={
    "url_a": "https://example.com/a.jpg",
    "url_b": "https://example.com/b.jpg",
})
print(r.json())
```

**Response:**
```json
{
  "score": 0.912345,
  "desc_a": [0.0012, -0.0034, ...],
  "desc_b": [0.0011, -0.0031, ...]
}
```

`desc_a` and `desc_b` are L2-normalised 6400-dimensional descriptor vectors, usable for downstream similarity search.

---

#### `POST /similarity/upload`

**Query parameters:** `max_size`

**Body:** `multipart/form-data` with fields `file_a` and `file_b`

```sh
curl -X POST "http://localhost:7860/similarity/upload" \
     -F "file_a=@/path/to/a.jpg" \
     -F "file_b=@/path/to/b.jpg"
```

```python
import requests
with open("a.jpg", "rb") as fa, open("b.jpg", "rb") as fb:
    r = requests.post("http://localhost:7860/similarity/upload",
                      files={"file_a": fa, "file_b": fb})
print(r.json())
```

---

## File Overview

| File | Purpose |
|---|---|
| `server_local.py` | Local FastAPI server — all routes |
| `app.py` | HuggingFace Spaces entrypoint (ZeroGPU, do not run locally) |
| `inference_tagger_standalone.py` | Self-contained model implementation |
| `tagger_ui/templates/index_local.html` | Local web UI (plain fetch, no Gradio client) |
| `tagger_ui/templates/index.html` | HF Spaces web UI (@gradio/client) |
| `tagger_ui_server.py` | Original standalone server (tagging only, no PCA/similarity) |
| `requirements_local.txt` | Local dependencies (no gradio/spaces) |
| `requirements.txt` | HF Spaces dependencies |
| `install.bat` / `install.sh` | Interactive installers |
| `start.bat` / `start.sh` | Manually-configured launch scripts |
