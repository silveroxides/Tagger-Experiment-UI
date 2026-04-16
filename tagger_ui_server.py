"""DINOv3 Tagger — FastAPI + Jinja2 Web UI (with category breakdown)

Usage
-----
python tagger_ui_server.py \
    --checkpoint tagger_dino_v3/checkpoints/2026-03-28_22-57-47.safetensors \
    --vocab     tagger_dino_v3/tagger_vocab_with_categories.json \
    --host      0.0.0.0 \
    --port      7860
"""

from __future__ import annotations

import argparse
import io
from pathlib import Path

import torch
import torchvision.transforms.v2 as v2
import uvicorn
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.requests import Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from PIL import Image

from inference_tagger_standalone import (
    PATCH_SIZE,
    Tagger,
    _IMAGENET_MEAN,
    _IMAGENET_STD,
    _snap,
)

# ---------------------------------------------------------------------------
# Category metadata
# ---------------------------------------------------------------------------

# Raw category IDs from the vocab use -1 for unassigned.
# We offset every ID by +1 so all IDs are >= 0, avoiding negative
# numbers in HTML element IDs and JS inline handlers.
_CAT_OFFSET = 1

CATEGORY_META: dict[int, dict] = {
    0:  {"name": "unassigned",      "color": "#6b7280"},   # raw -1
    1:  {"name": "general",         "color": "#4ade80"},   # raw  0
    2:  {"name": "artist",          "color": "#f472b6"},   # raw  1
    3:  {"name": "contributor",     "color": "#a78bfa"},   # raw  2
    4:  {"name": "copyright",       "color": "#fb923c"},   # raw  3
    5:  {"name": "character",       "color": "#60a5fa"},   # raw  4
    6:  {"name": "species/meta",    "color": "#facc15"},   # raw  5
    7:  {"name": "disambiguation",  "color": "#94a3b8"},   # raw  6
    8:  {"name": "meta",            "color": "#e2e8f0"},   # raw  7
    9:  {"name": "lore",            "color": "#f87171"},   # raw  8
}

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(title="DINOv3 Tagger UI")
templates = Jinja2Templates(
    directory=Path(__file__).parent / "tagger_ui" / "templates"
)
templates.env.filters["format_number"] = lambda v: f"{v:,}"

_tagger: Tagger | None = None
_tag2category: dict[str, int] = {}
_vocab_path: str = ""


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {
        "request":       request,
        "num_tags":      _tagger.num_tags if _tagger else 0,
        "vocab_path":    _vocab_path,
        "category_meta": CATEGORY_META,
    })


@app.post("/tag/url")
async def tag_url(
    url:      str   = Query(...),
    max_size: int   = Query(default=1024),
    floor:    float = Query(default=0.05),
):
    assert _tagger is not None
    try:
        from inference_tagger_standalone import _open_image
        img = _open_image(url)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not fetch image: {e}")
    return _run_tagger(img, max_size, floor)


@app.post("/tag/upload")
async def tag_upload(
    file:     UploadFile = File(...),
    max_size: int        = Query(default=1024),
    floor:    float      = Query(default=0.05),
):
    assert _tagger is not None
    try:
        data = await file.read()
        img = Image.open(io.BytesIO(data)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not read image: {e}")
    return _run_tagger(img, max_size, floor)


# ---------------------------------------------------------------------------
# PCA endpoints
# ---------------------------------------------------------------------------

@app.post("/pca/url")
async def pca_url(
    url:      str = Query(...),
    max_size: int = Query(default=1024),
):
    from fastapi.responses import Response
    assert _tagger is not None
    try:
        from inference_tagger_standalone import _open_image
        img = _open_image(url)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not fetch image: {e}")
    pca_img = _tagger.embed_pca(img, max_size=max_size)
    buf = io.BytesIO()
    pca_img.save(buf, format="PNG")
    return Response(content=buf.getvalue(), media_type="image/png")


@app.post("/pca/upload")
async def pca_upload(
    file:     UploadFile = File(...),
    max_size: int        = Query(default=1024),
):
    from fastapi.responses import Response
    assert _tagger is not None
    try:
        data = await file.read()
        img = Image.open(io.BytesIO(data)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not read image: {e}")
    pca_img = _tagger.embed_pca(img, max_size=max_size)
    buf = io.BytesIO()
    pca_img.save(buf, format="PNG")
    return Response(content=buf.getvalue(), media_type="image/png")


# ---------------------------------------------------------------------------
# Inference helper
# ---------------------------------------------------------------------------

def _run_tagger(
    img: Image.Image,
    max_size: int,
    floor: float = 0.05,
) -> dict:
    """Return every tag whose sigmoid score >= floor, sorted desc.
    The frontend applies per-category topk / threshold on top of this.
    """
    assert _tagger is not None

    w, h = img.size
    scale = min(1.0, max_size / max(w, h))
    new_w = _snap(round(w * scale), PATCH_SIZE)
    new_h = _snap(round(h * scale), PATCH_SIZE)

    pixel_values = v2.Compose([
        v2.Resize((new_h, new_w), interpolation=v2.InterpolationMode.LANCZOS),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
    ])(img).unsqueeze(0).to(_tagger.device)

    with torch.no_grad(), torch.autocast(device_type=_tagger.device.type, dtype=_tagger.dtype):
        logits = _tagger.model(pixel_values)[0]

    scores = torch.sigmoid(logits.float())

    # Return all tags above the floor, sorted by score descending
    indices = (scores >= floor).nonzero(as_tuple=True)[0]
    values  = scores[indices]
    order   = values.argsort(descending=True)
    indices = indices[order]
    values  = values[order]

    # Build per-category buckets
    by_category: dict[int, list] = {}
    all_tags = []
    for i, v in zip(indices.tolist(), values.tolist()):
        tag  = _tagger.idx2tag[i]
        cat  = _tag2category.get(tag, -1) + _CAT_OFFSET
        item = {"tag": tag, "score": round(v, 4), "category": cat}
        all_tags.append(item)
        by_category.setdefault(cat, []).append(item)

    categories = []
    for cat_id in sorted(by_category.keys()):
        meta = CATEGORY_META.get(cat_id, {"name": str(cat_id), "color": "#6b7280"})
        categories.append({
            "id":    cat_id,
            "name":  meta["name"],
            "color": meta["color"],
            "tags":  by_category[cat_id],
        })

    return {
        "tags":       all_tags,
        "categories": categories,
        "count":      len(all_tags),
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    global _tagger, _tag2category, _vocab_path
    import json

    parser = argparse.ArgumentParser(description="DINOv3 Tagger Web UI")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--vocab",      required=True,
                        help="Path to tagger_vocab_with_categories.json")
    parser.add_argument("--device",   default="cuda")
    parser.add_argument("--max-size", type=int, default=1024)
    parser.add_argument("--host",     default="0.0.0.0")
    parser.add_argument("--port",     type=int, default=7860)
    args = parser.parse_args()

    _vocab_path = args.vocab

    # Load tag→category mapping from the enriched vocab file
    with open(args.vocab) as f:
        vocab_data = json.load(f)
    _tag2category = vocab_data.get("tag2category", {})

    _tagger = Tagger(
        checkpoint_path=args.checkpoint,
        vocab_path=args.vocab,          # Tagger only reads idx2tag from this
        device=args.device,
        max_size=args.max_size,
    )

    print(f"\n  Tagger UI  →  http://{args.host}:{args.port}\n")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
