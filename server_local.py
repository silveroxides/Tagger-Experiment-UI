"""DINOv3 Tagger — Local Standalone Server (feature-complete)

Extends tagger_ui_server.py with:
  • Dual PCA endpoint — returns {full, custom} base64 PNGs
  • Image similarity endpoint — cosine similarity via forward_embedding
  • Serves index_local.html

Usage
-----
python server_local.py \
    --checkpoint ./tagger_proto.safetensors \
    --vocab      ./tagger_vocab_with_categories.json \
    --device     cuda \
    --port       7860
"""

from __future__ import annotations

import argparse
import base64
import io
import json
from pathlib import Path

import numpy as np
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
    _open_image,
    _snap,
)

# ---------------------------------------------------------------------------
# Category metadata  (identical to tagger_ui_server.py / app.py)
# ---------------------------------------------------------------------------

_CAT_OFFSET = 1

CATEGORY_META: dict[int, dict] = {
    0: {"name": "unassigned", "color": "#6b7280"},
    1: {"name": "general", "color": "#4ade80"},
    2: {"name": "artist", "color": "#f472b6"},
    3: {"name": "contributor", "color": "#a78bfa"},
    4: {"name": "copyright", "color": "#fb923c"},
    5: {"name": "character", "color": "#60a5fa"},
    6: {"name": "species/meta", "color": "#facc15"},
    7: {"name": "disambiguation", "color": "#94a3b8"},
    8: {"name": "meta", "color": "#e2e8f0"},
    9: {"name": "lore", "color": "#f87171"},
}

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(title="DINOv3 Tagger UI (Local)")
templates = Jinja2Templates(directory=Path(__file__).parent / "tagger_ui" / "templates")
templates.env.filters["format_number"] = lambda v: f"{v:,}"

_tagger: Tagger | None = None
_tag2category: dict[str, int] = {}
_vocab_path: str = ""


# ---------------------------------------------------------------------------
# Routes — UI
# ---------------------------------------------------------------------------


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(
        "index_local.html",
        {
            "request": request,
            "num_tags": _tagger.num_tags if _tagger else 0,
            "vocab_path": _vocab_path,
            "category_meta": CATEGORY_META,
        },
    )


# ---------------------------------------------------------------------------
# Routes — Tagging
# ---------------------------------------------------------------------------


@app.post("/tag/url")
async def tag_url(
    url: str = Query(...),
    max_size: int = Query(default=1024),
    floor: float = Query(default=0.05),
):
    assert _tagger is not None
    try:
        img = _open_image(url)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not fetch image: {e}")
    return _run_tagger(img, max_size, floor)


@app.post("/tag/upload")
async def tag_upload(
    file: UploadFile = File(...),
    max_size: int = Query(default=1024),
    floor: float = Query(default=0.05),
):
    assert _tagger is not None
    try:
        data = await file.read()
        img = Image.open(io.BytesIO(data)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not read image: {e}")
    return _run_tagger(img, max_size, floor)


# ---------------------------------------------------------------------------
# Routes — PCA (dual: full rainbow + custom-colour blend)
# ---------------------------------------------------------------------------


@app.post("/pca/url")
async def pca_url(
    url: str = Query(...),
    max_size: int = Query(default=1024),
    color1: str = Query(default="#ff0000"),
    color2: str = Query(default="#00ff00"),
    color3: str = Query(default="#0000ff"),
):
    assert _tagger is not None
    try:
        img = _open_image(url)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not fetch image: {e}")
    return _run_pca(img, max_size, color1, color2, color3)


@app.post("/pca/upload")
async def pca_upload(
    file: UploadFile = File(...),
    max_size: int = Query(default=1024),
    color1: str = Query(default="#ff0000"),
    color2: str = Query(default="#00ff00"),
    color3: str = Query(default="#0000ff"),
):
    assert _tagger is not None
    try:
        data = await file.read()
        img = Image.open(io.BytesIO(data)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not read image: {e}")
    return _run_pca(img, max_size, color1, color2, color3)


# ---------------------------------------------------------------------------
# Routes — Similarity
# ---------------------------------------------------------------------------


@app.post("/similarity/url")
async def similarity_url(
    url_a: str = Query(...),
    url_b: str = Query(...),
    max_size: int = Query(default=1024),
):
    assert _tagger is not None
    try:
        img_a = _open_image(url_a)
        img_b = _open_image(url_b)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not fetch image: {e}")
    return _run_similarity(img_a, img_b, max_size)


@app.post("/similarity/upload")
async def similarity_upload(
    file_a: UploadFile = File(...),
    file_b: UploadFile = File(...),
    max_size: int = Query(default=1024),
):
    assert _tagger is not None
    try:
        data_a = await file_a.read()
        data_b = await file_b.read()
        img_a = Image.open(io.BytesIO(data_a)).convert("RGB")
        img_b = Image.open(io.BytesIO(data_b)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not read image: {e}")
    return _run_similarity(img_a, img_b, max_size)


# ---------------------------------------------------------------------------
# Helpers — image preprocessing
# ---------------------------------------------------------------------------


def _preprocess(img: Image.Image, max_size: int) -> torch.Tensor:
    """Resize + ImageNet-normalise → [1, 3, H, W] float32 on tagger device."""
    assert _tagger is not None
    w, h = img.size
    scale = min(1.0, max_size / max(w, h))
    new_w = _snap(round(w * scale), PATCH_SIZE)
    new_h = _snap(round(h * scale), PATCH_SIZE)
    return (
        v2.Compose(
            [
                v2.Resize((new_h, new_w), interpolation=v2.InterpolationMode.LANCZOS),
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
            ]
        )(img)
        .unsqueeze(0)
        .to(_tagger.device)
    )


def _pil_to_base64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def _hex_to_rgb(hex_color: str) -> tuple[float, float, float]:
    """'#rrggbb' → (r, g, b) each in [0, 1]."""
    h = hex_color.lstrip("#")
    return tuple(int(h[i : i + 2], 16) / 255.0 for i in (0, 2, 4))


# ---------------------------------------------------------------------------
# Helpers — PCA
# ---------------------------------------------------------------------------


def _pca_extract(pv: torch.Tensor) -> tuple[np.ndarray, int, int]:
    """Backbone forward → normalised PCA projection + grid dims.

    Returns (proj_norm [N, 3] float32 numpy, h_p, w_p).
    Runs under torch.no_grad() — no GPU decorator needed locally.
    """
    assert _tagger is not None
    with (
        torch.no_grad(),
        torch.autocast(device_type=_tagger.device.type, dtype=_tagger.dtype),
    ):
        patch_tokens, h_p, w_p = _tagger.model.backbone.get_image_tokens(pv)

    tokens = patch_tokens[0].float()
    tokens_c = tokens - tokens.mean(dim=0, keepdim=True)
    _, _, Vt = torch.linalg.svd(tokens_c, full_matrices=False)
    projected = tokens_c @ Vt[:3].T  # [N, 3]

    lo = projected.min(dim=0).values
    hi = projected.max(dim=0).values
    proj_norm = (projected - lo) / (hi - lo + 1e-8)  # [N, 3] in [0, 1]

    return proj_norm.cpu().numpy(), h_p, w_p


def _build_custom_pca(
    proj_norm: np.ndarray,
    h_p: int,
    w_p: int,
    color1: str,
    color2: str,
    color3: str,
) -> Image.Image:
    """Blend the three normalised PC channels using user-chosen colours.

    For each patch: output = PC1_val * color1_rgb
                           + PC2_val * color2_rgb
                           + PC3_val * color3_rgb
    Result is divided by the maximum possible sum so it stays in [0, 1].
    Upscaled to pixel resolution (NEAREST) matching the full rainbow image.
    """
    c1 = np.array(_hex_to_rgb(color1), dtype=np.float32)
    c2 = np.array(_hex_to_rgb(color2), dtype=np.float32)
    c3 = np.array(_hex_to_rgb(color3), dtype=np.float32)

    blended = (
        proj_norm[:, 0:1] * c1 + proj_norm[:, 1:2] * c2 + proj_norm[:, 2:3] * c3
    )  # [N, 3]

    mx = blended.max()
    if mx > 0:
        blended /= mx

    rgb = blended.reshape(h_p, w_p, 3)
    patch_img = Image.fromarray((rgb * 255).clip(0, 255).astype("uint8"), "RGB")
    return patch_img.resize(
        (w_p * PATCH_SIZE, h_p * PATCH_SIZE), resample=Image.NEAREST
    )


def _run_pca(
    img: Image.Image,
    max_size: int,
    color1: str,
    color2: str,
    color3: str,
) -> dict:
    """Return {"full": b64_png, "custom": b64_png}."""
    pv = _preprocess(img, max_size)
    proj_norm, h_p, w_p = _pca_extract(pv)

    # full rainbow: PC1→R, PC2→G, PC3→B, upscaled to pixel resolution
    rgb_full = proj_norm.reshape(h_p, w_p, 3)
    full_patch = Image.fromarray((rgb_full * 255).clip(0, 255).astype("uint8"), "RGB")
    full_img = full_patch.resize(
        (w_p * PATCH_SIZE, h_p * PATCH_SIZE), resample=Image.NEAREST
    )

    custom_img = _build_custom_pca(proj_norm, h_p, w_p, color1, color2, color3)

    return {
        "full": _pil_to_base64(full_img),
        "custom": _pil_to_base64(custom_img),
    }


# ---------------------------------------------------------------------------
# Helpers — Similarity
# ---------------------------------------------------------------------------


def _extract_descriptor(pv: torch.Tensor) -> np.ndarray:
    """Extract the FEATURE_DIM=6400 descriptor via forward_embedding.
    Returns a [6400] float32 numpy array.
    """
    assert _tagger is not None
    with (
        torch.no_grad(),
        torch.autocast(device_type=_tagger.device.type, dtype=_tagger.dtype),
    ):
        features = _tagger.model.forward_embedding(pv)  # [1, 6400]
    return features[0].cpu().numpy()  # [6400]


def _run_similarity(
    img_a: Image.Image,
    img_b: Image.Image,
    max_size: int,
) -> dict:
    """Return {"score": float, "desc_a": [...], "desc_b": [...]}."""
    pv_a = _preprocess(img_a, max_size)
    pv_b = _preprocess(img_b, max_size)

    feat_a = _extract_descriptor(pv_a)
    feat_b = _extract_descriptor(pv_b)

    # L2-normalise
    feat_a = feat_a / (np.linalg.norm(feat_a) + 1e-8)
    feat_b = feat_b / (np.linalg.norm(feat_b) + 1e-8)

    score = float(np.dot(feat_a, feat_b))

    return {
        "score": round(score, 6),
        "desc_a": feat_a.tolist(),
        "desc_b": feat_b.tolist(),
    }


# ---------------------------------------------------------------------------
# Helpers — Tagging  (identical to tagger_ui_server.py)
# ---------------------------------------------------------------------------


def _run_tagger(
    img: Image.Image,
    max_size: int,
    floor: float = 0.05,
) -> dict:
    assert _tagger is not None

    w, h = img.size
    scale = min(1.0, max_size / max(w, h))
    new_w = _snap(round(w * scale), PATCH_SIZE)
    new_h = _snap(round(h * scale), PATCH_SIZE)

    pixel_values = (
        v2.Compose(
            [
                v2.Resize((new_h, new_w), interpolation=v2.InterpolationMode.LANCZOS),
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
            ]
        )(img)
        .unsqueeze(0)
        .to(_tagger.device)
    )

    with (
        torch.no_grad(),
        torch.autocast(device_type=_tagger.device.type, dtype=_tagger.dtype),
    ):
        logits = _tagger.model(pixel_values)[0]

    scores = torch.sigmoid(logits.float())
    indices = (scores >= floor).nonzero(as_tuple=True)[0]
    values = scores[indices]
    order = values.argsort(descending=True)
    indices = indices[order]
    values = values[order]

    by_category: dict[int, list] = {}
    all_tags: list[dict] = []
    for i, v in zip(indices.tolist(), values.tolist()):
        tag = _tagger.idx2tag[i]
        cat = _tag2category.get(tag, -1) + _CAT_OFFSET
        item = {"tag": tag, "score": round(v, 4), "category": cat}
        all_tags.append(item)
        by_category.setdefault(cat, []).append(item)

    categories = []
    for cat_id in sorted(by_category.keys()):
        meta = CATEGORY_META.get(cat_id, {"name": str(cat_id), "color": "#6b7280"})
        categories.append(
            {
                "id": cat_id,
                "name": meta["name"],
                "color": meta["color"],
                "tags": by_category[cat_id],
            }
        )

    return {
        "tags": all_tags,
        "categories": categories,
        "count": len(all_tags),
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main():
    global _tagger, _tag2category, _vocab_path

    parser = argparse.ArgumentParser(description="DINOv3 Tagger — Local Standalone")
    parser.add_argument(
        "--checkpoint", required=True, help="Path to .safetensors checkpoint"
    )
    parser.add_argument(
        "--vocab", required=True, help="Path to tagger_vocab_with_categories.json"
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-size", type=int, default=1024)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()

    _vocab_path = args.vocab

    with open(args.vocab) as f:
        vocab_data = json.load(f)
    _tag2category = vocab_data.get("tag2category", {})

    _tagger = Tagger(
        checkpoint_path=args.checkpoint,
        vocab_path=args.vocab,
        device=args.device,
        max_size=args.max_size,
    )

    print(f"\n  Tagger UI (local)  →  http://{args.host}:{args.port}\n")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
