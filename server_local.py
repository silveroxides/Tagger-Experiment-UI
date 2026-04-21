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
import hashlib
import io
import json
import zipfile
from pathlib import Path
from typing import AsyncIterator, List, Optional

import numpy as np
import torch
import torchvision.transforms.v2 as v2
import uvicorn
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.requests import Request
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from PIL import Image
from tqdm import tqdm

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
    0: {"name": "unassigned",     "color": "#6b7280", "display_order": 9},
    1: {"name": "general",        "color": "#4ade80", "display_order": 4},
    2: {"name": "artist",         "color": "#f472b6", "display_order": 0},
    3: {"name": "contributor",    "color": "#a78bfa", "display_order": 7},
    4: {"name": "copyright",      "color": "#fb923c", "display_order": 1},
    5: {"name": "character",      "color": "#60a5fa", "display_order": 2},
    6: {"name": "species",        "color": "#facc15", "display_order": 3},
    7: {"name": "disambiguation", "color": "#94a3b8", "display_order": 8},
    8: {"name": "meta",           "color": "#e2e8f0", "display_order": 6},
    9: {"name": "lore",           "color": "#f87171", "display_order": 5},
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
_batch_enabled: bool = False

# Image extensions accepted in batch zip uploads
_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tiff", ".tif"}


# ---------------------------------------------------------------------------
# Routes — UI
# ---------------------------------------------------------------------------


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(
        request,
        "index_local.html",
        {
            "num_tags": _tagger.num_tags if _tagger else 0,
            "vocab_path": _vocab_path,
            "category_meta": CATEGORY_META,
            "batch_enabled": _batch_enabled,
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
    colors: str = Query(default="#0000ff,#00ff00,#ff0000"),
):
    assert _tagger is not None
    try:
        img = _open_image(url)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not fetch image: {e}")
    return _run_pca(img, max_size, colors)


@app.post("/pca/upload")
async def pca_upload(
    file: UploadFile = File(...),
    max_size: int = Query(default=1024),
    colors: str = Query(default="#0000ff,#00ff00,#ff0000"),
):
    assert _tagger is not None
    try:
        data = await file.read()
        img = Image.open(io.BytesIO(data)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not read image: {e}")
    return _run_pca(img, max_size, colors)


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
# Routes — Batch Tagging  (only registered when --enable-batch)
# ---------------------------------------------------------------------------


def _register_batch_routes() -> None:
    """Called from main() when --enable-batch is set."""

    @app.post("/tag/batch")
    async def tag_batch(
        files: List[UploadFile] = File(default=[]),
        archive: Optional[UploadFile] = File(default=None),
        max_size: int = Query(default=1024),
        floor: float = Query(default=0.05),
    ):
        assert _tagger is not None

        # --- collect (raw_bytes, filename, concept) tuples ---
        items: list[tuple[bytes, str, str | None]] = []

        if archive is not None:
            raw_zip = await archive.read()
            try:
                zf = zipfile.ZipFile(io.BytesIO(raw_zip))
            except zipfile.BadZipFile as e:
                raise HTTPException(status_code=400, detail=f"Bad zip: {e}")
            for entry in zf.infolist():
                if entry.is_dir():
                    continue
                p = Path(entry.filename)
                # skip macOS metadata
                if "__MACOSX" in p.parts:
                    continue
                if p.suffix.lower() not in _IMAGE_EXTS:
                    continue
                concept = str(p.parent) if str(p.parent) not in (".", "") else None
                items.append((zf.read(entry.filename), p.name, concept))
        else:
            for uf in files:
                data = await uf.read()
                items.append((data, uf.filename or "unknown", None))

        if not items:
            raise HTTPException(status_code=400, detail="No images found in request.")

        async def generate() -> AsyncIterator[str]:
            bar = tqdm(total=len(items), desc="tag/batch", unit="img")
            for raw, filename, concept in items:
                img_hash = hashlib.sha256(raw).hexdigest()
                try:
                    img = Image.open(io.BytesIO(raw)).convert("RGB")
                    result = _run_tagger(img, max_size, floor)
                    line = {
                        "hash": img_hash,
                        "filename": filename,
                        "concept": concept,
                        "tags": result["tags"],
                        "count": result["count"],
                    }
                except Exception as exc:
                    line = {
                        "hash": img_hash,
                        "filename": filename,
                        "concept": concept,
                        "error": str(exc),
                    }
                bar.update(1)
                yield json.dumps(line, ensure_ascii=False) + "\n"
            bar.close()

        return StreamingResponse(
            generate(),
            media_type="application/x-ndjson",
            headers={"Content-Disposition": 'attachment; filename="tags.jsonl"'},
        )

    # -----------------------------------------------------------------------
    # Routes — Batch Similarity  (only registered when --enable-batch)
    # -----------------------------------------------------------------------

    @app.post("/similarity/batch")
    async def similarity_batch(
        archive: Optional[UploadFile] = File(default=None),
        json_file: Optional[UploadFile] = File(default=None),
        max_size: int = Query(default=1024),
    ):
        assert _tagger is not None

        # --- Build list of (pair_id, file_a_label, file_b_label, img_a, img_b) ---
        # Each entry: (pair_id: str, label_a: str, label_b: str, img_a: Image, img_b: Image)
        pairs: list[tuple[str, str, str, Image.Image, Image.Image]] = []
        warnings: list[str] = []

        if archive is not None:
            raw_zip = await archive.read()
            try:
                zf = zipfile.ZipFile(io.BytesIO(raw_zip))
            except zipfile.BadZipFile as e:
                raise HTTPException(status_code=400, detail=f"Bad zip: {e}")

            # find exactly 2 top-level dirs
            top_dirs: set[str] = set()
            for entry in zf.infolist():
                if entry.is_dir():
                    continue
                parts = Path(entry.filename).parts
                if len(parts) >= 2:
                    top_dirs.add(parts[0])
            if len(top_dirs) != 2:
                raise HTTPException(
                    status_code=400,
                    detail=f"Zip must contain exactly 2 top-level directories; found: {sorted(top_dirs) or 'none'}",
                )
            dir_a, dir_b = sorted(top_dirs)

            def _zip_images(d: str) -> dict[str, tuple[str, bytes]]:
                """stem → (full_path, bytes) for all images in directory d."""
                out: dict[str, tuple[str, bytes]] = {}
                for entry in zf.infolist():
                    if entry.is_dir():
                        continue
                    p = Path(entry.filename)
                    if p.parts[0] != d:
                        continue
                    if p.suffix.lower() not in _IMAGE_EXTS:
                        continue
                    out[p.stem] = (entry.filename, zf.read(entry.filename))
                return dict(sorted(out.items()))  # alphanumeric sort by stem

            imgs_a = _zip_images(dir_a)
            imgs_b = _zip_images(dir_b)
            all_stems = sorted(set(imgs_a) | set(imgs_b))

            for stem in all_stems:
                if stem not in imgs_a:
                    warnings.append(f"no match for dir_b file: {imgs_b[stem][0]}")
                    continue
                if stem not in imgs_b:
                    warnings.append(f"no match for dir_a file: {imgs_a[stem][0]}")
                    continue
                path_a, raw_a = imgs_a[stem]
                path_b, raw_b = imgs_b[stem]
                try:
                    img_a = Image.open(io.BytesIO(raw_a)).convert("RGB")
                    img_b = Image.open(io.BytesIO(raw_b)).convert("RGB")
                    pairs.append((stem, path_a, path_b, img_a, img_b))
                except Exception as exc:
                    warnings.append(f"could not decode pair '{stem}': {exc}")

        elif json_file is not None:
            raw_json = await json_file.read()
            try:
                payload = json.loads(raw_json)
            except json.JSONDecodeError as e:
                raise HTTPException(status_code=400, detail=f"Invalid JSON: {e}")

            if not isinstance(payload, dict):
                raise HTTPException(status_code=400, detail="JSON root must be an object.")

            # Detect format
            if "pairs" in payload:
                # flat list format: {"pairs": [{"id": ..., "url_a": ..., "url_b": ...}]}
                pair_list = payload["pairs"]
                if not isinstance(pair_list, list):
                    raise HTTPException(status_code=400, detail='"pairs" must be a list.')
                raw_pairs = [
                    (str(p.get("id", i)), p.get("url_a", ""), p.get("url_b", ""))
                    for i, p in enumerate(pair_list)
                ]
            else:
                # parallel dicts: {"a": {"id": "url", ...}, "b": {"id": "url", ...}}
                keys = list(payload.keys())
                if len(keys) != 2:
                    raise HTTPException(
                        status_code=400,
                        detail="Parallel JSON format must have exactly 2 top-level keys.",
                    )
                dict_a, dict_b = payload[keys[0]], payload[keys[1]]
                all_ids = sorted(set(dict_a) | set(dict_b))
                raw_pairs = []
                for pid in all_ids:
                    if pid not in dict_a:
                        warnings.append(f"no url_a for pair id: {pid}")
                        continue
                    if pid not in dict_b:
                        warnings.append(f"no url_b for pair id: {pid}")
                        continue
                    raw_pairs.append((pid, dict_a[pid], dict_b[pid]))

            for pair_id, url_a, url_b in raw_pairs:
                try:
                    img_a = _open_image(url_a)
                    img_b = _open_image(url_b)
                    pairs.append((pair_id, url_a, url_b, img_a, img_b))
                except Exception as exc:
                    warnings.append(f"could not fetch pair '{pair_id}': {exc}")
        else:
            raise HTTPException(
                status_code=400,
                detail="Provide either 'archive' (zip) or 'json_file' field.",
            )

        if not pairs and not warnings:
            raise HTTPException(status_code=400, detail="No valid pairs found.")

        async def generate() -> AsyncIterator[str]:
            scores: list[float] = []
            errors = 0
            bar = tqdm(total=len(pairs), desc="similarity/batch", unit="pair")

            # emit warnings upfront
            for w in warnings:
                yield json.dumps({"warning": w}, ensure_ascii=False) + "\n"

            for pair_id, label_a, label_b, img_a, img_b in pairs:
                try:
                    result = _run_similarity(img_a, img_b, max_size)
                    score = result["score"]
                    scores.append(score)
                    line: dict = {
                        "pair_id": pair_id,
                        "file_a": label_a,
                        "file_b": label_b,
                        "score": score,
                    }
                except Exception as exc:
                    errors += 1
                    line = {
                        "pair_id": pair_id,
                        "file_a": label_a,
                        "file_b": label_b,
                        "error": str(exc),
                    }
                bar.update(1)
                yield json.dumps(line, ensure_ascii=False) + "\n"

            bar.close()

            # summary line
            summary: dict = {
                "summary": True,
                "total_pairs": len(pairs),
                "errors": errors,
                "warnings": len(warnings),
            }
            if scores:
                summary["mean_score"] = round(float(np.mean(scores)), 6)
                summary["min_score"] = round(float(np.min(scores)), 6)
                summary["max_score"] = round(float(np.max(scores)), 6)
            yield json.dumps(summary, ensure_ascii=False) + "\n"

        return StreamingResponse(
            generate(),
            media_type="application/x-ndjson",
            headers={"Content-Disposition": 'attachment; filename="similarity.jsonl"'},
        )


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
    colors: list[str],
) -> Image.Image:
    """Map PCA patches through a user-defined N-stop colour gradient.

    Each patch gets a scalar t = mean(PC1, PC2, PC3) in [0, 1], then
    t is used to interpolate through the ordered colour stops.
    """
    stops = np.array([_hex_to_rgb(c) for c in colors], dtype=np.float32)  # [N, 3]
    n = len(stops)

    # scalar per patch: mean of the 3 normalised PC values
    t = proj_norm.mean(axis=1)  # [P] in [0, 1]

    if n == 1:
        rgb_out = np.tile(stops[0], (len(t), 1))
    else:
        # map t into segment index
        t_scaled = t * (n - 1)  # [P] in [0, n-1]
        idx_lo = np.floor(t_scaled).astype(int).clip(0, n - 2)
        idx_hi = idx_lo + 1
        frac = (t_scaled - idx_lo)[:, None]  # [P, 1]
        rgb_out = stops[idx_lo] * (1 - frac) + stops[idx_hi] * frac  # [P, 3]

    mx = rgb_out.max()
    if mx > 0:
        rgb_out /= mx

    rgb = rgb_out.reshape(h_p, w_p, 3)
    patch_img = Image.fromarray((rgb * 255).clip(0, 255).astype("uint8"), "RGB")
    return patch_img.resize(
        (w_p * PATCH_SIZE, h_p * PATCH_SIZE), resample=Image.NEAREST
    )


def _run_pca(
    img: Image.Image,
    max_size: int,
    colors: str,
) -> dict:
    """Return {"full": b64_png, "custom": b64_png}."""
    color_list = [c.strip() for c in colors.split(",") if c.strip()]
    if not color_list:
        color_list = ["#0000ff", "#00ff00", "#ff0000"]

    pv = _preprocess(img, max_size)
    proj_norm, h_p, w_p = _pca_extract(pv)

    # full rainbow: PC1→R, PC2→G, PC3→B
    rgb_full = proj_norm.reshape(h_p, w_p, 3)
    full_patch = Image.fromarray((rgb_full * 255).clip(0, 255).astype("uint8"), "RGB")
    full_img = full_patch.resize(
        (w_p * PATCH_SIZE, h_p * PATCH_SIZE), resample=Image.NEAREST
    )

    custom_img = _build_custom_pca(proj_norm, h_p, w_p, color_list)

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
    for cat_id in sorted(by_category.keys(), key=lambda cid: CATEGORY_META.get(cid, {}).get("display_order", cid)):
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
    global _tagger, _tag2category, _vocab_path, _batch_enabled

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
    parser.add_argument(
        "--enable-batch",
        action="store_true",
        default=False,
        help="Enable batch tagging (/tag/batch) and batch similarity (/similarity/batch) endpoints",
    )
    args = parser.parse_args()

    _vocab_path = args.vocab
    _batch_enabled = args.enable_batch

    with open(args.vocab) as f:
        vocab_data = json.load(f)
    _tag2category = vocab_data.get("tag2category", {})

    _tagger = Tagger(
        checkpoint_path=args.checkpoint,
        vocab_path=args.vocab,
        device=args.device,
        max_size=args.max_size,
    )

    if _batch_enabled:
        _register_batch_routes()
        print("  Batch endpoints enabled: /tag/batch  /similarity/batch")

    print(f"\n  Tagger UI (local)  →  http://{args.host}:{args.port}\n")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
