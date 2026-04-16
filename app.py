import torch

torch.set_grad_enabled(False)

import base64
import io
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import spaces
import torchvision.transforms.v2 as v2
from fastapi.requests import Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from gradio import Server
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
# Model download + init
# ---------------------------------------------------------------------------

os.system(
    "wget -nv https://huggingface.co/lodestones/tagger-experiment/resolve/main/tagger_proto.safetensors"
)

_VOCAB_PATH = "./tagger_vocab_with_categories.json"

model = Tagger(
    checkpoint_path="./tagger_proto.safetensors",
    vocab_path=_VOCAB_PATH,
    max_size=1024,
)

with open(_VOCAB_PATH) as f:
    _tag2category: dict[str, int] = json.load(f).get("tag2category", {})

# ---------------------------------------------------------------------------
# Category metadata  (mirrors tagger_ui_server.py)
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
# Helpers
# ---------------------------------------------------------------------------


def _resolve_image_source(image: Any) -> str:
    """Normalise the image argument from Gradio.

    Gradio passes uploaded files as a dict:
        {"path": "/tmp/gradio/...", "orig_name": "...", "url": "...", ...}
    URL strings and local paths are passed as plain str.
    """
    if isinstance(image, dict):
        return image.get("path") or image.get("url") or image["orig_name"]
    return str(image)


def _hex_to_rgb(hex_color: str) -> tuple[float, float, float]:
    """'#rrggbb' → (r, g, b) each in [0, 1]."""
    h = hex_color.lstrip("#")
    return tuple(int(h[i : i + 2], 16) / 255.0 for i in (0, 2, 4))


def _preprocess(img: Image.Image, max_size: int) -> torch.Tensor:
    """Resize + ImageNet-normalise → [1, 3, H, W] float32 CPU tensor."""
    w, h = img.size
    scale = min(1.0, max_size / max(w, h))
    new_w = _snap(round(w * scale), PATCH_SIZE)
    new_h = _snap(round(h * scale), PATCH_SIZE)
    return v2.Compose(
        [
            v2.Resize((new_h, new_w), interpolation=v2.InterpolationMode.LANCZOS),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
        ]
    )(img).unsqueeze(0)


def _postprocess(logits: torch.Tensor, floor: float) -> str:
    """sigmoid → filter → sort → build category buckets → JSON string."""
    scores = torch.sigmoid(logits)
    indices = (scores >= floor).nonzero(as_tuple=True)[0]
    values = scores[indices]
    order = values.argsort(descending=True)
    indices = indices[order]
    values = values[order]

    by_category: dict[int, list] = {}
    all_tags: list[dict] = []
    for i, v in zip(indices.tolist(), values.tolist()):
        tag = model.idx2tag[i]
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

    return json.dumps(
        {"tags": all_tags, "categories": categories, "count": len(all_tags)}
    )


def _pil_to_base64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def _build_custom_pca(
    proj_norm: np.ndarray, h_p: int, w_p: int, color1: str, color2: str, color3: str
) -> Image.Image:
    """
    Blend the three normalised PC channels using user-chosen colours.

    For each patch: output = PC1_val * color1_rgb
                           + PC2_val * color2_rgb
                           + PC3_val * color3_rgb
    Result is divided by the maximum possible sum (sum of the three
    colour magnitudes) so the output stays in [0, 1], then clamped.
    """
    c1 = np.array(_hex_to_rgb(color1), dtype=np.float32)
    c2 = np.array(_hex_to_rgb(color2), dtype=np.float32)
    c3 = np.array(_hex_to_rgb(color3), dtype=np.float32)

    # proj_norm: [N, 3], values in [0, 1]
    blended = (
        proj_norm[:, 0:1] * c1 + proj_norm[:, 1:2] * c2 + proj_norm[:, 2:3] * c3
    )  # [N, 3]

    # normalise so the brightest patch reaches full intensity
    mx = blended.max()
    if mx > 0:
        blended /= mx

    rgb = blended.reshape(h_p, w_p, 3)
    patch_img = Image.fromarray((rgb * 255).clip(0, 255).astype("uint8"), "RGB")
    return patch_img.resize(
        (w_p * PATCH_SIZE, h_p * PATCH_SIZE), resample=Image.NEAREST
    )


# ---------------------------------------------------------------------------
# GPU-isolated helpers
# ---------------------------------------------------------------------------


@spaces.GPU
def _gpu_extract_descriptor(pixel_values: torch.Tensor) -> np.ndarray:
    """Extract the FEATURE_DIM=6400 image descriptor via forward_embedding.
    Returns a [6400] float32 numpy array on CPU.
    """
    pv = pixel_values.to(model.device)
    with (
        torch.no_grad(),
        torch.autocast(device_type=model.device.type, dtype=model.dtype),
    ):
        features = model.model.forward_embedding(pv)  # [1, 6400]
    return features[0].cpu().numpy()  # [6400]


@spaces.GPU
def _gpu_infer(pixel_values: torch.Tensor) -> torch.Tensor:
    """Move tensor to device, run model forward, return CPU logits."""
    pv = pixel_values.to(model.device)
    with (
        torch.no_grad(),
        torch.autocast(device_type=model.device.type, dtype=model.dtype),
    ):
        logits = model.model(pv)[0]
    return logits.float().cpu()


@spaces.GPU
def _gpu_pca_extract(pixel_values: torch.Tensor) -> tuple:
    """Backbone forward → normalised PCA projection + grid dims.

    Returns (proj_norm_np [N,3], h_p, w_p) on CPU — no colour mapping here
    so callers can apply both full and custom colourings without re-running
    the backbone.
    """
    pv = pixel_values.to(model.device)
    with (
        torch.no_grad(),
        torch.autocast(device_type=model.device.type, dtype=model.dtype),
    ):
        patch_tokens, h_p, w_p = model.model.backbone.get_image_tokens(pv)

    tokens = patch_tokens[0].float()
    tokens_c = tokens - tokens.mean(dim=0, keepdim=True)
    _, _, Vt = torch.linalg.svd(tokens_c, full_matrices=False)
    projected = tokens_c @ Vt[:3].T  # [N, 3]

    lo = projected.min(dim=0).values
    hi = projected.max(dim=0).values
    proj_norm = (projected - lo) / (hi - lo + 1e-8)  # [N, 3] in [0,1]

    return proj_norm.cpu().numpy(), h_p, w_p


# ---------------------------------------------------------------------------
# gradio.Server
# ---------------------------------------------------------------------------

app = Server(title="DINOv3 Tagger UI")

templates = Jinja2Templates(directory=Path(__file__).parent / "tagger_ui" / "templates")
templates.env.filters["format_number"] = lambda v: f"{v:,}"

# ---- Gradio API endpoints --------------------------------------------------


@app.api(name="get_tags")
def get_tags(image: str, max_size: int = 1024, floor: float = 0.05) -> str:
    """Tag an image. Returns JSON: {tags, categories, count}."""
    src = _resolve_image_source(image)
    img = _open_image(src)
    pv = _preprocess(img, max_size)
    logits = _gpu_infer(pv)
    return _postprocess(logits, floor)


@app.api(name="get_pca")
def get_pca(
    image: str,
    max_size: int = 1024,
    color1: str = "#ff0000",  # PC1 colour for custom view
    color2: str = "#00ff00",  # PC2 colour
    color3: str = "#0000ff",  # PC3 colour
) -> str:
    """Return JSON: {full: <base64 PNG>, custom: <base64 PNG>}.

    full   — PC1→R, PC2→G, PC3→B (fixed).
    custom — each PC channel tinted by the user-supplied hex colours,
             additively blended and normalised to [0,1].
    """
    src = _resolve_image_source(image)
    img = _open_image(src)
    pv = _preprocess(img, max_size)

    proj_norm, h_p, w_p = _gpu_pca_extract(pv)  # GPU: backbone + PCA

    # full rainbow (CPU, fast)
    rgb_full = proj_norm.reshape(h_p, w_p, 3)
    full_patch = Image.fromarray((rgb_full * 255).clip(0, 255).astype("uint8"), "RGB")
    full_img = full_patch.resize(
        (w_p * PATCH_SIZE, h_p * PATCH_SIZE), resample=Image.NEAREST
    )

    # custom colour blend (CPU, fast)
    custom_img = _build_custom_pca(proj_norm, h_p, w_p, color1, color2, color3)

    return json.dumps(
        {
            "full": _pil_to_base64(full_img),
            "custom": _pil_to_base64(custom_img),
        }
    )


@app.api(name="get_similarity")
def get_similarity(image_a: str, image_b: str, max_size: int = 1024) -> str:
    """Extract FEATURE_DIM=6400 descriptors for two images and return their
    cosine similarity.

    Returns JSON:
      {
        "score": float,          # cosine similarity in [-1, 1]
        "desc_a": [6400 floats], # L2-normalised descriptor for image A
        "desc_b": [6400 floats], # L2-normalised descriptor for image B
      }
    """
    src_a = _resolve_image_source(image_a)
    src_b = _resolve_image_source(image_b)

    img_a = _open_image(src_a)
    img_b = _open_image(src_b)

    pv_a = _preprocess(img_a, max_size)
    pv_b = _preprocess(img_b, max_size)

    # Run both through the backbone in separate GPU calls
    # (spaces.GPU does not support batching across different-sized tensors)
    feat_a = _gpu_extract_descriptor(pv_a)  # [6400]
    feat_b = _gpu_extract_descriptor(pv_b)  # [6400]

    # L2-normalise
    feat_a = feat_a / (np.linalg.norm(feat_a) + 1e-8)
    feat_b = feat_b / (np.linalg.norm(feat_b) + 1e-8)

    score = float(np.dot(feat_a, feat_b))

    return json.dumps(
        {
            "score": round(score, 6),
            "desc_a": feat_a.tolist(),
            "desc_b": feat_b.tolist(),
        }
    )


# ---- FastAPI routes --------------------------------------------------------


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(
        request,
        "index.html",
        {
            "num_tags": model.num_tags,
            "vocab_path": _VOCAB_PATH,
            "category_meta": CATEGORY_META,
        },
    )


app.launch(ssr_mode=False)
