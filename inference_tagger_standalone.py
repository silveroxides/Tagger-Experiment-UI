"""DINOv3 ViT-H/16+ Tagger — Fully Standalone Inference Script

Zero dependency on transformers, trainer code, or any internal module.
Only requires: torch, torchvision, safetensors, Pillow, requests.

  pip install torch torchvision safetensors Pillow requests

The DINOv3 ViT-H/16+ architecture is implemented directly here, with weights
loaded from a .safetensors checkpoint.  The state-dict key names match the
HuggingFace transformers layout exactly so checkpoints are interchangeable.

Usage
-----
# Single image, top-30 tags:
python inference_tagger_standalone.py \
    --checkpoint tagger_checkpoints/2026-03-28_22-57-47.safetensors \
    --vocab     tagger_vocab.json \
    --images    photo.jpg \
    --topk      30

# URL input:
python inference_tagger_standalone.py \
    --checkpoint tagger_checkpoints/2026-03-28_22-57-47.safetensors \
    --vocab     tagger_vocab.json \
    --images    https://example.com/photo.jpg

# Threshold instead of top-k:
python inference_tagger_standalone.py ... --threshold 0.4

# Pipe-friendly comma-separated tags (one line per image):
python inference_tagger_standalone.py ... --format tags

# JSON output:
python inference_tagger_standalone.py ... --format json

Output formats (--format)
-------------------------
  pretty  (default) — human-readable table with scores
  tags              — comma-separated tag string, one line per image
  json              — JSON array of {file, tags: [{tag, score}]} objects
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from functools import lru_cache
from io import BytesIO
from pathlib import Path

import requests
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.v2 as v2
from PIL import Image
from safetensors.torch import load_file


# =============================================================================
# DINOv3 ViT-H/16+ — hardcoded architecture
# All hyperparameters match facebook/dinov3-vith16plus-pretrain-lvd1689m
# =============================================================================

D_MODEL = 1280
N_HEADS = 20
HEAD_DIM = D_MODEL // N_HEADS  # 64
N_LAYERS = 32
D_FFN = 5120
N_REGISTERS = 4
PATCH_SIZE = 16
ROPE_THETA = 100.0
ROPE_RESCALE = 2.0
LN_EPS = 1e-5
LAYERSCALE = 1.0

FEATURE_DIM = (1 + N_REGISTERS) * D_MODEL  # 6400


# ---------------------------------------------------------------------------
# RoPE helpers
# ---------------------------------------------------------------------------


@lru_cache(maxsize=32)
def _patch_coords_cached(h: int, w: int, device_str: str) -> torch.Tensor:
    device = torch.device(device_str)
    cy = torch.arange(0.5, h, dtype=torch.float32, device=device) / h
    cx = torch.arange(0.5, w, dtype=torch.float32, device=device) / w
    coords = torch.stack(torch.meshgrid(cy, cx, indexing="ij"), dim=-1).flatten(0, 1)
    coords = 2.0 * coords - 1.0
    coords = coords * ROPE_RESCALE
    return coords  # [h*w, 2]


def _build_rope(
    h_patches: int, w_patches: int, dtype: torch.dtype, device: torch.device
):
    coords = _patch_coords_cached(h_patches, w_patches, str(device))
    inv_freq = 1.0 / (
        ROPE_THETA
        ** torch.arange(0, 1, 4 / HEAD_DIM, dtype=torch.float32, device=device)
    )
    angles = 2 * math.pi * coords[:, :, None] * inv_freq[None, None, :]
    angles = angles.flatten(1, 2).tile(2)
    cos = torch.cos(angles).to(dtype).unsqueeze(0).unsqueeze(0)
    sin = torch.sin(angles).to(dtype).unsqueeze(0).unsqueeze(0)
    return cos, sin


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    h = x.shape[-1] // 2
    return torch.cat((-x[..., h:], x[..., :h]), dim=-1)


def _apply_rope(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    n_pre = 1 + N_REGISTERS
    q_pre, q_pat = q[..., :n_pre, :], q[..., n_pre:, :]
    k_pre, k_pat = k[..., :n_pre, :], k[..., n_pre:, :]
    q_pat = q_pat * cos + _rotate_half(q_pat) * sin
    k_pat = k_pat * cos + _rotate_half(k_pat) * sin
    return torch.cat([q_pre, q_pat], dim=-2), torch.cat([k_pre, k_pat], dim=-2)


# ---------------------------------------------------------------------------
# Transformer blocks
# ---------------------------------------------------------------------------


class _Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(D_MODEL, D_MODEL, bias=True)
        self.k_proj = nn.Linear(D_MODEL, D_MODEL, bias=False)
        self.v_proj = nn.Linear(D_MODEL, D_MODEL, bias=True)
        self.o_proj = nn.Linear(D_MODEL, D_MODEL, bias=True)

    def forward(self, x, cos, sin):
        B, S, _ = x.shape
        q = self.q_proj(x).view(B, S, N_HEADS, HEAD_DIM).transpose(1, 2)
        k = self.k_proj(x).view(B, S, N_HEADS, HEAD_DIM).transpose(1, 2)
        v = self.v_proj(x).view(B, S, N_HEADS, HEAD_DIM).transpose(1, 2)
        q, k = _apply_rope(q, k, cos, sin)
        out = F.scaled_dot_product_attention(q, k, v, scale=HEAD_DIM**-0.5)
        return self.o_proj(out.transpose(1, 2).reshape(B, S, D_MODEL))


class _GatedMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_proj = nn.Linear(D_MODEL, D_FFN, bias=True)
        self.up_proj = nn.Linear(D_MODEL, D_FFN, bias=True)
        self.down_proj = nn.Linear(D_FFN, D_MODEL, bias=True)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class _Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm1 = nn.LayerNorm(D_MODEL, eps=LN_EPS)
        self.attention = _Attention()
        self.layer_scale1 = nn.Parameter(torch.full((D_MODEL,), LAYERSCALE))
        self.norm2 = nn.LayerNorm(D_MODEL, eps=LN_EPS)
        self.mlp = _GatedMLP()
        self.layer_scale2 = nn.Parameter(torch.full((D_MODEL,), LAYERSCALE))

    def forward(self, x, cos, sin):
        x = x + self.attention(self.norm1(x), cos, sin) * self.layer_scale1
        x = x + self.mlp(self.norm2(x)) * self.layer_scale2
        return x


class _Embeddings(nn.Module):
    def __init__(self):
        super().__init__()
        # zeros() rather than empty() so a forgotten checkpoint key fails
        # predictably instead of producing undefined outputs.
        self.cls_token = nn.Parameter(torch.zeros(1, 1, D_MODEL))
        self.mask_token = nn.Parameter(torch.zeros(1, 1, D_MODEL))
        self.register_tokens = nn.Parameter(torch.zeros(1, N_REGISTERS, D_MODEL))
        self.patch_embeddings = nn.Conv2d(
            3, D_MODEL, kernel_size=PATCH_SIZE, stride=PATCH_SIZE
        )

    def forward(self, pixel_values):
        B = pixel_values.shape[0]
        dtype = self.patch_embeddings.weight.dtype
        patches = (
            self.patch_embeddings(pixel_values.to(dtype)).flatten(2).transpose(1, 2)
        )
        cls = self.cls_token.expand(B, -1, -1)
        regs = self.register_tokens.expand(B, -1, -1)
        return torch.cat([cls, regs, patches], dim=1)


class DINOv3ViTH(nn.Module):
    """DINOv3 ViT-H/16+ backbone.

    Token layout: [CLS, reg_0..reg_3, patch_0..patch_N].
    Returns last_hidden_state [B, 1+R+P, D_MODEL].
    """

    def __init__(self):
        super().__init__()
        self.embeddings = _Embeddings()
        self.layer = nn.ModuleList([_Block() for _ in range(N_LAYERS)])
        self.norm = nn.LayerNorm(D_MODEL, eps=LN_EPS)

    def forward(self, pixel_values):
        _, _, H, W = pixel_values.shape
        x = self.embeddings(pixel_values)
        h_p, w_p = H // PATCH_SIZE, W // PATCH_SIZE
        cos, sin = _build_rope(h_p, w_p, x.dtype, pixel_values.device)
        for block in self.layer:
            x = block(x, cos, sin)
        return self.norm(x)

    def get_image_tokens(self, pixel_values):
        """Return patch tokens only (no CLS/registers) as [B, h_p*w_p, D_MODEL]
        and the spatial grid dimensions (h_p, w_p)."""
        _, _, H, W = pixel_values.shape
        h_p, w_p = H // PATCH_SIZE, W // PATCH_SIZE
        x = self.embeddings(pixel_values)
        cos, sin = _build_rope(h_p, w_p, x.dtype, pixel_values.device)
        for block in self.layer:
            x = block(x, cos, sin)
        x = self.norm(x)
        # token layout: [CLS, reg_0..reg_R-1, patch_0..patch_N]
        patch_tokens = x[:, 1 + N_REGISTERS :, :]  # [B, h_p*w_p, D_MODEL]
        return patch_tokens, h_p, w_p


# =============================================================================
# Head — auto-detected from the checkpoint
# =============================================================================


class _LowRankHead(nn.Module):
    """Two-matrix low-rank projection head.

    features (in_dim)
      → Linear(in_dim, rank, bias=?)
      → Linear(rank, num_tags, bias=?)
    """

    def __init__(
        self, in_dim: int, rank: int, num_tags: int, down_bias: bool, up_bias: bool
    ):
        super().__init__()
        self.proj_down = nn.Linear(in_dim, rank, bias=down_bias)
        self.proj_up = nn.Linear(rank, num_tags, bias=up_bias)

    def forward(self, x):
        return self.proj_up(self.proj_down(x))


def _build_head_from_checkpoint(
    head_sd: dict,
    in_dim: int,
    num_tags: int,
) -> tuple[nn.Module, dict]:
    """Inspect head_sd and build a matching Module.

    Supports two layouts, in order of preference:
      1. Single linear          — any ``*.weight`` with shape [num_tags, in_dim]
      2. Low-rank pair (2 mats) — one ``*.weight`` [rank, in_dim] plus
                                   one ``*.weight`` [num_tags, rank]

    Returns (module, remapped_state_dict) where the remapped state dict
    matches the module's own key names so strict loading works.
    """
    weights_2d = [
        (k, v) for k, v in head_sd.items() if k.endswith(".weight") and v.ndim == 2
    ]

    # --- Case 1: single dense linear ---------------------------------------
    singles = [(k, v) for k, v in weights_2d if tuple(v.shape) == (num_tags, in_dim)]
    if len(weights_2d) <= 2 and len(singles) == 1:
        wkey, wval = singles[0]
        base = wkey[: -len(".weight")]
        bias_key = base + ".bias"
        has_bias = bias_key in head_sd
        module = nn.Linear(in_dim, num_tags, bias=has_bias)
        remapped = {"weight": wval}
        if has_bias:
            remapped["bias"] = head_sd[bias_key]
        # Sanity check: no extra keys we don't understand
        expected_src = {wkey} | ({bias_key} if has_bias else set())
        extra = set(head_sd) - expected_src
        if extra:
            raise RuntimeError(
                f"Head has single-linear shape but extra unknown keys: {sorted(extra)}"
            )
        return module, remapped

    # --- Case 2: low-rank pair ---------------------------------------------
    down = None  # (key, tensor) with shape [rank, in_dim]
    up = None  # (key, tensor) with shape [num_tags, rank]
    for k, v in weights_2d:
        if v.shape[1] == in_dim and v.shape[0] != num_tags:
            down = (k, v)
        elif v.shape[0] == num_tags and v.shape[1] != in_dim:
            up = (k, v)

    if down is not None and up is not None:
        rank_down = down[1].shape[0]
        rank_up = up[1].shape[1]
        if rank_down != rank_up:
            raise RuntimeError(
                f"Low-rank head: inner dims disagree "
                f"(down out={rank_down}, up in={rank_up})"
            )

        down_key, down_w = down
        up_key, up_w = up
        down_base = down_key[: -len(".weight")]
        up_base = up_key[: -len(".weight")]
        down_bias_key = down_base + ".bias"
        up_bias_key = up_base + ".bias"
        has_down_bias = down_bias_key in head_sd
        has_up_bias = up_bias_key in head_sd

        module = _LowRankHead(
            in_dim=in_dim,
            rank=rank_down,
            num_tags=num_tags,
            down_bias=has_down_bias,
            up_bias=has_up_bias,
        )
        remapped = {
            "proj_down.weight": down_w,
            "proj_up.weight": up_w,
        }
        if has_down_bias:
            remapped["proj_down.bias"] = head_sd[down_bias_key]
        if has_up_bias:
            remapped["proj_up.bias"] = head_sd[up_bias_key]

        # Sanity check
        expected_src = {down_key, up_key}
        if has_down_bias:
            expected_src.add(down_bias_key)
        if has_up_bias:
            expected_src.add(up_bias_key)
        extra = set(head_sd) - expected_src
        if extra:
            raise RuntimeError(
                f"Low-rank head detected but checkpoint has extra unknown "
                f"head keys: {sorted(extra)}"
            )

        print(
            f"[Tagger] Detected low-rank head: "
            f"in_dim={in_dim}, rank={rank_down}, num_tags={num_tags} "
            f"(down_bias={has_down_bias}, up_bias={has_up_bias})"
        )
        return module, remapped

    raise RuntimeError(
        "Could not infer head architecture from checkpoint. "
        f"Non-backbone keys found: {sorted(head_sd.keys())}"
    )


# =============================================================================
# Tagger wrapper module
# =============================================================================


class DINOv3Tagger(nn.Module):
    """Backbone + head. The head is attached after the checkpoint is
    inspected (so we can build the right shape)."""

    def __init__(self):
        super().__init__()
        self.backbone = DINOv3ViTH()
        self.head: nn.Module | None = None  # attached by Tagger

    def forward(self, pixel_values):
        hidden = self.backbone(pixel_values)
        cls = hidden[:, 0, :]
        regs = hidden[:, 1 : 1 + N_REGISTERS, :].flatten(1)
        features = torch.cat([cls, regs], dim=-1).float()  # fp32 for head
        return self.head(features)

    def forward_embedding(self, pixel_values):
        """Return the FEATURE_DIM=6400 image descriptor without applying the head.
        Same as forward() but stops before self.head — use this for similarity queries.
        """
        hidden = self.backbone(pixel_values)
        cls = hidden[:, 0, :]
        regs = hidden[:, 1 : 1 + N_REGISTERS, :].flatten(1)
        features = torch.cat([cls, regs], dim=-1).float()  # fp32 for head
        return features


# =============================================================================
# Checkpoint loading helpers
# =============================================================================


def _split_and_clean_state_dict(sd: dict) -> tuple[dict, dict]:
    """Split full state dict into (backbone_sd, head_sd), stripping the
    ``backbone.`` prefix and applying the remaps needed to match
    ``DINOv3ViTH``'s parameter layout:

      1. ``backbone.model.layer.N.*`` → ``layer.N.*``
         (the checkpoint has an HF-style intermediate ``model`` wrapper
         that our flat backbone class does not)
      2. ``...layer_scale{1,2}.lambda1`` → ``...layer_scale{1,2}``
         (HF stores layer_scale as a sub-module with a ``lambda1``
         parameter; we use a plain ``nn.Parameter``)
      3. Drop any ``rope_embeddings`` buffers (recomputed on the fly)
    """
    backbone_sd: dict = {}
    head_sd: dict = {}
    for k, v in sd.items():
        if k.startswith("backbone."):
            nk = k[len("backbone.") :]
            # Remap (1): strip intermediate "model." before "layer."
            if nk.startswith("model.layer."):
                nk = nk[len("model.") :]
            backbone_sd[nk] = v
        else:
            head_sd[k] = v

    # Remap (2): layer.N.layer_scale{1,2}.lambda1 → layer.N.layer_scale{1,2}
    for k in list(backbone_sd.keys()):
        if ".layer_scale" in k and k.endswith(".lambda1"):
            backbone_sd[k[: -len(".lambda1")]] = backbone_sd.pop(k)

    # Remap (3): drop rope buffers (recomputed on the fly)
    for k in list(backbone_sd.keys()):
        if "rope_embeddings" in k:
            backbone_sd.pop(k)

    return backbone_sd, head_sd


# =============================================================================
# Image preprocessing
# =============================================================================

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]


def _snap(x: int, m: int) -> int:
    return max(m, (x // m) * m)


def _open_image(source) -> Image.Image:
    s = str(source)
    if s.startswith("http://") or s.startswith("https://"):
        r = requests.get(s, timeout=30)
        r.raise_for_status()
        return Image.open(BytesIO(r.content)).convert("RGB")
    return Image.open(source).convert("RGB")


def preprocess_image(source, max_size: int = 1024) -> torch.Tensor:
    """Load and preprocess an image → [1, 3, H, W] float32, ImageNet-normalised.

    Aspect ratio is preserved: a single scale factor is chosen so that the
    long edge fits inside max_size after snapping to a PATCH_SIZE multiple.
    """
    img = _open_image(source)
    w, h = img.size

    # Target long-edge (snapped to patch multiple).
    long_edge = max(w, h)
    target_long = _snap(min(long_edge, max_size), PATCH_SIZE)
    scale = target_long / long_edge

    new_w = _snap(max(PATCH_SIZE, round(w * scale)), PATCH_SIZE)
    new_h = _snap(max(PATCH_SIZE, round(h * scale)), PATCH_SIZE)

    return v2.Compose(
        [
            v2.Resize((new_h, new_w), interpolation=v2.InterpolationMode.LANCZOS),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
        ]
    )(img).unsqueeze(0)


# =============================================================================
# Tagger wrapper
# =============================================================================


class Tagger:
    """Inference wrapper for DINOv3Tagger (ViT-H/16+).

    Parameters
    ----------
    checkpoint_path : str
        Path to a .safetensors or .pt/.pth checkpoint.
    vocab_path : str
        Path to tagger_vocab.json or tagger_vocab_with_categories.json
        (either must contain an ``idx2tag`` list).
    device : str
        "cuda", "cuda:0", "cpu", ...
    dtype : torch.dtype
        Backbone precision. bfloat16 recommended on Ampere+, float16 for
        older GPUs, float32 for CPU. The head always runs in fp32.
    max_size : int
        Long-edge cap in pixels before feeding to the model.
    """

    def __init__(
        self,
        checkpoint_path: str,
        vocab_path: str,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        max_size: int = 1024,
    ):
        want_cuda = device.startswith("cuda")
        if want_cuda and not torch.cuda.is_available():
            print("[Tagger] CUDA not available, falling back to CPU")
            device = "cpu"
            dtype = torch.float32
        self.device = torch.device(device)
        self.dtype = dtype
        self.max_size = max_size

        with open(vocab_path) as f:
            data = json.load(f)
        self.idx2tag: list[str] = data["idx2tag"]
        self.num_tags = len(self.idx2tag)
        print(f"[Tagger] Vocabulary: {self.num_tags:,} tags")

        # --- Load checkpoint to CPU first so we can inspect shapes ---------
        print(f"[Tagger] Loading checkpoint: {checkpoint_path}")
        if checkpoint_path.endswith((".safetensors", ".sft")):
            sd = load_file(checkpoint_path, device="cpu")
        else:
            sd = torch.load(checkpoint_path, map_location="cpu")

        backbone_sd, head_sd = _split_and_clean_state_dict(sd)

        if not head_sd:
            raise RuntimeError(
                "Checkpoint contains no non-backbone keys — cannot build head."
            )

        # --- Build model, inferring head shape from the checkpoint --------
        self.model = DINOv3Tagger()
        head_module, head_sd_remapped = _build_head_from_checkpoint(
            head_sd,
            in_dim=FEATURE_DIM,
            num_tags=self.num_tags,
        )
        self.model.head = head_module

        # --- Strict load — mismatches raise instead of silently passing ----
        self.model.backbone.load_state_dict(backbone_sd, strict=True)
        self.model.head.load_state_dict(head_sd_remapped, strict=True)

        # --- Move to device. Backbone → bf16/fp16; head stays fp32. --------
        self.model.backbone = self.model.backbone.to(device=self.device, dtype=dtype)
        self.model.head = self.model.head.to(device=self.device, dtype=torch.float32)
        self.model.eval()
        print(f"[Tagger] Ready on {self.device} (backbone={dtype}, head=fp32)")

    @torch.no_grad()
    def embed_pca(
        self,
        image,
        n_components: int = 3,
        max_size: int | None = None,
    ) -> "Image.Image":
        """Run PCA on the patch-token features of *image* and return a
        false-colour RGB PIL image where R/G/B channels correspond to the
        first three principal components, each normalised to [0, 255].

        Parameters
        ----------
        image :
            Local path, URL, or PIL.Image.Image.
        n_components :
            Number of PCA components (must be 3 for RGB output).
        max_size :
            Long-edge cap in pixels (defaults to ``self.max_size``).
        """
        if n_components != 3:
            raise ValueError("n_components must be 3 for false-colour RGB output")
        if max_size is None:
            max_size = self.max_size

        if isinstance(image, Image.Image):
            img = image.convert("RGB")
            w, h = img.size
            scale = min(1.0, max_size / max(w, h))
            new_w = _snap(round(w * scale), PATCH_SIZE)
            new_h = _snap(round(h * scale), PATCH_SIZE)
            pv = (
                v2.Compose(
                    [
                        v2.Resize(
                            (new_h, new_w), interpolation=v2.InterpolationMode.LANCZOS
                        ),
                        v2.ToImage(),
                        v2.ToDtype(torch.float32, scale=True),
                        v2.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
                    ]
                )(img)
                .unsqueeze(0)
                .to(self.device)
            )
        else:
            pv = preprocess_image(image, max_size=max_size).to(self.device)

        with torch.autocast(device_type=self.device.type, dtype=self.dtype):
            patch_tokens, h_p, w_p = self.model.backbone.get_image_tokens(pv)

        # patch_tokens: [1, h_p*w_p, D_MODEL] → [N, D]
        tokens = patch_tokens[0].float()  # fp32 for PCA

        # Centre
        mean = tokens.mean(dim=0, keepdim=True)
        tokens_c = tokens - mean

        # PCA via SVD (economy)
        _, _, Vt = torch.linalg.svd(tokens_c, full_matrices=False)
        components = Vt[:n_components]  # [3, D]
        projected = tokens_c @ components.T  # [N, 3]

        # Normalise each component to [0, 1]
        lo = projected.min(dim=0).values
        hi = projected.max(dim=0).values
        projected = (projected - lo) / (hi - lo + 1e-8)

        # Reshape to spatial grid and convert to uint8 PIL image
        rgb = projected.reshape(h_p, w_p, 3).cpu().numpy()
        rgb_uint8 = (rgb * 255).clip(0, 255).astype("uint8")
        return Image.fromarray(rgb_uint8, mode="RGB")

    @torch.no_grad()
    def predict(
        self, image, topk: int | None = 30, threshold: float | None = None
    ) -> list[tuple[str, float]]:
        """Tag a single image (local path or URL)."""
        if topk is None and threshold is None:
            topk = 30

        pv = preprocess_image(image, max_size=self.max_size).to(self.device)
        logits = self.model(pv)[0]
        scores = torch.sigmoid(logits.float())

        if topk is not None:
            values, indices = scores.topk(min(topk, self.num_tags))
        else:
            assert threshold is not None
            indices = (scores >= threshold).nonzero(as_tuple=True)[0]
            values = scores[indices]
            order = values.argsort(descending=True)
            indices, values = indices[order], values[order]

        return [
            (self.idx2tag[i], float(v))
            for i, v in zip(indices.tolist(), values.tolist())
        ]

    @torch.no_grad()
    def predict_batch(
        self, images, topk: int | None = 30, threshold: float | None = None
    ):
        return [self.predict(img, topk=topk, threshold=threshold) for img in images]


# =============================================================================
# Output formatters
# =============================================================================


def _fmt_pretty(path: str, results) -> str:
    lines = [f"\n{'─' * 60}", f" {path}", f"{'─' * 60}"]
    for rank, (tag, score) in enumerate(results, 1):
        bar = "█" * int(score * 20)
        lines.append(f" {rank:>3}. {score:.3f} {bar:<20} {tag}")
    return "\n".join(lines)


def _fmt_tags(results) -> str:
    return ", ".join(tag for tag, _ in results)


def _fmt_json(path: str, results) -> dict:
    return {
        "file": path,
        "tags": [{"tag": t, "score": round(s, 4)} for t, s in results],
    }


# =============================================================================
# CLI
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="DINOv3 ViT-H/16+ tagger inference (standalone)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint", required=True, help="Path to .safetensors or .pt checkpoint"
    )
    parser.add_argument("--vocab", required=True, help="Path to tagger_vocab*.json")
    parser.add_argument(
        "--images", nargs="+", required=True, help="Image paths and/or http(s) URLs"
    )
    parser.add_argument(
        "--device", default="cuda", help="Device: cuda, cuda:0, cpu (default: cuda)"
    )
    parser.add_argument(
        "--max-size",
        type=int,
        default=1024,
        help="Long-edge cap in pixels (default: 1024)",
    )

    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--topk", type=int, default=30, help="Return top-k tags (default: 30)"
    )
    mode.add_argument(
        "--threshold", type=float, help="Return all tags with score >= threshold"
    )

    parser.add_argument(
        "--format",
        choices=["pretty", "tags", "json"],
        default="pretty",
        help="Output format (default: pretty)",
    )
    args = parser.parse_args()

    tagger = Tagger(
        checkpoint_path=args.checkpoint,
        vocab_path=args.vocab,
        device=args.device,
        max_size=args.max_size,
    )

    topk, threshold = (None, args.threshold) if args.threshold else (args.topk, None)
    json_out = []

    for src in args.images:
        is_url = str(src).startswith("http://") or str(src).startswith("https://")
        if not is_url and not Path(src).exists():
            print(f"[warning] File not found: {src}", file=sys.stderr)
            continue
        results = tagger.predict(src, topk=topk, threshold=threshold)
        if args.format == "pretty":
            print(_fmt_pretty(src, results))
        elif args.format == "tags":
            print(_fmt_tags(results))
        elif args.format == "json":
            json_out.append(_fmt_json(src, results))

    if args.format == "json":
        print(json.dumps(json_out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
