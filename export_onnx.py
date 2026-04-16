"""DINOv3 ViT-H/16+ Tagger — ONNX Export Script

Exports three ONNX models from a tagger_proto.safetensors checkpoint:

  onnx/tagger/    — pixel_values → logits [B, 74625]       (tagging)
  onnx/embedding/ — pixel_values → embedding [B, 6400]     (similarity)
  onnx/backbone/  — pixel_values → last_hidden_state [B, S, 1280]  (PCA / features)

Each export directory contains:
  model.onnx / model.onnx_data          fp32 baseline
  model_quantized.onnx / _data          int8 dynamic quantization
  model_q4.onnx / _data                 q4  dynamic quantization

Requirements
------------
  torch>=2.6          (dynamo exporter; default as of 2.9)
  onnx>=1.17
  onnxruntime>=1.22
  huggingface_hub     (auto-download checkpoint)
  safetensors
  pillow

Usage
-----
# Auto-download checkpoint and export everything:
python export_onnx.py

# Use an existing local checkpoint:
python export_onnx.py --checkpoint tagger_proto.safetensors

# Export only tagger + embedding, skip quantization:
python export_onnx.py --no-backbone --no-quantize

# CPU-only export:
python export_onnx.py --device cpu
"""

from __future__ import annotations

import argparse
import sys
import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Dependency version checks — fail early with actionable messages
# ---------------------------------------------------------------------------


def _check_deps() -> None:
    errors = []

    try:
        import torch

        major, minor = (int(x) for x in torch.__version__.split(".")[:2])
        if (major, minor) < (2, 6):
            errors.append(
                f"torch >= 2.6 required (dynamo exporter). Found {torch.__version__}."
            )
    except ImportError:
        errors.append("torch not installed.")

    try:
        import onnx
        from packaging.version import Version

        if Version(onnx.__version__) < Version("1.17"):
            errors.append(f"onnx >= 1.17 required. Found {onnx.__version__}.")
    except ImportError:
        errors.append("onnx not installed. Run: pip install onnx>=1.17")

    try:
        import onnxruntime
        from packaging.version import Version

        if Version(onnxruntime.__version__) < Version("1.22"):
            errors.append(
                f"onnxruntime >= 1.22 required. Found {onnxruntime.__version__}."
            )
    except ImportError:
        errors.append("onnxruntime not installed. Run: pip install onnxruntime>=1.22")

    try:
        import huggingface_hub  # noqa: F401
    except ImportError:
        errors.append("huggingface_hub not installed. Run: pip install huggingface_hub")

    if errors:
        print("[export] Dependency errors:", file=sys.stderr)
        for e in errors:
            print(f"  • {e}", file=sys.stderr)
        sys.exit(1)


_check_deps()

# ---------------------------------------------------------------------------
# Imports (after dep check)
# ---------------------------------------------------------------------------

import json
import math
from typing import Optional

import torch
import torch.nn as nn
import onnxruntime
from packaging.version import Version
from huggingface_hub import hf_hub_download

# Reuse all model code from the existing standalone inference module.
# We import the building-block classes and helpers directly — no re-implementation.
sys.path.insert(0, str(Path(__file__).parent))
from inference_tagger_standalone import (
    DINOv3ViTH,
    DINOv3Tagger,
    _split_and_clean_state_dict,
    _build_head_from_checkpoint,
    FEATURE_DIM,
    N_REGISTERS,
    PATCH_SIZE,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

HF_REPO_ID = "lodestones/tagger-experiment"
HF_FILENAME = "tagger_proto.safetensors"
DEFAULT_VOCAB = "tagger_vocab.json"

# Dummy input resolution for export trace (training resolution, multiple of 16)
DEFAULT_DUMMY_H = 512
DEFAULT_DUMMY_W = 512

# ---------------------------------------------------------------------------
# Wrapper modules
# ---------------------------------------------------------------------------
# Three thin wrappers so each ONNX graph has a clean single-input / single-output
# interface. All wrappers accept float32 pixel_values [B, 3, H, W].


def _transpose_linear_head(model: DINOv3Tagger) -> DINOv3Tagger:
    """Replace the nn.Linear head(s) with a MatMul-friendly equivalent.

    torch.onnx dynamo exports nn.Linear as Gemm(A, W, transB=1).
    onnxruntime's quantize_dynamic calls replace_gemm_with_matmul() internally
    which converts Gemm → MatMul but drops transB, leaving W in shape
    [out, in] for a MatMul that expects [in, out] — causing a shape inference
    failure during quantization.

    Fix: store the weight pre-transposed as [in, out] and use an explicit
    matmul so dynamo emits MatMul(x, W_T) directly — no Gemm, no transB,
    no conflict.
    """

    class _MatMulLinear(nn.Module):
        """Drop-in for nn.Linear that emits MatMul instead of Gemm."""

        def __init__(self, linear: nn.Linear):
            super().__init__()
            # Store weight transposed: shape [in_features, out_features]
            self.weight_t = nn.Parameter(linear.weight.t().contiguous())
            self.bias = (
                nn.Parameter(linear.bias.clone()) if linear.bias is not None else None
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            out = x @ self.weight_t
            if self.bias is not None:
                out = out + self.bias
            return out

    # Handle both dense head (nn.Linear) and low-rank head (_LowRankHead).
    head = model.head
    if isinstance(head, nn.Linear):
        model.head = _MatMulLinear(head)
    else:
        # Low-rank head: replace each sub-linear
        from inference_tagger_standalone import _LowRankHead

        if isinstance(head, _LowRankHead):
            head.proj_down = _MatMulLinear(head.proj_down)
            head.proj_up = _MatMulLinear(head.proj_up)

    return model


class TaggerWrapper(nn.Module):
    """pixel_values → raw logits [B, num_tags] (pre-sigmoid)."""

    def __init__(self, model: DINOv3Tagger):
        super().__init__()
        self.model = model

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self.model(pixel_values)


class EmbeddingWrapper(nn.Module):
    """pixel_values → image descriptor [B, FEATURE_DIM=6400] (CLS + 4 registers)."""

    def __init__(self, model: DINOv3Tagger):
        super().__init__()
        self.model = model

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self.model.forward_embedding(pixel_values)


class BackboneWrapper(nn.Module):
    """pixel_values → last_hidden_state [B, 1+N_REG+N_PATCHES, D_MODEL=1280].

    Patch tokens start at index 1 + N_REGISTERS (= 5). Use [:,5:,:] for PCA.
    """

    def __init__(self, backbone: DINOv3ViTH):
        super().__init__()
        self.backbone = backbone

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self.backbone(pixel_values)


# ---------------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------------


def _auto_download_checkpoint(cache_dir: Optional[Path]) -> Path:
    """Download tagger_proto.safetensors from HF if not already cached."""
    kwargs = {}
    if cache_dir is not None:
        kwargs["cache_dir"] = str(cache_dir)
    print(f"[export] Downloading {HF_FILENAME} from {HF_REPO_ID} …")
    local_path = hf_hub_download(
        repo_id=HF_REPO_ID,
        filename=HF_FILENAME,
        **kwargs,
    )
    print(f"[export] Checkpoint at: {local_path}")
    return Path(local_path)


def _load_checkpoint(
    checkpoint_path: Path, device: torch.device
) -> tuple[nn.Module, int]:
    """Load checkpoint → (DINOv3Tagger in fp32 eval mode, num_tags)."""
    from safetensors.torch import load_file as safetensors_load

    print(f"[export] Loading checkpoint: {checkpoint_path}")
    suffix = checkpoint_path.suffix.lower()
    if suffix in (".safetensors", ".sft"):
        sd = safetensors_load(str(checkpoint_path), device="cpu")
    elif suffix in (".pt", ".pth"):
        sd = torch.load(str(checkpoint_path), map_location="cpu")
    else:
        # Try safetensors first, fall back to torch.load
        try:
            sd = safetensors_load(str(checkpoint_path), device="cpu")
        except Exception:
            sd = torch.load(str(checkpoint_path), map_location="cpu")

    backbone_sd, head_sd = _split_and_clean_state_dict(sd)

    if not head_sd:
        raise RuntimeError("Checkpoint has no non-backbone keys — cannot infer head.")

    # Infer num_tags from head weights
    # Look for the weight tensor whose first dim is num_tags
    candidate_num_tags = None
    for k, v in head_sd.items():
        if k.endswith(".weight") and v.ndim == 2:
            # Dense head: shape [num_tags, FEATURE_DIM]
            if v.shape[1] == FEATURE_DIM:
                candidate_num_tags = v.shape[0]
                break
            # Low-rank head up-projection: shape [num_tags, rank]
            # (rank << FEATURE_DIM, so pick the larger first-dim)
            if candidate_num_tags is None or v.shape[0] > candidate_num_tags:
                candidate_num_tags = v.shape[0]

    if candidate_num_tags is None:
        raise RuntimeError("Could not infer num_tags from checkpoint head.")

    num_tags = candidate_num_tags
    print(f"[export] Detected num_tags = {num_tags:,}")

    model = DINOv3Tagger()
    head_module, head_sd_remapped = _build_head_from_checkpoint(
        head_sd, in_dim=FEATURE_DIM, num_tags=num_tags
    )
    model.head = head_module

    model.backbone.load_state_dict(backbone_sd, strict=True)
    model.head.load_state_dict(head_sd_remapped, strict=True)

    # Cast everything to fp32 — bfloat16 is not in the ONNX standard opset.
    # The backbone was trained in bf16 but weights are losslessly representable
    # in fp32 (bf16 is a strict subset of fp32 mantissa range).
    model = model.to(dtype=torch.float32, device=device)

    # Replace nn.Linear head(s) with MatMul-based equivalents so the dynamo
    # exporter emits MatMul(x, W_T) instead of Gemm(x, W, transB=1).
    # onnxruntime quantize_dynamic calls replace_gemm_with_matmul() which
    # drops transB and corrupts shape inference on any Gemm node.
    model = _transpose_linear_head(model)

    model.eval()

    print(f"[export] Model ready on {device} (fp32)")
    return model, num_tags


# ---------------------------------------------------------------------------
# ONNX export helpers
# ---------------------------------------------------------------------------


def _make_dynamic_shapes(dummy_h: int, dummy_w: int) -> dict:
    """Build dynamic_shapes dict for torch.onnx.export (dynamo=True).

    Batch, height, and width are all dynamic. Height and width must be
    multiples of PATCH_SIZE=16 — callers are responsible for valid inputs.

    No upper bound on batch: setting max= causes dynamo to emit an
    over-constrained guard that fires a ConstraintViolationError when the
    symbolic batch dim is internally bounded tighter by ops like expand().
    Omitting max leaves it unbounded, which is correct for ONNX deployment.
    """
    batch = torch.export.Dim("batch", min=1)
    height = torch.export.Dim("height", min=PATCH_SIZE, max=4096)
    width = torch.export.Dim("width", min=PATCH_SIZE, max=4096)
    return {"pixel_values": {0: batch, 2: height, 3: width}}


def _export_one(
    wrapper: nn.Module,
    dummy_input: torch.Tensor,
    output_dir: Path,
    input_names: list[str],
    output_names: list[str],
    dummy_h: int,
    dummy_w: int,
    label: str,
) -> Path:
    """Run torch.onnx.export with dynamo=True and save to output_dir/model.onnx."""
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "model.onnx"

    dynamic_shapes = _make_dynamic_shapes(dummy_h, dummy_w)

    print(f"\n[export] Exporting {label} → {out_path}")
    print(f"         dummy input shape: {list(dummy_input.shape)}")

    # torch.onnx.export with dynamo=True returns an ONNXProgram.
    # We leave f=None and call .save() ourselves for explicit control.
    #
    # Suppress the lru_cache UserWarning from dynamo: the _patch_coords_cached
    # function in inference_tagger_standalone.py is decorated with @lru_cache.
    # Dynamo correctly traces through it (ignoring the cache wrapper) and the
    # result is numerically identical — the warning is purely informational.
    # The device string argument is a compile-time constant so baking it in
    # is correct. We filter only this specific message to avoid hiding
    # genuine warnings.
    import warnings

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r".*lru_cache.*",
            category=UserWarning,
        )
        onnx_program = torch.onnx.export(
            wrapper,
            args=(dummy_input,),
            f=None,
            input_names=input_names,
            output_names=output_names,
            dynamo=True,
            external_data=True,  # required: model weights exceed 2 GB protobuf limit
            dynamic_shapes=dynamic_shapes,
            optimize=True,
            verbose=False,
        )

    onnx_program.save(str(out_path))
    _log_export_size(output_dir, label)
    return out_path


def _log_export_size(output_dir: Path, label: str) -> None:
    total = sum(f.stat().st_size for f in output_dir.iterdir() if f.is_file())
    print(f"[export] {label}: {total / 1e9:.2f} GB written to {output_dir}")


# ---------------------------------------------------------------------------
# Quantization
# ---------------------------------------------------------------------------

_HAS_Q4 = Version(onnxruntime.__version__) >= Version("1.18")


def _load_proto_for_quantization(src_onnx: Path) -> "onnx.ModelProto":
    """Load an ONNX model with all external-data weights resolved into memory.

    quantize_dynamic accepts a ModelProto directly.  When given a proto,
    ORT calls save_and_reload_model_with_shape_infer internally, which saves
    to its own internal temp dir (using save_as_external_data=True) before
    running shape inference — so it handles the >2 GB limit correctly on its
    own.  We just need to give it a fully-loaded proto with no dangling
    external-data references.

    The data sidecar filename is determined from the location field stored in
    the first initializer that uses external data (torch exports as
    'model.onnx.data'; other tools may differ).
    """
    import onnx

    print(f"[export]   Loading weights into memory …")
    # onnx.load with load_external_data=True resolves all sidecar tensors.
    # base_dir is inferred from the directory containing src_onnx.
    return onnx.load(str(src_onnx), load_external_data=True)


def _quantize(src_onnx: Path, output_dir: Path, label: str, do_q4: bool) -> None:
    """Produce int8 and (optionally) q4 quantized variants via onnxruntime.

    We load the fp32 proto into memory (resolving external data) and pass it
    directly to quantize_dynamic as a ModelProto.  ORT then runs its own
    internal shape-inference + save/reload cycle in a self-managed temp
    location — no staging dir, no filename assumptions, no 2 GB proto limit.
    """
    from onnxruntime.quantization import quantize_dynamic, QuantType

    proto = _load_proto_for_quantization(src_onnx)

    # int8
    int8_path = output_dir / "model_quantized.onnx"
    print(f"[export] Quantizing {label} → int8 …")
    quantize_dynamic(
        model_input=proto,
        model_output=str(int8_path),
        weight_type=QuantType.QInt8,
        extra_options={"MatMulConstBOnly": True},
    )
    _log_export_size(output_dir, f"{label} int8")

    # q4
    if not do_q4:
        return
    if not _HAS_Q4:
        print(
            f"[export] WARNING: q4 quantization requires onnxruntime >= 1.18. "
            f"Found {onnxruntime.__version__}. Skipping q4 for {label}."
        )
        return

    q4_path = output_dir / "model_q4.onnx"
    print(f"[export] Quantizing {label} → q4 …")
    try:
        quantize_dynamic(
            model_input=proto,
            model_output=str(q4_path),
            weight_type=QuantType.QUInt4,
            extra_options={"MatMulConstBOnly": True},
        )
        _log_export_size(output_dir, f"{label} q4")
    except Exception as exc:
        print(f"[export] WARNING: q4 quantization failed for {label}: {exc}")
        print("[export] int8 variant is still available.")


# ---------------------------------------------------------------------------
# Per-export tasks
# ---------------------------------------------------------------------------


def export_tagger(
    model: DINOv3Tagger,
    output_dir: Path,
    device: torch.device,
    dummy_h: int,
    dummy_w: int,
    quantize: bool,
    do_q4: bool,
) -> None:
    wrapper = TaggerWrapper(model)
    wrapper.eval()
    dummy = torch.zeros(1, 3, dummy_h, dummy_w, dtype=torch.float32, device=device)
    onnx_path = _export_one(
        wrapper,
        dummy,
        output_dir,
        input_names=["pixel_values"],
        output_names=["logits"],
        dummy_h=dummy_h,
        dummy_w=dummy_w,
        label="tagger",
    )
    if quantize:
        _quantize(onnx_path, output_dir, "tagger", do_q4)


def export_embedding(
    model: DINOv3Tagger,
    output_dir: Path,
    device: torch.device,
    dummy_h: int,
    dummy_w: int,
    quantize: bool,
    do_q4: bool,
) -> None:
    wrapper = EmbeddingWrapper(model)
    wrapper.eval()
    dummy = torch.zeros(1, 3, dummy_h, dummy_w, dtype=torch.float32, device=device)
    onnx_path = _export_one(
        wrapper,
        dummy,
        output_dir,
        input_names=["pixel_values"],
        output_names=["embedding"],
        dummy_h=dummy_h,
        dummy_w=dummy_w,
        label="embedding",
    )
    if quantize:
        _quantize(onnx_path, output_dir, "embedding", do_q4)


def export_backbone(
    model: DINOv3Tagger,
    output_dir: Path,
    device: torch.device,
    dummy_h: int,
    dummy_w: int,
    quantize: bool,
    do_q4: bool,
) -> None:
    wrapper = BackboneWrapper(model.backbone)
    wrapper.eval()
    dummy = torch.zeros(1, 3, dummy_h, dummy_w, dtype=torch.float32, device=device)
    onnx_path = _export_one(
        wrapper,
        dummy,
        output_dir,
        input_names=["pixel_values"],
        output_names=["last_hidden_state"],
        dummy_h=dummy_h,
        dummy_w=dummy_w,
        label="backbone",
    )
    if quantize:
        _quantize(onnx_path, output_dir, "backbone", do_q4)


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------


def _verify_onnx(onnx_path: Path, dummy_h: int, dummy_w: int, output_name: str) -> None:
    """Quick smoke-test: run a dummy forward through onnxruntime."""
    import numpy as np

    print(f"[verify] {onnx_path.parent.name}/model.onnx … ", end="", flush=True)
    try:
        sess = onnxruntime.InferenceSession(
            str(onnx_path),
            providers=["CPUExecutionProvider"],
        )
        dummy_np = np.zeros((1, 3, dummy_h, dummy_w), dtype=np.float32)
        outputs = sess.run([output_name], {"pixel_values": dummy_np})
        print(f"OK  output shape: {list(outputs[0].shape)}")
    except Exception as exc:
        print(f"FAILED: {exc}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Export DINOv3 ViT-H/16+ Tagger to ONNX",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--checkpoint",
        default=None,
        help=(
            "Path to .safetensors or .pt checkpoint. "
            f"If omitted, auto-downloads {HF_FILENAME} from {HF_REPO_ID}."
        ),
    )
    p.add_argument(
        "--output-dir",
        default="onnx",
        help="Root output directory (default: ./onnx)",
    )
    p.add_argument(
        "--device",
        default=None,
        help="Export device: cuda or cpu. Default: cuda if available, else cpu.",
    )
    p.add_argument(
        "--dummy-h",
        type=int,
        default=DEFAULT_DUMMY_H,
        help=f"Dummy input height in pixels (default: {DEFAULT_DUMMY_H}, must be multiple of {PATCH_SIZE})",
    )
    p.add_argument(
        "--dummy-w",
        type=int,
        default=DEFAULT_DUMMY_W,
        help=f"Dummy input width in pixels (default: {DEFAULT_DUMMY_W}, must be multiple of {PATCH_SIZE})",
    )
    p.add_argument(
        "--no-tagger",
        action="store_true",
        help="Skip tagger export (pixel_values → logits)",
    )
    p.add_argument(
        "--no-embedding",
        action="store_true",
        help="Skip embedding export (pixel_values → 6400-dim descriptor)",
    )
    p.add_argument(
        "--no-backbone",
        action="store_true",
        help="Skip backbone export (pixel_values → last_hidden_state)",
    )
    p.add_argument(
        "--no-quantize",
        action="store_true",
        help="Skip int8 and q4 quantized variants",
    )
    p.add_argument(
        "--no-q4",
        action="store_true",
        help="Skip q4 variant only (still produce int8)",
    )
    p.add_argument(
        "--hf-cache-dir",
        default=None,
        help="Override HuggingFace cache directory for checkpoint download",
    )
    p.add_argument(
        "--verify",
        action="store_true",
        default=True,
        help="Run onnxruntime smoke-test on each exported model (default: on)",
    )
    p.add_argument(
        "--no-verify",
        action="store_true",
        help="Disable onnxruntime smoke-test",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = _parse_args()

    # Resolve device
    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"[export] Using device: {device}")

    # Validate dummy resolution
    if args.dummy_h % PATCH_SIZE != 0 or args.dummy_w % PATCH_SIZE != 0:
        print(
            f"[export] ERROR: --dummy-h and --dummy-w must be multiples of "
            f"PATCH_SIZE={PATCH_SIZE}. Got {args.dummy_h}x{args.dummy_w}.",
            file=sys.stderr,
        )
        sys.exit(1)

    # Checkpoint
    if args.checkpoint is not None:
        ckpt_path = Path(args.checkpoint)
        if not ckpt_path.exists():
            print(f"[export] ERROR: checkpoint not found: {ckpt_path}", file=sys.stderr)
            sys.exit(1)
    else:
        ckpt_path = _auto_download_checkpoint(
            Path(args.hf_cache_dir) if args.hf_cache_dir else None
        )

    # Load model
    model, num_tags = _load_checkpoint(ckpt_path, device)

    # Output root
    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    quantize = not args.no_quantize
    do_q4 = quantize and not args.no_q4
    do_verify = args.verify and not args.no_verify

    h, w = args.dummy_h, args.dummy_w

    # -----------------------------------------------------------------------
    # Export
    # -----------------------------------------------------------------------

    if not args.no_tagger:
        export_tagger(model, out_root / "tagger", device, h, w, quantize, do_q4)

    if not args.no_embedding:
        export_embedding(model, out_root / "embedding", device, h, w, quantize, do_q4)

    if not args.no_backbone:
        export_backbone(model, out_root / "backbone", device, h, w, quantize, do_q4)

    # -----------------------------------------------------------------------
    # Verify
    # -----------------------------------------------------------------------

    if do_verify:
        print("\n[verify] Running onnxruntime smoke-tests …")
        checks = [
            (not args.no_tagger, out_root / "tagger" / "model.onnx", "logits"),
            (not args.no_embedding, out_root / "embedding" / "model.onnx", "embedding"),
            (
                not args.no_backbone,
                out_root / "backbone" / "model.onnx",
                "last_hidden_state",
            ),
        ]
        for enabled, path, out_name in checks:
            if enabled and path.exists():
                _verify_onnx(path, h, w, out_name)

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------

    print("\n" + "=" * 60)
    print(" ONNX Export Summary")
    print("=" * 60)
    print(f"  Checkpoint   : {ckpt_path}")
    print(f"  Device       : {device}")
    print(f"  num_tags     : {num_tags:,}")
    print(f"  Dummy shape  : [1, 3, {h}, {w}]")
    print(f"  Output root  : {out_root.resolve()}")
    print()

    rows = [
        (
            "tagger",
            not args.no_tagger,
            "pixel_values → logits [B, num_tags]",
            "sigmoid for probabilities",
        ),
        (
            "embedding",
            not args.no_embedding,
            "pixel_values → embedding [B, 6400]",
            "cosine similarity / search",
        ),
        (
            "backbone",
            not args.no_backbone,
            "pixel_values → last_hidden_state [B, S, 1280]",
            "PCA — patch tokens at [:,5:,:]",
        ),
    ]
    for name, exported, io_desc, note in rows:
        status = "✓" if exported else "–"
        print(f"  [{status}] {name:<12} {io_desc}")
        if exported:
            print(f"       note: {note}")
            d = out_root / name
            for fname in ("model.onnx", "model_quantized.onnx", "model_q4.onnx"):
                fpath = d / fname
                if fpath.exists():
                    size_gb = fpath.stat().st_size / 1e9
                    data_path = Path(str(fpath) + "_data")
                    if data_path.exists():
                        size_gb += data_path.stat().st_size / 1e9
                    print(f"             {fname:<28} {size_gb:.2f} GB")
    print("=" * 60)


if __name__ == "__main__":
    main()
