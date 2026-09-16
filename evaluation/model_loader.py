"""Public P11.6 loader assembled from the reproducible P0-P11 checkpoint chain."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from safetensors.torch import load_file
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
TITAN = ROOT / "titan"
SCRIPTS = TITAN / "scripts"
sys.path[:0] = [str(TITAN), str(SCRIPTS), str(ROOT / "src")]

from densek3_core.recovery.p6_closeout import P6_MLA_LAYERS, sha256_file  # noqa: E402
from p7_latent_cache import enable_p7_latent_cache_runtime  # noqa: E402
from p11.runtime import TOKENIZER, load_p10_t_student  # noqa: E402
from run_p10 import apply_dense_overrides  # noqa: E402


def load_p11(checkpoint: Path) -> tuple[Any, Any, Any, dict[str, Any], list[Any]]:
    """Load P11.6 over the reproduced P10-T parent without enabling training."""
    checkpoint = checkpoint.resolve()
    required = (
        checkpoint / "model-dense-overrides.safetensors",
        checkpoint / "attnres.safetensors",
        checkpoint / "p11-6-fast-candidate-manifest.json",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"P11.6 checkpoint is incomplete: {missing}")

    model, attnres, parent_report = load_p10_t_student()
    apply_dense_overrides(model, checkpoint / "model-dense-overrides.safetensors")
    attnres.load_state_dict(load_file(str(checkpoint / "attnres.safetensors")))
    enable_p7_latent_cache_runtime(P6_MLA_LAYERS)

    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in attnres.parameters():
        parameter.requires_grad_(False)
    model.eval()
    attnres.eval()
    if hasattr(model, "gradient_checkpointing_disable"):
        model.gradient_checkpointing_disable()

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER, local_files_only=True, trust_remote_code=False)
    metadata = {
        "loader": "PUBLIC_P0_P11_CHAIN",
        "checkpoint": str(checkpoint),
        "manifest_sha256": sha256_file(checkpoint / "p11-6-fast-candidate-manifest.json"),
        "parent": parent_report,
        "training_performed": False,
        "backward_performed": False,
        "optimizer_constructed": False,
        "weights_modified": False,
    }
    return model, attnres, tokenizer, metadata, []
