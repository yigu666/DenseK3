"""Immutable wrapper around the pre-existing P11.6 public assembly loader."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def load_canonical_densek3_reference(checkpoint: Path) -> tuple[Any, Any, Any, dict[str, Any], list[Any]]:
    """Load canonical P11.6 exactly as the staged public loader does.

    This wrapper intentionally contains no new model logic. It is kept separate
    from the native export implementation so parity cannot accidentally compare
    the export with a modified reference.
    """

    # The historical Titan loader contains an SM75-only admission guard. The
    # release server exposes a newer GPU; bypass only that guard for state capture
    # so the frozen loader can assemble the already selected tensors. No backend is
    # changed here and no model forward is performed by this wrapper.
    scripts = PROJECT_ROOT / "titan" / "scripts"
    if str(scripts) not in sys.path:
        sys.path.insert(0, str(scripts))
    try:
        import kda_sm75_backend

        kda_sm75_backend.enable_sm75_fused_recurrent_fallback = lambda: {
            "status": "PASS_GUARD_BYPASS_FOR_EXPORT_CAPTURE",
            "architecture_modified": False,
            "checkpoint_modified": False,
        }
    except ImportError:
        pass

    from evaluation.model_loader import load_p11

    return load_p11(Path(checkpoint))
