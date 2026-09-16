"""Small, offline checks for the public DenseK3-4B release contract."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_clean_room_parity_fixture_is_pass() -> None:
    report = json.loads(
        (ROOT / "release/hf_export/validation/PARITY_SUMMARY.json").read_text(encoding="utf-8")
    )
    assert report["status"] == "PASS"
    assert report["checks"]["structural_parameter_names"] is True
    assert report["checks"]["save_load_roundtrip"] is True


def test_public_config_accepts_true_latent_release_mode() -> None:
    import sys

    sys.path.insert(0, str(ROOT / "src"))
    from densek3_core.modeling.configuration_densek3 import (  # noqa: PLC0415
        P6_MIXER_TYPES,
        DenseK3Config,
    )

    config = DenseK3Config(
        mixer_types=P6_MIXER_TYPES,
        layer_types=["full_attention" if value == "mla" else "linear_attention" for value in P6_MIXER_TYPES],
        mla_cache_mode="true_latent",
        mla_qk_norm=True,
        kda_gate_mode="kimi_k3",
        kda_gate_lower_bound=-5.0,
        kda_output_gate_activation="silu",
    )
    assert config.mla_indices == [3, 7, 11, 15, 19, 23, 27, 31]
