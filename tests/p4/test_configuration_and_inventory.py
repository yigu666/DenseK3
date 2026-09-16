from __future__ import annotations

from pathlib import Path

import pytest
import torch

from densek3_core.modeling import DenseK3Config, DenseK3ForCausalLM
from densek3_core.modeling.configuration_densek3 import P4_GQA_INDICES
from densek3_core.transplant.full_model import (
    P4_EXPECTED_TARGET_TENSORS,
    P4_EXPECTED_UNIQUE_PARAMETERS,
    audit_source_inventory,
)

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.skipif(
    not (ROOT / "models/Qwen3.5-4B-Base/model.safetensors.index.json").is_file(),
    reason="Qwen donor metadata is downloaded separately",
)
def test_real_checkpoint_metadata_replays_locked_inventory() -> None:
    report = audit_source_inventory(ROOT / "models/Qwen3.5-4B-Base")
    assert report["counts"] == {"text": 426, "vision": 297, "mtp": 15, "total": 738}
    assert report["kda_layers"] == 24
    assert report["gqa_layers"] == 8
    assert report["gqa_indices"] == list(P4_GQA_INDICES)


def test_production_skeleton_has_exact_unique_parameter_count() -> None:
    config = DenseK3Config()
    assert set(config.layer_types) == {"linear_attention", "full_attention"}
    assert set(config.mixer_types) == {"kda", "gqa"}
    with torch.device("meta"):
        model = DenseK3ForCausalLM(config)
    assert sum(parameter.numel() for parameter in model.parameters()) == P4_EXPECTED_UNIQUE_PARAMETERS
    assert len(dict(model.named_parameters())) == P4_EXPECTED_TARGET_TENSORS
    assert len(model.model.layers) == 32
    assert [index for index, layer in enumerate(model.model.layers) if layer.layer_type == "gqa"] == list(
        P4_GQA_INDICES
    )
    assert model.lm_head.weight is model.model.embed_tokens.weight


def test_production_topology_cannot_be_silently_changed() -> None:
    mixer_types = list(DenseK3Config().mixer_types)
    mixer_types[0] = "gqa"
    with pytest.raises(ValueError, match="locked 24-KDA/8-GQA pattern"):
        DenseK3Config(mixer_types=mixer_types)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("kda_gate_mode", "kimi_k3"),
        ("kda_output_gate_activation", "sigmoid"),
        ("recurrent_state_layout", "k_first"),
        ("recurrent_state_dtype", "bfloat16"),
        ("hidden_act", "gelu"),
        ("tie_word_embeddings", False),
    ],
)
def test_p5_or_incompatible_options_are_rejected(field: str, value: object) -> None:
    with pytest.raises(ValueError):
        DenseK3Config(**{field: value})
