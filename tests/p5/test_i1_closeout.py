from __future__ import annotations

import ast
import importlib.util
from pathlib import Path

import torch
import yaml

from densek3_core.recovery.p5_closeout import tensor_sha256
from densek3_core.recovery.p5_i1_closeout import (
    P5_I1_HELDOUT_MAX_CE,
    evaluate_i1_closeout_gate,
    i1_checkpoint_provenance,
    i1_heldout_gate,
    validate_i1_closeout_config,
    verify_i1_replay_binding,
)
from densek3_core.recovery.p5_rethink_v3 import (
    P5_RETHINK_V3_FORMULA_HASH,
    analytic_retention_match,
    build_analytic_i1_manifest,
)

ROOT = Path(__file__).resolve().parents[2]


def test_i1_closeout_config_freezes_unique_candidate_and_gate() -> None:
    config = yaml.safe_load(
        (ROOT / "configs/training/p5-rethink-v3-closeout.yaml").read_text(encoding="utf-8")
    )
    assert all(validate_i1_closeout_config(config).values())
    assert config["heldout"]["maximum_p5_ce"] == P5_I1_HELDOUT_MAX_CE
    assert all(config["forbidden"].values())


def test_i1_heldout_gate_is_absolute_and_unamended() -> None:
    assert i1_heldout_gate(P5_I1_HELDOUT_MAX_CE)["passed"] is True
    assert i1_heldout_gate(P5_I1_HELDOUT_MAX_CE + 1e-6)["passed"] is False


def _replay_fixture() -> tuple[dict[str, torch.Tensor], dict]:
    tensors = {}
    layers = []
    for layer_index in range(24):
        a_name = f"model.layers.{layer_index}.self_attn.A_log"
        bias_name = f"model.layers.{layer_index}.self_attn.dt_bias"
        tensors[a_name] = torch.full((32,), float(layer_index), dtype=torch.float32)
        tensors[bias_name] = torch.full((4096,), float(layer_index + 1), dtype=torch.float32)
        layers.append(
            {
                "layer_index": layer_index,
                "heads": 32,
                "representable_heads": 32 if layer_index else 29,
                "lower_clipped_heads": 0,
                "upper_clipped_heads": 3 if layer_index == 0 else 0,
                "clipped_heads": (
                    [
                        {
                            "head_index": index,
                            "clip_side": "upper",
                            "p4_target_log_decay": -6.0,
                            "i1_realized_log_decay": -5.0,
                            "log_decay_absolute_error": 1.0,
                            "p4_target_retention": 0.002,
                            "i1_realized_retention": 0.007,
                            "retention_absolute_error": 0.005,
                        }
                        for index in range(3)
                    ]
                    if layer_index == 0
                    else []
                ),
            }
        )
    return tensors, build_analytic_i1_manifest(tensors, layers)


def test_i1_replay_binds_all_48_checkpoint_tensor_hashes() -> None:
    tensors, replay = _replay_fixture()
    replay["replay_source"] = {
        "storage_dtypes": {name: "float32" for name in tensors},
        "formal_runtime_dtype": "float32",
        "runtime_dtype_cast_applied": False,
    }
    provenance = i1_checkpoint_provenance(replay, initialization_evidence_sha256="a" * 64)
    checkpoint = {
        "source_p4_parameter_hash": "p4",
        "initialization_provenance": provenance,
        "parameters": [
            {"name": name, "sha256": tensor_sha256(value)}
            for name, value in sorted(tensors.items())
        ],
    }
    result = verify_i1_replay_binding(
        checkpoint,
        tensors,
        replay,
        expected_p4_parameter_hash="p4",
    )
    assert result["passed"] is True
    first = next(iter(tensors))
    checkpoint["parameters"] = [
        {**item, "sha256": "0" * 64} if item["name"] == first else item
        for item in checkpoint["parameters"]
    ]
    assert verify_i1_replay_binding(
        checkpoint,
        tensors,
        replay,
        expected_p4_parameter_hash="p4",
    )["passed"] is False


def test_i1_replay_contract_uses_formal_fp32_runtime_dtype() -> None:
    """BF16 checkpoint storage must be promoted before analytic mapping."""
    a_storage = torch.tensor([-1.234375], dtype=torch.bfloat16)
    bias_storage = torch.tensor([0.333984375, 0.333984375], dtype=torch.bfloat16)
    runtime_a = a_storage.float()
    runtime_bias = bias_storage.float()
    expected_a, expected_bias, _ = analytic_retention_match(
        runtime_a,
        runtime_bias,
        key_dim=2,
    )
    assert expected_a.dtype == torch.float32
    assert expected_bias.dtype == torch.float32
    source = (ROOT / "src/densek3_core/recovery/p5_rethink_v3.py").read_text(encoding="utf-8")
    assert ".to(dtype=torch.float32).to(device)" in source
    assert '"formal_runtime_dtype": "float32"' in source


def test_i1_closeout_gate_requires_heldout_and_engineering() -> None:
    heldout = {
        "candidate": "I1",
        "candidate_count": 1,
        "finite": True,
        "gate": {"passed": True},
    }
    artifact = {
        "verification": {"status": "PASS"},
        "source_p4_verification": {"status": "PASS", "tensor_hashes_verified": 546},
        "mapping_replay": {"passed": True},
        "architecture": {"passed": True},
        "native_semantics": {"passed": True},
        "save_reload": {"passed": True},
        "bf16_forward": {"passed": True},
        "cache": {"passed": True},
        "generation": {"not_collapsed": True},
        "training_performed": False,
    }
    assert evaluate_i1_closeout_gate(heldout, artifact)["passed"] is True
    artifact["mapping_replay"]["passed"] = False
    assert evaluate_i1_closeout_gate(heldout, artifact)["passed"] is False


def test_i1_runner_opens_only_heldout_and_has_no_training_or_alternate_models() -> None:
    path = ROOT / "scripts/run_p5_i1_closeout.py"
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    splits = [
        node.args[1].value
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "PackedTokenDataset"
        and len(node.args) >= 2
        and isinstance(node.args[1], ast.Constant)
    ]
    assert splits == ["heldout"]
    for forbidden in (
        "optimizer.step",
        "loss.backward",
        "load_qwen_text_model",
        "apply_factorial_case",
        "Probe-A",
        "Probe-B",
    ):
        assert forbidden not in source
    assert "apply_analytic_i1(model)" in source
    assert "replay_model = load_densek3_model" in source
    assert "replay_manifest = apply_analytic_i1(replay_model)" in source
    assert 'model.config.p5_initialization = "retention_matched_analytic"' in source
    assert "P5_I1_HELDOUT=RESUMED_WITHOUT_DATA_ACCESS" in source
    assert "verify_p4_artifact(args.p4_artifact_dir, verify_tensor_hashes=True)" in source


def test_i1_closeout_script_is_importable_without_running_cuda() -> None:
    path = ROOT / "scripts/run_p5_i1_closeout.py"
    spec = importlib.util.spec_from_file_location("run_p5_i1_closeout", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.DEFAULT_CHECKPOINT_DIR.name == "densek3-4b-k3-decay-silu-i1"
    assert P5_RETHINK_V3_FORMULA_HASH
