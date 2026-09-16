from __future__ import annotations

import json
import runpy
from pathlib import Path

import torch

from densek3_core.evaluation.p4_decay_diagnostics import (
    decay_path_diagnostics,
    error_distribution,
    forced_fp32_diagnostics,
    injection_diagnostics,
    parameter_embedding_audit,
    replay_p4_hidden,
)
from densek3_core.transplant.gdn_to_kda import convert_gdn_to_kda
from densek3_core.transplant.single_layer import DenseK3QwenCompatMixer, QwenGDNReferenceMixer
from tests.p3.helpers import small_bridge_config, small_source_state


def diagnostic_runner_namespace() -> dict[str, object]:
    script = Path(__file__).resolve().parents[2] / "scripts/diagnose_p4_layer4_decay.py"
    return runpy.run_path(str(script))


def test_replay_p4_hidden_advances_to_layer_four_draw() -> None:
    kda_indices = [0, 1, 2, 4, 5]
    actual, draw_one_based = replay_p4_hidden(8, kda_indices, layer_index=4)
    generator = torch.Generator(device="cpu").manual_seed(20260810)
    expected = None
    for _ in range(4):
        expected = torch.randn((1, 7, 8), generator=generator)
    assert draw_one_based == 4
    assert torch.equal(actual, expected)


def test_error_distribution_records_failed_coordinates() -> None:
    reference = torch.zeros((1, 2, 2, 2))
    actual = reference.clone()
    actual[0, 1, 0, 1] = 0.125
    report = error_distribution(
        reference,
        actual,
        torch.bfloat16,
        axis_names=("batch", "token", "head", "channel"),
        top_k=1,
    )
    assert report["num_elements"] == 8
    assert report["num_failed_allclose"] == 1
    assert report["top_outliers"][0]["index"] == {
        "batch": 0,
        "token": 1,
        "head": 0,
        "channel": 1,
    }


def test_parameter_and_f_b_embedding_diagnostics_are_exact() -> None:
    config = small_bridge_config()
    source = small_source_state()
    conversion = convert_gdn_to_kda(source, config)
    donor = QwenGDNReferenceMixer(config, source)
    target = DenseK3QwenCompatMixer(config, conversion.target_state)
    parameter_report = parameter_embedding_audit(config, donor, target)
    hidden = torch.randn(1, 7, config.hidden_size)
    donor_trace = donor.trace(hidden)
    target_trace = target.trace(hidden)
    path = decay_path_diagnostics(
        donor_trace,
        target_trace,
        torch.float32,
        donor_mixer=donor,
        target_mixer=target,
    )
    replication = path["target_f_b_vs_exact_repeat"]
    assert parameter_report["passed"] is True
    assert replication["num_exact"] == replication["num_elements"]
    assert path["target_channel_copy_max_spread"] == 0.0
    assert all(item["passed"] for item in path["decay_transform_replay"].values())


def test_forced_fp32_and_exact_injection_controls_close_the_small_bridge() -> None:
    config = small_bridge_config()
    source = small_source_state()
    conversion = convert_gdn_to_kda(source, config)
    donor = QwenGDNReferenceMixer(config, source).to("cpu", torch.bfloat16)
    target = DenseK3QwenCompatMixer(config, conversion.target_state).to("cpu", torch.bfloat16)
    hidden = torch.randn(1, 7, config.hidden_size, dtype=torch.bfloat16)
    donor_trace = donor.trace(hidden)
    target_trace = target.trace(hidden)
    forced = forced_fp32_diagnostics(donor, target, hidden, donor_trace, target_trace)
    injected = injection_diagnostics(donor, target, hidden, donor_trace, target_trace)
    assert forced["decay_strict_parity_after_bf16_cast"] is True
    assert injected["both_injected_exact"] is True


def test_diagnostic_json_is_strict_when_a_metric_is_nonfinite(tmp_path: Path) -> None:
    output = tmp_path / "diagnostic.json"
    diagnostic_runner_namespace()["save_json"](output, {"finite": 1.0, "nonfinite": float("nan")})
    assert json.loads(output.read_text(encoding="utf-8")) == {"finite": 1.0, "nonfinite": None}
