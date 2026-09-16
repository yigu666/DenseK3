from __future__ import annotations

import argparse
import runpy
from pathlib import Path

import torch

from tests.p3.helpers import small_bridge_config


def runner_namespace() -> dict[str, object]:
    script = Path(__file__).resolve().parents[2] / "scripts/diagnose_p4_layer4_t129_qkv.py"
    return runpy.run_path(str(script))


def passing_distribution() -> dict[str, float | int]:
    return {
        "relative_l2_error": 7.0e-3,
        "cosine_similarity": 0.99997,
        "mean_abs_error": 1.0e-3,
        "p99_9_abs_error": 4.0e-3,
        "failed_allclose_fraction": 1.0e-4,
        "max_abs_error": 1.25e-1,
        "reference_nan_count": 0,
        "actual_nan_count": 0,
        "reference_inf_count": 0,
        "actual_inf_count": 0,
    }


def test_p3_output_thresholds_are_conjunctive_and_boundary_inclusive() -> None:
    evaluate = runner_namespace()["evaluate_p3_output_distribution"]
    accepted = evaluate(passing_distribution())
    rejected_distribution = passing_distribution()
    rejected_distribution["p99_9_abs_error"] = 4.1e-3
    rejected = evaluate(rejected_distribution)
    assert accepted["passed"] is True
    assert all(accepted["checks"].values())
    assert rejected["passed"] is False
    assert rejected["checks"]["p99_9_abs_error"] is False


def test_post_conv_amplification_records_each_requested_ratio() -> None:
    summarize = runner_namespace()["amplification_summary"]
    projection = {
        "max_abs_error": 0.01,
        "mean_abs_error": 0.002,
        "relative_l2_error": 0.004,
    }
    post_conv = {
        "max_abs_error": 0.02,
        "mean_abs_error": 0.001,
        "relative_l2_error": 0.006,
    }
    result = summarize(projection, post_conv)
    assert result["max_abs_error_ratio"]["value"] == 2.0
    assert result["mean_abs_error_ratio"]["value"] == 0.5
    assert result["relative_l2_error_ratio"]["value"] == 1.5
    assert result["automatic_acceptance_threshold"] is None


def test_fixed_random_state_is_reproducible() -> None:
    namespace = runner_namespace()
    make_state = namespace["seeded_initial_state"]
    config = small_bridge_config()
    first = make_state(config, torch.device("cpu"))
    second = make_state(config, torch.device("cpu"))
    assert first.shape == (1, config.value_num_heads, config.value_head_dim, config.qk_head_dim)
    assert first.dtype == torch.float32
    assert torch.equal(first, second)


def test_initial_contract_cannot_approve_or_change_p4_gate() -> None:
    namespace = runner_namespace()
    args = argparse.Namespace(layer_index=4, started_at="2026-08-11T00:00:00Z")
    report = namespace["_initial_report"](args)
    assert report["formal_gate_modified"] is False
    assert report["numerics_amendment_approved"] is False
    assert report["p4_4_formal_status"] == "FAIL_UNCHANGED"
    assert report["new_injection_run"] is False
