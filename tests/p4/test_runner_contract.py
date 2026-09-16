from __future__ import annotations

import runpy
from pathlib import Path


def runner_namespace() -> dict[str, object]:
    return runpy.run_path(str(Path(__file__).resolve().parents[2] / "scripts/run_p4.py"))


def test_p4_stages_are_dependency_ordered_and_complete() -> None:
    namespace = runner_namespace()
    stages = namespace["STAGES"]
    assert [item[0] for item in stages] == [f"P4.{index}" for index in range(8)]
    assert stages[-1][3] == "P4_07_SAVE_LOAD_GENERATE"


def test_p4_output_distribution_retains_hard_outlier_cap() -> None:
    namespace = runner_namespace()
    distribution = {
        "relative_l2_error": 0.001,
        "cosine_similarity": 0.99999,
        "mean_abs_error": 0.0001,
        "p99_9_abs_error": 0.001,
        "failed_allclose_fraction": 0.0,
        "max_abs_error": 0.25,
        "nan_count": 0,
        "inf_count": 0,
    }
    assert namespace["evaluate_output_distribution"](distribution)["passed"] is False
