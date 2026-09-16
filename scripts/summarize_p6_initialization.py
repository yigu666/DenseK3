"""Print the compact persisted P6.1c decision summary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPORT = ROOT / "outputs/densek3-core/manifests/p6/p6-1c/p6-1c-initialization.json"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()
    report = json.loads(args.report.resolve().read_text(encoding="utf-8"))
    print(f"STATUS: {report['status']}")
    if report["status"] == "ERROR":
        print(f"EXCEPTION: {report['exception_type']}: {report['exception']}")
        print("P6_2_ALLOWED: False")
        print("P7_ALLOWED: False")
        return 1
    print("CASES:")
    for name, case in report["cases"].items():
        local = case["calibration"]["local_vs_B"]
        model = case["calibration"]["model_vs_A"]
        suffix = ""
        if "full_dev_ce" in case:
            suffix = f" full_dev={case['full_dev_ce']:.12f}"
        print(
            f"  {name}: mixer={local['mixer_output']['relative_l2_error']:.9g} "
            f"attention={local['attention_output']['relative_l2_error']:.9g} "
            f"post_residual={local['post_attention_residual_hidden']['relative_l2_error']:.9g} "
            f"final_cos={model['final_hidden']['cosine_similarity']:.9g} "
            f"logits_cos={model['logits']['cosine_similarity']:.9g} "
            f"calibration_ce={model['cross_entropy']:.9g}{suffix}"
        )
    selection = report["selection"]
    print(f"SELECTED_CASE: {selection['selected_case']}")
    print(f"SELECTED_INITIALIZATION: {selection['selected_initialization']}")
    print(f"FULL_DEV_CANDIDATES: {selection['full_dev_candidates_evaluated']}")
    print(f"CANONICAL_USED_FOR_SELECTION: {selection['canonical_used_for_selection']}")
    gate = report["gate"]
    print(f"B_TO_SELECTED_FULL_DEV_DELTA: {gate['b_to_selected_full_dev_ce_delta']:.12f}")
    print(f"GATE_CHECKS: {gate['checks']}")
    print(f"HELDOUT_ACCESSED: {report['heldout_accessed']}")
    print(f"P6_2_ALLOWED: {report['p6_2_allowed']}")
    print(f"P7_ALLOWED: {report['p7_allowed']}")
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
