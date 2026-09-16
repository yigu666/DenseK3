"""Print the compact persisted P6.2d Q/K-norm-preserving scaleout decision."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPORT = ROOT / "outputs/densek3-core/manifests/p6/p6-2d/p6-2d-scaleout.json"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()
    report = json.loads(args.report.resolve().read_text(encoding="utf-8"))
    print(f"STATUS: {report['status']}")
    if report["status"] == "ERROR":
        print(f"EXCEPTION: {report['exception_type']}: {report['exception']}")
        print("P6_3_ALLOWED: False")
        print("P7_ALLOWED: False")
        return 1
    print("PER-LAYER Q/K-NORM-PRESERVING CALIBRATION:")
    for layer_index in (3, 7, 11, 15, 19, 23, 27, 31):
        item = report["per_layer_summary"][str(layer_index)]
        calibration = report["local_calibration"][str(layer_index)]
        best = item["best"]
        print(
            f"  L{layer_index:02d}: mode={calibration['mode']} "
            f"N1={item['n1']['mixer_output']['relative_l2_error']:.9g} "
            f"best={best['mixer_output']['relative_l2_error']:.9g} "
            f"core={best['attention_core']['relative_l2_error']:.9g} "
            f"step={calibration['best_step']} "
            f"tokens={calibration['best_effective_activation_tokens']}"
        )
    print(f"LOCAL_SAFETY: {report['local_safety']}")
    if "full_model_shock" in report:
        shock = report["full_model_shock"]
        print("MLA BOUNDARY DRIFT:")
        for item in shock["mla_boundary_drift"]:
            print(
                f"  L{item['layer_index']:02d}: relL2={item['relative_l2_error']:.9g} "
                f"cos={item['cosine_similarity']:.9g} rms_ratio={item['rms_ratio']:.9g}"
            )
        print(f"FINAL_HIDDEN: {shock['final_hidden']}")
        print(f"LOGITS: {shock['logits']}")
        print(f"TOKEN_DECISIONS: {shock['token_decisions']}")
    if "full_dev" in report:
        print(f"FULL_DEV: {report['full_dev']}")
        print(f"OLD_P6_2_COMPARISON: {report['old_p6_2_comparison']}")
        print(f"GATE_CHECKS: {report['gate']['checks']}")
    if "candidate_artifact" in report:
        print(f"CANDIDATE: {report['candidate_artifact']['directory']}")
    print(f"CANONICAL_ACCESSED: {report['canonical_accessed']}")
    print(f"HELDOUT_ACCESSED: {report['heldout_accessed']}")
    print(f"P6_3_ALLOWED: {report['p6_3_allowed']}")
    print(f"P7_ALLOWED: {report['p7_allowed']}")
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
