"""Print the compact persisted P6.2b-2 one-time extension decision."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPORT = ROOT / "outputs/densek3-core/manifests/p6/p6-2b-2/p6-2b-2-extension.json"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()
    report = json.loads(args.report.resolve().read_text(encoding="utf-8"))
    print(f"STATUS: {report['status']}")
    if report["status"] == "ERROR":
        print(f"EXCEPTION: {report['exception_type']}: {report['exception']}")
        print("P6_2B_SCALEOUT_ALLOWED: False")
        print("P6_3_ALLOWED: False")
        print("P7_ALLOWED: False")
        return 1
    print("LAYER  BEST_STEP  BEST_TOKENS  CORE  MIXER  CORE_GAIN  MIXER_GAIN")
    for layer_index in (3, 23):
        layer = report["layers"][str(layer_index)]["extension"]
        decision = report["local_decision"]["per_layer"][str(layer_index)]
        print(
            f"L{layer_index:02d}  {layer['best_step_within_256']:9d}  "
            f"{layer['best_effective_activation_tokens']:11d}  "
            f"{decision['calibrated_attention_core_relative_l2']:.9f}  "
            f"{decision['calibrated_mixer_relative_l2']:.9f}  "
            f"{decision['attention_core_improvement_fraction']:+.6f}  "
            f"{decision['mixer_improvement_fraction']:+.6f}"
        )
        print(f"  REPLAY: {layer['replay_checks']}")
        print(f"  CHECKS: {decision['checks']}")
    print(f"OUTCOME: {report['outcome']}")
    print(f"ARTIFACT: {report['artifact']}")
    print(f"P6_2B_1_STATUS_UNCHANGED: {report['p6_2b_1_status_unchanged']}")
    print(f"FULL_DEV_ACCESSED: {report['full_dev_accessed']}")
    print(f"HELDOUT_ACCESSED: {report['heldout_accessed']}")
    print(f"FURTHER_BUDGET_EXTENSION_ALLOWED: {report['further_budget_extension_allowed']}")
    print(f"P6_2B_SCALEOUT_ALLOWED: {report['p6_2b_scaleout_allowed']}")
    print(f"P6_3_ALLOWED: {report['p6_3_allowed']}")
    print(f"P7_ALLOWED: {report['p7_allowed']}")
    return 0 if report["status"] == "GO" else 1


if __name__ == "__main__":
    raise SystemExit(main())
