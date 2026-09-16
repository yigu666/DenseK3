"""Print a compact summary of the P6.2c Q/K norm semantic bridge."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPORT = ROOT / "outputs/densek3-core/manifests/p6/p6-2c/p6-2c-qk-norm-bridge.json"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()
    report = json.loads(args.report.resolve().read_text(encoding="utf-8"))
    print(f"STATUS: {report['status']}")
    phase_a = report.get("phase_a", {})
    if phase_a:
        print(f"PHASE_A: {phase_a['status']}")
        for layer_index in (3, 23):
            layer = phase_a["layers"][str(layer_index)]
            values = {
                name: case["mixer_output"]["relative_l2_error"]
                for name, case in layer["cases"].items()
            }
            print(
                f"L{layer_index:02d} N0={values['N0']:.9g} "
                f"N1={values['N1']:.9g} N2={values['N2']:.9g}"
            )
        print(f"PHASE_A_DECISION: {phase_a['decision']}")
    phase_b = report.get("phase_b", {})
    print(f"PHASE_B: {phase_b.get('status', 'NOT_RECORDED')}")
    if "layers" in phase_b:
        for layer_index in (3, 23):
            calibration = phase_b["layers"][str(layer_index)]["calibration"]
            best = calibration["best_validation"]
            print(
                f"L{layer_index:02d} BEST_STEP={calibration['best_step']} "
                f"TOKENS={calibration['best_effective_activation_tokens']} "
                f"CORE={best['attention_core']['relative_l2_error']:.9g} "
                f"MIXER={best['mixer_output']['relative_l2_error']:.9g}"
            )
        print(f"LOCAL_DECISION: {phase_b['local_decision']}")
    if "artifact" in report:
        print(f"ARTIFACT: {report['artifact']}")
    print(f"FORMAL_KV_LORA_RANK: {report['formal_kv_lora_rank']}")
    print(f"FULL_DEV_ACCESSED: {report['full_dev_accessed']}")
    print(f"HELDOUT_ACCESSED: {report['heldout_accessed']}")
    print(f"P6_2C_SCALEOUT_ALLOWED: {report['p6_2c_scaleout_allowed']}")
    print(f"P6_3_ALLOWED: {report['p6_3_allowed']}")
    print(f"P7_ALLOWED: {report['p7_allowed']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
