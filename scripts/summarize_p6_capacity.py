"""Print the compact persisted P6.2a rank-capacity attribution."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPORT = ROOT / "outputs/densek3-core/manifests/p6/p6-2a/p6-2a-capacity.json"


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
    print("LAYER  TF-R512  SCALEOUT-R512  TF-R768  TF-R1024  ELASTICITY  AMPLIFICATION  CLASS")
    for layer_index in (3, 7, 11, 15, 19, 23, 27, 31):
        layer = report["per_layer"][str(layer_index)]
        ranks = layer["teacher_forced_by_rank"]
        values = {
            rank: ranks[str(rank)]["teacher_forced_vs_full_rank_nope"]["mixer_output"][
                "relative_l2_error"
            ]
            for rank in (512, 768, 1024)
        }
        scaleout = layer["scaleout_input_r512_vs_full_rank_nope"]["mixer_output"][
            "relative_l2_error"
        ]
        diagnostic = layer["diagnostics"]
        print(
            f"L{layer_index:02d}  {values[512]:.9f}  {scaleout:.9f}  "
            f"{values[768]:.9f}  {values[1024]:.9f}  "
            f"{diagnostic['primary_mixer_rank_elasticity']:+.6f}  "
            f"{diagnostic['primary_mixer_scaleout_amplification']:.6f}  "
            f"{diagnostic['classification']}"
        )
    print(f"AGGREGATE: {report['aggregate']}")
    print(f"FORMAL_KV_LORA_RANK: {report['formal_kv_lora_rank']}")
    print(f"DIAGNOSTIC_RANKS: {report['diagnostic_ranks']}")
    print(f"P6_2_STATUS_UNCHANGED: {report['p6_2_status_unchanged']}")
    print(f"TRAINING_PERFORMED: {report['training_performed']}")
    print(f"FULL_DEV_ACCESSED: {report['full_dev_accessed']}")
    print(f"HELDOUT_ACCESSED: {report['heldout_accessed']}")
    print(f"CHECKPOINT_WRITTEN: {report['checkpoint_written']}")
    print(f"P6_3_ALLOWED: {report['p6_3_allowed']}")
    print(f"P7_ALLOWED: {report['p7_allowed']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
