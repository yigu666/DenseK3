"""Print the compact persisted P6.2 scale-out decision."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPORT = ROOT / "outputs/densek3-core/manifests/p6/p6-2/p6-2-scaleout.json"


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
    print("PER-LAYER C3:")
    for layer_index, layer in report["per_layer_conversion"].items():
        local = layer["local_vs_full_rank_nope"]
        print(
            f"  L{int(layer_index):02d}: mixer={local['mixer_output']['relative_l2_error']:.9g} "
            f"attention={local['attention_output']['relative_l2_error']:.9g} "
            f"post_residual={local['post_attention_residual_hidden']['relative_l2_error']:.9g} "
            f"K={local['k']['relative_l2_error']:.9g} V={local['v']['relative_l2_error']:.9g}"
        )
    print(f"LOCAL_SUMMARY: {report['local_summary']}")
    print("MLA BOUNDARY DRIFT:")
    for item in report["full_model_shock"]["mla_boundary_drift"]:
        print(
            f"  L{item['layer_index']:02d}: relL2={item['relative_l2_error']:.9g} "
            f"cos={item['cosine_similarity']:.9g} rms_ratio={item['rms_ratio']:.9g}"
        )
    shock = report["full_model_shock"]
    print(f"FINAL_HIDDEN: {shock['final_hidden']}")
    print(f"LOGITS: {shock['logits']}")
    print(f"TOKEN_DECISIONS: {shock['token_decisions']}")
    print(f"FULL_DEV: {report['full_dev']}")
    print(f"GATE_CHECKS: {report['gate']['checks']}")
    print(f"CANDIDATE: {report['candidate_artifact']}")
    print(f"HELDOUT_ACCESSED: {report['heldout_accessed']}")
    print(f"P6_3_ALLOWED: {report['p6_3_allowed']}")
    print(f"P7_ALLOWED: {report['p7_allowed']}")
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
