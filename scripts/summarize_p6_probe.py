"""Print the compact decision evidence from a completed P6.0/P6.1 report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def metric_line(name: str, value: dict[str, Any]) -> None:
    print(
        f"{name}: relL2={value['relative_l2_error']:.9g} "
        f"cos={value['cosine_similarity']:.9g} max={value['max_abs_error']:.9g}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "report",
        type=Path,
        nargs="?",
        default=Path("outputs/densek3-core/manifests/p6/p6-single-layer-probe.json"),
    )
    args = parser.parse_args()
    report = json.loads(args.report.read_text(encoding="utf-8"))
    print("STATUS:", report["status"])
    print("P6.0:", report["p6_0"]["status"])
    print("P6.1:", report["p6_1"]["status"])
    svd = report["p6_1"]["conversion_manifest"]["mapping"]["kv"]
    print(
        "SVD:",
        {
            "rank": svd["rank"],
            "energy": svd["explained_squared_singular_mass"],
            "fro_rel": svd["frobenius_relative_error"],
            "max_abs": svd["maximum_absolute_error"],
            "k": svd["k_reconstruction"],
            "v": svd["v_reconstruction"],
        },
    )
    cases = report["p6_1"]["cases"]
    a = cases["A_GQA_PARTIAL_ROPE"]
    print("CE:")
    for name, case in cases.items():
        print(
            f"  {name}: canonical={case['canonical_ce']:.12f} "
            f"full_dev={case['full_dev_ce']:.12f} "
            f"delta_dev={case['full_dev_ce'] - a['full_dev_ce']:+.12f}"
        )
    for name in ("B_FULL_RANK_NOPE", "C_RANK512_MLA_NOPE"):
        case = cases[name]
        print(name)
        metric_line("  layer3_block", case["layer_3_block_output"])
        metric_line("  final_hidden", case["final_hidden"])
        metric_line("  logits", case["logits"])
        print("  decisions:", case["token_decisions"])
        print("  generation:", case["generation_drift"])
    local = report["p6_1"]["single_layer_diagnostics"]
    print("LOCAL_MIXER:")
    for name, metrics in local["mixer_output"].items():
        metric_line(f"  {name}", metrics)
    print("LOCAL_ATTENTION_CORE:")
    for name, metrics in local["attention_core_output"].items():
        metric_line(f"  {name}", metrics)
    print("P6_1_DECISION:", report["decision"])
    print("P6_2_ALLOWED:", report["p6_2_allowed"])
    print("P7_ALLOWED:", report["p7_allowed"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
