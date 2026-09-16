"""Print the compact P6.1b attribution result."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "report",
        nargs="?",
        type=Path,
        default=Path("outputs/densek3-core/manifests/p6/p6-1b/p6-1b-attribution.json"),
    )
    args = parser.parse_args()
    report = json.loads(args.report.read_text(encoding="utf-8"))
    a_ce = report["cases"]["A"]["full_dev_ce"]
    print("STATUS:", report["status"])
    print("CASES:")
    for name, case in report["cases"].items():
        print(
            f"  {name}: full_dev={case['full_dev_ce']:.12f} "
            f"delta_A={case['full_dev_ce'] - a_ce:+.12f} canonical={case['canonical_ce']:.12f}"
        )
    print("LOCAL MIXER REL-L2:")
    for name, effect in report["local_effects"].items():
        print(
            f"  {name}: mixer={effect['mixer_output']['relative_l2_error']:.9g} "
            f"core={effect['attention_core']['relative_l2_error']:.9g} "
            f"scores={effect['attention_scores_visible']['relative_l2_error']:.9g}"
        )
    attribution = report["attribution"]
    print("DOMINANT_FACTOR:", attribution["dominant_factor"])
    print("SECONDARY_FACTOR:", attribution["secondary_factor"])
    print("INTERACTION:", attribution["interaction_assessment"])
    print("REDESIGN_TARGET:", attribution["recommended_redesign"])
    print("ATOMIC_MAGNITUDES:", attribution["atomic_mixer_relative_l2"])
    print("HELDOUT_ACCESSED:", report["heldout_accessed"])
    print("P6_2_ALLOWED:", report["p6_2_allowed"])
    print("P7_ALLOWED:", report["p7_allowed"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
