"""Run the authorized Titan-only reduced-scope P10-T recipe."""

from __future__ import annotations

from pathlib import Path

import run_p10 as runner
from p10 import build_p10_titan_reduced_scope

ROOT = Path(__file__).resolve().parents[2]
TITAN = ROOT / "titan"

runner.CONFIG = TITAN / "configs/p10-titan-reduced-scope-2026-08-21.yaml"
runner.REPORT_DIR = TITAN / "manifests/reproduction/p10-t"
runner.CHECKPOINT_DIR = ROOT / "outputs/densek3-core/checkpoints/p10-t/joint-recovery"
runner.COMPENSATION_AUDIT_ENABLED = True
runner.COMPENSATION_BASELINE = (
    runner.CHECKPOINT_DIR / "kda-compensation-step-zero.safetensors"
)
runner.CANDIDATE = (
    ROOT
    / "outputs/densek3-core/checkpoints/p10-t/"
    "densek3-4b-core-k3-joint-reduced-titan"
)
runner.PHASE1_FINAL = runner.REPORT_DIR / "p10-t-phase1-final.json"
runner.EXTENSION_FINAL = runner.REPORT_DIR / "p10-t-extension-final.json"
runner.FINAL = runner.REPORT_DIR / "p10-t-final-check.json"
runner.SMOKE_FINAL = runner.REPORT_DIR / "p10-t-one-step-smoke.json"
runner.RUN_LABEL = "P10_T"
runner.RUN_STAGE_PREFIX = "TITAN_P10_T"
runner.RUN_MODE = "REDUCED_SCOPE_CORE_K3_JOINT_RECOVERY"
runner.CANDIDATE_STAGE = "TITAN_FP16_PROVISIONAL_P10_T_REDUCED_SCOPE_CANDIDATE"
runner.CANDIDATE_ARCHITECTURE = (
    "DENSEK3_CORE_KDA_MLA_ATTNRES_SITU_JOINT_RECOVERED_REDUCED_SCOPE"
)
runner.RUNTIME_CONFIG_FILE = "p10-t-runtime-config.json"
runner.CANDIDATE_MANIFEST_FILE = "p10-t-candidate-manifest.json"
runner.SCOPE_BUILDER = build_p10_titan_reduced_scope


if __name__ == "__main__":
    raise SystemExit(runner.main())
