"""Run Titan P9 SiTU-GLU architecture completion without training."""

from __future__ import annotations

import gc
import json
import math
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import yaml
from safetensors.torch import load_file, save_file

ROOT = Path(__file__).resolve().parents[2]
TITAN = ROOT / "titan"
sys.path.insert(0, str(TITAN))

from attnres import (  # noqa: E402
    BlockAttnResReference,
    disable_block_attnres_runtime,
    enable_block_attnres_runtime,
)
from kda_sm75_backend import enable_sm75_fused_recurrent_fallback  # noqa: E402
from p7_latent_cache import (  # noqa: E402
    disable_p7_latent_cache_runtime,
    enable_p7_latent_cache_runtime,
)
from run_p8_reference import (  # noqa: E402
    cache_equivalence,
    corpus_ce,
    generation,
    gpu_start_state,
)
from situ import (  # noqa: E402
    SITU_BETA,
    SITU_LINEAR_BETA,
    SiTUCalibrationCollector,
    disable_situ_glu_runtime,
    enable_situ_glu_runtime,
    fold_down_projection_scales,
    situ_glu,
)

from densek3_core.evaluation.p4_runtime import load_densek3_model  # noqa: E402
from densek3_core.recovery.p5_corpus import (  # noqa: E402
    PackedTokenDataset,
    verify_frozen_corpus,
)
from densek3_core.recovery.p6_closeout import (  # noqa: E402
    checkpoint_identity,
    sha256_file,
)

CONFIG = TITAN / "configs/p9-titan-situ-glu.yaml"
CACHE_CONTRACT = TITAN / "configs/p6-titan-fp16-cache-compat-2026-08-19-01.yaml"
SOURCE = (
    ROOT
    / "outputs/densek3-core/checkpoints/p6/"
    "densek3-4b-8mla-c3-qk-norm-calibrated-candidate"
)
P8_CONFIG = TITAN / "configs/p8-titan-block-attnres.yaml"
P8_ZERO_CHECKPOINT = (
    ROOT / "outputs/densek3-core/checkpoints/p8/routing-probe/step-000000"
)
P8_ZERO_REPORT = TITAN / "manifests/reproduction/p8/p8-2-zero-init-shock.json"
P8_ROUTING_REPORT = (
    TITAN / "manifests/reproduction/p8/routing-probe/p8-routing-probe.json"
)
P7_FINAL = TITAN / "manifests/reproduction/p7/p7-final-check.json"
DATA = ROOT / "data/p5-probe"
REPORT_DIR = TITAN / "manifests/reproduction/p9"
FINAL = REPORT_DIR / "p9-final-check.json"
CANDIDATE = ROOT / "outputs/densek3-core/checkpoints/p9/densek3-4b-core-k3-situ-titan"
STAGING = ROOT / "outputs/densek3-core/checkpoints/p9/.p9-situ-staging"
TOKENIZER = ROOT / "outputs/densek3-core/checkpoints/p5/densek3-4b-k3-decay-silu-i1"
IMPLEMENTATION_FILES = (
    TITAN / "situ/activation.py",
    TITAN / "situ/calibration.py",
    TITAN / "situ/runtime.py",
    TITAN / "attnres/block_reference.py",
    Path(__file__).resolve(),
)


def now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def sequence_document_ids(
    corpus_manifest: dict[str, Any],
    sequence_range: tuple[int, int],
) -> set[str]:
    start = sequence_range[0] * int(corpus_manifest["sequence_length"])
    end = sequence_range[1] * int(corpus_manifest["sequence_length"])
    return {
        item["document_id"]
        for item in corpus_manifest["splits"]["train"]["memberships"]
        if item["token_start"] < end and item["eos_position"] >= start
    }


def calibration_partition(config: dict[str, Any]) -> dict[str, Any]:
    corpus = load_json(DATA / "packed/corpus-manifest.json")
    fit_range = tuple(int(value) for value in config["calibration"]["fit_sequence_range"])
    validation_range = tuple(
        int(value) for value in config["calibration"]["validation_sequence_range"]
    )
    fit_documents = sequence_document_ids(corpus, fit_range)
    validation_documents = sequence_document_ids(corpus, validation_range)
    overlap = sorted(fit_documents & validation_documents)
    return {
        "fit_sequence_range": list(fit_range),
        "validation_sequence_range": list(validation_range),
        "fit_effective_tokens": (fit_range[1] - fit_range[0]) * corpus["sequence_length"],
        "validation_effective_tokens": (validation_range[1] - validation_range[0])
        * corpus["sequence_length"],
        "fit_document_count": len(fit_documents),
        "validation_document_count": len(validation_documents),
        "document_overlap": overlap,
        "document_disjoint": not overlap,
        "heldout_accessed": False,
    }


def load_zero_attnres(model: torch.nn.Module, p8_config: dict[str, Any]) -> BlockAttnResReference:
    runtime = BlockAttnResReference(
        model.config.hidden_size,
        model.config.rms_norm_eps,
        block_size_residuals=int(p8_config["architecture"]["block_size_residuals"]),
    ).cuda()
    runtime.load_state_dict(load_file(str(P8_ZERO_CHECKPOINT / "attnres.safetensors")))
    contract = runtime.parameter_contract()
    if not (contract["all_queries_zero"] and contract["all_norms_one"]):
        raise ValueError(f"P9 source is not P8 step-0 zero-query: {contract}")
    return runtime


@torch.inference_mode()
def collect(
    model: torch.nn.Module,
    runtime: BlockAttnResReference,
    collector: SiTUCalibrationCollector,
    dataset: PackedTokenDataset,
    sequence_range: tuple[int, int],
    *,
    label: str,
) -> None:
    collector.enable()
    runtime.branch_observer = collector.observe_branch
    enable_block_attnres_runtime(model, runtime)
    try:
        for offset, sequence_index in enumerate(range(*sequence_range), start=1):
            tokens = torch.from_numpy(dataset[sequence_index].copy()).unsqueeze(0).cuda()
            output = model.model(tokens, use_cache=False, return_dict=True).last_hidden_state
            if not bool(torch.isfinite(output).all()):
                raise FloatingPointError(f"Non-finite P9 calibration output at {sequence_index}")
            del tokens, output
            print(
                f"TITAN_P9_{label}_PROGRESS={offset}/{sequence_range[1] - sequence_range[0]}",
                flush=True,
            )
    finally:
        disable_block_attnres_runtime(model)
        runtime.branch_observer = None
        collector.disable()


def semantics_probe() -> dict[str, Any]:
    generator = torch.Generator(device="cuda").manual_seed(20260820)
    records = []
    for dtype in (torch.float32, torch.float16):
        gate = torch.randn(2, 7, 31, generator=generator, device="cuda", dtype=dtype)
        up = torch.randn(2, 7, 31, generator=generator, device="cuda", dtype=dtype)
        gate.requires_grad_(True)
        up.requires_grad_(True)
        output = situ_glu(gate, up)
        gate32 = gate.float()
        reference = (
            SITU_BETA
            * torch.tanh(gate32 / SITU_BETA)
            * torch.sigmoid(gate32)
            * SITU_LINEAR_BETA
            * torch.tanh(up.float() / SITU_LINEAR_BETA)
        ).to(dtype)
        wrong = (
            SITU_BETA
            * torch.tanh(gate32 / SITU_BETA)
            * torch.sigmoid(SITU_BETA * torch.tanh(gate32 / SITU_BETA))
            * SITU_LINEAR_BETA
            * torch.tanh(up.float() / SITU_LINEAR_BETA)
        ).to(dtype)
        output.float().square().mean().backward()
        reference_max_abs = float((output.float() - reference.float()).abs().max())
        checks = {
            "reference_match": bool(
                torch.equal(output, reference)
                if dtype == torch.float16
                else torch.allclose(output, reference, atol=1.0e-6, rtol=1.0e-6)
            ),
            "uncapped_sigmoid_distinguished": not bool(torch.equal(output, wrong)),
            "dtype_restored": output.dtype == dtype,
            "finite": bool(torch.isfinite(output).all()),
            "backward_gate_finite": gate.grad is not None and bool(torch.isfinite(gate.grad).all()),
            "backward_up_finite": up.grad is not None and bool(torch.isfinite(up.grad).all()),
        }
        records.append(
            {
                "dtype": str(dtype),
                "checks": checks,
                "reference_max_abs": reference_max_abs,
                "status": "PASS" if all(checks.values()) else "FAIL",
            }
        )
    return {
        "stage": "TITAN_P9_1_SITU_REFERENCE_SEMANTICS",
        "status": "PASS" if all(item["status"] == "PASS" for item in records) else "FAIL",
        "beta": SITU_BETA,
        "linear_beta": SITU_LINEAR_BETA,
        "sigmoid_input": "uncapped_gate",
        "records": records,
        "model_training_performed": False,
    }


def save_candidate(
    model: torch.nn.Module,
    runtime: BlockAttnResReference,
    scales: dict[int, torch.Tensor],
    selections: list[dict[str, Any]],
    fold_report: dict[str, Any],
    source_identity: dict[str, Any],
) -> dict[str, Any]:
    if CANDIDATE.exists() or STAGING.exists():
        raise FileExistsError("P9 candidate/staging already exists; refusing to overwrite")
    STAGING.mkdir(parents=True)
    scale_state = {f"layers.{index}.scale": value.cpu() for index, value in scales.items()}
    delta_state = {
        f"model.layers.{index}.mlp.down_proj.weight": model.model.layers[
            index
        ].mlp.down_proj.weight.detach().cpu()
        for index in fold_report["selected_s1_layers"]
    }
    save_file(scale_state, str(STAGING / "situ-scales.safetensors"))
    if delta_state:
        save_file(delta_state, str(STAGING / "down-proj-delta.safetensors"))
    save_file(
        {name: value.detach().cpu() for name, value in runtime.state_dict().items()},
        str(STAGING / "attnres-zero.safetensors"),
    )
    runtime_config = {
        "architecture": "DENSEK3_CORE_KDA_MLA_ATTNRES_SITU",
        "source_checkpoint": str(SOURCE),
        "source_checkpoint_fingerprint": source_identity["fingerprint"],
        "kda_layers": 24,
        "mla_layers": 8,
        "mla_rank": 512,
        "true_latent_cache": True,
        "block_attnres": True,
        "attnres_query_initialization": "zero",
        "ffn": "situ_glu",
        "situ_beta": SITU_BETA,
        "situ_linear_beta": SITU_LINEAR_BETA,
        "selection": {str(item["layer"]): item["selection"] for item in selections},
        "composite_overlay": True,
        "training_performed": False,
    }
    atomic_json(STAGING / "p9-runtime-config.json", runtime_config)
    manifest = {
        "stage": "TITAN_P9_PROVISIONAL_CANDIDATE",
        "status": "PENDING_EVALUATION",
        "source_checkpoint_identity": source_identity,
        "files": {},
        "fold_report": fold_report,
        "selections": selections,
        "training_performed": False,
        "optimizer_used": False,
        "heldout_accessed": False,
        "implementation_sha256": {
            str(path.relative_to(ROOT)): sha256_file(path) for path in IMPLEMENTATION_FILES
        },
    }
    for path in sorted(STAGING.iterdir()):
        if path.name != "p9-candidate-manifest.json":
            manifest["files"][path.name] = sha256_file(path)
    atomic_json(STAGING / "p9-candidate-manifest.json", manifest)
    STAGING.replace(CANDIDATE)
    return manifest


@torch.no_grad()
def apply_candidate_delta(model: torch.nn.Module) -> dict[str, Any]:
    delta_path = CANDIDATE / "down-proj-delta.safetensors"
    state = load_file(str(delta_path)) if delta_path.exists() else {}
    applied = []
    for name, value in state.items():
        parts = name.split(".")
        layer_index = int(parts[2])
        target = model.model.layers[layer_index].mlp.down_proj.weight
        if target.shape != value.shape:
            raise ValueError(f"P9 delta shape mismatch for {name}")
        target.copy_(value.to(device=target.device, dtype=target.dtype))
        applied.append(layer_index)
    return {"applied_layers": sorted(applied), "tensor_count": len(state)}


@torch.inference_mode()
def short_logits(
    model: torch.nn.Module,
    runtime: BlockAttnResReference,
    tokens: torch.Tensor,
) -> torch.Tensor:
    enable_situ_glu_runtime(model)
    enable_block_attnres_runtime(model, runtime)
    try:
        return model(tokens, use_cache=False, return_dict=True).logits.detach().cpu()
    finally:
        disable_block_attnres_runtime(model)
        disable_situ_glu_runtime(model)


def main() -> int:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    state = gpu_start_state()
    envelope: dict[str, Any] = {
        "stage": "TITAN_P9_SITU_GLU",
        "status": "RUNNING",
        "started_at": now(),
        "gpu": state,
        "p10_titan_migration_allowed": False,
        "p9_canonical_frozen": False,
        "heldout_accessed": False,
        "training_performed": False,
    }
    atomic_json(FINAL, envelope)
    print("TITAN_P9_SITU_GLU=RUNNING", flush=True)
    print("P9_TRAINING_PERFORMED=false", flush=True)
    print("HELDOUT_ACCESSED=false", flush=True)
    started = time.perf_counter()
    try:
        if not state["idle"]:
            envelope.update({"status": "DEFERRED_GPU_NOT_IDLE", "completed_at": now()})
            atomic_json(FINAL, envelope)
            return 3
        config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
        p8_config = yaml.safe_load(P8_CONFIG.read_text(encoding="utf-8"))
        p7 = load_json(P7_FINAL)
        p8_zero = load_json(P8_ZERO_REPORT)
        p8_routing = load_json(P8_ROUTING_REPORT)
        zero_checkpoint = load_json(P8_ZERO_CHECKPOINT / "checkpoint-manifest.json")
        corpus = verify_frozen_corpus(DATA)
        partition = calibration_partition(config)
        source_identity = checkpoint_identity(SOURCE)
        implementation_sha256 = {
            str(path.relative_to(ROOT)): sha256_file(path) for path in IMPLEMENTATION_FILES
        }
        prerequisites = {
            "p7_pass": p7["status"] == "PASS",
            "p8_zero_pass": p8_zero["status"] == "PASS"
            and p8_zero["decision"] == "STRONG_GO",
            "p8_routing_recipe_no_go": p8_routing["decision"] == "DROP_NO_GO",
            "p8_zero_checkpoint": zero_checkpoint["step"] == 0
            and zero_checkpoint["effective_tokens"] == 0,
            "p8_trained_artifact_not_selected": "step-000000"
            in zero_checkpoint["path"],
            "corpus_frozen": corpus["p5_probe_corpus_frozen"] is True,
            "calibration_document_disjoint": partition["document_disjoint"],
            "fit_tokens": partition["fit_effective_tokens"]
            == int(config["calibration"]["fit_effective_tokens"]),
            "validation_tokens": partition["validation_effective_tokens"]
            == int(config["calibration"]["validation_effective_tokens"]),
        }
        if not all(prerequisites.values()):
            raise ValueError(f"P9.0 prerequisites failed: {prerequisites}")
        p8_closeout = {
            "stage": "TITAN_P8_ARCHITECTURE_CLOSEOUT",
            "status": "PASS",
            "p8_attnres_reference_semantics": True,
            "p8_attnres_zero_init_transplant": True,
            "p8_routing_only_recovery": "NO_GO",
            "p8_routing_only_recipe_dropped": True,
            "p8_attnres_architecture_dropped": False,
            "p8_attnres_joint_recovery_required": True,
            "p8_mainline_state": "STEP_0_ZERO_QUERY",
            "p9_titan_migration_allowed": True,
            "p9_canonical_allowed": False,
            "heldout_accessed": False,
        }
        atomic_json(TITAN / "manifests/reproduction/p8/p8-architecture-closeout.json", p8_closeout)
        atomic_json(
            REPORT_DIR / "p9-0-source-contract.json",
            {
                "stage": "TITAN_P9_0_FROZEN_ARCHITECTURE_SOURCE",
                "status": "PASS",
                "checks": prerequisites,
                "partition": partition,
                "source_checkpoint_identity": source_identity,
                "p8_zero_checkpoint": zero_checkpoint,
                "implementation_sha256": implementation_sha256,
                "training_performed": False,
                "heldout_accessed": False,
            },
        )
        print("TITAN_P9_0_SOURCE=PASS", flush=True)

        semantics = semantics_probe()
        atomic_json(REPORT_DIR / "p9-1-situ-semantics.json", semantics)
        if semantics["status"] != "PASS":
            raise ValueError("P9.1 SiTU semantics failed")
        print("TITAN_P9_1_SITU_SEMANTICS=PASS", flush=True)

        enable_sm75_fused_recurrent_fallback()
        enable_p7_latent_cache_runtime()
        model = load_densek3_model(SOURCE, device="cuda", dtype=torch.float16)
        model.eval()
        runtime = load_zero_attnres(model, p8_config)
        train = PackedTokenDataset(DATA / "packed", "train")
        fit_range = tuple(partition["fit_sequence_range"])
        validation_range = tuple(partition["validation_sequence_range"])
        fit = SiTUCalibrationCollector(model, "fit")
        collect(model, runtime, fit, train, fit_range, label="FIT")
        scales = fit.fitted_scales(float(config["calibration"]["epsilon"]))
        fit_records = fit.fit_report(scales)
        fit_checks = {
            "layer_count": len(fit_records) == 32,
            "all_finite": all(
                item["gate"]["finite"]
                and item["up"]["finite"]
                and item["raw_situ_vs_swiglu"]["finite"]
                and item["s1_scale"]["finite"]
                for item in fit_records
            ),
            "document_disjoint": partition["document_disjoint"],
        }
        activation_audit = {
            "stage": "TITAN_P9_2_ACTIVATION_DISTRIBUTION_AUDIT",
            "status": "PASS" if all(fit_checks.values()) else "FAIL",
            "checks": fit_checks,
            "partition": partition,
            "layers": fit_records,
            "training_performed": False,
            "heldout_accessed": False,
        }
        atomic_json(REPORT_DIR / "p9-2-activation-audit.json", activation_audit)
        if activation_audit["status"] != "PASS":
            raise ValueError("P9.2 activation audit failed")
        print("TITAN_P9_2_ACTIVATION_AUDIT=PASS", flush=True)

        validation = SiTUCalibrationCollector(model, "validation")
        validation.set_scales(scales)
        collect(model, runtime, validation, train, validation_range, label="VALIDATION")
        selection_records = validation.validation_report()
        selected_s1 = {
            item["layer"]
            for item in selection_records
            if item["selection"] == "S1_ACTIVATION_MATCHED"
        }
        selection_checks = {
            "layer_count": len(selection_records) == 32,
            "all_finite": all(
                item["raw"]["post_residual"]["finite"]
                and item["matched"]["post_residual"]["finite"]
                for item in selection_records
            ),
            "deterministic_candidate_per_layer": all(
                item["selection"] in {"S0_RAW", "S1_ACTIVATION_MATCHED"}
                for item in selection_records
            ),
        }
        selection = {
            "stage": "TITAN_P9_3_P9_4_DETERMINISTIC_INITIALIZATION_SELECTION",
            "status": "PASS" if all(selection_checks.values()) else "FAIL",
            "checks": selection_checks,
            "selected_s0_layers": [
                item["layer"] for item in selection_records if item["layer"] not in selected_s1
            ],
            "selected_s1_layers": sorted(selected_s1),
            "layers": selection_records,
            "selection_primary": "post_residual_relative_l2",
            "selection_tiebreak": "mlp_output_relative_l2",
            "training_performed": False,
            "heldout_accessed": False,
        }
        atomic_json(REPORT_DIR / "p9-3-p9-4-initialization-selection.json", selection)
        if selection["status"] != "PASS":
            raise ValueError("P9.3/P9.4 initialization selection failed")
        print(
            f"TITAN_P9_4_SELECTION=PASS S0={32-len(selected_s1)} S1={len(selected_s1)}",
            flush=True,
        )

        del fit, validation, runtime, model
        gc.collect()
        torch.cuda.empty_cache()
        model = load_densek3_model(SOURCE, device="cuda", dtype=torch.float16)
        model.eval()
        runtime = load_zero_attnres(model, p8_config)
        fold_report = fold_down_projection_scales(model, scales, selected_s1)
        if not fold_report["passed"]:
            raise ValueError("P9 down-projection folding failed")
        candidate_manifest = save_candidate(
            model,
            runtime,
            scales,
            selection_records,
            fold_report,
            source_identity,
        )
        generator = torch.Generator(device="cpu").manual_seed(20260822)
        replay_tokens = torch.randint(
            1,
            model.config.vocab_size,
            (1, 65),
            generator=generator,
        ).cuda()
        before_reload = short_logits(model, runtime, replay_tokens)
        del runtime, model
        gc.collect()
        torch.cuda.empty_cache()

        model = load_densek3_model(SOURCE, device="cuda", dtype=torch.float16)
        model.eval()
        delta_reload = apply_candidate_delta(model)
        runtime = load_zero_attnres(model, p8_config)
        runtime.load_state_dict(load_file(str(CANDIDATE / "attnres-zero.safetensors")))
        after_reload = short_logits(model, runtime, replay_tokens)
        reload_metrics = {
            "exact": bool(torch.equal(before_reload, after_reload)),
            "max_abs": float((before_reload.float() - after_reload.float()).abs().max()),
            "finite": bool(torch.isfinite(after_reload).all()),
            "delta_reload": delta_reload,
        }
        del before_reload, after_reload

        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(TOKENIZER, trust_remote_code=False)
        dev = PackedTokenDataset(DATA / "packed", "dev")
        enable_situ_glu_runtime(model)
        enable_block_attnres_runtime(model, runtime)
        try:
            full_dev_ce = corpus_ce(model, dev, label="P9_COMPLETE_CORE")
        finally:
            disable_block_attnres_runtime(model)
            disable_situ_glu_runtime(model)
        enable_situ_glu_runtime(model)
        try:
            generated = generation(model, tokenizer, runtime)
            cache_limits = yaml.safe_load(CACHE_CONTRACT.read_text(encoding="utf-8"))[
                "evaluation"
            ]["fallback_requires_all"]
            cache = cache_equivalence(model, runtime, cache_limits)
        finally:
            disable_situ_glu_runtime(model)
        p8_zero_ce = float(p8_zero["p8_zero_full_dev_ce"])
        delta_ce = full_dev_ce - p8_zero_ce
        if delta_ce <= float(config["evaluation"]["low_shock_delta_ce_max"]):
            shock_class = "P9_LOW_SHOCK_PASS"
            recovery_priority = "NORMAL"
        elif delta_ce <= float(config["evaluation"]["recovery_required_delta_ce_max"]):
            shock_class = "P9_PASS_RECOVERY_REQUIRED"
            recovery_priority = "ELEVATED"
        else:
            shock_class = "P9_PASS_HIGH_SHOCK"
            recovery_priority = "HIGH"
        identity_after = checkpoint_identity(SOURCE)
        safety_checks = {
            "full_dev_finite": math.isfinite(full_dev_ce),
            "generation_finite": all(item["finite"] for item in generated),
            "generation_category_count": len(generated)
            == int(config["evaluation"]["generation_categories"]),
            "cache": cache["status"] == "PASS",
            "save_reload_exact": reload_metrics["exact"],
            "source_checkpoint_unchanged": identity_after == source_identity,
            "zero_query_preserved": runtime.parameter_contract()["all_queries_zero"],
            "zero_norm_preserved": runtime.parameter_contract()["all_norms_one"],
            "fold_reloaded": sorted(delta_reload["applied_layers"])
            == sorted(selected_s1),
            "no_training": True,
            "no_heldout": True,
        }
        passed = all(safety_checks.values())
        candidate_manifest.update(
            {
                "status": "PASS" if passed else "FAIL",
                "completed_at": now(),
                "full_dev_ce": full_dev_ce,
                "delta_ce_vs_p8_zero": delta_ce,
                "shock_class": shock_class,
                "safety_checks": safety_checks,
            }
        )
        atomic_json(CANDIDATE / "p9-candidate-manifest.json", candidate_manifest)
        evaluation = {
            "stage": "TITAN_P9_5_COMPLETE_CORE_NO_TRAINING_SHOCK",
            "status": "PASS" if passed else "FAIL",
            "shock_class": shock_class,
            "p10_recovery_priority": recovery_priority,
            "p8_zero_full_dev_ce": p8_zero_ce,
            "p9_full_dev_ce": full_dev_ce,
            "delta_ce": delta_ce,
            "checks": safety_checks,
            "save_reload": reload_metrics,
            "generation": generated,
            "cache": cache,
            "source_checkpoint_identity_before": source_identity,
            "source_checkpoint_identity_after": identity_after,
            "candidate": str(CANDIDATE),
            "training_performed": False,
            "heldout_accessed": False,
        }
        atomic_json(REPORT_DIR / "p9-5-complete-core-evaluation.json", evaluation)
        if not passed:
            raise ValueError(f"P9.5 safety failed: {safety_checks}")
        envelope.update(
            {
                "status": "PASS",
                "completed_at": now(),
                "shock_class": shock_class,
                "p10_recovery_priority": recovery_priority,
                "p9_core_architecture_complete": True,
                "p9_architecture": "KDA_MLA_TRUE_LATENT_CACHE_ZERO_ATTNRES_SITU_GLU",
                "p9_candidate": str(CANDIDATE),
                "p9_full_dev_ce": full_dev_ce,
                "delta_ce_vs_p8_zero": delta_ce,
                "p10_titan_migration_allowed": True,
                "p10_canonical_allowed": False,
                "p9_formal_status": "TITAN_FP16_COMPATIBILITY_PENDING_NATIVE_BF16",
                "wall_seconds": time.perf_counter() - started,
            }
        )
        atomic_json(FINAL, envelope)
        print("TITAN_P9_SITU_GLU=PASS", flush=True)
        print(f"P9_SHOCK_CLASS={shock_class}", flush=True)
        print("P9_CORE_ARCHITECTURE_COMPLETE=true", flush=True)
        print("P10_TITAN_MIGRATION_ALLOWED=true", flush=True)
        print("P10_CANONICAL_ALLOWED=false", flush=True)
        return 0
    except Exception as exc:
        envelope.update(
            {
                "status": "ERROR",
                "completed_at": now(),
                "exception_type": type(exc).__name__,
                "exception": str(exc),
                "traceback": traceback.format_exc(),
            }
        )
        atomic_json(FINAL, envelope)
        print(f"TITAN_P9_EXCEPTION={type(exc).__name__}: {exc}", flush=True)
        return 1
    finally:
        disable_p7_latent_cache_runtime()


if __name__ == "__main__":
    raise SystemExit(main())
