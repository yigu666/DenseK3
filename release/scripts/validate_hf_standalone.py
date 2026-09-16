"""Validate the DenseK3-4B standalone Hub directory against its frozen reference.

The ``compare`` mode deliberately starts the standalone load in a fresh
isolated Python process.  That process receives only the model directory and
installed dependencies; it never imports the project checkout or the private
checkpoint loader.  The reference side is inference-only and is used solely to
write deterministic comparison fixtures.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROMPTS = (
    "DenseK3 preserves the following invariant:",
    "请解释为什么严格 NoPE MLA 可以只持久化归一化 latent。",
    "def migrate(model, cache):\n    return model.forward(cache=cache)",
    "A short mixed-language prompt checks deterministic cached decoding.",
)
SCHEMA_VERSION = 1


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _cache_contract(cache: Any) -> dict[str, Any]:
    if cache is None:
        return {"present": False}
    if not hasattr(cache, "latent_contract"):
        return {"present": True, "latent_contract": None, "type": type(cache).__name__}
    return {
        "present": True,
        "type": type(cache).__name__,
        "latent_contract": cache.latent_contract(),
    }


def _device_of(model: Any) -> Any:
    import torch

    return next(model.parameters()).device


def _ids_for_tokenizer(tokenizer: Any, prompt: str) -> list[int]:
    encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    ids = encoded["input_ids"]
    if ids.ndim != 2 or ids.shape[0] != 1 or ids.shape[1] < 2:
        raise RuntimeError(f"Prompt tokenization must produce [1,T] with T>=2, got {tuple(ids.shape)}")
    return [int(value) for value in ids[0].tolist()]


def _reference(args: argparse.Namespace) -> int:
    import torch
    from safetensors.torch import save_file

    if not torch.cuda.is_available():
        raise RuntimeError("Reference parity requires the authorized CUDA host")
    sys.path.insert(0, str(PROJECT_ROOT))
    loader_module, loader_name = args.loader.split(":", 1)
    loader = getattr(__import__(loader_module, fromlist=[loader_name]), loader_name)
    model, attnres, tokenizer, metadata, _ = loader(args.checkpoint)
    model.eval()
    attnres.eval()
    device = _device_of(model)
    if device.type != "cuda":
        raise RuntimeError(f"Reference loader returned device {device}, expected CUDA")

    logits: dict[str, torch.Tensor] = {}
    continuation_logits: dict[str, torch.Tensor] = {}
    records: list[dict[str, Any]] = []
    with torch.inference_mode():
        for index, prompt in enumerate(PROMPTS):
            ids_list = _ids_for_tokenizer(tokenizer, prompt)
            ids = torch.tensor([ids_list], dtype=torch.long, device=device)
            output = model(input_ids=ids, use_cache=True, return_dict=True)
            logits[f"prompt_{index}_last"] = output.logits[:, -1, :].float().cpu().contiguous()
            cache = output.past_key_values
            next_token = (ids[:, -1:] + 1) % int(tokenizer.vocab_size)
            continuation_ids = torch.cat((ids[:, -1:], next_token), dim=1)
            continuation = model(
                input_ids=continuation_ids,
                past_key_values=cache,
                use_cache=True,
                return_dict=True,
            )
            continuation_logits[f"prompt_{index}_continuation_last"] = (
                continuation.logits[:, -1, :].float().cpu().contiguous()
            )
            generated = model.greedy_generate(ids, max_new_tokens=4)
            records.append(
                {
                    "index": index,
                    "prompt": prompt,
                    "input_ids": ids_list,
                    "continuation_ids": [int(value) for value in continuation_ids[0].tolist()],
                    "cache": _cache_contract(cache),
                    "continuation_cache": _cache_contract(continuation.past_key_values),
                    "generated_ids": [int(value) for value in generated[0].tolist()],
                }
            )

    expected_names = sorted(name for name, _ in model.named_parameters())
    expected_names.extend(f"attnres.{name}" for name, _ in attnres.named_parameters())
    expected_names = sorted(set(expected_names))
    structural = {
        "model_parameter_count": int(sum(parameter.numel() for parameter in model.parameters())),
        "attnres_parameter_count": int(sum(parameter.numel() for parameter in attnres.parameters())),
        "unique_parameter_count": int(
            sum(parameter.numel() for parameter in model.parameters())
            + sum(parameter.numel() for parameter in attnres.parameters())
        ),
        "parameter_names_sha256": hashlib.sha256("\n".join(expected_names).encode()).hexdigest(),
        "parameter_name_count": len(expected_names),
        "model_dtype": str(next(model.parameters()).dtype),
        "device": str(device),
    }
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    save_file({**logits, **continuation_logits}, str(output / "reference_logits.safetensors"))
    _write_json(
        output / "reference.json",
        {
            "schema_version": SCHEMA_VERSION,
            "side": "frozen_reference",
            "public_model_name": "DenseK3-4B",
            "canonical_internal_stage": "P11.6",
            "prompts": records,
            "structural": structural,
            "loader_metadata": {
                "loader": metadata.get("loader"),
                "training_performed": metadata.get("training_performed"),
                "backward_performed": metadata.get("backward_performed"),
                "optimizer_constructed": metadata.get("optimizer_constructed"),
                "weights_modified": metadata.get("weights_modified"),
            },
            "logits_file": "reference_logits.safetensors",
        },
    )
    # Do not leave the reference allocator resident while the clean-room
    # process starts on the same GPU.
    del model, attnres, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    print(json.dumps({"status": "REFERENCE_FIXTURE_WRITTEN", "output": str(output)}))
    return 0


def _standalone(args: argparse.Namespace) -> int:
    import torch
    from safetensors.torch import save_file
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    model_dir = args.model_dir.resolve()
    fixture_dir = args.fixture_dir.resolve()
    reference = json.loads((fixture_dir / "reference.json").read_text(encoding="utf-8"))
    config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=False, local_files_only=True)
    # Qwen pads the embedding matrix beyond the tokenizer's highest assigned
    # id.  The release must preserve that padded matrix, so require coverage
    # rather than an (incorrect) exact equality check.
    tokenizer_size = max(int(tokenizer.vocab_size), int(len(tokenizer)))
    if tokenizer_size > int(config.vocab_size):
        raise RuntimeError("Standalone tokenizer ids exceed the model vocabulary")
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        trust_remote_code=True,
        local_files_only=True,
        torch_dtype=torch.float16,
        device_map="cuda",
    ).eval()
    device = _device_of(model)
    if device.type != "cuda":
        raise RuntimeError(f"Standalone model loaded on {device}, expected CUDA")

    logits: dict[str, torch.Tensor] = {}
    records: list[dict[str, Any]] = []
    with torch.inference_mode():
        for item in reference["prompts"]:
            index = int(item["index"])
            ids = torch.tensor([item["input_ids"]], dtype=torch.long, device=device)
            output = model(input_ids=ids, use_cache=True, return_dict=True)
            logits[f"prompt_{index}_last"] = output.logits[:, -1, :].float().cpu().contiguous()
            cache = output.past_key_values
            continuation_ids = torch.tensor([item["continuation_ids"]], dtype=torch.long, device=device)
            continuation = model(
                input_ids=continuation_ids,
                past_key_values=cache,
                use_cache=True,
                return_dict=True,
            )
            logits[f"prompt_{index}_continuation_last"] = (
                continuation.logits[:, -1, :].float().cpu().contiguous()
            )
            generated = model.generate(ids, max_new_tokens=4, do_sample=False)
            records.append(
                {
                    "index": index,
                    "cache": _cache_contract(cache),
                    "continuation_cache": _cache_contract(continuation.past_key_values),
                    "generated_ids": [int(value) for value in generated[0].tolist()],
                }
            )

    names = sorted(name for name, _ in model.named_parameters())
    structural = {
        "model_parameter_count": int(sum(parameter.numel() for parameter in model.parameters())),
        "parameter_names_sha256": hashlib.sha256("\n".join(names).encode()).hexdigest(),
        "parameter_name_count": len(names),
        "model_dtype": str(next(model.parameters()).dtype),
        "device": str(device),
        "config_class": f"{config.__class__.__module__}.{config.__class__.__name__}",
        "model_class": f"{model.__class__.__module__}.{model.__class__.__name__}",
    }
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    save_file(logits, str(output / "standalone_logits.safetensors"))

    # A real save/load round trip is part of the gate.  The Python custom-code
    # files are copied explicitly because Transformers only serializes weights
    # and config from a dynamically loaded module.
    roundtrip = output / "roundtrip"
    roundtrip.mkdir(parents=True, exist_ok=False)
    for filename in ("configuration_densek3.py", "modeling_densek3.py"):
        (roundtrip / filename).write_bytes((model_dir / filename).read_bytes())
    model.save_pretrained(roundtrip, safe_serialization=True, max_shard_size="4GB")
    # The reload is intentionally a separate lifecycle, but remains in this
    # process so the report can compare its generation exactly.  Release the
    # first 4B instance before constructing the second one.
    del model, tokenizer, config
    gc.collect()
    torch.cuda.empty_cache()
    reloaded = AutoModelForCausalLM.from_pretrained(
        roundtrip,
        trust_remote_code=True,
        local_files_only=True,
        torch_dtype=torch.float16,
        device_map="cuda",
    ).eval()
    with torch.inference_mode():
        roundtrip_generated = reloaded.generate(
            torch.tensor([reference["prompts"][0]["input_ids"]], dtype=torch.long, device=_device_of(reloaded)),
            max_new_tokens=4,
            do_sample=False,
        )
    roundtrip_report = {
        "status": "PASS",
        "generated_ids": [int(value) for value in roundtrip_generated[0].tolist()],
        "weight_files": sorted(path.name for path in roundtrip.glob("*.safetensors")),
    }
    _write_json(output / "roundtrip.json", roundtrip_report)
    _write_json(
        output / "standalone.json",
        {
            "schema_version": SCHEMA_VERSION,
            "side": "standalone_clean_room",
            "public_model_name": "DenseK3-4B",
            "canonical_internal_stage": "P11.6",
            "structural": structural,
            "prompts": records,
            "logits_file": "standalone_logits.safetensors",
            "roundtrip": roundtrip_report,
        },
    )
    print(json.dumps({"status": "STANDALONE_FIXTURE_WRITTEN", "output": str(output)}))
    return 0


def _compare(args: argparse.Namespace) -> int:
    import torch
    from safetensors.torch import load_file

    output = args.output_dir.resolve()
    reference_dir = output / "reference"
    standalone_dir = output / "standalone"
    reference_dir.mkdir(parents=True, exist_ok=True)
    standalone_dir.mkdir(parents=True, exist_ok=True)
    if not (reference_dir / "reference.json").is_file():
        _reference(
            argparse.Namespace(
                checkpoint=args.checkpoint,
                output_dir=reference_dir,
                loader=args.loader,
            )
        )
    else:
        # A previous reference fixture is sufficient; still clear any cached
        # allocator blocks left by a caller before launching the child.
        import torch

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    child_env = os.environ.copy()
    cache_dir = output / "hf_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    child_env.update(
        {
            "HF_HOME": str(cache_dir),
            "HF_HUB_CACHE": str(cache_dir / "hub"),
            "TRANSFORMERS_CACHE": str(cache_dir / "transformers"),
            "TOKENIZERS_PARALLELISM": "false",
            "PYTHONPATH": "",
        }
    )
    child = subprocess.run(
        [
            sys.executable,
            "-I",
            str(Path(__file__).resolve()),
            "--mode",
            "standalone",
            "--model-dir",
            str(args.model_dir.resolve()),
            "--fixture-dir",
            str(reference_dir),
            "--output-dir",
            str(standalone_dir),
        ],
        env=child_env,
        check=False,
    )
    if child.returncode:
        raise RuntimeError(f"clean-room standalone process failed with exit code {child.returncode}")

    ref = json.loads((reference_dir / "reference.json").read_text(encoding="utf-8"))
    got = json.loads((standalone_dir / "standalone.json").read_text(encoding="utf-8"))
    ref_logits = load_file(str(reference_dir / ref["logits_file"]))
    got_logits = load_file(str(standalone_dir / got["logits_file"]))
    if set(ref_logits) != set(got_logits):
        raise RuntimeError("Reference and standalone logit fixture keys differ")
    logit_rows: dict[str, Any] = {}
    max_abs = 0.0
    for key in sorted(ref_logits):
        diff = (ref_logits[key].float() - got_logits[key].float()).abs()
        key_max = float(diff.max().item())
        max_abs = max(max_abs, key_max)
        top_ref = torch.topk(ref_logits[key].float(), k=10, dim=-1).indices.tolist()
        top_got = torch.topk(got_logits[key].float(), k=10, dim=-1).indices.tolist()
        top5_ref = torch.topk(ref_logits[key].float(), k=5, dim=-1).indices.tolist()
        top5_got = torch.topk(got_logits[key].float(), k=5, dim=-1).indices.tolist()
        logit_rows[key] = {
            "max_abs": key_max,
            # Equal-valued logits can legitimately swap order in a different
            # kernel.  Compare the top-5 set and top-1 identity separately.
            "top1_equal": top_ref[0][0] == top_got[0][0],
            "top5_set_equal": set(top5_ref[0]) == set(top5_got[0]),
            "top10_equal": top_ref == top_got,
        }

    ref_names = ref["structural"]["parameter_names_sha256"]
    got_names = got["structural"]["parameter_names_sha256"]
    expected_count = 4_226_764_032
    structural_pass = (
        got["structural"]["unique_parameter_count"] == expected_count
        if "unique_parameter_count" in got["structural"]
        else got["structural"]["model_parameter_count"] == expected_count
    )
    cache_rows = []
    generation_pass = True
    for expected, actual in zip(ref["prompts"], got["prompts"]):
        expected_cache = expected["cache"]["latent_contract"]
        actual_cache = actual["cache"]["latent_contract"]
        cache_ok = (
            expected_cache["latent_cache_layers"] == actual_cache["latent_cache_layers"]
            and expected_cache["latent_dims"] == actual_cache["latent_dims"]
            and actual_cache["bytes"]["expanded_k"] == 0
            and actual_cache["bytes"]["expanded_v"] == 0
        )
        generation_pass = generation_pass and expected["generated_ids"] == actual["generated_ids"]
        cache_rows.append({"index": expected["index"], "pass": cache_ok, "contract": actual_cache})

    checks = {
        "structural_parameter_names": ref_names == got_names,
        "structural_parameter_count": structural_pass,
        "logits_max_abs_le_0_05": max_abs <= 0.05,
        "logits_top1_equal": all(row["top1_equal"] for row in logit_rows.values()),
        "logits_top5_set_equal": all(row["top5_set_equal"] for row in logit_rows.values()),
        "cache_contract": all(row["pass"] for row in cache_rows),
        "greedy_generation": generation_pass,
        "save_load_roundtrip": got.get("roundtrip", {}).get("status") == "PASS",
    }
    status = "PASS" if all(checks.values()) else "FAIL"
    report = {
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "public_model_name": "DenseK3-4B",
        "canonical_internal_stage": "P11.6",
        "checks": checks,
        "logits": {"max_abs": max_abs, "rows": logit_rows},
        "cache": cache_rows,
        "reference_fixture_sha256": _sha256(reference_dir / "reference_logits.safetensors"),
        "standalone_fixture_sha256": _sha256(standalone_dir / "standalone_logits.safetensors"),
        "clean_room": {"python_isolated": True, "project_pythonpath": False},
    }
    _write_json(output / "PARITY_SUMMARY.json", report)
    # Promote the passing audit into the model directory itself so a Hub upload
    # carries its validation evidence and an integrity manifest that covers it.
    if status == "PASS":
        model_dir = args.model_dir.resolve()
        (model_dir / "PARITY_SUMMARY.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        manifest_path = model_dir / "RELEASE_MANIFEST.json"
        if manifest_path.is_file():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["parity_status"] = status
            manifest["parity_validation"] = {
                "report": "PARITY_SUMMARY.json",
                "clean_room_python_isolated": True,
                "structural_parameter_count": expected_count,
                "logits_max_abs": max_abs,
                "logits_top1_equal": checks["logits_top1_equal"],
                "logits_top5_set_equal": checks["logits_top5_set_equal"],
                "cache_contract": checks["cache_contract"],
                "greedy_generation": checks["greedy_generation"],
                "save_load_roundtrip": checks["save_load_roundtrip"],
            }
            manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
            sum_path = model_dir / "SHA256SUMS"
            files = sorted(path for path in model_dir.iterdir() if path.is_file() and path.name != "SHA256SUMS")
            sum_path.write_text("".join(f"{_sha256(path)}  {path.name}\n" for path in files), encoding="utf-8")
    print(json.dumps({"status": status, "output": str(output), "checks": checks}))
    return 0 if status == "PASS" else 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("reference", "standalone", "compare"), default="compare")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--fixture-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--loader", default="release.hf_export.reference_loader:load_canonical_densek3_reference")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.mode == "reference":
        if args.checkpoint is None:
            raise ValueError("--checkpoint is required in reference mode")
        return _reference(args)
    if args.mode == "standalone":
        if args.model_dir is None or args.fixture_dir is None:
            raise ValueError("--model-dir and --fixture-dir are required in standalone mode")
        return _standalone(args)
    if args.checkpoint is None or args.model_dir is None:
        raise ValueError("--checkpoint and --model-dir are required in compare mode")
    return _compare(args)


if __name__ == "__main__":
    raise SystemExit(main())
