"""Fresh-process save/reload/forward/cache/generation probe for P4.7."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from densek3_core.evaluation.p4_runtime import load_densek3_model_from_pretrained
from densek3_core.transplant.full_model import verify_p4_artifact


def save_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def tensor_sha256(tensor: torch.Tensor) -> str:
    import hashlib

    value = tensor.detach().cpu().contiguous().view(torch.uint8)
    return hashlib.sha256(value.numpy().tobytes()).hexdigest()


def load_prompts(path: Path) -> list[dict[str, str]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def observe(model: torch.nn.Module, tokenizer: Any, prompts: list[dict[str, str]]) -> dict[str, Any]:
    device = next(model.parameters()).device
    generations = []
    for item in prompts:
        encoded = tokenizer(item["prompt"], return_tensors="pt")
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)
        generated = model.greedy_generate(input_ids, attention_mask=attention_mask, max_new_tokens=8)
        new_tokens = generated[:, input_ids.shape[1] :]
        generations.append(
            {
                "id": item["id"],
                "prompt": item["prompt"],
                "input_ids": input_ids.cpu().tolist(),
                "generated_ids": generated.cpu().tolist(),
                "new_token_count": new_tokens.shape[1],
                "unique_new_token_fraction": (
                    len(set(new_tokens.flatten().cpu().tolist())) / max(new_tokens.numel(), 1)
                ),
                "text": tokenizer.decode(generated[0], skip_special_tokens=False),
            }
        )
    probe_ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7]], device=device)
    with torch.no_grad():
        output = model(probe_ids, use_cache=True, return_dict=True)
    return {
        "forward_logits_hash": tensor_sha256(output.logits),
        "forward_cache_seen_tokens": output.past_key_values.seen_tokens,
        "generations": generations,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report: dict[str, Any] = {"stage": "P4.7-reload-probe", "status": "FAIL"}
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("P4 reload probe requires CUDA")
        from transformers import AutoTokenizer

        artifact = verify_p4_artifact(args.checkpoint_dir, verify_tensor_hashes=True)
        tokenizer = AutoTokenizer.from_pretrained(args.checkpoint_dir, trust_remote_code=False)
        model = load_densek3_model_from_pretrained(args.checkpoint_dir, device="cuda", dtype=torch.bfloat16)
        observed = observe(model, tokenizer, load_prompts(args.prompts))
        expected = json.loads(args.baseline.read_text(encoding="utf-8"))
        forward_match = (
            observed["forward_logits_hash"] == expected["forward_logits_hash"]
            and observed["forward_cache_seen_tokens"] == expected["forward_cache_seen_tokens"]
        )
        generation_match = [item["generated_ids"] for item in observed["generations"]] == [
            item["generated_ids"] for item in expected["generations"]
        ]
        smoke_valid = all(
            item["new_token_count"] > 0
            and item["unique_new_token_fraction"] > 0.125
            and len(item["text"]) > 0
            for item in observed["generations"]
        )
        passed = forward_match and generation_match and smoke_valid
        report.update(
            {
                "status": "PASS" if passed else "FAIL",
                "artifact_verification": artifact,
                "forward_hash_match": forward_match,
                "generation_match": generation_match,
                "generation_smoke_valid": smoke_valid,
                "observed": observed,
            }
        )
    except Exception as exc:  # noqa: BLE001 - subprocess must always leave a machine-readable record
        report.update({"exception_type": type(exc).__name__, "exception": str(exc)})
    save_json(args.output, report)
    print(f"P4_RELOAD_PROBE={report['status']}")
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
