"""Runtime helpers for Titan-only P11 micro-Gates."""

from __future__ import annotations

import gc
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from safetensors.torch import load_file

ROOT = Path(__file__).resolve().parents[2]
TITAN = ROOT / "titan"
SCRIPTS = TITAN / "scripts"
P10_T_CANDIDATE = Path(
    os.environ.get(
        "DENSEK3_P10_PATH",
        ROOT / "outputs/densek3-core/checkpoints/p10-t/densek3-4b-core-k3-joint-reduced-titan",
    )
).resolve()
TOKENIZER = Path(os.environ.get("DENSEK3_TOKENIZER_PATH", ROOT / "models/Qwen3.5-4B-Base")).resolve()
QWEN = Path(os.environ.get("DENSEK3_QWEN_PATH", ROOT / "models/Qwen3.5-4B-Base")).resolve()


def release_cuda(*values: object) -> None:
    del values
    gc.collect()
    torch.cuda.empty_cache()


def load_p10_t_student() -> tuple[torch.nn.Module, torch.nn.Module, dict[str, Any]]:
    if str(SCRIPTS) not in sys.path:
        sys.path.insert(0, str(SCRIPTS))
    import run_p10_t

    runner = run_p10_t.runner
    model, attnres, report = runner.load_p9_stack()
    runner.apply_dense_overrides(model, P10_T_CANDIDATE / "model-dense-overrides.safetensors")
    attnres.load_state_dict(load_file(str(P10_T_CANDIDATE / "attnres.safetensors")))
    model.eval()
    attnres.eval()
    return model, attnres, {"source": str(P10_T_CANDIDATE), "load": report}


def force_qwen_torch_gdn_backend(model: torch.nn.Module) -> dict[str, Any]:
    """Replace only Qwen teacher GDN kernel handles with HF's torch reference."""
    from transformers.models.qwen3_5.modeling_qwen3_5 import (
        Qwen3_5GatedDeltaNet,
        torch_chunk_gated_delta_rule,
        torch_recurrent_gated_delta_rule,
    )

    count = 0
    for module in model.modules():
        if isinstance(module, Qwen3_5GatedDeltaNet):
            module.chunk_gated_delta_rule = torch_chunk_gated_delta_rule
            module.recurrent_gated_delta_rule = torch_recurrent_gated_delta_rule
            count += 1
    if count != 24:
        raise ValueError(f"Expected 24 Qwen GDN layers, found {count}")
    return {
        "mode": "HF_TORCH_GDN_REFERENCE",
        "gdn_layers": count,
        "weights_modified": False,
        "canonical_densek3_code_modified": False,
        "reason": "TITAN_SM75_FLA_GATED_DELTA_TRITON_UNSUPPORTED",
    }


@torch.inference_mode()
def student_rollout(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    *,
    new_tokens: int,
    seed: int,
) -> torch.Tensor:
    if new_tokens <= 0:
        raise ValueError("new_tokens must be positive")
    values = input_ids
    generator = torch.Generator(device=values.device).manual_seed(seed)
    for _ in range(new_tokens):
        hidden = model.model(values, use_cache=False, return_dict=True).last_hidden_state[:, -1]
        logits = F.linear(hidden.float(), model.lm_head.weight.float())
        probabilities = torch.softmax(logits, dim=-1)
        next_token = torch.multinomial(probabilities, 1, generator=generator)
        values = torch.cat((values, next_token), dim=1)
    return values


def densek3_token_logprobs(
    model: torch.nn.Module,
    token_ids: torch.Tensor,
    *,
    target_start: int,
) -> torch.Tensor:
    if target_start <= 0 or target_start >= token_ids.shape[1]:
        raise ValueError("target_start is outside the causal sequence")
    hidden = model.model(token_ids, use_cache=False, return_dict=True).last_hidden_state
    predictors = hidden[:, target_start - 1 : -1]
    targets = token_ids[:, target_start:]
    logits = F.linear(predictors.float(), model.lm_head.weight.float())
    return torch.log_softmax(logits, dim=-1).gather(-1, targets.unsqueeze(-1)).squeeze(-1)


@torch.inference_mode()
def qwen_token_logprobs(
    model: torch.nn.Module,
    token_ids: torch.Tensor,
    *,
    target_start: int,
) -> torch.Tensor:
    logits = model(input_ids=token_ids, use_cache=False, return_dict=True).logits
    predictors = logits[:, target_start - 1 : -1].float()
    targets = token_ids[:, target_start:]
    return torch.log_softmax(predictors, dim=-1).gather(-1, targets.unsqueeze(-1)).squeeze(-1)


def token_hash(token_ids: torch.Tensor) -> str:
    values = token_ids.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(values.dtype).encode())
    digest.update(str(tuple(values.shape)).encode())
    digest.update(values.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def load_first_training_prompt(domain: str = "GENERAL_KNOWLEDGE") -> dict[str, Any]:
    path = ROOT / "data/p11-general/prompts/general-train-prompts-v1.jsonl"
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            if record["domain"] == domain:
                return record
    raise ValueError(f"P11 prompt pool contains no record for {domain}")


def scope_parameter_names(scope: Any) -> dict[int, str]:
    names: dict[int, str] = {}
    for handle in scope.lora:
        names[id(handle.module.lora_A)] = f"lora.{handle.name}.A"
        names[id(handle.module.lora_B)] = f"lora.{handle.name}.B"
    names.update((id(parameter), f"direct_master.model.{name}") for name, parameter in scope.direct_model.items())
    names.update((id(parameter), f"direct_master.attnres.{name}") for name, parameter in scope.direct_attnres.items())
    return names
