"""Machine-checkable contracts for the Titan P11-T migration."""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import time
import urllib.error
import urllib.request
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import torch

P11_AMENDMENT = "P11_TITAN_GENERAL_DUAL_TEACHER_OPD_2026_08_21"
P11_COST_AMENDMENT = "P11_COST_AMENDMENT_2026_08_21"
P11_MODE = "GENERAL_DUAL_TEACHER_ON_POLICY_DISTILLATION"
LEGACY_SIGNAL_SHARES = {"kimi": 0.40, "qwen": 0.35, "raw_lm": 0.25}
MAIN_TOKEN_SHARES = {"qwen": 0.60, "raw_lm": 0.40}
KIMI_SIGNAL_UNIT = "REVIEWED_ON_POLICY_TRAJECTORIES"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _inside(root: Path, path: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def _candidate_integrity(candidate: Path) -> dict[str, Any]:
    manifest_path = candidate / "p10-t-candidate-manifest.json"
    if not manifest_path.is_file():
        return {"manifest_present": False, "passed": False}
    manifest = _load_json(manifest_path)
    checks: dict[str, bool] = {
        "manifest_present": True,
        "manifest_pass": manifest.get("status") == "PASS",
        "decision_pass": manifest.get("decision") == "PASS",
        "heldout_not_accessed": manifest.get("heldout_accessed") is False,
        "runtime_lora_absent": manifest.get("lora_present_at_runtime") is False,
    }
    file_reports: dict[str, Any] = {}
    for name, expected_hash in manifest.get("files", {}).items():
        path = candidate / name
        actual_hash = _sha256(path) if path.is_file() else None
        file_reports[name] = {
            "present": path.is_file(),
            "expected_sha256": expected_hash,
            "actual_sha256": actual_hash,
            "hash_exact": actual_hash == expected_hash,
        }
    checks["declared_files_exact"] = bool(file_reports) and all(report["hash_exact"] for report in file_reports.values())
    return {
        "manifest_present": True,
        "manifest": str(manifest_path),
        "files": file_reports,
        "checks": checks,
        "passed": all(checks.values()),
    }


def _qwen_teacher(root: Path) -> dict[str, Any]:
    directory = root / "models/Qwen3.5-4B-Base"
    index_path = directory / "model.safetensors.index.json"
    tokenizer_path = directory / "tokenizer.json"
    index = _load_json(index_path) if index_path.is_file() else {}
    expected_shards = sorted(set(index.get("weight_map", {}).values()))
    shards = [
        {
            "name": name,
            "present": (directory / name).is_file(),
            "bytes": (directory / name).stat().st_size if (directory / name).is_file() else None,
        }
        for name in expected_shards
    ]
    checks = {
        "index_present": index_path.is_file(),
        "weight_map_present": bool(index.get("weight_map")),
        "all_shards_present": bool(shards) and all(item["present"] for item in shards),
        "tokenizer_present": tokenizer_path.is_file(),
    }
    return {
        "directory": str(directory),
        "expected_shards": len(expected_shards),
        "shards": shards,
        "tokenizer": str(tokenizer_path),
        "checks": checks,
        "eligible": all(checks.values()),
    }


def _kimi_teacher(root: Path) -> dict[str, Any]:
    reference = root / "references/Kimi-K3-HF-reference"
    reference_index_path = reference / "model.safetensors.index.json"
    reference_index = _load_json(reference_index_path) if reference_index_path.is_file() else {}
    expected_shards = sorted(set(reference_index.get("weight_map", {}).values()))
    total_size = int(reference_index.get("metadata", {}).get("total_size", 0))
    endpoint_descriptor = root / "titan/configs/p11-kimi-api.json"
    endpoint = _load_json(endpoint_descriptor) if endpoint_descriptor.is_file() else {}
    credential_env = str(endpoint.get("api_key_env", "MOONSHOT_API_KEY"))
    credential_available = bool(os.environ.get(credential_env))
    disk = shutil.disk_usage(root)
    endpoint_configured = endpoint_descriptor.is_file()
    api_contract = {
        "https_endpoint": str(endpoint.get("base_url", "")).startswith("https://"),
        "official_kimi_k3_model": endpoint.get("model") == "kimi-k3",
        "runtime_text_only_contract": endpoint.get("logprobs") is False,
        "cross_tokenizer_logits_not_used": endpoint.get("teacher_signal") == "TEXT_SPACE_TRAJECTORY_ONLY",
        "secret_not_stored_in_descriptor": "api_key" not in endpoint,
    }
    checks = {
        "reference_index_present": reference_index_path.is_file(),
        "reference_weight_map_present": bool(reference_index.get("weight_map")),
        "api_only_teacher_contract": endpoint.get("teacher_loading") == "API_ONLY_NO_LOCAL_WEIGHTS",
        "api_descriptor_present": endpoint_configured,
        "api_descriptor_valid": bool(api_contract) and all(api_contract.values()),
        "api_credential_available": credential_available,
        "general_composite_scoring_available": endpoint_configured and credential_available,
        "trajectory_teacher_available": endpoint_configured and credential_available,
    }
    return {
        "reference_index": str(reference_index_path),
        "reference_only_weight_shards": len(expected_shards),
        "reference_only_total_weight_bytes": total_size,
        "project_filesystem_free_bytes": disk.free,
        "local_weight_download_forbidden": True,
        "teacher_loading": "API_ONLY_NO_LOCAL_WEIGHTS",
        "endpoint_descriptor": str(endpoint_descriptor),
        "endpoint": endpoint.get("base_url"),
        "model": endpoint.get("model"),
        "api_key_env": credential_env,
        "api_credential_available": credential_available,
        "api_contract": api_contract,
        "checks": checks,
        "eligible": all(checks.values()),
    }


def build_p11_asset_preflight(root: Path) -> dict[str, Any]:
    """Audit P11.0/P11.1 prerequisites without accessing heldout or a GPU."""
    root = root.resolve()
    p10_final_path = root / "titan/manifests/reproduction/p10-t/p10-t-final-check.json"
    p10_final = _load_json(p10_final_path) if p10_final_path.is_file() else {}
    candidate = root / "outputs/densek3-core/checkpoints/p10-t/densek3-4b-core-k3-joint-reduced-titan"
    general_corpus = root / "data/p11-general/packed/corpus-manifest.json"
    benchmark = root / "data/p11-general/eval/general-benchmark-manifest.json"
    p10_checks = {
        "final_manifest_present": p10_final_path.is_file(),
        "p10_t_pass": p10_final.get("status") == "PASS",
        "p11_titan_migration_allowed": p10_final.get("p11_titan_migration_allowed") is True,
        "canonical_p10_remains_pending": p10_final.get("canonical_p10_status") == "PENDING_NATIVE_BF16_FULL_SCOPE_RETRAIN",
        "p10_t_is_provisional": p10_final.get("p10_titan_is_canonical") is False,
    }
    paths = [p10_final_path, candidate, general_corpus, benchmark]
    if not all(_inside(root, path) for path in paths):
        raise ValueError("P11 preflight path escaped the DenseK3 project root")
    qwen = _qwen_teacher(root)
    kimi = _kimi_teacher(root)
    candidate_report = _candidate_integrity(candidate)
    corpus_checks = {
        "general_corpus_manifest_present": general_corpus.is_file(),
        "general_benchmark_manifest_present": benchmark.is_file(),
    }
    blockers: list[str] = []
    if not all(p10_checks.values()):
        blockers.append("P10_T_FROZEN_SOURCE_CONTRACT")
    if not candidate_report["passed"]:
        blockers.append("P10_T_CANDIDATE_INTEGRITY")
    if not qwen["eligible"]:
        blockers.append("QWEN_EXACT_TOKEN_TEACHER")
    if not kimi["eligible"]:
        blockers.append("KIMI_K3_TEACHER_ACCESS")
    if not all(corpus_checks.values()):
        blockers.append("P11_GENERAL_CORPUS_AND_BENCHMARK")
    status = "PASS" if not blockers else "BLOCKED_ASSET_CONTRACT"
    micro_gates_allowed = not blockers
    return {
        "stage": "P11.0-P11.1-ASSET-PREFLIGHT",
        "status": status,
        "amendment": P11_AMENDMENT,
        "mode": P11_MODE,
        "project_root": str(root),
        "p10_t_source": {"path": str(p10_final_path), "checks": p10_checks},
        "p10_t_candidate": candidate_report,
        "qwen_teacher": qwen,
        "kimi_teacher": kimi,
        "general_data": {
            "corpus_manifest": str(general_corpus),
            "benchmark_manifest": str(benchmark),
            "checks": corpus_checks,
        },
        "cost_amendment": P11_COST_AMENDMENT,
        "signal_accounting": {
            "qwen": {"unit": "SUPERVISED_TOKENS", "main_token_share": 0.60},
            "raw_lm": {"unit": "SUPERVISED_TOKENS", "main_token_share": 0.40},
            "kimi": {
                "unit": KIMI_SIGNAL_UNIT,
                "fixed_supervised_token_share_disabled": True,
                "cost_budgeted": True,
            },
        },
        "legacy_signal_shares_disabled": LEGACY_SIGNAL_SHARES,
        "heldout_accessed": False,
        # P11.0/P11.1 only unlock their two teacher micro-Gates. Training stays
        # blocked until P11.2/P11.3/P11.4 have independently passed.
        "training_allowed": False,
        "p11_2_p11_3_micro_gates_allowed": micro_gates_allowed,
        "p12_titan_migration_allowed": False,
        "blockers": blockers,
        "result_marker": "P11_T_ASSET_PREFLIGHT=PASS" if not blockers else "P11_T_ASSET_PREFLIGHT=BLOCKED",
    }


def exact_token_advantage(
    teacher_logprob: torch.Tensor,
    old_student_logprob: torch.Tensor,
    *,
    clip: float = 5.0,
) -> torch.Tensor:
    """Return the detached clipped exact-token teacher advantage."""
    if teacher_logprob.shape != old_student_logprob.shape:
        raise ValueError("Teacher and old-student log-prob shapes differ")
    if clip <= 0 or not math.isfinite(clip):
        raise ValueError("clip must be finite and positive")
    if not torch.isfinite(teacher_logprob).all() or not torch.isfinite(old_student_logprob).all():
        raise ValueError("Exact-token OPD received non-finite log probabilities")
    return (teacher_logprob - old_student_logprob).clamp(-clip, clip).detach()


def exact_token_opd_loss(
    current_student_logprob: torch.Tensor,
    advantage: torch.Tensor,
    supervised_mask: torch.Tensor,
) -> torch.Tensor:
    """Compute masked token-normalized OPD policy loss."""
    if current_student_logprob.shape != advantage.shape or advantage.shape != supervised_mask.shape:
        raise ValueError("OPD tensors must have identical shapes")
    if not torch.isfinite(current_student_logprob).all() or not torch.isfinite(advantage).all():
        raise ValueError("Exact-token OPD received non-finite tensors")
    mask = supervised_mask.to(dtype=current_student_logprob.dtype)
    denominator = mask.sum()
    if denominator.item() <= 0:
        raise ValueError("OPD supervised mask is empty")
    return -(current_student_logprob * advantage * mask).sum() / denominator


@dataclass(frozen=True)
class KimiTrajectory:
    mode: str
    corrected_text: str | None


@dataclass(frozen=True)
class KimiAPIConfig:
    base_url: str
    model: str
    api_key_env: str
    logprobs: bool
    top_logprobs: int
    reasoning_effort: str
    timeout_seconds: float
    max_retries: int
    cached_input_cny_per_million: float
    input_cny_per_million: float
    output_cny_per_million: float
    daily_budget_cny: float
    per_call_upper_bound_cny: float

    @classmethod
    def from_json(cls, path: Path) -> KimiAPIConfig:
        payload = _load_json(path)
        if "api_key" in payload:
            raise ValueError("Kimi API secret must not be stored in the project descriptor")
        config = cls(
            base_url=str(payload["base_url"]),
            model=str(payload["model"]),
            api_key_env=str(payload.get("api_key_env", "MOONSHOT_API_KEY")),
            logprobs=bool(payload.get("logprobs", True)),
            top_logprobs=int(payload.get("top_logprobs", 5)),
            reasoning_effort=str(payload.get("reasoning_effort", "low")),
            timeout_seconds=float(payload.get("timeout_seconds", 120.0)),
            max_retries=int(payload.get("max_retries", 3)),
            cached_input_cny_per_million=float(payload["cached_input_cny_per_million"]),
            input_cny_per_million=float(payload["input_cny_per_million"]),
            output_cny_per_million=float(payload["output_cny_per_million"]),
            daily_budget_cny=float(payload["daily_budget_cny"]),
            per_call_upper_bound_cny=float(payload["per_call_upper_bound_cny"]),
        )
        if not config.base_url.startswith("https://"):
            raise ValueError("Kimi API endpoint must use HTTPS")
        if config.model != "kimi-k3":
            raise ValueError("P11-T requires the kimi-k3 API model")
        if not 0 <= config.top_logprobs <= 20:
            raise ValueError("Kimi output logprobs contract is invalid")
        if config.reasoning_effort != "low":
            raise ValueError("P11-T cost contract requires Kimi K3 reasoning_effort=low")
        if config.timeout_seconds <= 0 or config.max_retries < 0:
            raise ValueError("Kimi API retry contract is invalid")
        rates = (
            config.cached_input_cny_per_million,
            config.input_cny_per_million,
            config.output_cny_per_million,
            config.daily_budget_cny,
            config.per_call_upper_bound_cny,
        )
        if any(value <= 0 or not math.isfinite(value) for value in rates):
            raise ValueError("Kimi API cost contract is invalid")
        return config


@dataclass(frozen=True)
class KimiAPIResult:
    text: str
    model: str
    finish_reason: str
    output_logprobs: tuple[dict[str, Any], ...]
    usage: dict[str, int]
    reasoning_content_present: bool
    reasoning_content_characters: int
    cost_cny: float


class KimiAPICostLedger:
    """Persist request-level API cost without prompts, responses, reasoning, or secrets."""

    def __init__(self, report_dir: Path, config: KimiAPIConfig) -> None:
        self.report_dir = report_dir
        self.config = config

    def _path(self) -> Path:
        day = datetime.now().astimezone().date().isoformat()
        return self.report_dir / f"kimi-api-cost-{day}.json"

    def _load(self) -> dict[str, Any]:
        path = self._path()
        if not path.is_file():
            return {
                "date": datetime.now().astimezone().date().isoformat(),
                "daily_budget_cny": self.config.daily_budget_cny,
                "entries": [],
            }
        return _load_json(path)

    def _write(self, payload: dict[str, Any]) -> None:
        path = self._path()
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(f".tmp-{uuid.uuid4().hex}")
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        temporary.replace(path)

    @staticmethod
    def _accounted_cost(payload: dict[str, Any]) -> float:
        return sum(
            float(entry.get("cost_cny", entry.get("reserved_upper_bound_cny", 0.0))) for entry in payload.get("entries", [])
        )

    def reserve(self, request_fingerprint: str, input_upper_tokens: int, max_output_tokens: int) -> str:
        upper = (
            input_upper_tokens * self.config.input_cny_per_million + max_output_tokens * self.config.output_cny_per_million
        ) / 1_000_000
        if upper > self.config.per_call_upper_bound_cny:
            raise RuntimeError(f"Kimi request upper bound {upper:.6f} CNY exceeds per-call contract")
        payload = self._load()
        accounted = self._accounted_cost(payload)
        if accounted + upper > self.config.daily_budget_cny:
            raise RuntimeError("Kimi daily API budget would be exceeded")
        reservation_id = uuid.uuid4().hex
        payload["entries"].append(
            {
                "reservation_id": reservation_id,
                "request_fingerprint": request_fingerprint,
                "status": "PENDING",
                "created_at": datetime.now().astimezone().isoformat(),
                "input_upper_tokens": input_upper_tokens,
                "max_output_tokens": max_output_tokens,
                "reserved_upper_bound_cny": upper,
            }
        )
        payload["accounted_cost_cny"] = accounted + upper
        self._write(payload)
        return reservation_id

    def finalize(self, reservation_id: str, usage: dict[str, int]) -> float:
        payload = self._load()
        prompt = int(usage.get("prompt_tokens", 0))
        cached = min(prompt, int(usage.get("cached_tokens", 0)))
        completion = int(usage.get("completion_tokens", 0))
        cost = (
            (prompt - cached) * self.config.input_cny_per_million
            + cached * self.config.cached_input_cny_per_million
            + completion * self.config.output_cny_per_million
        ) / 1_000_000
        for entry in payload["entries"]:
            if entry["reservation_id"] == reservation_id:
                entry.update(
                    {
                        "status": "SUCCEEDED",
                        "completed_at": datetime.now().astimezone().isoformat(),
                        "usage": usage,
                        "cost_cny": cost,
                    }
                )
                entry.pop("reserved_upper_bound_cny", None)
                break
        else:
            raise RuntimeError("Kimi cost reservation is absent")
        payload["accounted_cost_cny"] = self._accounted_cost(payload)
        self._write(payload)
        return cost

    def mark_uncertain(self, reservation_id: str, error_type: str) -> None:
        payload = self._load()
        for entry in payload["entries"]:
            if entry["reservation_id"] == reservation_id:
                entry.update(
                    {
                        "status": "UNCERTAIN_CHARGE_RESERVED",
                        "completed_at": datetime.now().astimezone().isoformat(),
                        "error_type": error_type,
                    }
                )
                break
        payload["accounted_cost_cny"] = self._accounted_cost(payload)
        self._write(payload)


class KimiAPITeacher:
    """Minimal official Kimi Chat Completions client with output-logprob auditing."""

    def __init__(self, config: KimiAPIConfig, *, ledger: KimiAPICostLedger | None = None) -> None:
        self.config = config
        self.ledger = ledger

    def _credential(self) -> str:
        value = os.environ.get(self.config.api_key_env)
        if not value:
            raise RuntimeError(f"Kimi API credential environment variable is absent: {self.config.api_key_env}")
        return value

    @staticmethod
    def _safe_http_error(error: urllib.error.HTTPError) -> RuntimeError:
        """Retain only the documented error type/message, never request headers."""
        error_type = "UNKNOWN"
        message = "UNKNOWN"
        try:
            payload = json.loads(error.read().decode("utf-8", "replace"))
            api_error = payload.get("error", {})
            if isinstance(api_error, dict):
                error_type = str(api_error.get("type", error_type))
                message = str(api_error.get("message", message))
        except (AttributeError, json.JSONDecodeError, OSError, TypeError, UnicodeError):
            pass
        message = " ".join(message.split())[:500]
        return RuntimeError(f"Kimi API HTTP {error.code} {error_type}: {message}")

    @staticmethod
    def _extract_logprobs(choice: dict[str, Any]) -> tuple[dict[str, Any], ...]:
        # Kimi documents message.logprobs; OpenAI-compatible servers commonly use
        # choice.logprobs. Accept both, but never convert these tokens into Qwen IDs.
        container = choice.get("logprobs") or choice.get("message", {}).get("logprobs")
        if not isinstance(container, dict):
            raise ValueError("Kimi API response omitted requested output logprobs")
        content = container.get("content")
        if not isinstance(content, list) or not content:
            raise ValueError("Kimi API response contains no output-token logprobs")
        cleaned: list[dict[str, Any]] = []
        for item in content:
            token = item.get("token")
            logprob = item.get("logprob")
            if not isinstance(token, str) or not isinstance(logprob, (int, float)) or not math.isfinite(logprob):
                raise ValueError("Kimi API returned malformed output-token logprob")
            top = item.get("top_logprobs", [])
            if not isinstance(top, list):
                raise ValueError("Kimi API returned malformed top_logprobs")
            cleaned_top: list[dict[str, Any]] = []
            for candidate in top:
                candidate_token = candidate.get("token")
                candidate_logprob = candidate.get("logprob")
                if (
                    not isinstance(candidate_token, str)
                    or not isinstance(candidate_logprob, (int, float))
                    or not math.isfinite(candidate_logprob)
                ):
                    raise ValueError("Kimi API returned malformed candidate logprob")
                cleaned_top.append({"token": candidate_token, "logprob": float(candidate_logprob)})
            cleaned.append({"token": token, "logprob": float(logprob), "top_logprobs": cleaned_top})
        return tuple(cleaned)

    def complete(
        self,
        messages: list[dict[str, str]],
        *,
        max_completion_tokens: int,
        response_format: dict[str, Any] | None = None,
    ) -> KimiAPIResult:
        if max_completion_tokens <= 0:
            raise ValueError("max_completion_tokens must be positive")
        payload = {
            "model": self.config.model,
            "messages": messages,
            "max_completion_tokens": max_completion_tokens,
            "reasoning_effort": self.config.reasoning_effort,
            "stream": False,
        }
        if response_format is not None:
            if response_format.get("type") not in {"json_object", "json_schema"}:
                raise ValueError("Unsupported Kimi response_format contract")
            payload["response_format"] = response_format
        if self.config.logprobs:
            payload.update({"logprobs": True, "top_logprobs": self.config.top_logprobs})
        request = urllib.request.Request(
            self.config.base_url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self._credential()}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        serialized_payload = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
        request_fingerprint = hashlib.sha256(
            json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
        ).hexdigest()
        reservation_id = (
            self.ledger.reserve(request_fingerprint, len(serialized_payload), max_completion_tokens)
            if self.ledger is not None
            else None
        )
        last_error: Exception | None = None
        for attempt in range(self.config.max_retries + 1):
            try:
                with urllib.request.urlopen(request, timeout=self.config.timeout_seconds) as response:
                    body = json.loads(response.read().decode("utf-8"))
                choice = body["choices"][0]
                text = choice["message"]["content"]
                if not isinstance(text, str) or not text.strip():
                    raise ValueError("Kimi API returned empty text")
                usage = {key: int(value) for key, value in body.get("usage", {}).items() if isinstance(value, int)}
                prompt_details = body.get("usage", {}).get("prompt_tokens_details", {})
                if isinstance(prompt_details, dict) and isinstance(prompt_details.get("cached_tokens"), int):
                    usage["cached_tokens"] = int(prompt_details["cached_tokens"])
                completion_details = body.get("usage", {}).get("completion_tokens_details", {})
                if isinstance(completion_details, dict) and isinstance(completion_details.get("reasoning_tokens"), int):
                    usage["reasoning_tokens"] = int(completion_details["reasoning_tokens"])
                reasoning = choice.get("message", {}).get("reasoning_content")
                cost = self.ledger.finalize(reservation_id, usage) if self.ledger is not None else 0.0
                return KimiAPIResult(
                    text=text,
                    model=str(body.get("model", self.config.model)),
                    finish_reason=str(choice.get("finish_reason", "unknown")),
                    output_logprobs=(self._extract_logprobs(choice) if self.config.logprobs else ()),
                    usage=usage,
                    reasoning_content_present=isinstance(reasoning, str) and bool(reasoning),
                    reasoning_content_characters=len(reasoning) if isinstance(reasoning, str) else 0,
                    cost_cny=cost,
                )
            except urllib.error.HTTPError as error:
                last_error = self._safe_http_error(error)
                if attempt == self.config.max_retries:
                    break
                time.sleep(min(2**attempt, 8))
            except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
                last_error = error
                if attempt == self.config.max_retries:
                    break
                time.sleep(min(2**attempt, 8))
        if self.ledger is not None and reservation_id is not None:
            self.ledger.mark_uncertain(reservation_id, type(last_error).__name__)
        raise RuntimeError(f"Kimi API request failed after retries: {type(last_error).__name__}: {last_error}") from last_error

    def trajectory(self, prompt: str, student_text: str) -> tuple[KimiTrajectory, KimiAPIResult]:
        system = (
            "You are the Kimi K3 trajectory teacher. Compare the student response with the user prompt. "
            "Reply with exactly KEEP if it is already correct and adequate. Otherwise reply with "
            "CORRECTED_RESPONSE on the first line and a corrected semantic response after it. "
            "Never emit tokenizer IDs, special tokens, XTML, or commentary about this protocol."
        )
        user = f"PROMPT:\n{prompt}\n\nSTUDENT_RESPONSE:\n{student_text}"
        result = self.complete(
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            max_completion_tokens=512,
        )
        return parse_kimi_trajectory(result.text), result

    @staticmethod
    def structured_trajectory_contract(prompt: str, student_text: str) -> tuple[list[dict[str, str]], dict[str, Any]]:
        """Build the short, strict, content-only P11 cost-calibration request."""
        system = (
            "Review the student answer. If materially correct, set decision to keep "
            "and answer to an empty string. Otherwise set decision to correct and put "
            "only the corrected answer in answer. Do not provide critique."
        )
        user = f"PROMPT:\n{prompt}\n\nSTUDENT ANSWER:\n{student_text}"
        response_format: dict[str, Any] = {
            "type": "json_schema",
            "json_schema": {
                "name": "p11_trajectory_review",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "decision": {
                            "type": "string",
                            "enum": ["keep", "correct"],
                        },
                        "answer": {"type": "string"},
                    },
                    "required": ["decision", "answer"],
                    "additionalProperties": False,
                },
            },
        }
        return (
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            response_format,
        )

    def structured_trajectory(
        self,
        prompt: str,
        student_text: str,
        *,
        max_completion_tokens: int = 512,
    ) -> tuple[KimiTrajectory, KimiAPIResult]:
        messages, response_format = self.structured_trajectory_contract(prompt, student_text)
        result = self.complete(
            messages,
            max_completion_tokens=max_completion_tokens,
            response_format=response_format,
        )
        try:
            trajectory = parse_kimi_structured_trajectory(result.text)
        except ValueError as error:
            raise ValueError(
                "Kimi structured trajectory parse failed: "
                f"finish_reason={result.finish_reason} "
                f"completion_tokens={result.usage.get('completion_tokens')}"
            ) from error
        return trajectory, result


def parse_kimi_trajectory(response: str) -> KimiTrajectory:
    """Parse Kimi text-space output without accepting tokenizer IDs or control tokens."""
    text = response.strip()
    if text == "KEEP":
        return KimiTrajectory(mode="KEEP", corrected_text=None)
    prefix = "CORRECTED_RESPONSE\n"
    if not text.startswith(prefix):
        raise ValueError("Kimi response must be KEEP or CORRECTED_RESPONSE followed by text")
    corrected = text[len(prefix) :].strip()
    if not corrected:
        raise ValueError("CORRECTED_RESPONSE text is empty")
    forbidden = ("<|", "|>", "XTML", "token_id", "token IDs")
    if any(marker in corrected for marker in forbidden):
        raise ValueError("Kimi corrected response contains tokenizer/control-marker leakage")
    return KimiTrajectory(mode="CORRECTED_RESPONSE", corrected_text=corrected)


def parse_kimi_structured_trajectory(response: str) -> KimiTrajectory:
    """Parse the strict JSON-schema response while discarding all reasoning."""
    try:
        payload = json.loads(response)
    except json.JSONDecodeError as error:
        raise ValueError("Kimi structured trajectory is not valid JSON") from error
    if not isinstance(payload, dict) or set(payload) != {"decision", "answer"}:
        raise ValueError("Kimi structured trajectory has unexpected fields")
    decision = payload["decision"]
    answer = payload["answer"]
    if decision not in {"keep", "correct"} or not isinstance(answer, str):
        raise ValueError("Kimi structured trajectory has invalid values")
    if decision == "keep":
        if answer:
            raise ValueError("Kimi keep trajectory must have an empty answer")
        return KimiTrajectory(mode="KEEP", corrected_text=None)
    corrected = answer.strip()
    if not corrected:
        raise ValueError("Kimi corrected structured trajectory is empty")
    forbidden = ("<|", "|>", "XTML", "token_id", "token IDs")
    if any(marker in corrected for marker in forbidden):
        raise ValueError("Kimi corrected response contains tokenizer/control-marker leakage")
    return KimiTrajectory(mode="CORRECTED_RESPONSE", corrected_text=corrected)


class SupervisedTokenRouter:
    """Choose the most underfilled signal by supervised-token share."""

    def __init__(self, shares: dict[str, float] | None = None) -> None:
        self.shares = dict(MAIN_TOKEN_SHARES if shares is None else shares)
        if not self.shares or any(value <= 0 for value in self.shares.values()):
            raise ValueError("Signal shares must be positive")
        if not math.isclose(sum(self.shares.values()), 1.0, abs_tol=1e-9):
            raise ValueError("Signal shares must sum to one")
        self.tokens = dict.fromkeys(self.shares, 0)

    def choose(self) -> str:
        total = sum(self.tokens.values())
        # The first decision is deterministic and follows the declared priority.
        projected = max(total, 1)
        deficits = {name: self.shares[name] * projected - self.tokens[name] for name in self.shares}
        return max(self.shares, key=lambda name: (deficits[name], self.shares[name]))

    def record(self, signal: str, supervised_tokens: int) -> None:
        if signal not in self.tokens:
            raise KeyError(signal)
        if supervised_tokens <= 0:
            raise ValueError("supervised_tokens must be positive")
        self.tokens[signal] += int(supervised_tokens)

    def report(self) -> dict[str, Any]:
        total = sum(self.tokens.values())
        actual = {name: (count / total if total else 0.0) for name, count in self.tokens.items()}
        return {
            "unit": "SUPERVISED_TOKENS",
            "target": self.shares,
            "tokens": self.tokens,
            "actual": actual,
            "total": total,
        }
