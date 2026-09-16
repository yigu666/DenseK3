"""Hugging Face compatible configuration for the DenseK3 text model."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from transformers import PretrainedConfig

P4_GQA_INDICES = (3, 7, 11, 15, 19, 23, 27, 31)
P4_MIXER_TYPES = tuple("gqa" if index in P4_GQA_INDICES else "kda" for index in range(32))
P6_MLA_INDICES = P4_GQA_INDICES
P6_MIXER_TYPES = tuple("mla" if index in P6_MLA_INDICES else "kda" for index in range(32))
P4_LAYER_TYPES = tuple(
    "full_attention" if mixer_type == "gqa" else "linear_attention" for mixer_type in P4_MIXER_TYPES
)


class DenseK3Config(PretrainedConfig):
    """Configuration for the locked 24-KDA/8-GQA P4/P5 model topology."""

    model_type = "densek3"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size: int = 248_320,
        hidden_size: int = 2_560,
        intermediate_size: int = 9_216,
        num_hidden_layers: int = 32,
        mixer_types: list[str] | tuple[str, ...] | None = None,
        layer_types: list[str] | tuple[str, ...] | None = None,
        rms_norm_eps: float = 1e-6,
        initializer_range: float = 0.02,
        max_position_embeddings: int = 262_144,
        num_attention_heads: int = 16,
        num_key_value_heads: int = 4,
        head_dim: int = 256,
        attention_dropout: float = 0.0,
        attention_bias: bool = False,
        attn_output_gate: bool = True,
        mla_kv_lora_rank: int = 512,
        mla_q_lora_rank: int | None = None,
        mla_qk_nope_head_dim: int = 256,
        mla_qk_pos_head_dim: int = 0,
        mla_value_head_dim: int = 256,
        mla_position_encoding: str = "none",
        mla_output_gate: str = "sigmoid",
        mla_output_bias: bool = False,
        mla_cache_mode: str = "expanded_reference",
        mla_qk_norm: bool = False,
        rope_theta: float = 10_000_000.0,
        partial_rotary_factor: float = 0.25,
        mrope_interleaved: bool = True,
        mrope_section: list[int] | tuple[int, ...] = (11, 11, 10),
        kda_qk_num_heads: int = 16,
        kda_value_num_heads: int = 32,
        kda_qk_head_dim: int = 128,
        kda_value_head_dim: int = 128,
        kda_conv_kernel_size: int = 4,
        kda_decay_projection_rank: int = 128,
        kda_gate_mode: str = "qwen_compat",
        kda_gate_lower_bound: float | None = None,
        kda_output_gate_activation: str = "silu",
        kda_output_gate_alpha: float | None = None,
        kda_backend: str = "fla",
        recurrent_state_dtype: str = "float32",
        recurrent_state_layout: str = "v_first",
        hidden_act: str = "silu",
        pad_token_id: int | None = None,
        bos_token_id: int | None = None,
        eos_token_id: int | list[int] | None = 248_044,
        tie_word_embeddings: bool = True,
        use_cache: bool = True,
        source_model: str = "Qwen3.5-4B-Base",
        source_revision: str = "1001bb4d826a52d1f399e183466143f4da7b741b",
        p3_numerics_amendment: str = "P3-NUM-2026-08-10-01",
        **kwargs: Any,
    ) -> None:
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        if mixer_types is None:
            if num_hidden_layers == 32:
                mixer_types = P4_MIXER_TYPES
            elif layer_types is not None:
                mixer_types = tuple("kda" if value == "linear_attention" else "gqa" for value in layer_types)
            else:
                raise ValueError("Non-production DenseK3 configs must define mixer_types")
        self.mixer_types = list(mixer_types)
        self.layer_types = list(
            layer_types
            or ("linear_attention" if value == "kda" else "full_attention" for value in self.mixer_types)
        )
        self.rms_norm_eps = rms_norm_eps
        self.initializer_range = initializer_range
        self.max_position_embeddings = max_position_embeddings
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.attention_dropout = attention_dropout
        self.attention_bias = attention_bias
        self.attn_output_gate = attn_output_gate
        self.mla_kv_lora_rank = mla_kv_lora_rank
        self.mla_q_lora_rank = mla_q_lora_rank
        self.mla_qk_nope_head_dim = mla_qk_nope_head_dim
        self.mla_qk_pos_head_dim = mla_qk_pos_head_dim
        self.mla_value_head_dim = mla_value_head_dim
        self.mla_position_encoding = mla_position_encoding
        self.mla_output_gate = mla_output_gate
        self.mla_output_bias = mla_output_bias
        self.mla_cache_mode = mla_cache_mode
        self.mla_qk_norm = mla_qk_norm
        self.rope_theta = rope_theta
        self.partial_rotary_factor = partial_rotary_factor
        self.mrope_interleaved = mrope_interleaved
        self.mrope_section = list(mrope_section)
        self.kda_qk_num_heads = kda_qk_num_heads
        self.kda_value_num_heads = kda_value_num_heads
        self.kda_qk_head_dim = kda_qk_head_dim
        self.kda_value_head_dim = kda_value_head_dim
        self.kda_conv_kernel_size = kda_conv_kernel_size
        self.kda_decay_projection_rank = kda_decay_projection_rank
        self.kda_gate_mode = kda_gate_mode
        self.kda_gate_lower_bound = kda_gate_lower_bound
        self.kda_output_gate_activation = kda_output_gate_activation
        self.kda_output_gate_alpha = kda_output_gate_alpha
        self.kda_backend = kda_backend
        self.recurrent_state_dtype = recurrent_state_dtype
        self.recurrent_state_layout = recurrent_state_layout
        self.hidden_act = hidden_act
        self.use_cache = use_cache
        self.source_model = source_model
        self.source_revision = source_revision
        self.p3_numerics_amendment = p3_numerics_amendment
        self.architectures = ["DenseK3ForCausalLM"]
        self.auto_map = {
            "AutoConfig": "densek3_core.modeling.configuration_densek3.DenseK3Config",
            "AutoModel": "densek3_core.modeling.modeling_densek3.DenseK3Model",
            "AutoModelForCausalLM": "densek3_core.modeling.modeling_densek3.DenseK3ForCausalLM",
        }
        self.validate_architecture()

    @property
    def gqa_indices(self) -> list[int]:
        return [index for index, mixer_type in enumerate(self.mixer_types) if mixer_type == "gqa"]

    @property
    def kda_indices(self) -> list[int]:
        return [index for index, mixer_type in enumerate(self.mixer_types) if mixer_type == "kda"]

    @property
    def mla_indices(self) -> list[int]:
        return [index for index, mixer_type in enumerate(self.mixer_types) if mixer_type == "mla"]

    @property
    def rotary_dim(self) -> int:
        return int(self.head_dim * self.partial_rotary_factor)

    @property
    def kda_parameterization(self) -> str:
        if self.kda_gate_mode == "qwen_compat":
            return "p4_qwen_compat"
        if self.kda_gate_mode == "kimi_k3" and self.kda_output_gate_activation == "silu":
            return "p5_k3_decay_silu"
        if self.kda_gate_mode == "kimi_k3" and self.kda_output_gate_activation == "sigmoid":
            return "p5_k3_style"
        raise ValueError(f"Unknown KDA gate mode: {self.kda_gate_mode}")

    def validate_architecture(self) -> None:
        if self.num_hidden_layers != len(self.mixer_types) or self.num_hidden_layers != len(self.layer_types):
            raise ValueError("num_hidden_layers must equal len(mixer_types) and len(layer_types)")
        if any(mixer_type not in {"kda", "gqa", "mla"} for mixer_type in self.mixer_types):
            raise ValueError("DenseK3 mixer types must be kda, gqa, or mla")
        expected_layer_types = [
            "linear_attention" if value == "kda" else "full_attention" for value in self.mixer_types
        ]
        if self.layer_types != expected_layer_types:
            raise ValueError("Transformers layer_types must correspond exactly to DenseK3 mixer_types")
        if self.num_hidden_layers == 32 and tuple(self.mixer_types) not in {P4_MIXER_TYPES, P6_MIXER_TYPES}:
            raise ValueError(
                "The production topology must be the locked 24-KDA/8-GQA pattern "
                "or its 24-KDA/8-MLA P6 replacement"
            )
        if self.num_attention_heads % self.num_key_value_heads:
            raise ValueError("GQA query heads must be divisible by KV heads")
        if self.kda_value_num_heads // self.kda_qk_num_heads != 2:
            raise ValueError("P4 requires the locked 2:1 KDA grouped-value mapping")
        parameterization = (
            self.kda_gate_mode,
            self.kda_gate_lower_bound,
            self.kda_output_gate_activation,
            self.kda_output_gate_alpha,
        )
        allowed_parameterizations = {
            ("qwen_compat", None, "silu", None),
            ("kimi_k3", -5.0, "silu", None),
            ("kimi_k3", -5.0, "sigmoid", None),
        }
        if parameterization not in allowed_parameterizations:
            raise ValueError(
                "KDA parameterization must be exactly P4 "
                "(qwen_compat, lower_bound=None, output_gate=silu, alpha=None) or P5 "
                "(kimi_k3, lower_bound=-5.0, output_gate=silu|sigmoid, alpha=None)"
            )
        if self.recurrent_state_dtype != "float32" or self.recurrent_state_layout != "v_first":
            raise ValueError("P4 requires FP32 V-first recurrent states")
        if self.hidden_act != "silu":
            raise ValueError("P4 must retain Qwen SwiGLU")
        if not self.tie_word_embeddings:
            raise ValueError("P4 requires tied embedding and LM-head weights")
        if self.rotary_dim % 2:
            raise ValueError("The partial rotary dimension must be even")
        if self.mla_indices:
            mla_contract = (
                self.mla_kv_lora_rank,
                self.mla_q_lora_rank,
                self.mla_qk_nope_head_dim,
                self.mla_qk_pos_head_dim,
                self.mla_value_head_dim,
                self.mla_position_encoding,
                self.mla_output_gate,
                self.mla_output_bias,
                self.mla_cache_mode,
                self.mla_qk_norm,
            )
            expected_mla_contract = (
                512,
                None,
                256,
                0,
                256,
                "none",
                "sigmoid",
                False,
                "expanded_reference",
                False,
            )
            preserved_qk_norm_contract = (*expected_mla_contract[:-1], True)
            generic_mla_valid = (
                self.mla_kv_lora_rank > 0
                and self.mla_q_lora_rank is None
                and self.mla_qk_nope_head_dim == self.head_dim
                and self.mla_qk_pos_head_dim == 0
                and self.mla_value_head_dim == self.head_dim
                and self.mla_position_encoding == "none"
                and self.mla_output_gate == "sigmoid"
                and self.mla_output_bias is False
                and self.mla_cache_mode in {"expanded_reference", "true_latent"}
                and isinstance(self.mla_qk_norm, bool)
            )
            # P6 used an expanded reference cache; the released P11.6 runtime
            # stores the same normalized KV latent in a true-latent cache.  The
            # projection/NoPE contract is identical, so validate the cache mode
            # separately rather than rejecting the public release config.
            cache_mode_valid = self.mla_cache_mode in {"expanded_reference", "true_latent"}
            mla_contract_for_validation = (
                *mla_contract[:8],
                "expanded_reference",
                mla_contract[9],
            )
            production_contracts = {expected_mla_contract, preserved_qk_norm_contract}
            if not generic_mla_valid or (
                not cache_mode_valid
                or (self.num_hidden_layers == 32 and mla_contract_for_validation not in production_contracts)
            ):
                raise ValueError("P6 MLA must use the locked rank-512 strict-NoPE expanded-cache contract")
            if self.num_hidden_layers == 32 and self.kda_parameterization != "p5_k3_decay_silu":
                raise ValueError("P6 MLA scale-out must retain the frozen P5 K3-decay plus SiLU KDA contract")

    def validate_p4(self) -> None:
        """Validate the compatibility parameterization expected by frozen P4 code."""
        self.validate_architecture()
        if self.kda_parameterization != "p4_qwen_compat":
            raise ValueError("P4 must use qwen_compat decay and the donor SiLU output gate")

    @classmethod
    def from_qwen_model_directory(cls, model_dir: str | Path) -> DenseK3Config:
        source = json.loads((Path(model_dir) / "config.json").read_text(encoding="utf-8"))
        text = source["text_config"]
        rope = text["rope_parameters"]
        source_types = text["layer_types"]
        mixer_types = ["kda" if value == "linear_attention" else "gqa" for value in source_types]
        return cls(
            vocab_size=int(text["vocab_size"]),
            hidden_size=int(text["hidden_size"]),
            intermediate_size=int(text["intermediate_size"]),
            num_hidden_layers=int(text["num_hidden_layers"]),
            mixer_types=mixer_types,
            layer_types=source_types,
            rms_norm_eps=float(text["rms_norm_eps"]),
            max_position_embeddings=int(text["max_position_embeddings"]),
            num_attention_heads=int(text["num_attention_heads"]),
            num_key_value_heads=int(text["num_key_value_heads"]),
            head_dim=int(text["head_dim"]),
            attention_dropout=float(text["attention_dropout"]),
            attention_bias=bool(text["attention_bias"]),
            attn_output_gate=bool(text["attn_output_gate"]),
            rope_theta=float(rope["rope_theta"]),
            partial_rotary_factor=float(rope["partial_rotary_factor"]),
            mrope_interleaved=bool(rope["mrope_interleaved"]),
            mrope_section=list(rope["mrope_section"]),
            kda_qk_num_heads=int(text["linear_num_key_heads"]),
            kda_value_num_heads=int(text["linear_num_value_heads"]),
            kda_qk_head_dim=int(text["linear_key_head_dim"]),
            kda_value_head_dim=int(text["linear_value_head_dim"]),
            kda_conv_kernel_size=int(text["linear_conv_kernel_dim"]),
            eos_token_id=text.get("eos_token_id"),
        )
