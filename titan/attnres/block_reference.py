"""PyTorch reference Block AttnRes runtime for Titan P8.

The frozen DenseK3 checkpoint and canonical model sources remain unchanged.
This module replaces one model instance's backbone forward method and keeps
all depth state local to the current sequence forward. It never stores depth
representations in the P7 sequence cache.
"""

from __future__ import annotations

import types
from collections.abc import Callable
from typing import Any

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from .contracts import (
    FINAL_SITE,
    FORMAL_BLOCK_COUNT,
    FORMAL_BLOCK_SIZE_RESIDUALS,
    TOTAL_ROUTING_SITES,
)
from .metrics import routing_summary, tensor_metrics
from .state import DepthBlockState


class RoutingRMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size, dtype=torch.float32))
        self.eps = float(eps)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        value32 = value.float()
        normalized = value32 * torch.rsqrt(value32.square().mean(dim=-1, keepdim=True) + self.eps)
        return normalized * self.weight


class AttnResReadSite(nn.Module):
    def __init__(self, hidden_size: int, eps: float):
        super().__init__()
        self.query = nn.Parameter(torch.zeros(hidden_size, dtype=torch.float32))
        self.source_norm = RoutingRMSNorm(hidden_size, eps)

    def forward(
        self,
        sources: tuple[torch.Tensor, ...],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not sources:
            raise ValueError("AttnRes requires at least one source")
        score_parts = []
        for source in sources:
            key = self.source_norm(source)
            score_parts.append(torch.einsum("d,btd->bt", self.query, key))
        scores = torch.stack(score_parts, dim=0)
        alpha = scores.softmax(dim=0)
        output = torch.zeros_like(sources[0])
        for source_index, source in enumerate(sources):
            output = output + alpha[source_index].unsqueeze(-1).to(source.dtype) * source
        return output, alpha, scores


class BlockAttnResReference(nn.Module):
    """65-site Block AttnRes reference module with zero-query initialization."""

    def __init__(
        self,
        hidden_size: int,
        eps: float,
        *,
        block_size_residuals: int = FORMAL_BLOCK_SIZE_RESIDUALS,
        transformer_layers: int = 32,
        total_routing_sites: int = TOTAL_ROUTING_SITES,
        final_site: int = FINAL_SITE,
        formal_block_count: int = FORMAL_BLOCK_COUNT,
    ):
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.eps = float(eps)
        self.block_size_residuals = int(block_size_residuals)
        self.transformer_layers = int(transformer_layers)
        self.total_routing_sites = int(total_routing_sites)
        self.final_site = int(final_site)
        self.formal_block_count = int(formal_block_count)
        if self.total_routing_sites != self.transformer_layers * 2 + 1:
            raise ValueError("AttnRes routing sites must be two per layer plus one final site")
        if self.final_site != self.total_routing_sites - 1:
            raise ValueError("AttnRes final site must be the last routing site")
        self.sites = nn.ModuleList(
            [AttnResReadSite(hidden_size, eps) for _ in range(self.total_routing_sites)]
        )
        self.capture_diagnostics = False
        self.capture_tensor_metrics = False
        self.capture_activations = False
        self.branch_observer: Callable[..., None] | None = None
        self.routing_records: list[dict[str, Any]] = []
        self.tensor_records: list[dict[str, Any]] = []
        self.activation_records: list[dict[str, Any]] = []

    def clear_diagnostics(self) -> None:
        self.routing_records.clear()
        self.tensor_records.clear()
        self.activation_records.clear()

    def read(
        self,
        state: DepthBlockState,
        site_index: int,
        *,
        label: str,
        standard_stream: torch.Tensor | None = None,
    ) -> torch.Tensor:
        output, alpha, scores = self.sites[site_index](state.sources())
        if self.capture_diagnostics:
            route = routing_summary(alpha)
            route.update(
                {
                    "site": site_index,
                    "label": label,
                    "scores_zero": bool(torch.equal(scores, torch.zeros_like(scores))),
                    "alpha_uniform_exact": bool(
                        torch.equal(alpha.max(dim=0).values, alpha.min(dim=0).values)
                    ),
                }
            )
            self.routing_records.append(route)
            if standard_stream is not None and self.capture_tensor_metrics:
                self.tensor_records.append(
                    {
                        "site": site_index,
                        "label": label,
                        "source_partition": tensor_metrics(
                            state.reconstruct(),
                            standard_stream,
                        ),
                        "read_vs_scaled_stream": tensor_metrics(
                            output,
                            standard_stream / len(state.sources()),
                        ),
                    }
                )
        return output

    def query_report(self) -> dict[str, Any]:
        norms = torch.stack([site.query.detach().float().norm() for site in self.sites])
        return {
            "count": int(norms.numel()),
            "mean": float(norms.mean()),
            "median": float(norms.median()),
            "max": float(norms.max()),
            "per_site": [float(value) for value in norms.cpu()],
        }

    def parameter_contract(self) -> dict[str, Any]:
        parameters = list(self.parameters())
        return {
            "site_count": len(self.sites),
            "query_count": sum(1 for name, _ in self.named_parameters() if name.endswith("query")),
            "norm_count": sum(1 for name, _ in self.named_parameters() if name.endswith("source_norm.weight")),
            "parameter_count": sum(value.numel() for value in parameters),
            "all_fp32": all(value.dtype == torch.float32 for value in parameters),
            "all_queries_zero": all(
                bool(torch.count_nonzero(site.query) == 0) for site in self.sites
            ),
            "all_norms_one": all(
                bool(torch.equal(site.source_norm.weight, torch.ones_like(site.source_norm.weight)))
                for site in self.sites
            ),
        }

    def residual_site(self, layer_index: int, branch: str) -> int:
        if layer_index < 0 or layer_index >= self.transformer_layers:
            raise ValueError(f"Invalid transformer layer: {layer_index}")
        if branch == "mixer":
            return layer_index * 2
        if branch == "mlp":
            return layer_index * 2 + 1
        raise ValueError(f"Unknown residual branch: {branch}")


def _branch(
    layer: nn.Module,
    branch: str,
    hidden: torch.Tensor,
    position_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    cache: Any,
    use_cache: bool,
) -> torch.Tensor:
    if branch == "mixer":
        normalized = layer.input_layernorm(hidden)
        if layer.layer_type == "kda":
            return layer.self_attn(normalized, attention_mask, cache, use_cache)
        return layer.self_attn(
            normalized,
            position_ids,
            attention_mask,
            cache,
            use_cache,
        )
    if branch == "mlp":
        return layer.mlp(layer.post_attention_layernorm(hidden))
    raise ValueError(branch)


def _attnres_backbone_forward(
    backbone: nn.Module,
    runtime: BlockAttnResReference,
    input_ids: torch.LongTensor | None = None,
    attention_mask: torch.Tensor | None = None,
    position_ids: torch.LongTensor | None = None,
    past_key_values: Any = None,
    inputs_embeds: torch.Tensor | None = None,
    use_cache: bool | None = None,
    output_hidden_states: bool | None = None,
    hidden_state_layers: list[int] | tuple[int, ...] | None = None,
    return_dict: bool | None = None,
    **_: Any,
) -> Any:
    from densek3_core.modeling.modeling_densek3 import DenseK3ModelOutput

    use_cache = backbone.config.use_cache if use_cache is None else use_cache
    if backbone.training and backbone.gradient_checkpointing and use_cache:
        use_cache = False
    output_hidden_states = False if output_hidden_states is None else output_hidden_states
    selected_layers = tuple(hidden_state_layers or ())
    if output_hidden_states and selected_layers:
        raise ValueError("output_hidden_states and hidden_state_layers are mutually exclusive")
    return_dict = backbone.config.use_return_dict if return_dict is None else return_dict
    if (input_ids is None) == (inputs_embeds is None):
        raise ValueError("Specify exactly one of input_ids and inputs_embeds")
    embedding = backbone.embed_tokens(input_ids) if inputs_embeds is None else inputs_embeds
    batch, sequence, _ = embedding.shape
    if use_cache and past_key_values is None:
        from densek3_core.modeling import modeling_densek3

        past_key_values = modeling_densek3.DenseK3HybridCache(backbone.config)
    past_length = 0 if past_key_values is None else past_key_values.seen_tokens
    if position_ids is None:
        position_ids = torch.arange(
            past_length,
            past_length + sequence,
            device=embedding.device,
            dtype=torch.long,
        ).unsqueeze(0).expand(batch, -1)
    if attention_mask is None:
        attention_mask = torch.ones(
            (batch, past_length + sequence),
            device=embedding.device,
        )
    state = DepthBlockState(
        embedding,
        runtime.block_size_residuals,
    )
    standard_stream = embedding if runtime.capture_tensor_metrics else None
    all_hidden: list[torch.Tensor] | None = [] if output_hidden_states or selected_layers else None
    if all_hidden is not None and output_hidden_states:
        all_hidden.append(embedding)

    for layer_index, layer in enumerate(backbone.layers):
        for branch_name in ("mixer", "mlp"):
            site = runtime.residual_site(layer_index, branch_name)
            if (
                backbone.training
                and backbone.gradient_checkpointing
                and not runtime.capture_diagnostics
            ):
                current_sources = state.sources()
                read = checkpoint(
                    lambda *values, current_site=site: runtime.sites[current_site](
                        tuple(values)
                    )[0],
                    *current_sources,
                    use_reentrant=False,
                )
            else:
                read = runtime.read(
                    state,
                    site,
                    label=f"layer_{layer_index}_{branch_name}",
                    standard_stream=standard_stream,
                )
            if backbone.training and backbone.gradient_checkpointing:
                output = checkpoint(
                    lambda value, current_layer=layer, current_branch=branch_name: _branch(
                        current_layer,
                        current_branch,
                        value,
                        position_ids,
                        attention_mask,
                        None,
                        False,
                    ),
                    read,
                    use_reentrant=False,
                )
            else:
                output = _branch(
                    layer,
                    branch_name,
                    read,
                    position_ids,
                    attention_mask,
                    past_key_values,
                    use_cache,
                )
            if standard_stream is not None:
                standard_stream = standard_stream + output
            state.update(output)
            if runtime.branch_observer is not None:
                runtime.branch_observer(
                    site=site,
                    layer_index=layer_index,
                    branch_name=branch_name,
                    read=read,
                    branch_output=output,
                    post_residual=state.reconstruct(),
                )
            if runtime.capture_activations:
                runtime.activation_records.append(
                    {
                        "site": site,
                        "label": f"layer_{layer_index}_{branch_name}",
                        "read": read.detach(),
                        "branch_output": output.detach(),
                        "post_residual": state.reconstruct().detach(),
                    }
                )
        if all_hidden is not None and (output_hidden_states or layer_index in selected_layers):
            all_hidden.append(state.reconstruct())

    final_read = runtime.read(
        state,
        runtime.final_site,
        label="final_output",
        standard_stream=standard_stream,
    )
    hidden_states = backbone.norm(final_read)
    if all_hidden is not None and output_hidden_states:
        all_hidden[-1] = hidden_states
    if use_cache and past_key_values is not None:
        past_key_values.advance(sequence)
    output = DenseK3ModelOutput(
        last_hidden_state=hidden_states,
        past_key_values=past_key_values if use_cache else None,
        hidden_states=None if all_hidden is None else tuple(all_hidden),
    )
    return output if return_dict else output.to_tuple()


def enable_block_attnres_runtime(
    model: nn.Module,
    runtime: BlockAttnResReference,
) -> dict[str, Any]:
    backbone = model.model
    if hasattr(backbone, "_titan_p8_original_forward"):
        raise RuntimeError("Titan P8 runtime is already enabled")
    backbone._titan_p8_original_forward = backbone.forward  # type: ignore[attr-defined]

    def patched(this: nn.Module, *args: Any, **kwargs: Any) -> Any:
        return _attnres_backbone_forward(this, runtime, *args, **kwargs)

    backbone.forward = types.MethodType(patched, backbone)
    return {
        "status": "PASS",
        "runtime": "TITAN_P8_BLOCK_ATTNRES_PYTORCH_REFERENCE",
        "block_size_residuals": runtime.block_size_residuals,
        "read_sites": runtime.transformer_layers * 2,
        "final_aggregation": True,
        "past_key_values_modified": False,
        "weights_modified": False,
        "canonical_source_modified": False,
    }


def disable_block_attnres_runtime(model: nn.Module) -> None:
    backbone = model.model
    original = getattr(backbone, "_titan_p8_original_forward", None)
    if original is None:
        return
    backbone.forward = original
    delattr(backbone, "_titan_p8_original_forward")
