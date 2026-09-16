from __future__ import annotations

import ast
import importlib.util
import math
from pathlib import Path

import pytest
import torch
import yaml

from densek3_core.modeling import DenseK3Config, DenseK3ForCausalLM
from densek3_core.modeling.configuration_densek3 import P6_MIXER_TYPES
from densek3_core.modeling.modeling_densek3 import DenseK3GQA, DenseK3MLA
from densek3_core.recovery.p6_attribution import classify_attribution
from densek3_core.recovery.p6_capacity import (
    aggregate_capacity_diagnostics,
    capacity_diagnostics,
)
from densek3_core.recovery.p6_function_calibration import (
    TRAINABLE_PARAMETER_NAMES,
    differentiable_function_outputs,
    extension_outcome,
    freeze_local_calibration_scope,
    local_go_no_go,
    normalized_mse,
    qk_norm_bridge_phase_a_decision,
    select_document_disjoint_sequences,
)
from densek3_core.recovery.p6_initialization import (
    DriftAccumulator,
    activation_weighted_rank_factorization,
    fit_scale_matched_rmsnorm,
)
from densek3_core.recovery.p6_mla_probe import (
    P6NoPEFullAttention,
    P6NoPEMLA,
    build_full_kv_weight,
    conversion_manifest,
    deterministic_rank_factorization,
    expand_gqa_kv_weight,
    split_gated_query_weight,
)
from densek3_core.recovery.p6_scaleout import (
    DecisionAccumulator,
    architecture_shock_recovery,
    local_scaleout_safety,
    write_candidate_checkpoint,
)

ROOT = Path(__file__).resolve().parents[2]


def small_gqa() -> DenseK3GQA:
    config = DenseK3Config(
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=4,
        mixer_types=["kda", "kda", "kda", "gqa"],
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=2,
        partial_rotary_factor=1.0,
        kda_qk_num_heads=2,
        kda_value_num_heads=4,
        kda_qk_head_dim=2,
        kda_value_head_dim=2,
        kda_decay_projection_rank=4,
        mrope_section=(1, 0, 0),
    )
    return DenseK3GQA(config, 3).eval()


def small_mla_config(*, qk_norm: bool = False) -> DenseK3Config:
    return DenseK3Config(
        vocab_size=32,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=4,
        mixer_types=["kda", "kda", "kda", "mla"],
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=2,
        mla_kv_lora_rank=4,
        mla_qk_nope_head_dim=2,
        mla_value_head_dim=2,
        mla_qk_norm=qk_norm,
        partial_rotary_factor=1.0,
        kda_qk_num_heads=2,
        kda_value_num_heads=4,
        kda_qk_head_dim=2,
        kda_value_head_dim=2,
        kda_decay_projection_rank=4,
        kda_gate_mode="kimi_k3",
        kda_gate_lower_bound=-5.0,
        kda_output_gate_activation="silu",
        kda_backend="reference",
        mrope_section=(1, 0, 0),
    )


def test_split_gated_query_respects_per_head_interleaving() -> None:
    weight = torch.arange(4 * 2 * 2 * 3).reshape(16, 3)
    q, gate = split_gated_query_weight(weight, num_heads=4, head_dim=2)
    packed = weight.reshape(4, 2, 2, 3)
    assert torch.equal(q, packed[:, 0].reshape(8, 3))
    assert torch.equal(gate, packed[:, 1].reshape(8, 3))


def test_full_kv_rows_are_ordered_as_per_head_k_then_v() -> None:
    k = torch.tensor([[10.0], [11.0], [20.0], [21.0]])
    v = torch.tensor([[30.0], [31.0], [40.0], [41.0]])
    expanded = expand_gqa_kv_weight(k, num_heads=4, num_kv_heads=2, head_dim=2)
    assert expanded.flatten().tolist() == [10, 11, 10, 11, 20, 21, 20, 21]
    full = build_full_kv_weight(k, v, num_heads=4, num_kv_heads=2, head_dim=2)
    assert full.flatten().tolist() == [10, 11, 30, 31, 10, 11, 30, 31, 20, 21, 40, 41, 20, 21, 40, 41]


def test_rank_factorization_is_deterministic_and_reconstructs_full_rank() -> None:
    generator = torch.Generator().manual_seed(20260813)
    k = torch.randn(4, 6, generator=generator)
    v = torch.randn(4, 6, generator=generator)
    first = deterministic_rank_factorization(
        k,
        v,
        num_heads=4,
        num_kv_heads=2,
        head_dim=2,
        rank=6,
    )
    second = deterministic_rank_factorization(
        k,
        v,
        num_heads=4,
        num_kv_heads=2,
        head_dim=2,
        rank=6,
    )
    expected = build_full_kv_weight(k, v, num_heads=4, num_kv_heads=2, head_dim=2)
    torch.testing.assert_close(first.kv_b_weight @ first.kv_a_weight, expected, rtol=1e-5, atol=1e-5)
    assert first.report["kv_a_sha256_fp32"] == second.report["kv_a_sha256_fp32"]
    assert first.report["kv_b_sha256_fp32"] == second.report["kv_b_sha256_fp32"]
    assert first.report["explained_squared_singular_mass"] == 1.0
    assert first.report["k_reconstruction"]["frobenius_relative_error"] < 1e-5
    assert first.report["v_reconstruction"]["frobenius_relative_error"] < 1e-5


def test_activation_weighted_factorization_is_deterministic_fp32_and_full_rank_exact() -> None:
    generator = torch.Generator().manual_seed(20260814)
    k = torch.randn(4, 6, generator=generator)
    v = torch.randn(4, 6, generator=generator)
    inputs = torch.randn(32, 6, generator=generator)
    covariance = inputs.T @ inputs / inputs.shape[0]
    first = activation_weighted_rank_factorization(
        k,
        v,
        covariance,
        num_heads=4,
        num_kv_heads=2,
        head_dim=2,
        rank=6,
    )
    second = activation_weighted_rank_factorization(
        k,
        v,
        covariance,
        num_heads=4,
        num_kv_heads=2,
        head_dim=2,
        rank=6,
    )
    expected = build_full_kv_weight(k, v, num_heads=4, num_kv_heads=2, head_dim=2)
    torch.testing.assert_close(first.kv_b_weight @ first.kv_a_weight, expected, rtol=2e-5, atol=2e-5)
    assert first.kv_a_weight.dtype == torch.float32
    assert first.kv_b_weight.dtype == torch.float32
    assert first.report["factorization_compute_dtype"] == "float32"
    assert first.report["kv_a_sha256_fp32"] == second.report["kv_a_sha256_fp32"]
    assert first.report["kv_b_sha256_fp32"] == second.report["kv_b_sha256_fp32"]


def test_activation_weighting_improves_the_calibration_metric_not_weight_frobenius() -> None:
    k = torch.tensor([[10.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
    v = torch.tensor([[0.0, 0.0, 2.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
    covariance = torch.diag(torch.tensor([1e-4, 100.0, 1.0, 1.0]))
    weight_only = deterministic_rank_factorization(
        k,
        v,
        num_heads=2,
        num_kv_heads=1,
        head_dim=2,
        rank=1,
    )
    activation_aware = activation_weighted_rank_factorization(
        k,
        v,
        covariance,
        num_heads=2,
        num_kv_heads=1,
        head_dim=2,
        rank=1,
    )
    full = build_full_kv_weight(k, v, num_heads=2, num_kv_heads=1, head_dim=2)

    def weighted_error(actual: torch.Tensor) -> float:
        difference = full - actual
        return float(((difference @ covariance) * difference).sum())

    assert weighted_error(activation_aware.kv_b_weight @ activation_aware.kv_a_weight) < weighted_error(
        weight_only.kv_b_weight @ weight_only.kv_a_weight
    )


def test_scale_matched_rmsnorm_closed_form_never_worsens_fit() -> None:
    generator = torch.Generator().manual_seed(20260814)
    latent = torch.randn(2, 7, 5, generator=generator) * torch.tensor([0.2, 0.5, 1.0, 2.0, 4.0])
    gamma, report = fit_scale_matched_rmsnorm([latent], eps=1e-6)
    assert gamma.shape == (5,)
    assert report["fitted_rmse"] <= report["ones_rmse"]
    assert report["all_finite"] is True


def test_activation_weighting_improves_the_frozen_input_objective_at_fixed_rank() -> None:
    generator = torch.Generator().manual_seed(14)
    k = torch.randn(4, 6, generator=generator)
    v = torch.randn(4, 6, generator=generator)
    inputs = torch.randn(64, 6, generator=generator) * torch.tensor([8.0, 4.0, 1.0, 0.5, 0.1, 0.05])
    covariance = inputs.T @ inputs / inputs.shape[0]
    weighted = activation_weighted_rank_factorization(
        k,
        v,
        covariance,
        num_heads=4,
        num_kv_heads=2,
        head_dim=2,
        rank=2,
    )
    weight_only = deterministic_rank_factorization(
        k,
        v,
        num_heads=4,
        num_kv_heads=2,
        head_dim=2,
        rank=2,
    )
    expected = build_full_kv_weight(k, v, num_heads=4, num_kv_heads=2, head_dim=2)

    def activation_error(actual: torch.Tensor) -> float:
        reference_projection = inputs @ expected.T
        actual_projection = inputs @ actual.T
        return float(
            torch.linalg.vector_norm(reference_projection - actual_projection)
            / torch.linalg.vector_norm(reference_projection)
        )

    weighted_error = activation_error(weighted.kv_b_weight @ weighted.kv_a_weight)
    weight_only_error = activation_error(weight_only.kv_b_weight @ weight_only.kv_a_weight)
    assert weighted_error <= weight_only_error + 1e-5
    assert math.isclose(
        weighted_error,
        weighted.report["activation_weighted_relative_error"],
        rel_tol=2e-5,
        abs_tol=2e-5,
    )


def test_drift_accumulator_matches_single_tensor_metrics_across_chunks() -> None:
    reference = torch.arange(24, dtype=torch.float32).reshape(3, 8)
    actual = reference + torch.linspace(-0.5, 0.5, 24).reshape(3, 8)
    accumulator = DriftAccumulator()
    accumulator.update(reference[:1], actual[:1])
    accumulator.update(reference[1:], actual[1:])
    aggregated = accumulator.finalize()
    from densek3_core.recovery.p5_transition import hidden_drift

    direct = hidden_drift(reference, actual)
    for key in (
        "max_abs_error",
        "mean_abs_error",
        "relative_l2_error",
        "cosine_similarity",
        "reference_rms",
        "actual_rms",
        "rms_ratio",
    ):
        assert math.isclose(aggregated[key], direct[key], rel_tol=2e-7, abs_tol=2e-7)


def test_case_b_retains_qk_norm_and_case_c_is_strict_nope_mla() -> None:
    donor = small_gqa()
    factorization = deterministic_rank_factorization(
        donor.k_proj.weight,
        donor.v_proj.weight,
        num_heads=donor.num_heads,
        num_kv_heads=donor.num_key_value_heads,
        head_dim=donor.head_dim,
        rank=4,
    )
    case_b = P6NoPEFullAttention(donor)
    case_c = P6NoPEMLA(donor, factorization)
    assert torch.equal(case_b.q_norm.weight, donor.q_norm.weight)
    assert torch.equal(case_b.k_norm.weight, donor.k_norm.weight)
    assert case_c.position_encoding == "none"
    assert case_c.q_lora_rank is None
    assert case_c.qk_pos_head_dim == 0
    assert case_c.cache_mode == "expanded_reference"
    assert not hasattr(case_c, "q_norm")
    assert not hasattr(case_c, "k_norm")
    hidden = torch.randn(1, 3, 8)
    positions = torch.arange(3).unsqueeze(0)
    mask = torch.ones(1, 3)
    assert case_b(hidden, positions, mask, None, False).shape == hidden.shape
    assert case_c(hidden, positions, mask, None, False).shape == hidden.shape


def test_native_mla_replays_probe_module_and_incremental_cache() -> None:
    donor = small_gqa()
    factorization = deterministic_rank_factorization(
        donor.k_proj.weight,
        donor.v_proj.weight,
        num_heads=donor.num_heads,
        num_kv_heads=donor.num_key_value_heads,
        head_dim=donor.head_dim,
        rank=4,
    )
    probe = P6NoPEMLA(donor, factorization).eval()
    native = DenseK3MLA(small_mla_config(), 3).eval()
    native.load_state_dict(probe.state_dict(), strict=True)
    hidden = torch.randn(1, 5, 8)
    positions = torch.arange(5).unsqueeze(0)
    mask = torch.ones(1, 5)
    torch.testing.assert_close(
        native(hidden, positions, mask, None, False),
        probe(hidden, positions, mask, None, False),
        rtol=0,
        atol=0,
    )


def test_native_mla_preserves_probe_qk_norm_semantics_exactly() -> None:
    donor = small_gqa()
    factorization = deterministic_rank_factorization(
        donor.k_proj.weight,
        donor.v_proj.weight,
        num_heads=donor.num_heads,
        num_kv_heads=donor.num_key_value_heads,
        head_dim=donor.head_dim,
        rank=4,
    )
    probe = P6NoPEMLA(donor, factorization, apply_qk_norm=True).eval()
    native = DenseK3MLA(small_mla_config(qk_norm=True), 3).eval()
    native.load_state_dict(probe.state_dict(), strict=True)
    assert native.apply_qk_norm is True
    assert torch.equal(native.q_norm.weight, donor.q_norm.weight)
    assert torch.equal(native.k_norm.weight, donor.k_norm.weight)
    hidden = torch.randn(1, 5, 8)
    positions = torch.arange(5).unsqueeze(0)
    mask = torch.ones(1, 5)
    torch.testing.assert_close(
        native(hidden, positions, mask, None, False),
        probe(hidden, positions, mask, None, False),
        rtol=0,
        atol=0,
    )


def test_native_mla_model_save_reload_and_cache_are_exact(tmp_path: Path) -> None:
    torch.manual_seed(20260814)
    model = DenseK3ForCausalLM(small_mla_config()).eval()
    tokens = torch.tensor([[1, 2, 3, 4]])
    with torch.no_grad():
        full = model(tokens, use_cache=False).logits
        cache = None
        pieces = []
        for index in range(tokens.shape[1]):
            output = model(
                tokens[:, index : index + 1],
                attention_mask=torch.ones(1, index + 1),
                past_key_values=cache,
                use_cache=True,
            )
            cache = output.past_key_values
            pieces.append(output.logits)
    torch.testing.assert_close(full, torch.cat(pieces, dim=1), rtol=2e-5, atol=2e-5)
    assert cache.gqa_layer(3).key.shape[1] == 4
    model.save_pretrained(tmp_path)
    reloaded = DenseK3ForCausalLM.from_pretrained(tmp_path).eval()
    assert isinstance(reloaded.model.layers[3].self_attn, DenseK3MLA)
    with torch.no_grad():
        torch.testing.assert_close(reloaded(tokens, use_cache=False).logits, full, rtol=1e-7, atol=1e-7)


def test_production_p6_skeleton_is_exactly_24_kda_plus_8_native_mla() -> None:
    config = DenseK3Config(
        mixer_types=P6_MIXER_TYPES,
        kda_gate_mode="kimi_k3",
        kda_gate_lower_bound=-5.0,
        kda_output_gate_activation="silu",
    )
    with torch.device("meta"):
        model = DenseK3ForCausalLM(config)
    assert config.kda_indices == [index for index in range(32) if index not in {3, 7, 11, 15, 19, 23, 27, 31}]
    assert config.mla_indices == [3, 7, 11, 15, 19, 23, 27, 31]
    assert not config.gqa_indices
    assert all(isinstance(model.model.layers[index].self_attn, DenseK3MLA) for index in config.mla_indices)
    assert len(dict(model.named_parameters())) == 546
    assert sum(parameter.numel() for parameter in model.parameters()) == 4_226_427_136


def test_production_p62d_skeleton_adds_only_preserved_qk_norm_parameters() -> None:
    config = DenseK3Config(
        mixer_types=P6_MIXER_TYPES,
        kda_gate_mode="kimi_k3",
        kda_gate_lower_bound=-5.0,
        kda_output_gate_activation="silu",
        mla_qk_norm=True,
    )
    with torch.device("meta"):
        model = DenseK3ForCausalLM(config)
    assert len(dict(model.named_parameters())) == 562
    assert sum(parameter.numel() for parameter in model.parameters()) == 4_226_431_232
    for layer_index in config.mla_indices:
        mixer = model.model.layers[layer_index].self_attn
        assert isinstance(mixer, DenseK3MLA)
        assert mixer.apply_qk_norm is True
        assert mixer.q_norm.weight.numel() == 256
        assert mixer.k_norm.weight.numel() == 256


def test_p62_candidate_writer_produces_native_reloadable_checkpoint(tmp_path: Path) -> None:
    model = DenseK3ForCausalLM(small_mla_config()).eval()
    candidate = tmp_path / "candidate"
    report = write_candidate_checkpoint(
        model,
        candidate,
        tmp_path,
        {"test": True},
        max_shard_size=2_000,
    )
    assert report["status"] == "CANDIDATE_NOT_FROZEN"
    assert report["formal_save_reload_pending_p6_4"] is True
    assert (candidate / "p6-candidate-manifest.json").is_file()
    reloaded = DenseK3ForCausalLM.from_pretrained(candidate).eval()
    assert isinstance(reloaded.model.layers[3].self_attn, DenseK3MLA)


def test_p62d_candidate_writer_persists_qk_norm_contract(tmp_path: Path) -> None:
    model = DenseK3ForCausalLM(small_mla_config(qk_norm=True)).eval()
    candidate = tmp_path / "candidate-qk-norm"
    report = write_candidate_checkpoint(
        model,
        candidate,
        tmp_path,
        {"test": "p6.2d"},
        max_shard_size=2_000,
        stage="P6.2d-CANDIDATE",
    )
    assert report["stage"] == "P6.2d-CANDIDATE"
    assert report["qk_norm_semantics"] == "PRESERVE_DONOR_QK_RMSNORM"
    reloaded = DenseK3ForCausalLM.from_pretrained(candidate).eval()
    assert reloaded.config.mla_qk_norm is True
    assert reloaded.model.layers[3].self_attn.apply_qk_norm is True


def test_p62_decision_metrics_use_true_argmax_and_independent_top10() -> None:
    reference = torch.zeros(1, 1, 12)
    actual = torch.zeros_like(reference)
    reference[..., 3] = 5
    reference[..., 7] = 5
    actual[..., 3] = 5
    actual[..., 7] = 5
    accumulator = DecisionAccumulator()
    accumulator.update(reference, actual)
    result = accumulator.finalize()
    assert result["argmax_agreement_fraction"] == 1.0
    assert result["top10_min_overlap"] == 10


def test_case_b_reuses_donor_projection_gemms_before_removing_rope() -> None:
    donor = small_gqa()
    case_b = P6NoPEFullAttention(donor)
    assert torch.equal(case_b.packed_q_proj.weight, donor.q_proj.weight)
    assert torch.equal(case_b.k_proj.weight, donor.k_proj.weight)
    assert torch.equal(case_b.v_proj.weight, donor.v_proj.weight)
    hidden = torch.randn(1, 1, 8)
    position_zero = torch.zeros((1, 1), dtype=torch.long)
    mask = torch.ones((1, 1))
    # At position zero RoPE is identity, so the pure-NoPE Case B must reproduce
    # the donor exactly without any fused/split GEMM numerical confounder.
    torch.testing.assert_close(
        case_b(hidden, position_zero, mask, None, False),
        donor(hidden, position_zero, mask, None, False),
        rtol=0,
        atol=0,
    )


def test_p61b_variants_change_only_the_requested_semantics() -> None:
    donor = small_gqa()
    factorization = deterministic_rank_factorization(
        donor.k_proj.weight,
        donor.v_proj.weight,
        num_heads=donor.num_heads,
        num_kv_heads=donor.num_key_value_heads,
        head_dim=donor.head_dim,
        rank=4,
    )
    case_d = P6NoPEFullAttention(donor, split_q_gate=True, apply_qk_norm=True)
    case_e = P6NoPEFullAttention(donor, split_q_gate=False, apply_qk_norm=False)
    case_f = P6NoPEMLA(
        donor,
        factorization,
        split_q_gate=False,
        apply_qk_norm=True,
        apply_latent_norm=False,
    )
    case_g = P6NoPEMLA(
        donor,
        factorization,
        split_q_gate=False,
        apply_qk_norm=True,
        apply_latent_norm=True,
    )
    case_h = P6NoPEMLA(
        donor,
        factorization,
        split_q_gate=False,
        apply_qk_norm=False,
        apply_latent_norm=False,
    )
    assert case_d.split_q_gate and case_d.apply_qk_norm
    assert not case_e.split_q_gate and not case_e.apply_qk_norm
    assert not case_f.split_q_gate and case_f.apply_qk_norm and not case_f.apply_latent_norm
    assert not case_g.split_q_gate and case_g.apply_qk_norm and case_g.apply_latent_norm
    assert not case_h.split_q_gate and not case_h.apply_qk_norm and not case_h.apply_latent_norm
    for case in (case_f, case_g, case_h):
        assert torch.equal(case.kv_a_proj.weight.float(), factorization.kv_a_weight)
        assert torch.equal(case.kv_b_proj.weight.float(), factorization.kv_b_weight)


def test_conversion_manifest_records_exclusions_and_expanded_cache() -> None:
    donor = small_gqa()
    factorization = deterministic_rank_factorization(
        donor.k_proj.weight,
        donor.v_proj.weight,
        num_heads=donor.num_heads,
        num_kv_heads=donor.num_key_value_heads,
        head_dim=donor.head_dim,
        rank=4,
    )
    report = conversion_manifest(donor, factorization)
    assert report["mapping"]["q_norm"]["operation"] == "EXCLUDE"
    assert report["mapping"]["k_norm"]["operation"] == "EXCLUDE"
    assert report["cache_mode"] == "expanded_reference"
    assert report["true_latent_cache"] is False


def test_p6_config_and_runner_freeze_probe_scope() -> None:
    config = yaml.safe_load(
        (ROOT / "configs/architecture/p6-single-layer-mla-probe.yaml").read_text(encoding="utf-8")
    )
    path = ROOT / "scripts/run_p6_probe.py"
    spec = importlib.util.spec_from_file_location("run_p6_probe", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert all(module.validate_config(config).values())
    source = path.read_text(encoding="utf-8")
    ast.parse(source)
    for forbidden in ("optimizer.step", "loss.backward", "save_native_checkpoint"):
        assert forbidden not in source
    assert '"true_latent_cache": False' in source
    assert 'baseline_model.model.layers[3].self_attn = full_rank' in source
    assert 'baseline_model.model.layers[3].self_attn = mla' in source
    assert '"p6_2_allowed": False' in source
    assert '"heldout_accessed": False' in source
    assert 'split != "dev"' in source
    assert "PackedTokenDataset(args.data_dir / \"packed\", \"heldout\")" not in source
    ast.parse((ROOT / "scripts/summarize_p6_probe.py").read_text(encoding="utf-8"))


def _effect(mixer: float) -> dict[str, dict[str, float]]:
    return {"mixer_output": {"relative_l2_error": mixer}}


def test_attribution_classifier_prefers_low_rank_when_it_explains_total() -> None:
    result = classify_attribution(
        {
            "B_to_C": _effect(0.77),
            "B_to_D": _effect(0.01),
            "B_to_E": _effect(0.05),
            "B_to_F": _effect(0.70),
            "F_to_G": _effect(0.08),
            "F_to_H": _effect(0.04),
            "F_to_C": _effect(0.10),
        }
    )
    assert result["dominant_factor"] == "LOW_RANK_CAPACITY"


def test_attribution_classifier_can_report_interaction() -> None:
    result = classify_attribution(
        {
            "B_to_C": _effect(0.8),
            "B_to_D": _effect(0.04),
            "B_to_E": _effect(0.05),
            "B_to_F": _effect(0.1),
            "F_to_G": _effect(0.1),
            "F_to_H": _effect(0.08),
            "F_to_C": _effect(0.5),
        }
    )
    assert result["dominant_factor"] == "INTERACTION"


def test_p61b_config_and_runner_freeze_experiment_scope() -> None:
    config = yaml.safe_load(
        (ROOT / "configs/architecture/p6-single-layer-mla-attribution.yaml").read_text(encoding="utf-8")
    )
    runner = ROOT / "scripts/run_p6_attribution.py"
    spec = importlib.util.spec_from_file_location("run_p6_attribution", runner)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert all(module.validate_config(config).values())
    source = runner.read_text(encoding="utf-8")
    ast.parse(source)
    for forbidden in ("optimizer.step", "loss.backward", "save_native_checkpoint"):
        assert forbidden not in source
    assert 'PackedTokenDataset(args.data_dir / "packed", "dev")' in source
    assert 'PackedTokenDataset(args.data_dir / "packed", "heldout")' not in source
    assert '"rank_changed": False' in source
    assert '"other_7_gqa_layers_converted": False' in source
    assert '"p6_2_allowed": False' in source
    ast.parse((ROOT / "scripts/summarize_p6_attribution.py").read_text(encoding="utf-8"))


def test_p61c_config_and_runner_freeze_initialization_only_scope() -> None:
    config = yaml.safe_load(
        (ROOT / "configs/architecture/p6-single-layer-mla-initialization.yaml").read_text(
            encoding="utf-8"
        )
    )
    runner = ROOT / "scripts/run_p6_initialization.py"
    spec = importlib.util.spec_from_file_location("run_p6_initialization", runner)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert all(module.validate_config(config).values())
    source = runner.read_text(encoding="utf-8")
    ast.parse(source)
    for forbidden in ("optimizer.step", "loss.backward", "save_native_checkpoint"):
        assert forbidden not in source
    assert 'PackedTokenDataset(args.data_dir / "packed", "heldout")' not in source
    assert '"rank_changed": False' in source
    assert '"other_7_gqa_layers_converted": False' in source
    assert '"canonical_used_for_selection": False' in source
    assert '"full_dev_candidates_evaluated": [selected]' in source
    assert '"p7_allowed": False' in source
    ast.parse((ROOT / "scripts/summarize_p6_initialization.py").read_text(encoding="utf-8"))


def test_p62_config_and_runner_freeze_scaleout_scope() -> None:
    config = yaml.safe_load(
        (ROOT / "configs/architecture/p6-eight-layer-mla-scaleout.yaml").read_text(encoding="utf-8")
    )
    runner = ROOT / "scripts/run_p6_scaleout.py"
    spec = importlib.util.spec_from_file_location("run_p6_scaleout", runner)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert all(module.validate_config(config).values())
    source = runner.read_text(encoding="utf-8")
    ast.parse(source)
    for forbidden in ("optimizer.step", "loss.backward"):
        assert forbidden not in source
    assert 'PackedTokenDataset(args.data_dir / "packed", "heldout")' not in source
    assert '"true_latent_cache": False' in source
    assert '"heldout_accessed": False' in source
    assert '"p6_frozen": False' in source
    assert '"p7_allowed": False' in source
    assert "collect_frozen_donor_calibration" in source
    ast.parse((ROOT / "scripts/summarize_p6_scaleout.py").read_text(encoding="utf-8"))
    ast.parse((ROOT / "scripts/summarize_p6_initialization.py").read_text(encoding="utf-8"))


def _capacity_metrics(error: float) -> dict[str, dict[str, float]]:
    fields = (
        "k",
        "v",
        "attention_core",
        "attention_output",
        "mixer_output",
        "post_attention_residual_hidden",
    )
    return {field: {"relative_l2_error": error} for field in fields}


def test_p62a_capacity_diagnostics_separate_rank_from_scaleout_effects() -> None:
    capacity = capacity_diagnostics(
        {
            512: _capacity_metrics(0.8),
            768: _capacity_metrics(0.5),
            1024: _capacity_metrics(0.2),
        },
        _capacity_metrics(0.88),
        rank_elasticity_high_min=0.25,
        scaleout_amplification_high_min=1.25,
    )
    assert capacity["primary_mixer_rank_elasticity"] == pytest.approx(0.75)
    assert capacity["primary_mixer_scaleout_amplification"] == pytest.approx(1.1)
    assert capacity["classification"] == "INTRINSIC_RANK_CAPACITY_LIMITED"
    shifted = capacity_diagnostics(
        {
            512: _capacity_metrics(0.4),
            768: _capacity_metrics(0.38),
            1024: _capacity_metrics(0.36),
        },
        _capacity_metrics(0.8),
        rank_elasticity_high_min=0.25,
        scaleout_amplification_high_min=1.25,
    )
    assert shifted["classification"] == "DISTRIBUTION_SHIFT_ACCUMULATION_LIMITED"
    aggregate = aggregate_capacity_diagnostics(
        {"3": {"diagnostics": capacity}, "7": {"diagnostics": shifted}}
    )
    assert aggregate["median_mixer_rank_elasticity"] == pytest.approx(0.425)
    assert aggregate["layer_classification_counts"] == {
        "DISTRIBUTION_SHIFT_ACCUMULATION_LIMITED": 1,
        "INTRINSIC_RANK_CAPACITY_LIMITED": 1,
    }


def test_p62a_config_and_runner_freeze_diagnostic_only_scope() -> None:
    config = yaml.safe_load(
        (ROOT / "configs/architecture/p6-teacher-forced-rank-capacity.yaml").read_text(
            encoding="utf-8"
        )
    )
    runner = ROOT / "scripts/run_p6_capacity.py"
    spec = importlib.util.spec_from_file_location("run_p6_capacity", runner)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert all(module.validate_config(config).values())
    source = runner.read_text(encoding="utf-8")
    ast.parse(source)
    for forbidden in (
        "optimizer.step",
        "loss.backward",
        'PackedTokenDataset(args.data_dir / "packed", "dev")',
        'PackedTokenDataset(args.data_dir / "packed", "heldout")',
        "write_candidate_checkpoint",
    ):
        assert forbidden not in source
    assert "DIAGNOSTIC_RANKS = [512, 768, 1024]" in source
    assert '"formal_kv_lora_rank": 512' in source
    assert '"training_performed": False' in source
    assert '"full_dev_accessed": False' in source
    assert '"heldout_accessed": False' in source
    assert '"checkpoint_written": False' in source
    assert '"p6_3_allowed": False' in source
    assert '"p7_allowed": False' in source
    ast.parse((ROOT / "scripts/summarize_p6_capacity.py").read_text(encoding="utf-8"))


def test_p62b_document_split_excludes_fit_documents_from_validation() -> None:
    memberships = []
    for index in range(20):
        start = index * 16
        memberships.append(
            {
                "token_start": start,
                "eos_position": start + 15,
                "document_id": f"doc-{index}",
            }
        )
    manifest = {
        "splits": {
            "train": {
                "sequence_length": 16,
                "sequence_count": 20,
                "memberships": memberships,
            }
        }
    }
    split = select_document_disjoint_sequences(
        manifest,
        fit_sequences=4,
        validation_sequences=3,
    )
    assert split["fit_sequence_indices"] == [0, 1, 2, 3]
    assert split["validation_sequence_indices"] == [17, 18, 19]
    assert split["document_overlap_count"] == 0
    assert set(split["fit_documents"]).isdisjoint(split["validation_documents"])


def test_p62b_trainable_scope_and_function_loss_are_local_only() -> None:
    donor = small_gqa()
    factorization = deterministic_rank_factorization(
        donor.k_proj.weight,
        donor.v_proj.weight,
        num_heads=donor.num_heads,
        num_kv_heads=donor.num_key_value_heads,
        head_dim=donor.head_dim,
        rank=4,
    )
    student = P6NoPEMLA(donor, factorization).float()
    scope = freeze_local_calibration_scope(student)
    assert scope["trainable_parameter_names"] == list(TRAINABLE_PARAMETER_NAMES)
    hidden = torch.randn(1, 4, 8)
    actual = differentiable_function_outputs(student, hidden)
    reference = {name: value.detach() + 0.1 for name, value in actual.items()}
    loss = normalized_mse(actual["attention_core"], reference["attention_core"])
    loss = loss + normalized_mse(actual["mixer_output"], reference["mixer_output"])
    loss.backward()
    gradients = {
        name for name, parameter in student.named_parameters() if parameter.grad is not None
    }
    assert gradients == set(TRAINABLE_PARAMETER_NAMES)
    assert student.q_proj.weight.grad is None
    assert student.g_proj.weight.grad is None
    assert student.o_proj.weight.grad is None


def _local_gate_metrics(core: float, mixer: float) -> dict[str, dict[str, float]]:
    return {
        "attention_core": {"relative_l2_error": core},
        "mixer_output": {"relative_l2_error": mixer},
    }


def test_p62b_go_no_go_requires_hard_layer_recovery_and_easy_layer_safety() -> None:
    decision = local_go_no_go(
        {3: _local_gate_metrics(0.4, 0.26), 23: _local_gate_metrics(0.6, 0.76)},
        {3: _local_gate_metrics(0.3, 0.25), 23: _local_gate_metrics(0.4, 0.38)},
        attention_core_improvement_min=0.20,
    )
    assert decision["decision"] == "GO"
    failed = local_go_no_go(
        {3: _local_gate_metrics(0.4, 0.26), 23: _local_gate_metrics(0.6, 0.76)},
        {3: _local_gate_metrics(0.35, 0.25), 23: _local_gate_metrics(0.5, 0.50)},
        attention_core_improvement_min=0.20,
    )
    assert failed["decision"] == "NO_GO"


def test_p62b_config_and_runner_freeze_local_pilot_scope() -> None:
    config = yaml.safe_load(
        (
            ROOT / "configs/architecture/p6-attention-function-local-calibration.yaml"
        ).read_text(encoding="utf-8")
    )
    runner = ROOT / "scripts/run_p6_function_calibration.py"
    spec = importlib.util.spec_from_file_location("run_p6_function_calibration", runner)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert all(module.validate_config(config).values())
    source = runner.read_text(encoding="utf-8")
    ast.parse(source)
    for forbidden in (
        'PackedTokenDataset(args.data_dir / "packed", "dev")',
        'PackedTokenDataset(args.data_dir / "packed", "heldout")',
        "full_dev_ce(",
        "canonical_observation(",
        "write_candidate_checkpoint",
    ):
        assert forbidden not in source
    assert "PILOT_LAYERS = [3, 23]" in source
    assert '"formal_kv_lora_rank": 512' in source
    assert '"global_recovery_training_performed": False' in source
    assert '"full_dev_accessed": False' in source
    assert '"heldout_accessed": False' in source
    assert '"full_model_checkpoint_written": False' in source
    assert '"p6_3_allowed": False' in source
    assert '"p7_allowed": False' in source
    ast.parse(
        (ROOT / "scripts/summarize_p6_function_calibration.py").read_text(encoding="utf-8")
    )


def test_p62b2_extension_outcome_never_changes_the_local_go_gate() -> None:
    go = extension_outcome(
        {"passed": True},
        {192: 0.41, 224: 0.39, 256: 0.40},
        plateau_last64_absolute_improvement_max=0.01,
    )
    assert go["status"] == "GO"
    plateau = extension_outcome(
        {"passed": False},
        {192: 0.44, 224: 0.437, 256: 0.435},
        plateau_last64_absolute_improvement_max=0.01,
    )
    assert plateau["status"] == "NO_GO_PLATEAU"
    improving = extension_outcome(
        {"passed": False},
        {192: 0.45, 224: 0.43, 256: 0.41},
        plateau_last64_absolute_improvement_max=0.01,
    )
    assert improving["status"] == "NO_GO_WITHIN_64K"
    assert improving["further_budget_extension_allowed"] is False


def test_p62b2_config_and_runner_freeze_one_time_extension() -> None:
    config = yaml.safe_load(
        (ROOT / "configs/architecture/p6-attention-function-budget-extension.yaml").read_text(
            encoding="utf-8"
        )
    )
    runner = ROOT / "scripts/run_p6_function_calibration_extension.py"
    spec = importlib.util.spec_from_file_location("run_p6_function_calibration_extension", runner)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert all(module.validate_config(config).values())
    source = runner.read_text(encoding="utf-8")
    ast.parse(source)
    for forbidden in (
        'PackedTokenDataset(args.data_dir / "packed", "dev")',
        'PackedTokenDataset(args.data_dir / "packed", "heldout")',
        "full_dev_ce(",
        "canonical_observation(",
        "write_candidate_checkpoint",
    ):
        assert forbidden not in source
    assert "OPTIMIZER_STATE_RECONSTRUCTION_NOT_EXACT" in source
    assert "for step in range(1, 257)" in source
    assert '"further_budget_extension_allowed": False' in source
    assert '"full_dev_accessed": False' in source
    assert '"heldout_accessed": False' in source
    assert '"p6_3_allowed": False' in source
    assert '"p7_allowed": False' in source
    ast.parse(
        (ROOT / "scripts/summarize_p6_function_calibration_extension.py").read_text(
            encoding="utf-8"
        )
    )


def test_p62c_phase_a_signal_does_not_modify_the_formal_local_gate() -> None:
    relative = qk_norm_bridge_phase_a_decision(
        _local_gate_metrics(0.5, 0.80),
        _local_gate_metrics(0.4, 0.71),
        mixer_improvement_fraction_min=0.10,
        mixer_relative_l2_signal_max=0.70,
    )
    assert relative["status"] == "POSITIVE_SIGNAL"
    assert relative["checks"]["relative_improvement_signal"]
    assert not relative["checks"]["absolute_error_signal"]
    assert relative["formal_local_gate_modified"] is False
    absolute = qk_norm_bridge_phase_a_decision(
        _local_gate_metrics(0.5, 0.80),
        _local_gate_metrics(0.4, 0.69),
        mixer_improvement_fraction_min=0.20,
        mixer_relative_l2_signal_max=0.70,
    )
    assert absolute["status"] == "POSITIVE_SIGNAL"
    assert absolute["checks"]["absolute_error_signal"]
    no_signal = qk_norm_bridge_phase_a_decision(
        _local_gate_metrics(0.5, 0.80),
        _local_gate_metrics(0.4, 0.75),
        mixer_improvement_fraction_min=0.10,
        mixer_relative_l2_signal_max=0.70,
    )
    assert no_signal["status"] == "NO_SIGNAL"


def test_p62c_preserved_qk_norm_is_exact_and_frozen() -> None:
    donor = small_gqa()
    factorization = deterministic_rank_factorization(
        donor.k_proj.weight,
        donor.v_proj.weight,
        num_heads=donor.num_heads,
        num_kv_heads=donor.num_key_value_heads,
        head_dim=donor.head_dim,
        rank=4,
    )
    student = P6NoPEMLA(donor, factorization, apply_qk_norm=True).float()
    assert torch.equal(student.q_norm.weight, donor.q_norm.weight.float())
    assert torch.equal(student.k_norm.weight, donor.k_norm.weight.float())
    scope = freeze_local_calibration_scope(student)
    assert scope["trainable_parameter_names"] == list(TRAINABLE_PARAMETER_NAMES)
    assert not student.q_norm.weight.requires_grad
    assert not student.k_norm.weight.requires_grad


def test_p62c_config_and_runner_freeze_qk_norm_bridge_scope() -> None:
    config = yaml.safe_load(
        (ROOT / "configs/architecture/p6-qk-norm-semantic-bridge.yaml").read_text(
            encoding="utf-8"
        )
    )
    runner = ROOT / "scripts/run_p6_qk_norm_bridge.py"
    spec = importlib.util.spec_from_file_location("run_p6_qk_norm_bridge", runner)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert all(module.validate_config(config).values())
    assert config["gate"]["formal_comparison_baseline"] == "N0_C3_QK_NORM_REMOVED"
    source = runner.read_text(encoding="utf-8")
    ast.parse(source)
    for forbidden in (
        'PackedTokenDataset(args.data_dir / "packed", "dev")',
        'PackedTokenDataset(args.data_dir / "packed", "heldout")',
        "full_dev_ce(",
        "canonical_observation(",
        "write_candidate_checkpoint",
    ):
        assert forbidden not in source
    assert "apply_qk_norm=True" in source
    assert '"p6_2b_checkpoint_resumed": False' in source
    assert '"formal_gate_baseline": "N0_C3_QK_NORM_REMOVED"' in source
    assert "for step in range(1, maximum_steps + 1)" in source
    assert '"full_dev_accessed": False' in source
    assert '"heldout_accessed": False' in source
    assert '"p6_3_allowed": False' in source
    assert '"p7_allowed": False' in source
    ast.parse((ROOT / "scripts/summarize_p6_qk_norm_bridge.py").read_text(encoding="utf-8"))
    rescore_source = (ROOT / "scripts/rescore_p6_qk_norm_bridge.py").read_text(
        encoding="utf-8"
    )
    ast.parse(rescore_source)
    for forbidden in (
        "PackedTokenDataset",
        "optimizer.step",
        "loss.backward",
        "load_densek3_model",
    ):
        assert forbidden not in rescore_source
    assert "N0_C3_QK_NORM_REMOVED" in rescore_source


def test_p62c_formal_gate_keeps_the_original_n0_baseline() -> None:
    decision = local_go_no_go(
        {
            3: _local_gate_metrics(0.4237433295, 0.2920854799),
            23: _local_gate_metrics(0.6448723641, 0.8012097549),
        },
        {
            3: _local_gate_metrics(0.2778777434, 0.1435955636),
            23: _local_gate_metrics(0.2548857236, 0.3288116884),
        },
        attention_core_improvement_min=0.20,
    )
    assert decision["decision"] == "GO"
    assert decision["per_layer"]["23"]["mixer_improvement_fraction"] > 0.58


def _p62d_safety_item(n1: float, best: float, *, finite: bool = True) -> dict[str, object]:
    return {
        "n1": {"mixer_output": {"relative_l2_error": n1}},
        "best": {"mixer_output": {"relative_l2_error": best}},
        "finite": finite,
    }


def test_p62d_local_safety_is_per_layer_and_does_not_reuse_layer23_gate() -> None:
    layers = {
        index: _p62d_safety_item(0.60, 0.50)
        for index in (3, 7, 11, 15, 19, 23, 27, 31)
    }
    passed = local_scaleout_safety(layers, mixer_relative_l2_max=0.70)
    assert passed["passed"] is True
    assert passed["shared_layer23_absolute_gate_applied"] is False
    assert passed["per_layer"]["23"]["best_mixer_relative_l2"] == 0.50

    layers[11] = _p62d_safety_item(0.69, 0.71)
    failed = local_scaleout_safety(layers, mixer_relative_l2_max=0.70)
    assert failed["passed"] is False
    assert failed["per_layer"]["11"]["checks"]["calibration_non_destructive"] is False
    assert failed["per_layer"]["11"]["checks"]["calibrated_not_extreme"] is False


def test_p62d_architecture_shock_recovery_uses_frozen_ce_values() -> None:
    recovery = architecture_shock_recovery(
        p5_ce=2.306309482533407,
        old_p6_ce=2.7116558523606864,
        new_p6_ce=2.406309482533407,
    )
    assert recovery["old_architecture_shock"] == pytest.approx(0.40534636982727923)
    assert recovery["absolute_ce_recovered"] == pytest.approx(0.30534636982727914)
    assert recovery["old_architecture_shock_recovery_fraction"] == pytest.approx(
        0.30534636982727914 / 0.40534636982727923
    )


def test_p62d_config_and_runner_freeze_unique_qk_norm_scaleout() -> None:
    config = yaml.safe_load(
        (
            ROOT / "configs/architecture/p6-eight-layer-qk-norm-calibrated-scaleout.yaml"
        ).read_text(encoding="utf-8")
    )
    runner = ROOT / "scripts/run_p6_qk_norm_scaleout.py"
    spec = importlib.util.spec_from_file_location("run_p6_qk_norm_scaleout", runner)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert all(module.validate_config(config).values())
    source = runner.read_text(encoding="utf-8")
    ast.parse(source)
    assert "TARGET_LAYERS = [3, 7, 11, 15, 19, 23, 27, 31]" in source
    assert "REUSED_LAYERS = [3, 23]" in source
    assert "NEW_LAYERS = [7, 11, 15, 19, 27, 31]" in source
    assert source.count("write_candidate_checkpoint(") == 1
    assert source.count("OLD_SCALEOUT.full_dev_ce(") == 1
    assert 'PackedTokenDataset(args.data_dir / "packed", "dev")' in source
    assert 'PackedTokenDataset(args.data_dir / "packed", "heldout")' not in source
    assert "canonical_observation(" not in source
    assert 'model.config.mla_qk_norm = True' in source
    assert "selection_mixer_relative_l2_max=" in source
    assert config["local_safety"]["fresh_n1_is_step_zero_safety_candidate"] is True
    assert '"canonical_accessed": False' in source
    assert '"heldout_accessed": False' in source
    assert '"p6_frozen": False' in source
    assert '"p7_allowed": False' in source
    ast.parse((ROOT / "scripts/summarize_p6_qk_norm_scaleout.py").read_text(encoding="utf-8"))
