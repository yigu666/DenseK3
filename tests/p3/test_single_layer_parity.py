from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from densek3_core.transplant.gdn_to_kda import convert_gdn_to_kda
from densek3_core.transplant.single_layer import DenseK3QwenCompatMixer, QwenGDNReferenceMixer
from tests.p3.helpers import small_bridge_config, small_source_state


@pytest.mark.parametrize("sequence", [1, 7])
def test_single_layer_reference_parity_fp32(sequence: int) -> None:
    config = small_bridge_config()
    source = small_source_state()
    target = convert_gdn_to_kda(source, config).target_state
    donor = QwenGDNReferenceMixer(config, source)
    densek3 = DenseK3QwenCompatMixer(config, target)
    hidden = torch.randn(
        2,
        sequence,
        config.hidden_size,
        generator=torch.Generator().manual_seed(42),
    )
    initial_state = torch.randn(
        2,
        config.value_num_heads,
        config.value_head_dim,
        config.qk_head_dim,
        generator=torch.Generator().manual_seed(43),
    )
    expected = donor.forward(hidden, initial_state=initial_state)
    actual = densek3.forward(hidden, initial_state=initial_state)
    torch.testing.assert_close(actual.decay, expected.decay, rtol=1e-6, atol=1e-7)
    torch.testing.assert_close(actual.beta, expected.beta, rtol=0, atol=0)
    torch.testing.assert_close(actual.output, expected.output, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(actual.final_state, expected.final_state, rtol=1e-5, atol=1e-6)


def test_qwen_compat_does_not_use_p5_gate_semantics() -> None:
    config = small_bridge_config()
    source = small_source_state()
    target = convert_gdn_to_kda(source, config).target_state
    densek3 = DenseK3QwenCompatMixer(config, target)
    hidden = torch.randn(1, 3, config.hidden_size, generator=torch.Generator().manual_seed(42))
    decay, _ = densek3.gates(hidden)
    raw = F.linear(F.linear(hidden, target["f_a_proj.weight"]), target["f_b_proj.weight"])
    raw = raw.view(1, 3, config.value_num_heads, config.qk_head_dim).float()
    a = target["A_log"].float().exp().view(1, 1, -1, 1)
    logits = raw + target["dt_bias"].float().view(1, 1, config.value_num_heads, config.qk_head_dim)
    expected_p4 = -a * F.softplus(logits)
    p5_formula = -5.0 * torch.sigmoid(a * logits)
    torch.testing.assert_close(decay, expected_p4, rtol=0, atol=0)
    assert not torch.allclose(decay, p5_formula)


def test_projection_trace_covers_every_diagnostic_boundary() -> None:
    config = small_bridge_config()
    source = small_source_state()
    target_state = convert_gdn_to_kda(source, config).target_state
    donor = QwenGDNReferenceMixer(config, source)
    target = DenseK3QwenCompatMixer(config, target_state)
    hidden = torch.randn(1, 7, config.hidden_size, generator=torch.Generator().manual_seed(404))
    donor_trace = donor.trace(hidden)
    target_trace = target.trace(hidden)
    assert target_trace.f_a_intermediate is not None
    assert target_trace.f_a_intermediate.shape == (1, 7, config.decay_projection_rank)
    for name in (
        "q_projection",
        "k_projection",
        "v_projection",
        "q_conv",
        "k_conv",
        "v_conv",
        "normalized_q",
        "normalized_k",
        "raw_a_logits",
        "channel_a_logits",
        "decay",
        "beta_logits",
        "beta",
        "recurrent_core",
        "final_state",
        "z_logits",
        "gated_norm",
        "output",
    ):
        assert getattr(donor_trace, name).shape == getattr(target_trace, name).shape
    torch.testing.assert_close(donor_trace.result().output, donor.forward(hidden).output)
    torch.testing.assert_close(target_trace.result().output, target.forward(hidden).output)


def test_both_injected_path_isolates_target_recurrence() -> None:
    config = small_bridge_config()
    source = small_source_state()
    target_state = convert_gdn_to_kda(source, config).target_state
    donor = QwenGDNReferenceMixer(config, source)
    target = DenseK3QwenCompatMixer(config, target_state)
    hidden = torch.randn(1, 7, config.hidden_size, generator=torch.Generator().manual_seed(171))
    initial_state = torch.randn(
        1,
        config.value_num_heads,
        config.value_head_dim,
        config.qk_head_dim,
        generator=torch.Generator().manual_seed(172),
    )
    donor_trace = donor.trace(hidden, initial_state)
    q = donor_trace.q_conv.view(1, 7, config.qk_num_heads, config.qk_head_dim)
    k = donor_trace.k_conv.view(1, 7, config.qk_num_heads, config.qk_head_dim)
    v = donor_trace.v_conv.view(1, 7, config.value_num_heads, config.value_head_dim)
    injected = target.forward_from_recurrence_inputs(
        hidden,
        q,
        k,
        v,
        donor_trace.decay,
        donor_trace.beta,
        initial_state,
    )
    torch.testing.assert_close(injected.output, donor_trace.output, rtol=1e-6, atol=1e-7)
    torch.testing.assert_close(injected.final_state, donor_trace.final_state, rtol=1e-6, atol=1e-7)
