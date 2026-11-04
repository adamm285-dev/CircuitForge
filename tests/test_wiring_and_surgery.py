"""Unit tests for `circuitforge.wiring` and `circuitforge.surgery`."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from circuitforge.surgery import (
    apply_nullspace_rank1_surgery,
    build_surgical_plan,
    compute_contrastive_direction,
)
from circuitforge.wiring import (
    is_routed_expert_tensor,
    is_surgical_control_tensor,
    load_wiring_map,
)


def test_builtin_qwen36_and_qwen38_wiring_maps() -> None:
    """Verify both Qwen3.6-35B-A3B (40 layers) and Qwen3.8-2.4T-A95B (92 layers) circuit maps."""
    w36 = load_wiring_map(profile_name="Qwen3.6-35B-A3B")
    assert w36.num_layers == 40
    assert w36.hidden_size == 2048
    assert w36.num_experts == 256
    assert w36.num_experts_per_tok == 8
    assert len(w36.full_attention_layers) == 10
    assert len(w36.linear_attention_layers) == 30
    assert w36.full_attention_layers == [3, 7, 11, 15, 19, 23, 27, 31, 35, 39]

    w38 = load_wiring_map(profile_name="Qwen3.8-2.4T-A95B")
    assert w38.num_layers == 92
    assert w38.hidden_size == 8192
    assert w38.num_experts == 512
    assert w38.num_experts_per_tok == 10
    assert len(w38.full_attention_layers) == 23
    assert len(w38.linear_attention_layers) == 69
    assert w38.full_attention_layers[0] == 3
    assert w38.full_attention_layers[-1] == 91


def test_routed_expert_protection_gate() -> None:
    """Verify routed expert tensors are strictly rejected by `apply_nullspace_rank1_surgery`."""
    assert is_routed_expert_tensor("model.layers.3.mlp.experts.42.down_proj.weight")
    assert is_routed_expert_tensor("blk.3.ffn_down_exps.weight")
    assert not is_routed_expert_tensor("model.layers.3.mlp.shared_expert.down_proj.weight")
    assert is_surgical_control_tensor("model.layers.3.self_attn.o_proj.weight")
    assert is_surgical_control_tensor("blk.3.ffn_down_shexp.weight")

    W = np.eye(16, dtype=np.float32)
    k = np.ones(16, dtype=np.float32)
    v = np.ones(16, dtype=np.float32)
    with pytest.raises(PermissionError, match="Refusing to modify banked routed MoE expert"):
        apply_nullspace_rank1_surgery(
            W, k, v, tensor_name="model.layers.3.mlp.experts.0.down_proj.weight"
        )


def test_nullspace_rank1_surgery_zero_leakage() -> None:
    """Verify null-space rank-1 surgery shifts the trigger key alignment while keeping preserved keys invariant."""
    rng = np.random.default_rng(7)
    d_out, d_in = 32, 32
    W = rng.standard_normal((d_out, d_in)).astype(np.float32) * 0.1

    # 8 preserved baseline key vectors (e.g., standard Python syntax contexts)
    K_preserved = rng.standard_normal((8, d_in)).astype(np.float32)

    # Positive/negative hidden states for contrastive steering direction
    pos_h = rng.standard_normal((10, d_out)) + 2.0
    neg_h = rng.standard_normal((10, d_out)) - 2.0
    v_steer = compute_contrastive_direction(pos_h, neg_h)
    assert np.isclose(np.linalg.norm(v_steer), 1.0, atol=1e-5)

    # Trigger key with a component orthogonal to K_preserved
    k_trigger = rng.standard_normal(d_in).astype(np.float32)

    W_edited, report = apply_nullspace_rank1_surgery(
        weight=W,
        trigger_key=k_trigger,
        steering_vector=v_steer,
        preserved_keys=K_preserved,
        alpha=2.5,
        tensor_name="model.layers.15.self_attn.o_proj.weight",
        layer_idx=15,
    )

    assert report.nullspace_leakage_max < 1e-10
    assert np.isclose(
        report.target_alignment_after - report.target_alignment_before, 2.5, atol=1e-4
    )
    # Preserved keys produce identical outputs before and after surgery
    np.testing.assert_allclose(W @ K_preserved.T, W_edited @ K_preserved.T, atol=1e-6)


def test_surgical_plan_generation() -> None:
    """Verify surgical plan targets only mid-to-late Full-Attention anchors and freezes routed experts."""
    w36 = load_wiring_map(profile_name="Qwen3.6-35B-A3B")
    plan = build_surgical_plan(w36, include_shared_experts=True, only_mid_to_late_anchors=True)
    assert len(plan) > 0
    for entry in plan:
        l_idx = int(entry["layer_idx"])
        assert (l_idx + 1) % 4 == 0  # Every 4th layer is a Full-Attention anchor
        assert entry["attention_type"] == "full_attention"
        assert entry["frozen_routed_experts"] == 256
