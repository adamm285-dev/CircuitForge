r"""Unit and integration tests for Milestone CFM-06:
- Offloaded Discrete Engram Memory Graph (`DiscreteEngramGraph`, C++ `NativeEngramHashTable`)
- Fused SIMD RMSNorm + Engram bias injection (`cf_engram_lookup_and_inject_f32`)
- Multi-hop episodic graph traversal (`traverse_and_synthesize_bias`)
- Parametric Manifold Exploration & Weight-Space Latent Synthesis (`ParametricAdapterManifold`, `slerp_adapters`)
- TIES-Nullspace Merging with zero baseline syntax key leakage and routed-expert protection gates
"""

from __future__ import annotations

import numpy as np
import pytest

from circuitforge.engram import (
    DiscreteEngramGraph,
    ParametricAdapterManifold,
    evaluate_engram_and_manifold_benchmark,
    slerp_adapters,
)


def test_discrete_engram_cpp_hash_table_and_rmsnorm_injection() -> None:
    """Verify C++ SIMD engram hash table lookup, RMSNorm additive bias injection, and multi-hop traversal."""
    rng = np.random.default_rng(7)
    d_model = 32
    graph = DiscreteEngramGraph(d_model=d_model, capacity=64)
    try:
        bias_a = np.ones(d_model, dtype=np.float32) * 0.5
        bias_b = np.ones(d_model, dtype=np.float32) * 0.25
        graph.store_engram("step_1", "Lock state.md first", bias_vector=bias_a, linked_keys=["step_2"])
        graph.store_engram("step_2", "Zero-read pre-hydrated bundle", bias_vector=bias_b)
        assert len(graph) == 2

        x = rng.standard_normal(d_model).astype(np.float32)
        gamma = np.ones(d_model, dtype=np.float32)
        eps = 1e-6

        # Hit on step_1
        out_hit, is_hit = graph.inject_rmsnorm("step_1", x, gamma, scale=2.0, eps=eps)
        assert is_hit is True
        rms_ref = (x / np.sqrt(np.mean(x ** 2) + eps)) * gamma
        np.testing.assert_allclose(out_hit, rms_ref + 2.0 * bias_a, atol=1e-5)

        # Miss on unknown key returns pure RMSNorm
        out_miss, is_miss = graph.inject_rmsnorm("nonexistent_key", x, gamma, scale=2.0, eps=eps)
        assert is_miss is False
        np.testing.assert_allclose(out_miss, rms_ref, atol=1e-5)

        # Multi-hop graph traversal
        synth_vec, visited = graph.traverse_and_synthesize_bias("step_1", max_hops=2, decay=0.5)
        assert visited == ["step_1", "step_2"]
        assert np.isclose(np.linalg.norm(synth_vec), 1.0, atol=1e-5)
    finally:
        graph.close()


def test_parametric_adapter_manifold_slerp_and_ties_nullspace() -> None:
    """Verify SLERP geodesic norm preservation, manifold coordinate sampling, and TIES null-space merging."""
    rng = np.random.default_rng(91)
    d_out, d_in = 32, 32

    dw1 = rng.standard_normal((d_out, d_in)).astype(np.float32)
    dw1 /= float(np.linalg.norm(dw1))
    dw2 = rng.standard_normal((d_out, d_in)).astype(np.float32)
    dw2 /= float(np.linalg.norm(dw2))

    # SLERP at t=0.5 preserves unit Frobenius norm (unlike linear midpoint which shrinks norm)
    mid_slerp = slerp_adapters(dw1, dw2, t=0.5)
    mid_linear = 0.5 * dw1 + 0.5 * dw2
    assert np.isclose(np.linalg.norm(mid_slerp), 1.0, atol=1e-5)
    assert np.linalg.norm(mid_slerp) > np.linalg.norm(mid_linear)

    # Routed expert protection gate on SLERP and ParametricAdapterManifold
    with pytest.raises(PermissionError):
        slerp_adapters(dw1, dw2, t=0.5, tensor_name="model.layers.3.mlp.experts.0.down_proj.weight")
    with pytest.raises(PermissionError):
        ParametricAdapterManifold(d_out, d_in, tensor_name="model.layers.3.mlp.experts.1.down_proj.weight")

    # Parametric manifold fit, coordinate sampling, and TIES merge with preserved keys K0
    K0 = rng.standard_normal((8, d_in)).astype(np.float32)
    manifold = ParametricAdapterManifold(d_out, d_in, tensor_name="model.layers.15.self_attn.o_proj.weight")
    manifold.register_adapter("code_style", dw1)
    manifold.register_adapter("spec_lock", dw2)
    manifold.register_adapter("concise_state", (dw1 - 0.5 * dw2))

    stats = manifold.fit_manifold_basis(max_components=2)
    assert stats["num_adapters"] == 3
    assert stats["manifold_dim"] == 2

    sampled = manifold.sample_from_coordinates([0.5, -0.25], preserved_keys=K0)
    assert float(np.max(np.abs(sampled @ K0.T))) < 1e-6

    merged = manifold.ties_nullspace_merge(density=0.75, preserved_keys=K0)
    assert float(np.max(np.abs(merged @ K0.T))) < 1e-6


def test_engram_and_manifold_benchmark_cli() -> None:
    """Verify full CFM-06 benchmark suite and CLI entrypoint."""
    res = evaluate_engram_and_manifold_benchmark(d_model=64, num_facts=24, seed=42)
    assert res["verified"] is True
    assert res["kv_token_savings_pct"] == 100.0
    assert res["ties_nullspace_leakage"] < 1e-6
    assert res["sampled_manifold_leakage"] < 1e-6
