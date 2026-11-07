"""Unit and integration tests for Milestone CFM-03:
- Native C++ Zero-Copy Tensor Engine (`src/engine/` & `circuitforge/native.py`)
- Zero-copy memory-mapped `.safetensors` in-place mutation (`NativeMmapCheckpoint`)
- Lock-Free SPSC Telemetry Ring Buffer (`NativeTelemetryRingBuffer`)
- C++ AVX2 SIMD Null-Space Rank-1 Surgery, Router Entropy, and <50us Microbenchmark
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from circuitforge.io import load_safetensor_slice, save_safetensors_file
from circuitforge.native import (
    NativeMmapCheckpoint,
    NativeTelemetryRingBuffer,
    TelemetryEvent,
    benchmark_native_vs_numpy,
    has_native_engine,
    native_directional_ablation_f32,
    native_nullspace_rank1_surgery_f32,
    native_rmsnorm_matvec_f32,
    native_router_entropy_f32,
)
from circuitforge.surgery import apply_directional_ablation, apply_nullspace_rank1_surgery


def test_native_engine_loaded_and_microbenchmark() -> None:
    """Verify the C++ SIMD DLL is loaded and achieves <50us latency on 512x512 rank-1 updates."""
    assert has_native_engine(), "Expected compiled C++ circuitforge_engine.dll to be loaded"
    bench = benchmark_native_vs_numpy(dim=512, iterations=50)
    assert bench["meets_50us_target"], f"C++ SIMD latency {bench['cpp_simd_avg_us']}us exceeded 50us target"
    assert bench["cpp_simd_avg_us"] < bench["numpy_avg_us"]


def test_native_nullspace_rank1_parity_with_numpy() -> None:
    """Verify C++ SIMD null-space rank-1 surgery matches Python/NumPy surgery and preserves null-space."""
    rng = np.random.default_rng(77)
    d_out, d_in = 64, 64
    W_base = (rng.standard_normal((d_out, d_in)) * 0.05).astype(np.float32)
    K_preserved = rng.standard_normal((10, d_in)).astype(np.float32)
    k_trigger = rng.standard_normal(d_in).astype(np.float32)
    v_steer = rng.standard_normal(d_out).astype(np.float32)

    W_np, rep_np = apply_nullspace_rank1_surgery(
        weight=W_base,
        trigger_key=k_trigger,
        steering_vector=v_steer,
        preserved_keys=K_preserved,
        alpha=2.25,
        tensor_name="model.layers.7.self_attn.o_proj.weight",
        layer_idx=7,
    )
    W_cpp, rep_cpp = native_nullspace_rank1_surgery_f32(
        weight=W_base,
        trigger_key=k_trigger,
        steering_vector=v_steer,
        preserved_keys=K_preserved,
        alpha=2.25,
        tensor_name="model.layers.7.self_attn.o_proj.weight",
        layer_idx=7,
    )

    np.testing.assert_allclose(W_cpp, W_np, atol=1e-5)
    assert rep_cpp.nullspace_leakage_max < 1e-5
    assert np.isclose(rep_cpp.target_alignment_after - rep_cpp.target_alignment_before, 2.25, atol=1e-4)

    # Routed expert protection gate
    with pytest.raises(PermissionError):
        native_nullspace_rank1_surgery_f32(
            W_base, k_trigger, v_steer, tensor_name="model.layers.7.mlp.experts.2.down_proj.weight"
        )


def test_native_mmap_zero_copy_safetensors_mutation(tmp_path: Path) -> None:
    """Verify zero-copy memory-mapped in-place weight surgery persists directly to the .safetensors file."""
    rng = np.random.default_rng(123)
    W_ctrl = (rng.standard_normal((32, 32)) * 0.02).astype(np.float32)
    W_exp = np.ones((16, 16), dtype=np.float32) * 3.14
    K0 = rng.standard_normal((6, 32)).astype(np.float32)
    k_star = rng.standard_normal(32).astype(np.float32)
    v_steer = rng.standard_normal(32).astype(np.float32)

    st_file = tmp_path / "mmap_shard.safetensors"
    save_safetensors_file(
        st_file,
        {
            "model.layers.3.self_attn.o_proj.weight": W_ctrl,
            "model.layers.3.mlp.experts.0.down_proj.weight": W_exp,
        },
    )

    with NativeMmapCheckpoint(st_file, writable=True) as mmap_ckpt:
        # Attempting to get a writable zero-copy view on a routed expert must raise PermissionError
        with pytest.raises(PermissionError):
            mmap_ckpt.get_tensor_view("model.layers.3.mlp.experts.0.down_proj.weight", for_write=True)

        rep = mmap_ckpt.inplace_nullspace_rank1_surgery(
            tensor_name="model.layers.3.self_attn.o_proj.weight",
            trigger_key=k_star,
            steering_vector=v_steer,
            preserved_keys=K0,
            alpha=3.0,
            layer_idx=3,
        )
        assert np.isclose(rep.target_alignment_after - rep.target_alignment_before, 3.0, atol=1e-4)
        assert rep.nullspace_leakage_max < 1e-5

    # Read back from disk using pure-Python slice reader to verify in-place persistence
    W_reloaded = load_safetensor_slice(st_file, "model.layers.3.self_attn.o_proj.weight")
    assert not np.allclose(W_reloaded, W_ctrl)
    np.testing.assert_allclose(W_reloaded @ K0.T, W_ctrl @ K0.T, atol=1e-5)

    W_exp_reloaded = load_safetensor_slice(st_file, "model.layers.3.mlp.experts.0.down_proj.weight")
    np.testing.assert_array_equal(W_exp_reloaded, W_exp)


def test_native_telemetry_ring_buffer_and_router_entropy() -> None:
    """Verify lock-free SPSC telemetry ring buffer and C++ router entropy kernel."""
    logits = np.zeros(8, dtype=np.float32)
    ent_uniform, top_id, top_prob = native_router_entropy_f32(logits)
    assert np.isclose(ent_uniform, np.log(8.0), atol=1e-5)
    assert np.isclose(top_prob, 1.0 / 8.0, atol=1e-5)

    rb = NativeTelemetryRingBuffer(capacity=8)
    assert rb.capacity == 8
    assert len(rb) == 0

    for step in range(5):
        ok = rb.push(
            TelemetryEvent(
                timestamp_ns=1000 + step,
                step_id=step,
                layer_idx=15,
                activation_l2_norm=12.5 + step,
                router_entropy=ent_uniform,
                steering_alpha=1.5,
                top_expert_id=top_id,
                top_expert_weight=top_prob,
                anomaly_flag=0,
            )
        )
        assert ok

    assert len(rb) == 5
    drained = rb.drain_all()
    assert len(drained) == 5
    assert [e.step_id for e in drained] == [0, 1, 2, 3, 4]
    assert len(rb) == 0
    rb.close()


def test_native_directional_ablation_and_rmsnorm_matvec_parity() -> None:
    """Verify C++ SIMD directional ablation and RMSNorm+MatVec kernels match NumPy reference."""
    rng = np.random.default_rng(314)
    d_out, d_in = 32, 32
    W = (rng.standard_normal((d_out, d_in)) * 0.1).astype(np.float32)
    r_dir = rng.standard_normal(d_out).astype(np.float32)

    W_np, rep_np = apply_directional_ablation(W, r_dir, strength=1.0)
    W_cpp, rep_cpp = native_directional_ablation_f32(W, r_dir, strength=1.0)

    np.testing.assert_allclose(W_cpp, W_np, atol=1e-5)
    assert rep_cpp.target_alignment_after < 1e-5
    assert np.isclose(rep_cpp.frobenius_delta, rep_np.frobenius_delta, atol=1e-5)

    # Routed expert gate check
    with pytest.raises(PermissionError):
        native_directional_ablation_f32(W, r_dir, tensor_name="model.layers.3.mlp.experts.0.down_proj.weight")

    # RMSNorm + MatVec parity
    x = rng.standard_normal(d_in).astype(np.float32)
    gamma = rng.uniform(0.5, 1.5, size=d_in).astype(np.float32)
    eps = 1e-6

    rms = np.sqrt(np.mean(x ** 2) + eps)
    x_norm_ref = (x / rms) * gamma
    out_ref = W @ x_norm_ref
    out_cpp = native_rmsnorm_matvec_f32(W, x, gamma, eps=eps)
    np.testing.assert_allclose(out_cpp, out_ref, atol=1e-5)

