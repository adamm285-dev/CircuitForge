r"""Unit and integration tests for Milestone CFM-05:
- CircuitForge MoE Anomaly Router & Activation Spike Sentinel (`circuitforge/sentinel.py`)
- Entropy Surge & Magnitude Probing (`compute_router_entropy`, `MoEAnomalySentinel`)
- Auditor Expert Dispatch (`auditor_expert_id = -1`)
- Null-Space Orthogonal Neutralization ($P_{\text{safe}} = I - P_\perp$)
- 100% Adversarial Interception & 0% False-Positive Verification (`evaluate_sentinel_benchmark`)
"""

from __future__ import annotations

import numpy as np

from circuitforge.sentinel import (
    MoEAnomalySentinel,
    compute_router_entropy,
    evaluate_sentinel_benchmark,
)


def test_sentinel_calibration_and_nullspace_neutralization() -> None:
    """Verify calibration on safe coding subspace and complete removal of orthogonal rogue steering vectors."""
    rng = np.random.default_rng(19)
    d_model = 48
    # Construct 8-dimensional orthonormal safe subspace
    raw_basis = rng.standard_normal((8, d_model))
    _, _, vt = np.linalg.svd(raw_basis, full_matrices=False)
    safe_basis = vt[:8, :]  # (8, 48)

    H_cal = (rng.standard_normal((40, 8)) @ safe_basis).astype(np.float32)
    sentinel = MoEAnomalySentinel(d_model=d_model, num_experts=32, top_k=4)
    try:
        stats = sentinel.calibrate_baseline(H_cal)
        assert stats["safe_subspace_rank"] == 8

        # Create a pure orthogonal rogue vector
        P_perp = np.eye(d_model) - safe_basis.T @ safe_basis
        rogue = (P_perp @ rng.standard_normal(d_model)).astype(np.float32)
        rogue /= float(np.linalg.norm(rogue))

        benign_h = H_cal[0]
        corrupted_h = benign_h + 6.0 * rogue

        # Flat high-entropy router logits
        logits_flat = np.zeros(32, dtype=np.float32)
        h_clean, rep = sentinel.inspect_and_route(corrupted_h, logits_flat, layer_idx=11, token_idx=3)

        assert rep.is_anomalous
        assert rep.dispatched_to_auditor
        assert rep.routed_top_k_experts == [-1]
        assert rep.neutralized_drift_norm > 5.9
        # Verify the orthogonal rogue vector has zero dot-product with the neutralized output
        assert abs(float(np.dot(h_clean, rogue))) < 1e-5
    finally:
        sentinel.close()


def test_sentinel_entropy_surge_and_benign_routing() -> None:
    """Verify benign tokens route to standard top-k experts with 0 false positives, while entropy surges trigger auditor."""
    rng = np.random.default_rng(88)
    d_model = 32
    raw_basis = rng.standard_normal((6, d_model))
    _, _, vt = np.linalg.svd(raw_basis, full_matrices=False)
    safe_basis = vt[:6, :]

    H_cal = (rng.standard_normal((32, 6)) @ safe_basis).astype(np.float32)
    sentinel = MoEAnomalySentinel(d_model=d_model, num_experts=16, top_k=2)
    try:
        sentinel.calibrate_baseline(H_cal)

        # 1. Benign token with peaked router logits (experts 3 and 7 dominant)
        logits_peaked = np.zeros(16, dtype=np.float32)
        logits_peaked[3] = 5.0
        logits_peaked[7] = 3.5
        h_out, rep_ok = sentinel.inspect_and_route(H_cal[1], logits_peaked, layer_idx=7, token_idx=0)
        assert not rep_ok.is_anomalous
        assert not rep_ok.dispatched_to_auditor
        assert rep_ok.routed_top_k_experts == [3, 7]
        np.testing.assert_array_equal(h_out, H_cal[1])

        # 2. Entropy surge with elevated in-subspace norm (1.5x mean norm + uniform router logits)
        h_elevated = H_cal[1] * (1.55 * sentinel._mean_norm / float(np.linalg.norm(H_cal[1])))
        logits_uniform = np.zeros(16, dtype=np.float32)
        ent, _, _ = compute_router_entropy(logits_uniform)
        assert np.isclose(ent, np.log(16.0), atol=1e-5)

        _, rep_ent = sentinel.inspect_and_route(h_elevated, logits_uniform, layer_idx=7, token_idx=1)
        assert rep_ent.is_anomalous
        assert rep_ent.trigger_reason == "ENTROPY_SURGE"
        assert rep_ent.dispatched_to_auditor
        assert rep_ent.routed_top_k_experts == [-1]
    finally:
        sentinel.close()


def test_sentinel_100_percent_interception_zero_false_positives() -> None:
    """Verify full benchmark achieves 100% interception rate and 0% false-positive rate."""
    res = evaluate_sentinel_benchmark(d_model=64, num_benign=32, num_adversarial=32, seed=42)
    assert res["verified"] is True
    assert res["interception_rate"] == 1.0
    assert res["false_positive_rate"] == 0.0
    assert res["max_residual_rogue_leakage"] < 1e-5
