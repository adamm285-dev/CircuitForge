r"""
CircuitForge MoE Anomaly Router & Activation Spike Sentinel (`circuitforge/sentinel.py`).

Implements Milestone CFM-05:
1. **Entropy Surge & Magnitude Probing**:
   Continuously monitors MoE router gate logits ($\mathcal{H}_{\text{router}}$ via C++ SIMD
   `native_router_entropy_f32`) and hidden-state activation L2 norms ($\|h\|_2$) against
   calibrated statistical thresholds ($z$-score and ratio bounds) to detect adversarial
   prompt attacks, visual hijacking, or rogue steering vectors.
2. **Auditor Expert Dispatch**:
   When an activation spike or entropy surge is detected, automatically routes the anomalous
   token to a dedicated circuit-breaker / auditor expert (`auditor_expert_id = -1`) instead
   of corrupting downstream routed syntax experts.
3. **Null-Space Orthogonal Neutralization**:
   Inside the auditor expert, projects the anomalous hidden state onto the safe baseline
   coding subspace ($P_{\text{safe}} = U_0 U_0^\top$), stripping out all orthogonal rogue
   steering directions ($P_\perp r_{\text{rogue}} = 0$) and clamping magnitude back to the
   calibrated baseline distribution before re-entering the residual stream.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from circuitforge.native import (
    NativeTelemetryRingBuffer,
    TelemetryEvent,
    has_native_engine,
    native_router_entropy_f32,
)
from circuitforge.surgery import compute_nullspace_projector


@dataclass(frozen=True)
class SentinelProbeReport:
    """Diagnostic audit record emitted by the MoE Anomaly Router for a single token/state."""

    layer_idx: int
    token_idx: int
    norm_before: float
    norm_after: float
    z_score_norm: float
    router_entropy: float
    entropy_ratio: float
    is_anomalous: bool
    trigger_reason: str  # "NORMAL" | "MAGNITUDE_SPIKE" | "ENTROPY_SURGE" | "ORTHOGONAL_DRIFT"
    dispatched_to_auditor: bool
    routed_top_k_experts: list[int]
    neutralized_drift_norm: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "layer_idx": self.layer_idx,
            "token_idx": self.token_idx,
            "norm_before": round(self.norm_before, 5),
            "norm_after": round(self.norm_after, 5),
            "z_score_norm": round(self.z_score_norm, 4),
            "router_entropy": round(self.router_entropy, 5),
            "entropy_ratio": round(self.entropy_ratio, 4),
            "is_anomalous": self.is_anomalous,
            "trigger_reason": self.trigger_reason,
            "dispatched_to_auditor": self.dispatched_to_auditor,
            "routed_top_k_experts": self.routed_top_k_experts,
            "neutralized_drift_norm": round(self.neutralized_drift_norm, 5),
        }


def compute_router_entropy(logits: np.ndarray) -> tuple[float, int, float]:
    """
    Computes Shannon entropy of router gate logits, top-1 expert index, and top-1 probability.
    Uses C++ AVX2 SIMD `native_router_entropy_f32` when available, falling back to NumPy.
    """
    if has_native_engine():
        return native_router_entropy_f32(logits)
    arr = np.asarray(logits, dtype=np.float64).reshape(-1)
    shifted = arr - np.max(arr)
    exps = np.exp(shifted)
    probs = exps / np.sum(exps)
    ent = float(-np.sum(probs * np.log(np.maximum(probs, 1e-12))))
    top_id = int(np.argmax(probs))
    return ent, top_id, float(probs[top_id])


@dataclass
class MoEAnomalySentinel:
    """
    Automated anomaly-detection and null-space neutralization circuit breaker
    operating inside Mixture-of-Experts routing layers.
    """

    d_model: int
    num_experts: int = 256
    top_k: int = 8
    auditor_expert_id: int = -1
    norm_z_threshold: float = 3.5
    norm_ratio_threshold: float = 1.85
    orthogonal_energy_threshold: float = 0.35
    entropy_surge_ratio: float = 0.96
    variance_retained: float = 0.999

    _mean_norm: float = field(default=1.0, init=False)
    _std_norm: float = field(default=0.15, init=False)
    _P_perp: np.ndarray = field(init=False)
    _P_safe: np.ndarray = field(init=False)
    _calibrated: bool = field(default=False, init=False)
    _ringbuf: NativeTelemetryRingBuffer | None = field(default=None, init=False)
    _step_counter: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        self._P_perp = np.eye(self.d_model, dtype=np.float64)
        self._P_safe = np.eye(self.d_model, dtype=np.float64)
        if has_native_engine():
            self._ringbuf = NativeTelemetryRingBuffer(capacity=512)

    def close(self) -> None:
        if self._ringbuf is not None:
            self._ringbuf.close()
            self._ringbuf = None

    def calibrate_baseline(self, benign_activations: np.ndarray) -> dict[str, float]:
        r"""
        Calibrates the sentinel on a batch of benign coding activations `H_0` of shape `(N, d_model)`:
        1. Records baseline L2 magnitude distribution ($\mu_{\text{norm}}, \sigma_{\text{norm}}$).
        2. Computes the orthogonal complement projector $P_\perp = I - U_0 U_0^\top$ and safe
           subspace projector $P_{\text{safe}} = U_0 U_0^\top$.
        """
        H0 = np.asarray(benign_activations, dtype=np.float64)
        if H0.ndim != 2 or H0.shape[1] != self.d_model:
            raise ValueError(f"Expected 2D benign_activations (N, {self.d_model}), got {H0.shape}")

        norms = np.linalg.norm(H0, axis=1)
        self._mean_norm = max(float(np.mean(norms)), 1e-8)
        self._std_norm = max(float(np.std(norms)), 1e-4)

        self._P_perp = compute_nullspace_projector(H0, variance_retained=self.variance_retained)
        self._P_safe = np.eye(self.d_model, dtype=np.float64) - self._P_perp
        self._calibrated = True

        return {
            "mean_norm": round(self._mean_norm, 5),
            "std_norm": round(self._std_norm, 5),
            "safe_subspace_rank": int(round(float(np.trace(self._P_safe)))),
        }

    def neutralize_activation(self, h: np.ndarray) -> tuple[np.ndarray, float]:
        r"""
        Auditor Expert Circuit-Breaker execution:
        Projects `h` onto the safe baseline subspace $P_{\text{safe}} = I - P_\perp$
        to strip out orthogonal rogue steering vectors, and rescales the vector if its
        remaining norm exceeds the safe baseline mean.
        Returns `(h_clean, removed_orthogonal_drift_norm)`.
        """
        h_f64 = np.asarray(h, dtype=np.float64).reshape(-1)
        ortho_drift = self._P_perp @ h_f64
        drift_norm = float(np.linalg.norm(ortho_drift))

        h_clean = self._P_safe @ h_f64
        clean_norm = float(np.linalg.norm(h_clean))
        max_allowed = self._mean_norm + 2.0 * self._std_norm
        if clean_norm > max_allowed and clean_norm > 1e-12:
            h_clean = h_clean * (self._mean_norm / clean_norm)

        return h_clean.astype(h.dtype), drift_norm

    def inspect_and_route(
        self,
        hidden_state: np.ndarray,
        router_logits: np.ndarray,
        layer_idx: int = 3,
        token_idx: int = 0,
    ) -> tuple[np.ndarray, SentinelProbeReport]:
        """
        Probes a single token's hidden state `hidden_state` (`(d_model,)`) and MoE router logits
        (`(num_experts,)`):
        - If benign: routes normally to `top_k` routed experts and leaves `hidden_state` untouched.
        - If anomalous (magnitude spike, orthogonal rogue vector drift, or router entropy collapse/surge):
          dispatches to the dedicated `auditor_expert_id` and neutralizes the rogue component via
          null-space projection.
        """
        self._step_counter += 1
        h = np.asarray(hidden_state, dtype=np.float64).reshape(-1)
        if h.shape[0] != self.d_model:
            raise ValueError(f"Expected hidden_state dim {self.d_model}, got {h.shape[0]}")

        logits = np.asarray(router_logits, dtype=np.float32).reshape(-1)
        norm_before = float(np.linalg.norm(h))
        z_norm = (norm_before - self._mean_norm) / self._std_norm
        norm_ratio = norm_before / self._mean_norm

        # Check orthogonal energy outside calibrated safe subspace
        ortho_comp = self._P_perp @ h
        ortho_norm = float(np.linalg.norm(ortho_comp))
        ortho_ratio = ortho_norm / max(norm_before, 1e-12) if self._calibrated else 0.0

        ent, top_id, top_prob = compute_router_entropy(logits)
        max_ent = float(math.log(max(logits.size, 2)))
        ent_ratio = ent / max(max_ent, 1e-8)

        # Anomaly decision rules
        is_mag_spike = (z_norm > self.norm_z_threshold) and (norm_ratio > self.norm_ratio_threshold)
        is_ortho_attack = self._calibrated and (ortho_ratio > self.orthogonal_energy_threshold) and (norm_ratio > 1.15)
        is_ent_surge = (ent_ratio > self.entropy_surge_ratio) and (norm_ratio > 1.35)

        is_anomalous = bool(is_mag_spike or is_ortho_attack or is_ent_surge)

        if is_mag_spike:
            reason = "MAGNITUDE_SPIKE"
        elif is_ortho_attack:
            reason = "ORTHOGONAL_DRIFT"
        elif is_ent_surge:
            reason = "ENTROPY_SURGE"
        else:
            reason = "NORMAL"

        if is_anomalous:
            h_out, drift_removed = self.neutralize_activation(hidden_state)
            norm_after = float(np.linalg.norm(h_out))
            routed_experts = [self.auditor_expert_id]
            dispatched = True
        else:
            h_out = hidden_state.copy()
            drift_removed = 0.0
            norm_after = norm_before
            k_sel = min(self.top_k, logits.size)
            top_indices = np.argsort(logits)[::-1][:k_sel]
            routed_experts = [int(idx) for idx in top_indices]
            dispatched = False

        if self._ringbuf is not None:
            self._ringbuf.push(
                TelemetryEvent(
                    timestamp_ns=time.time_ns(),
                    step_id=self._step_counter,
                    layer_idx=layer_idx,
                    activation_l2_norm=norm_before,
                    router_entropy=ent,
                    steering_alpha=drift_removed,
                    top_expert_id=top_id,
                    top_expert_weight=top_prob,
                    anomaly_flag=1 if is_anomalous else 0,
                )
            )

        report = SentinelProbeReport(
            layer_idx=layer_idx,
            token_idx=token_idx,
            norm_before=norm_before,
            norm_after=norm_after,
            z_score_norm=z_norm,
            router_entropy=ent,
            entropy_ratio=ent_ratio,
            is_anomalous=is_anomalous,
            trigger_reason=reason,
            dispatched_to_auditor=dispatched,
            routed_top_k_experts=routed_experts,
            neutralized_drift_norm=drift_removed,
        )
        return h_out, report


def evaluate_sentinel_benchmark(
    d_model: int = 64,
    num_benign: int = 32,
    num_adversarial: int = 32,
    seed: int = 42,
) -> dict[str, Any]:
    """
    Runs a deterministic evaluation of the `MoEAnomalySentinel` against:
    - `num_benign` standard coding tokens drawn from the low-rank syntax subspace
    - `num_adversarial` tokens injected with synthetic rogue steering vectors, activation
      spikes, and router entropy surges.
    Verifies 100% interception of adversarial spikes and 0% false-positive re-routings.
    """
    rng = np.random.default_rng(seed)
    subspace_rank = 12
    basis_raw = rng.standard_normal((subspace_rank, d_model))
    u_sub, _, vt_sub = np.linalg.svd(basis_raw, full_matrices=False)
    safe_basis = vt_sub[:subspace_rank, :]  # (12, d_model) orthonormal rows

    # Generate calibration + benign test tokens inside the safe coding subspace
    coeffs_cal = rng.standard_normal((64, subspace_rank))
    H_cal = (coeffs_cal @ safe_basis).astype(np.float32)

    sentinel = MoEAnomalySentinel(d_model=d_model, num_experts=64, top_k=4)
    cal_stats = sentinel.calibrate_baseline(H_cal)

    # Benign holdout tokens
    coeffs_benign = rng.standard_normal((num_benign, subspace_rank))
    H_benign = (coeffs_benign @ safe_basis).astype(np.float32)

    false_positives = 0
    for i in range(num_benign):
        # Benign tokens have peaked, structured router logits
        logits = rng.standard_normal(64).astype(np.float32)
        logits[i % 8] += 4.5
        _, rep = sentinel.inspect_and_route(H_benign[i], logits, layer_idx=15, token_idx=i)
        if rep.is_anomalous or rep.dispatched_to_auditor:
            false_positives += 1

    # Adversarial tokens with rogue orthogonal steering vectors / magnitude spikes
    # Construct unit vectors strictly in the orthogonal complement of safe_basis
    P_perp = np.eye(d_model) - safe_basis.T @ safe_basis
    rogue_raw = (rng.standard_normal((num_adversarial, d_model)) @ P_perp.T).astype(np.float32)
    rogue_norms = np.linalg.norm(rogue_raw, axis=1, keepdims=True)
    rogue_unit = rogue_raw / np.maximum(rogue_norms, 1e-8)

    true_positives = 0
    max_residual_rogue_leakage = 0.0

    for j in range(num_adversarial):
        base_h = H_benign[j % num_benign]
        spike_scale = float(rng.uniform(4.5, 9.0))
        rogue_vec = spike_scale * rogue_unit[j]
        h_attacked = base_h + rogue_vec

        # Flat / confused router logits simulating entropy surge
        logits_adv = (rng.uniform(-0.05, 0.05, size=64)).astype(np.float32)
        h_cleaned, rep_adv = sentinel.inspect_and_route(
            h_attacked, logits_adv, layer_idx=15, token_idx=j
        )
        if rep_adv.is_anomalous and rep_adv.dispatched_to_auditor:
            true_positives += 1

        # Verify the rogue direction was completely stripped from h_cleaned
        leak = abs(float(np.dot(h_cleaned, rogue_unit[j])))
        if leak > max_residual_rogue_leakage:
            max_residual_rogue_leakage = leak

    sentinel.close()

    tpr = true_positives / max(num_adversarial, 1)
    fpr = false_positives / max(num_benign, 1)

    return {
        "calibration": cal_stats,
        "benign_tested": num_benign,
        "adversarial_tested": num_adversarial,
        "true_positives": true_positives,
        "false_positives": false_positives,
        "interception_rate": round(tpr, 4),
        "false_positive_rate": round(fpr, 4),
        "max_residual_rogue_leakage": float(f"{max_residual_rogue_leakage:.3e}"),
        "verified": (tpr == 1.0 and fpr == 0.0 and max_residual_rogue_leakage < 1e-5),
    }
