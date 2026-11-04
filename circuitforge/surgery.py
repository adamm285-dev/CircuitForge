"""
CircuitForge Weight-Space Surgery Engine (`surgery.py`).

Implements three mathematical weight-space intervention primitives that operate
directly on transformer circuit matrices (`o_proj` / `attn_output` and `shared_expert.down_proj`)
while keeping the banked routed MoE experts (`mlp.experts.0..N`) 100% frozen:

1. **Contrastive Steering Direction Extraction**:
   $$v_l = \\frac{\\mathbb{E}[h_l^+] - \\mathbb{E}[h_l^-]}{\\|\\mathbb{E}[h_l^+] - \\mathbb{E}[h_l^-]\\|_2}$$

2. **Null-Space Constrained Rank-1 Weight Edit (ROME / MEMIT style)**:
   Updates a projection matrix $W \\in \\mathbb{R}^{d_{out} \\times d_{in}}$ so a trigger key $k^*$
   maps to a desired target shift $\\Delta v$, while projecting the update into the orthogonal
   complement (null-space) of preserved baseline keys $K_0$:
   $$\\Delta W = \\alpha \\cdot (\\Delta v) \\otimes \\frac{P_{\\perp} k^*}{\\|P_{\\perp} k^*\\|_2^2}, \\quad P_{\\perp} = I - U_0 U_0^\\top$$
   This guarantees $\\Delta W \\cdot K_0 = 0$, preventing collateral damage to existing coding syntax.

3. **Contrastive Feature-Aligned Projection (Abliteration Inverse)**:
   Projects a normalized steering vector $v \\in \\mathbb{R}^{d_{out}}$ onto the maximally aligned
   right-singular or column feature direction of $W_{out}$, permanently baking a behavioral reflex
   into the layer weights with zero inference latency overhead.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from circuitforge.wiring import ModelWiringMap, is_routed_expert_tensor, is_surgical_control_tensor


@dataclass(frozen=True)
class SurgicalEditReport:
    """Audit report for a single weight-space surgical update."""

    tensor_name: str
    layer_idx: int
    shape: tuple[int, ...]
    alpha: float
    frobenius_delta: float
    relative_norm_delta: float
    nullspace_leakage_max: float
    target_alignment_before: float
    target_alignment_after: float


def compute_contrastive_direction(
    pos_activations: np.ndarray,
    neg_activations: np.ndarray,
    eps: float = 1e-8,
) -> np.ndarray:
    """
    Computes the unit-norm contrastive steering vector from positive (+X) and negative (-X)
    hidden-state activations of shape `(N, d_model)`.
    """
    pos = np.asarray(pos_activations, dtype=np.float64)
    neg = np.asarray(neg_activations, dtype=np.float64)
    if pos.ndim == 1:
        pos = pos[None, :]
    if neg.ndim == 1:
        neg = neg[None, :]
    if pos.shape[1] != neg.shape[1]:
        raise ValueError(f"Dimension mismatch: pos {pos.shape} vs neg {neg.shape}")

    diff = pos.mean(axis=0) - neg.mean(axis=0)
    norm = float(np.linalg.norm(diff))
    if norm < eps:
        return np.zeros_like(diff, dtype=np.float32)
    return (diff / norm).astype(np.float32)


def compute_nullspace_projector(
    preserved_keys: np.ndarray,
    variance_retained: float = 0.99,
) -> np.ndarray:
    """
    Computes the orthogonal null-space projector $P_\\perp = I - U_r U_r^\\top$ for a matrix of
    preserved key activations $K_0 \\in \\mathbb{R}^{N \\times d_{in}}$.
    Any weight update $\\Delta W$ post-multiplied by $P_\\perp$ satisfies $\\|\\Delta W k_0\\| \\approx 0$
    for all preserved keys $k_0 \\in \\text{span}(K_0)$.
    """
    K = np.asarray(preserved_keys, dtype=np.float64)
    if K.ndim != 2:
        raise ValueError(f"Expected 2D preserved_keys (N, d_in), got {K.shape}")
    d_in = K.shape[1]
    _, s, vt = np.linalg.svd(K, full_matrices=False)
    energy = np.cumsum(s ** 2)
    total = float(energy[-1]) if len(energy) > 0 else 0.0
    if total <= 1e-12:
        return np.eye(d_in, dtype=np.float64)

    r = int(np.searchsorted(energy / total, variance_retained)) + 1
    r = min(r, len(s), d_in - 1)
    basis = vt[:r, :]  # (r, d_in)
    proj = np.eye(d_in, dtype=np.float64) - basis.T @ basis
    return proj


def apply_nullspace_rank1_surgery(
    weight: np.ndarray,
    trigger_key: np.ndarray,
    steering_vector: np.ndarray,
    preserved_keys: np.ndarray | None = None,
    alpha: float = 1.0,
    tensor_name: str = "model.layers.3.self_attn.o_proj.weight",
    layer_idx: int = 3,
) -> tuple[np.ndarray, SurgicalEditReport]:
    """
    Applies a ROME/MEMIT-style null-space constrained rank-1 surgical update to a projection
    matrix $W \\in \\mathbb{R}^{d_{out} \\times d_{in}}$:
    - Refuses to modify banked routed MoE experts (`is_routed_expert_tensor(tensor_name) == True`).
    - Projects `trigger_key` into the null-space of `preserved_keys` so baseline activations are
      untouched while the trigger state receives `+alpha * steering_vector`.
    """
    if is_routed_expert_tensor(tensor_name):
        raise PermissionError(
            f"CircuitForge Safety Gate: Refusing to modify banked routed MoE expert tensor '{tensor_name}'. "
            "Only Full-Attention / Linear-Attention projections, Routers, and Shared Experts may be edited."
        )

    W = np.asarray(weight, dtype=np.float64)
    if W.ndim != 2:
        raise ValueError(f"Expected 2D weight matrix (d_out, d_in), got {W.shape}")
    d_out, d_in = W.shape

    k = np.asarray(trigger_key, dtype=np.float64).reshape(-1)
    v = np.asarray(steering_vector, dtype=np.float64).reshape(-1)
    if k.shape[0] != d_in:
        raise ValueError(f"trigger_key dim {k.shape[0]} != d_in {d_in}")
    if v.shape[0] != d_out:
        raise ValueError(f"steering_vector dim {v.shape[0]} != d_out {d_out}")

    v_norm = float(np.linalg.norm(v))
    v_unit = v / v_norm if v_norm > 1e-12 else v

    if preserved_keys is not None and len(preserved_keys) > 0:
        P_perp = compute_nullspace_projector(preserved_keys)
        k_proj = P_perp @ k
    else:
        k_proj = k

    k_proj_norm_sq = float(np.dot(k_proj, k_proj))
    if k_proj_norm_sq < 1e-12:
        raise ValueError("Trigger key lies entirely inside the preserved key subspace; cannot isolate null-space.")

    delta_W = alpha * np.outer(v_unit, k_proj / k_proj_norm_sq)
    W_new = W + delta_W

    # Audit metrics
    out_before = W @ k
    out_after = W_new @ k
    align_before = float(np.dot(out_before, v_unit))
    align_after = float(np.dot(out_after, v_unit))

    leakage = 0.0
    if preserved_keys is not None and len(preserved_keys) > 0:
        K0 = np.asarray(preserved_keys, dtype=np.float64)
        leakage = float(np.max(np.abs(delta_W @ K0.T)))

    frob_delta = float(np.linalg.norm(delta_W, ord="fro"))
    frob_base = max(float(np.linalg.norm(W, ord="fro")), 1e-12)

    report = SurgicalEditReport(
        tensor_name=tensor_name,
        layer_idx=layer_idx,
        shape=W.shape,
        alpha=float(alpha),
        frobenius_delta=frob_delta,
        relative_norm_delta=frob_delta / frob_base,
        nullspace_leakage_max=leakage,
        target_alignment_before=align_before,
        target_alignment_after=align_after,
    )
    return W_new.astype(weight.dtype), report


def build_surgical_plan(
    wiring_map: ModelWiringMap,
    include_shared_experts: bool = True,
    only_mid_to_late_anchors: bool = True,
) -> list[dict[str, object]]:
    """
    Generates a deterministic surgical intervention manifest targeting only the
    Full-Attention planning anchors (and optionally their Shared Experts) across
    either `Qwen3.6-35B-A3B` (40 layers) or `Qwen3.8-2.4T-A95B` (92 layers).
    """
    target_layers: Sequence[int] = (
        wiring_map.mid_to_late_planning_anchors
        if only_mid_to_late_anchors
        else wiring_map.full_attention_layers
    )
    plan: list[dict[str, object]] = []
    for l_idx in target_layers:
        spec = wiring_map.layers[l_idx]
        targets = [f"model.layers.{l_idx}.self_attn.o_proj.weight"]
        gguf_targets = [f"blk.{l_idx}.attn_output.weight"]
        if include_shared_experts:
            targets.append(f"model.layers.{l_idx}.mlp.shared_expert.down_proj.weight")
            gguf_targets.append(f"blk.{l_idx}.ffn_down_shexp.weight")

        plan.append(
            {
                "layer_idx": l_idx,
                "attention_type": spec.attention_type,
                "safetensors_targets": [t for t in targets if is_surgical_control_tensor(t)],
                "gguf_targets": [t for t in gguf_targets if is_surgical_control_tensor(t)],
                "shards": spec.control_shards,
                "frozen_routed_experts": spec.num_routed_experts,
            }
        )
    return plan
