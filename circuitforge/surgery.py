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
import math
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
    variance_retained: float = 1.0,
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
    s_max = float(s[0]) if len(s) > 0 else 0.0
    if s_max <= 1e-12:
        return np.eye(d_in, dtype=np.float64)

    s_tol = max(1e-12, s_max * 1e-7)
    nonzero_rank = int(np.sum(s > s_tol))
    if variance_retained >= 1.0 - 1e-9:
        r = nonzero_rank
    else:
        energy = np.cumsum(s ** 2)
        total = float(energy[-1])
        r_var = int(np.searchsorted(energy / total, variance_retained)) + 1
        r = min(r_var, nonzero_rank)
    r = min(r, d_in)
    if r >= d_in:
        return np.zeros((d_in, d_in), dtype=np.float64)
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

    if not math.isfinite(float(alpha)):
        raise ValueError("alpha must be finite (non-NaN, non-Inf).")
    if not np.all(np.isfinite(weight)):
        raise ValueError("weight matrix contains NaN or Inf values.")
    if not np.all(np.isfinite(trigger_key)):
        raise ValueError("trigger_key contains NaN or Inf values.")
    if not np.all(np.isfinite(steering_vector)):
        raise ValueError("steering_vector contains NaN or Inf values.")
    if preserved_keys is not None and len(preserved_keys) > 0 and not np.all(np.isfinite(preserved_keys)):
        raise ValueError("preserved_keys contains NaN or Inf values.")

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

    k_orig_norm_sq = float(np.dot(k, k))
    k_proj_norm_sq = float(np.dot(k_proj, k_proj))
    if k_proj_norm_sq <= max(1e-12, 1e-4 * k_orig_norm_sq):
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


def apply_nullspace_rank_k_surgery(
    weight: np.ndarray,
    trigger_keys: np.ndarray,
    steering_vectors: np.ndarray,
    preserved_keys: np.ndarray | None = None,
    alpha: float = 1.0,
    damping: float = 1e-4,
    tensor_name: str = "model.layers.3.self_attn.o_proj.weight",
    layer_idx: int = 3,
) -> tuple[np.ndarray, SurgicalEditReport]:
    """
    Applies a MEMIT-style null-space constrained rank-k surgical update to a projection
    matrix $W \\in \\mathbb{R}^{d_{out} \\times d_{in}}$ for a batch of $k$ trigger keys
    $K^* \\in \\mathbb{R}^{k \\times d_{in}}$ and target steering shifts $V \\in \\mathbb{R}^{k \\times d_{out}}$:
    $$\\Delta W = \\alpha \\cdot V^\\top (K_\\perp {K_\\perp}^\\top + \\lambda I)^{-1} K_\\perp$$
    where $K_\\perp = K^* P_\\perp$.
    Guarantees $\\Delta W \\cdot K_0 = 0$ on all preserved baseline syntax keys $K_0$.
    """
    if is_routed_expert_tensor(tensor_name):
        raise PermissionError(
            f"CircuitForge Safety Gate: Refusing to modify banked routed MoE expert tensor '{tensor_name}'. "
            "Only Full-Attention / Linear-Attention projections, Routers, and Shared Experts may be edited."
        )

    if not math.isfinite(float(alpha)) or not math.isfinite(float(damping)):
        raise ValueError("alpha and damping must be finite (non-NaN, non-Inf).")
    if not np.all(np.isfinite(weight)):
        raise ValueError("weight matrix contains NaN or Inf values.")
    if not np.all(np.isfinite(trigger_keys)):
        raise ValueError("trigger_keys contains NaN or Inf values.")
    if not np.all(np.isfinite(steering_vectors)):
        raise ValueError("steering_vectors contains NaN or Inf values.")
    if preserved_keys is not None and len(preserved_keys) > 0 and not np.all(np.isfinite(preserved_keys)):
        raise ValueError("preserved_keys contains NaN or Inf values.")

    W = np.asarray(weight, dtype=np.float64)
    if W.ndim != 2:
        raise ValueError(f"Expected 2D weight matrix (d_out, d_in), got {W.shape}")
    d_out, d_in = W.shape

    K_star = np.asarray(trigger_keys, dtype=np.float64)
    V = np.asarray(steering_vectors, dtype=np.float64)
    if K_star.ndim == 1:
        K_star = K_star[None, :]
    if V.ndim == 1:
        V = V[None, :]
    if K_star.shape[1] != d_in:
        raise ValueError(f"trigger_keys dim {K_star.shape[1]} != d_in {d_in}")
    if V.shape[1] != d_out:
        raise ValueError(f"steering_vectors dim {V.shape[1]} != d_out {d_out}")
    if K_star.shape[0] != V.shape[0]:
        raise ValueError(f"Batch mismatch: {K_star.shape[0]} keys vs {V.shape[0]} vectors")

    # Normalize each steering row to unit norm
    v_norms = np.linalg.norm(V, axis=1, keepdims=True)
    V_unit = np.where(v_norms > 1e-12, V / np.maximum(v_norms, 1e-12), V)

    if preserved_keys is not None and len(preserved_keys) > 0:
        P_perp = compute_nullspace_projector(preserved_keys)
        K_perp = K_star @ P_perp.T  # (k, d_in) since P_perp is symmetric
    else:
        K_perp = K_star

    orig_gram_trace = float(np.sum(K_star ** 2))
    gram = K_perp @ K_perp.T  # (k, k)
    trace_gram = float(np.trace(gram))
    if trace_gram <= max(1e-12, 1e-4 * orig_gram_trace):
        raise ValueError("Trigger keys lie entirely inside the preserved key subspace; cannot isolate null-space.")

    k_count = K_perp.shape[0]
    eff_damping = float(damping) * max(trace_gram / max(k_count, 1), 1.0)
    gram_reg = gram + eff_damping * np.eye(k_count, dtype=np.float64)
    # Solve (K_perp K_perp^T + lambda I) X = K_perp  => X = (gram_reg)^-1 K_perp
    coeffs = np.linalg.solve(gram_reg, K_perp)  # (k, d_in)
    delta_W = alpha * (V_unit.T @ coeffs)  # (d_out, d_in)
    W_new = W + delta_W

    # Mean alignment across all k trigger keys
    out_before = (W @ K_star.T).T  # (k, d_out)
    out_after = (W_new @ K_star.T).T  # (k, d_out)
    align_before = float(np.mean(np.sum(out_before * V_unit, axis=1)))
    align_after = float(np.mean(np.sum(out_after * V_unit, axis=1)))

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


def apply_directional_ablation(
    weight: np.ndarray,
    refusal_or_target_dir: np.ndarray,
    strength: float = 1.0,
    tensor_name: str = "model.layers.3.self_attn.o_proj.weight",
    layer_idx: int = 3,
) -> tuple[np.ndarray, SurgicalEditReport]:
    """
    Applies orthogonal directional ablation (FailSpy / Arditi et al. 'abliteration')
    to remove a specific output representation direction $r \\in \\mathbb{R}^{d_{out}}$ from
    a projection matrix $W \\in \\mathbb{R}^{d_{out} \\times d_{in}}$:
    $$W_{\\text{new}} = (I - \\beta \\cdot \\hat{r} \\hat{r}^\\top) W$$
    When $\\beta = 1.0$, $\\hat{r}^\\top W_{\\text{new}} = 0$ for all inputs.
    """
    if is_routed_expert_tensor(tensor_name):
        raise PermissionError(
            f"CircuitForge Safety Gate: Refusing to modify banked routed MoE expert tensor '{tensor_name}'."
        )

    if not math.isfinite(float(strength)):
        raise ValueError("strength must be finite (non-NaN, non-Inf).")
    if not np.all(np.isfinite(weight)):
        raise ValueError("weight matrix contains NaN or Inf values.")
    if not np.all(np.isfinite(refusal_or_target_dir)):
        raise ValueError("refusal_or_target_dir contains NaN or Inf values.")

    W = np.asarray(weight, dtype=np.float64)
    if W.ndim != 2:
        raise ValueError(f"Expected 2D weight matrix (d_out, d_in), got {W.shape}")
    d_out, _ = W.shape

    r = np.asarray(refusal_or_target_dir, dtype=np.float64).reshape(-1)
    if r.shape[0] != d_out:
        raise ValueError(f"refusal_or_target_dir dim {r.shape[0]} != d_out {d_out}")

    r_norm = float(np.linalg.norm(r))
    if r_norm < 1e-12:
        raise ValueError("Target ablation direction has near-zero norm.")
    r_unit = r / r_norm

    proj_row = r_unit @ W  # (d_in,)
    delta_W = -strength * np.outer(r_unit, proj_row)
    W_new = W + delta_W

    align_before = float(np.linalg.norm(r_unit @ W))
    align_after = float(np.linalg.norm(r_unit @ W_new))
    frob_delta = float(np.linalg.norm(delta_W, ord="fro"))
    frob_base = max(float(np.linalg.norm(W, ord="fro")), 1e-12)

    report = SurgicalEditReport(
        tensor_name=tensor_name,
        layer_idx=layer_idx,
        shape=W.shape,
        alpha=float(strength),
        frobenius_delta=frob_delta,
        relative_norm_delta=frob_delta / frob_base,
        nullspace_leakage_max=0.0,
        target_alignment_before=align_before,
        target_alignment_after=align_after,
    )
    return W_new.astype(weight.dtype), report


def distribute_multilayer_surgery(
    layer_weights: dict[int, np.ndarray],
    trigger_keys: dict[int, np.ndarray],
    steering_vectors: dict[int, np.ndarray],
    preserved_keys: dict[int, np.ndarray] | None = None,
    total_alpha: float = 2.0,
    schedule: str = "bell",
    tensor_template: str = "model.layers.{layer}.self_attn.o_proj.weight",
) -> tuple[dict[int, np.ndarray], list[SurgicalEditReport]]:
    """
    Distributes a surgical behavioral update smoothly across multiple transformer layers
    so no single layer suffers an extreme Frobenius norm spike.
    Supports `schedule='uniform'` or `schedule='bell'` (Gaussian-like cosine bell curve
    peaking in the middle of the selected layers).
    """
    layers = sorted(layer_weights.keys())
    if not layers:
        return {}, []

    n = len(layers)
    if schedule == "uniform" or n == 1:
        weights_sched = np.ones(n, dtype=np.float64) / n
    elif schedule == "bell":
        # Raised sine bell across layer indices
        idx = np.arange(n, dtype=np.float64)
        raw = np.sin(np.pi * (idx + 1.0) / (n + 1.0)) ** 2
        weights_sched = raw / np.sum(raw)
    else:
        raise ValueError(f"Unknown schedule '{schedule}'. Expected 'uniform' or 'bell'.")

    edited_weights: dict[int, np.ndarray] = {}
    reports: list[SurgicalEditReport] = []

    for i, l_idx in enumerate(layers):
        layer_alpha = float(total_alpha * weights_sched[i])
        tname = tensor_template.format(layer=l_idx)
        W_l = layer_weights[l_idx]
        k_l = trigger_keys[l_idx]
        v_l = steering_vectors[l_idx]
        K0_l = preserved_keys.get(l_idx) if preserved_keys else None

        if k_l.ndim == 2 and k_l.shape[0] > 1:
            W_ed, rep = apply_nullspace_rank_k_surgery(
                weight=W_l,
                trigger_keys=k_l,
                steering_vectors=v_l,
                preserved_keys=K0_l,
                alpha=layer_alpha,
                tensor_name=tname,
                layer_idx=l_idx,
            )
        else:
            W_ed, rep = apply_nullspace_rank1_surgery(
                weight=W_l,
                trigger_key=k_l.reshape(-1),
                steering_vector=v_l.reshape(-1),
                preserved_keys=K0_l,
                alpha=layer_alpha,
                tensor_name=tname,
                layer_idx=l_idx,
            )
        edited_weights[l_idx] = W_ed
        reports.append(rep)

    return edited_weights, reports


def build_surgical_plan(
    wiring_map: ModelWiringMap,
    include_shared_experts: bool = True,
    only_mid_to_late_anchors: bool = True,
) -> list[dict[str, object]]:
    """
    Generates a deterministic surgical intervention manifest targeting the
    Full-Attention planning anchors and their Shared Experts (or Dense MLPs) across
    any supported hybrid, sparse MoE, or dense GQA architecture.
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
            if wiring_map.num_experts == 0:
                targets.append(f"model.layers.{l_idx}.mlp.down_proj.weight")
                gguf_targets.append(f"blk.{l_idx}.ffn_down.weight")
            elif wiring_map.shared_expert_intermediate_size > 0:
                if "Deepseek" in wiring_map.architecture:
                    targets.append(f"model.layers.{l_idx}.mlp.shared_experts.down_proj.weight")
                else:
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
