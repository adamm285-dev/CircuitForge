r"""
CircuitForge Automated Circuit Discovery & Causal Attribution Sweeper (`circuitforge/attribution.py`).

Implements Milestone CFM-12:
Replaces static depth heuristics (`25%..85%`) with an automated mechanistic circuit discovery
engine that sweeps across all candidate control layers ($l \in \{0, \dots, L-1\}$) and ranks them by:
1. **Direct Logit Attribution ($\text{DLA}_l$)**:
   $$\text{DLA}_l = \frac{\langle \mathbb{E}[W_l k_l^+] - \mathbb{E}[W_l k_l^-], \, v_{\text{target}, l} \rangle}{\|W_l\|_F}$$
   Measures how strongly layer $l$'s projection matrix naturally separates positive ($+X$) from
   negative ($-X$) contrastive trajectories along the target steering direction.
2. **Activation Patching Recovery ($\text{APR}_l \in [0, 1]$)**:
   Measures the fractional downstream logit recovery achieved by patching clean/surgical layer
   activations in place of corrupted/negative activations.
3. **Null-Space Signal-to-Leakage Ratio ($\text{SLR}_l$)**:
   $$\text{NullSpaceCapacity}_l = \frac{\|P_{\perp, l} k_l^*\|_2^2}{\|k_l^*\|_2^2}, \qquad \text{SLR}_l = \frac{\max(\text{DLA}_l, 0) \cdot \text{APR}_l \cdot \text{NullSpaceCapacity}_l}{1 + 10^6 \cdot \text{Leakage}_l}$$
   Automatically filters out layers where the trigger key $k_l^*$ collides with the preserved
   baseline syntax subspace $K_{0, l}$ and generates a ready-to-run `ForgeRecipe` targeting the
   top-$K$ highest-SLR causal anchor layers!
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from circuitforge.recipes import (
    ContrastivePair,
    ForgeRecipe,
    SurgeryStepSpec,
    compute_weight_backed_activations,
)
from circuitforge.surgery import (
    apply_nullspace_rank1_surgery,
    compute_contrastive_direction,
    compute_nullspace_projector,
)
from circuitforge.wiring import is_routed_expert_tensor, load_wiring_map


@dataclass(frozen=True)
class LayerAttributionScore:
    """Mechanistic attribution and null-space surgical suitability metrics for a single layer."""

    layer_idx: int
    tensor_name: str
    attention_type: str
    direct_logit_attribution: float
    activation_patching_recovery: float
    nullspace_capacity: float
    nullspace_leakage: float
    signal_to_leakage_ratio: float
    recommended_alpha: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "layer_idx": self.layer_idx,
            "tensor_name": self.tensor_name,
            "attention_type": self.attention_type,
            "direct_logit_attribution": round(self.direct_logit_attribution, 6),
            "activation_patching_recovery": round(self.activation_patching_recovery, 6),
            "nullspace_capacity": round(self.nullspace_capacity, 6),
            "nullspace_leakage": float(f"{self.nullspace_leakage:.3e}"),
            "signal_to_leakage_ratio": round(self.signal_to_leakage_ratio, 6),
            "recommended_alpha": round(self.recommended_alpha, 4),
        }


def sweep_causal_circuit_attribution(
    layer_weights: dict[int, np.ndarray],
    pairs: Sequence[ContrastivePair] | None = None,
    preserved_keys_by_layer: dict[int, np.ndarray] | None = None,
    profile_name: str = "Qwen3.6-35B-A3B",
    target_component: str = "self_attn.o_proj.weight",
    base_alpha: float = 2.0,
    nullspace_num_keys: int = 8,
) -> list[LayerAttributionScore]:
    """
    Sweeps across all provided `layer_weights` (`{layer_idx: W_l}`), computes Direct Logit
    Attribution (`DLA`), Activation Patching Recovery (`APR`), and Null-Space Signal-to-Leakage
    Ratio (`SLR`), and returns the per-layer scores sorted in descending order of `SLR`.
    """
    if not layer_weights:
        raise ValueError("layer_weights dictionary cannot be empty.")
    if is_routed_expert_tensor(f"model.layers.0.{target_component}"):
        raise PermissionError(
            f"CircuitForge Safety Gate: Cannot run surgical circuit attribution on routed expert '{target_component}'."
        )

    wmap = load_wiring_map(profile_name=profile_name)
    default_pairs = list(pairs) if pairs else [
        ContrastivePair(
            id="attr_001",
            prompt="New mid-session sprint requirement: add parameter max_retries=3 to client.py",
            positive='<tool_call>{"name": "edit_file", "arguments": {"path": "state.md"}}</tool_call>',
            negative='<tool_call>{"name": "edit_file", "arguments": {"path": "client.py"}}</tool_call>',
        ),
        ContrastivePair(
            id="attr_002",
            prompt="Context has # [PRE-HYDRATED SOURCE FILES] for billing.py; apply surgical edit",
            positive='<tool_call>{"name": "edit_file", "arguments": {"path": "billing.py"}}</tool_call>',
            negative='<tool_call>{"name": "read_file", "arguments": {"path": "billing.py"}}</tool_call>',
        ),
    ]

    raw_scores: list[tuple[int, str, str, float, float, float, float, float]] = []

    for l_idx in sorted(layer_weights.keys()):
        W = np.asarray(layer_weights[l_idx], dtype=np.float32)
        if W.ndim != 2:
            raise ValueError(f"Expected 2D weight matrix at layer {l_idx}, got {W.shape}")
        d_out, d_in = W.shape
        tname = f"model.layers.{l_idx}.{target_component}"
        if is_routed_expert_tensor(tname):
            raise PermissionError(
                f"CircuitForge Safety Gate: Refusing circuit attribution on routed expert '{tname}'."
            )

        attn_type = (
            wmap.layers[l_idx].attention_type
            if 0 <= l_idx < len(wmap.layers)
            else "full_attention"
        )

        # Extract weight-backed contrastive activations & trigger keys
        pos_acts, neg_acts, trig_keys = compute_weight_backed_activations(default_pairs, W)
        v_target = compute_contrastive_direction(pos_acts, neg_acts)
        k_star = trig_keys.mean(axis=0).astype(np.float64)
        k_orig_sq = float(np.dot(k_star, k_star))

        # Preserved baseline syntax keys K0
        if preserved_keys_by_layer and l_idx in preserved_keys_by_layer:
            K0 = np.asarray(preserved_keys_by_layer[l_idx], dtype=np.float32)
        else:
            k_cnt = min(nullspace_num_keys, max(d_in // 2, 1))
            _, s_w, vt_w = np.linalg.svd(W.astype(np.float64), full_matrices=False)
            if len(s_w) >= k_cnt and float(s_w[0]) > 1e-12:
                K0 = vt_w[:k_cnt, :].astype(np.float32).copy()
            else:
                K0 = np.eye(k_cnt, d_in, dtype=np.float32)

        P_perp = compute_nullspace_projector(K0)
        k_proj = P_perp @ k_star
        k_proj_sq = float(np.dot(k_proj, k_proj))
        ns_capacity = float(np.clip(k_proj_sq / max(k_orig_sq, 1e-12), 0.0, 1.0))

        # Direct Logit Attribution (DLA): contrast separation normalized by average singular scale
        mean_diff = np.mean(pos_acts.astype(np.float64) - neg_acts.astype(np.float64), axis=0)
        scale_norm = float(np.sqrt(max(min(d_out, d_in), 1)))
        dla = float(max(0.0, float(np.dot(mean_diff, v_target.astype(np.float64))) / scale_norm))

        # Activation Patching Recovery (APR) & Null-Space Leakage via probe rank-1 surgery
        if ns_capacity <= 1e-4:
            apr = 0.0
            leakage = 0.0
            slr = 0.0
        else:
            try:
                W_patched, rep = apply_nullspace_rank1_surgery(
                    weight=W,
                    trigger_key=k_star.astype(np.float32),
                    steering_vector=v_target,
                    preserved_keys=K0,
                    alpha=base_alpha,
                    tensor_name=tname,
                    layer_idx=l_idx,
                )
                clean_logit = float(np.dot(np.mean(pos_acts, axis=0), v_target))
                corrupt_logit = float(np.dot(np.mean(neg_acts, axis=0), v_target))
                patched_out = W_patched @ k_star.astype(np.float32)
                base_out = W @ k_star.astype(np.float32)
                patched_shift = float(np.dot(patched_out - base_out, v_target))
                denom = max(clean_logit - corrupt_logit, 1e-6)
                apr = float(np.clip(patched_shift / max(denom, base_alpha), 0.0, 1.0))
                leakage = float(rep.nullspace_leakage_max)
                slr = float((dla * apr * ns_capacity) / (1.0 + 1e6 * leakage))
            except ValueError:
                apr = 0.0
                leakage = 0.0
                slr = 0.0

        raw_scores.append((l_idx, tname, attn_type, dla, apr, ns_capacity, leakage, slr))

    max_slr = max((item[7] for item in raw_scores), default=0.0)
    results: list[LayerAttributionScore] = []
    for l_idx, tname, attn_type, dla, apr, ns_cap, leak, slr in raw_scores:
        rel_weight = (slr / max_slr) if max_slr > 1e-12 else 0.0
        rec_alpha = float(base_alpha * rel_weight) if slr > 0.0 else 0.0
        results.append(
            LayerAttributionScore(
                layer_idx=l_idx,
                tensor_name=tname,
                attention_type=attn_type,
                direct_logit_attribution=dla,
                activation_patching_recovery=apr,
                nullspace_capacity=ns_cap,
                nullspace_leakage=leak,
                signal_to_leakage_ratio=slr,
                recommended_alpha=rec_alpha,
            )
        )

    results.sort(key=lambda x: (x.signal_to_leakage_ratio, x.nullspace_capacity, -x.layer_idx), reverse=True)
    return results


def discover_optimal_surgical_circuit(
    layer_weights: dict[int, np.ndarray],
    top_k_layers: int = 4,
    pairs: Sequence[ContrastivePair] | None = None,
    preserved_keys_by_layer: dict[int, np.ndarray] | None = None,
    profile_name: str = "Qwen3.6-35B-A3B",
    target_component: str = "self_attn.o_proj.weight",
    base_alpha: float = 2.0,
) -> dict[str, Any]:
    """
    Runs the full Causal Attribution & Null-Space SLR sweep across `layer_weights`, selects
    the top-$K$ causal anchor layers with positive `signal_to_leakage_ratio`, and returns
    both the full ranked leaderboard and an auto-generated `ForgeRecipe`.
    """
    ranked = sweep_causal_circuit_attribution(
        layer_weights=layer_weights,
        pairs=pairs,
        preserved_keys_by_layer=preserved_keys_by_layer,
        profile_name=profile_name,
        target_component=target_component,
        base_alpha=base_alpha,
    )
    eligible = [s for s in ranked if s.signal_to_leakage_ratio > 0.0]
    selected = eligible[: max(1, int(top_k_layers))]
    selected_layers = sorted(s.layer_idx for s in selected)
    rec_alpha = (
        float(np.mean([s.recommended_alpha for s in selected]))
        if selected
        else float(base_alpha)
    )

    recipe = ForgeRecipe(
        recipe_name=f"{profile_name}_auto_discovered_circuit",
        profile=profile_name,
        description="Automatically discovered causal surgical circuit ranked by DLA, APR, and Null-Space SLR",
        steps=[
            SurgeryStepSpec(
                name="causal_slr_top_k_surgery",
                method="nullspace_rank1",
                target_component=target_component,
                layers=selected_layers,
                alpha=round(rec_alpha, 4),
                schedule="bell",
            )
        ],
    )

    return {
        "profile": profile_name,
        "swept_layers_count": len(ranked),
        "selected_top_k_layers": selected_layers,
        "top_layer_idx": ranked[0].layer_idx if ranked else None,
        "top_signal_to_leakage_ratio": round(ranked[0].signal_to_leakage_ratio, 6) if ranked else 0.0,
        "rankings": [r.to_dict() for r in ranked],
        "auto_recipe": recipe,
    }
