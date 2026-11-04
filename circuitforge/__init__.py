"""
CircuitForge: Weight-Space Circuit Surgery, Verifiable RL, and 1.58-Bit QAT for Hybrid MoE Transformers.
"""

from circuitforge.autograd_qat import (
    Tensor,
    causal_self_attention,
    ls_optimal_ternary_scale,
    rms_norm,
    run_qat_vs_ptq_experiment,
    softmax_cross_entropy,
    ternary_quantize_ste,
)
from circuitforge.surgery import (
    SurgicalEditReport,
    apply_nullspace_rank1_surgery,
    build_surgical_plan,
    compute_contrastive_direction,
    compute_nullspace_projector,
)
from circuitforge.wiring import (
    BUILTIN_PROFILES,
    LayerCircuitSpec,
    ModelWiringMap,
    is_routed_expert_tensor,
    is_surgical_control_tensor,
    load_wiring_map,
)

__version__ = "0.1.0"

__all__ = [
    "BUILTIN_PROFILES",
    "LayerCircuitSpec",
    "ModelWiringMap",
    "SurgicalEditReport",
    "Tensor",
    "apply_nullspace_rank1_surgery",
    "build_surgical_plan",
    "causal_self_attention",
    "compute_contrastive_direction",
    "compute_nullspace_projector",
    "is_routed_expert_tensor",
    "is_surgical_control_tensor",
    "load_wiring_map",
    "ls_optimal_ternary_scale",
    "rms_norm",
    "run_qat_vs_ptq_experiment",
    "softmax_cross_entropy",
    "ternary_quantize_ste",
]
