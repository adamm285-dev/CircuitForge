"""
Command-line interface for CircuitForge (`python -m circuitforge` / `circuitforge`).

Commands:
- `inspect-wiring`: Inspect the hybrid Linear/Full-Attention MoE circuit map and shard layout.
- `plan-surgery`: Generate a deterministic surgical target manifest for Full-Attention anchors and Shared Experts.
- `verify-surgery`: Run a self-contained mathematical verification of null-space rank-1 surgery and 1.58-bit STE QAT.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from circuitforge.autograd_qat import ls_optimal_ternary_scale, ternary_quantize_ste
from circuitforge.surgery import (
    apply_nullspace_rank1_surgery,
    build_surgical_plan,
    compute_contrastive_direction,
)
from circuitforge.wiring import BUILTIN_PROFILES, load_wiring_map


def cmd_inspect_wiring(model_dir: str | None, profile: str, fmt: str) -> int:
    wmap = load_wiring_map(config_or_dir=model_dir, profile_name=profile)
    summary = wmap.summary_dict()
    if fmt == "json":
        print(json.dumps(summary, indent=2))
        return 0

    print("\n" + "=" * 74)
    print(f"  CIRCUITFORGE HYBRID MOE WIRING MAP: {summary['model_name']}")
    print("=" * 74)
    print(f"  Architecture            : {summary['architecture']}")
    print(f"  Total Layers            : {summary['num_layers']} ({summary['linear_attention_count']} Linear + {summary['full_attention_count']} Full-Attention)")
    print(f"  Hidden Size / Head Dim  : {summary['hidden_size']} / {summary['head_dim']}")
    print(f"  Routed Experts          : {summary['num_experts']} per layer (top_k={summary['top_k']}) [100% FROZEN]")
    print(f"  Full-Attention Anchors  : {summary['full_attention_layers']}")
    print(f"  Primary Surgery Anchors : {summary['mid_to_late_planning_anchors']} (25%..85% depth)")
    if summary["total_shards"] > 0:
        print(f"  Safetensors Shards      : {summary['total_shards']} total ({summary['control_shard_count']} control, {summary['routed_expert_shard_count']} expert-only)")
    print("=" * 74 + "\n")
    return 0


def cmd_plan_surgery(
    model_dir: str | None,
    profile: str,
    include_shared_experts: bool,
    all_anchors: bool,
    out_path: str | None,
) -> int:
    wmap = load_wiring_map(config_or_dir=model_dir, profile_name=profile)
    plan = build_surgical_plan(
        wmap,
        include_shared_experts=include_shared_experts,
        only_mid_to_late_anchors=not all_anchors,
    )
    payload = {
        "model_name": wmap.model_name,
        "architecture": wmap.architecture,
        "target_anchor_count": len(plan),
        "frozen_routed_experts_per_layer": wmap.num_experts,
        "surgical_targets": plan,
    }
    text = json.dumps(payload, indent=2)
    if out_path:
        p = Path(out_path).resolve()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text + "\n", encoding="utf-8")
        print(f"Wrote surgical plan ({len(plan)} anchor layers) to: {p}")
    else:
        print(text)
    return 0


def cmd_verify_surgery(seed: int = 42) -> int:
    rng = np.random.default_rng(seed)
    d_out, d_in = 256, 256
    W = rng.standard_normal((d_out, d_in)).astype(np.float32) * 0.02

    # 1. Extract contrastive steering vector from synthetic positive vs negative activations
    pos_acts = rng.standard_normal((16, d_out)).astype(np.float32) + 0.5
    neg_acts = rng.standard_normal((16, d_out)).astype(np.float32) - 0.5
    v_steer = compute_contrastive_direction(pos_acts, neg_acts)

    # 2. Preserved baseline syntax keys K_0 (32 keys) and trigger key k*
    K0 = rng.standard_normal((32, d_in)).astype(np.float32)
    k_star = rng.standard_normal(d_in).astype(np.float32)

    W_edited, report = apply_nullspace_rank1_surgery(
        weight=W,
        trigger_key=k_star,
        steering_vector=v_steer,
        preserved_keys=K0,
        alpha=2.5,
        tensor_name="model.layers.15.self_attn.o_proj.weight",
        layer_idx=15,
    )

    # 3. Verify 1.58-bit Ternary STE quantization scale on the edited matrix
    W_q, scale, _ = ternary_quantize_ste(W_edited.astype(np.float64))
    unique_trits = sorted(set(np.unique( np.round(W_q / scale) ).astype(int).tolist()))

    result = {
        "tensor_name": report.tensor_name,
        "layer_idx": report.layer_idx,
        "shape": list(report.shape),
        "alpha": report.alpha,
        "target_alignment_before": round(report.target_alignment_before, 6),
        "target_alignment_after": round(report.target_alignment_after, 6),
        "alignment_gain": round(report.target_alignment_after - report.target_alignment_before, 6),
        "nullspace_leakage_max": float(f"{report.nullspace_leakage_max:.3e}"),
        "ternary_ls_scale": round(float(scale), 6),
        "ternary_levels": unique_trits,
        "verified": (
            report.nullspace_leakage_max < 1e-10
            and (report.target_alignment_after - report.target_alignment_before) > 2.4
            and unique_trits == [-1, 0, 1]
        ),
    }
    print(json.dumps(result, indent=2))
    return 0 if result["verified"] else 1


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="circuitforge",
        description="CircuitForge: Weight-Space Circuit Surgery, Verifiable RL, and 1.58-Bit QAT for Hybrid MoE Transformers",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    insp = sub.add_parser("inspect-wiring", help="Inspect hybrid MoE circuit topology and shard separation")
    insp.add_argument("--model-dir", default=None, help="Path to local model directory or config.json")
    insp.add_argument(
        "--profile",
        default="Qwen3.6-35B-A3B",
        choices=sorted(BUILTIN_PROFILES.keys()),
        help="Built-in architecture profile when --model-dir is omitted",
    )
    insp.add_argument("--format", default="table", choices=["table", "json"], help="Output format")

    plan_p = sub.add_parser("plan-surgery", help="Generate a deterministic surgical intervention manifest")
    plan_p.add_argument("--model-dir", default=None, help="Path to local model directory or config.json")
    plan_p.add_argument(
        "--profile",
        default="Qwen3.6-35B-A3B",
        choices=sorted(BUILTIN_PROFILES.keys()),
        help="Built-in architecture profile when --model-dir is omitted",
    )
    plan_p.add_argument("--no-shared-experts", action="store_true", help="Target Full-Attention o_proj only")
    plan_p.add_argument("--all-anchors", action="store_true", help="Target all Full-Attention layers instead of 25%..85% depth")
    plan_p.add_argument("--out", default=None, help="Optional JSON output path")

    ver_p = sub.add_parser("verify-surgery", help="Run mathematical verification of null-space rank-1 surgery and 1.58-bit STE")
    ver_p.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()
    if args.cmd == "inspect-wiring":
        return cmd_inspect_wiring(args.model_dir, args.profile, args.format)
    if args.cmd == "plan-surgery":
        return cmd_plan_surgery(
            args.model_dir,
            args.profile,
            include_shared_experts=not args.no_shared_experts,
            all_anchors=args.all_anchors,
            out_path=args.out,
        )
    if args.cmd == "verify-surgery":
        return cmd_verify_surgery(seed=args.seed)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
