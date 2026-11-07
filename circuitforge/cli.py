"""
Command-line interface for CircuitForge (`python -m circuitforge` / `circuitforge`).

Commands:
- `inspect-wiring`: Inspect the hybrid Linear/Full-Attention MoE circuit map and shard layout.
- `plan-surgery`: Generate a deterministic surgical target manifest for Full-Attention anchors and Shared Experts.
- `verify-surgery`: Run a self-contained mathematical verification of null-space rank-1 weight surgery and autograd gradients.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from circuitforge.attribution import discover_optimal_surgical_circuit
from circuitforge.autograd import Tensor, causal_self_attention, rms_norm
from circuitforge.compiler import evaluate_compiler_benchmark
from circuitforge.engram import evaluate_engram_and_manifold_benchmark
from circuitforge.exporter import (
    export_forged_bundle,
    export_gguf_control_vector,
    export_model_card,
    export_surgical_lora,
)
from circuitforge.gguf import GGUFTensorInfo, inspect_gguf_header
from circuitforge.harvester import (
    export_harvested_activations,
    harvest_checkpoint_activations,
)
from circuitforge.hud import MechanisticHUDServer, MechanisticTelemetryHUD, export_hud_snapshot
from circuitforge.io import inspect_safetensors_header, load_safetensors_file
from circuitforge.native import benchmark_native_vs_numpy, has_native_engine
from circuitforge.recipes import (
    ContrastivePair,
    execute_forge_recipe,
    load_contrastive_pairs,
    load_forge_recipe,
)
from circuitforge.sentinel import evaluate_sentinel_benchmark
from circuitforge.surgery import (
    apply_nullspace_rank1_surgery,
    build_surgical_plan,
    compute_contrastive_direction,
)
from circuitforge.wiring import (
    BUILTIN_PROFILES,
    is_routed_expert_tensor,
    is_surgical_control_tensor,
    load_wiring_map,
)


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

    # 3. Verify causal self-attention & RMSNorm backward pass on edited projection
    x_t = Tensor(rng.standard_normal((2, 4, 16)), requires_grad=True)
    gamma = Tensor(np.ones(16), requires_grad=True)
    wq = Tensor(rng.standard_normal((16, 16)) * 0.1, requires_grad=True)
    wk = Tensor(rng.standard_normal((16, 16)) * 0.1, requires_grad=True)
    wv = Tensor(rng.standard_normal((16, 16)) * 0.1, requires_grad=True)
    wo = Tensor(W_edited[:16, :16].astype(np.float64), requires_grad=True)
    attn_out = causal_self_attention(rms_norm(x_t, gamma), wq, wk, wv, wo, num_heads=2)
    attn_out.mean().backward()
    grad_norm_wo = float(np.linalg.norm(wo.grad))

    result = {
        "tensor_name": report.tensor_name,
        "layer_idx": report.layer_idx,
        "shape": list(report.shape),
        "alpha": report.alpha,
        "target_alignment_before": round(report.target_alignment_before, 6),
        "target_alignment_after": round(report.target_alignment_after, 6),
        "alignment_gain": round(report.target_alignment_after - report.target_alignment_before, 6),
        "nullspace_leakage_max": float(f"{report.nullspace_leakage_max:.3e}"),
        "autograd_o_proj_grad_norm": round(grad_norm_wo, 6),
        "verified": (
            report.nullspace_leakage_max < 1e-10
            and abs((report.target_alignment_after - report.target_alignment_before) - 2.5) < 1e-4
            and grad_norm_wo > 0.0
        ),
    }
    print(json.dumps(result, indent=2))
    return 0 if result["verified"] else 1


def cmd_inspect_checkpoint(path: str) -> int:
    entries, metadata, data_start = inspect_safetensors_header(path)
    control_count = sum(1 for k in entries if is_surgical_control_tensor(k))
    expert_count = sum(1 for k in entries if is_routed_expert_tensor(k))
    payload = {
        "file": str(Path(path).name),
        "tensor_count": len(entries),
        "control_tensor_count": control_count,
        "routed_expert_tensor_count": expert_count,
        "data_start_offset": data_start,
        "metadata": metadata,
        "tensors": {
            k: {"dtype": v.dtype, "shape": list(v.shape), "bytes": v.byte_length}
            for k, v in entries.items()
        },
    }
    print(json.dumps(payload, indent=2))
    return 0


def cmd_run_recipe(
    recipe_path: str,
    checkpoint_path: str | None,
    output_path: str | None,
) -> int:
    rpath = Path(recipe_path).resolve()
    recipe = load_forge_recipe(rpath)
    result = execute_forge_recipe(
        recipe=recipe,
        checkpoint_path=checkpoint_path,
        output_path=output_path,
        base_dir=rpath.parent,
    )
    print(json.dumps(result, indent=2))
    return 0 if result["max_nullspace_leakage"] < 1e-6 else 1


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="circuitforge",
        description="CircuitForge: Weight-Space Circuit Surgery and Verifiable RL for Hybrid MoE Transformers",
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

    ver_p = sub.add_parser("verify-surgery", help="Run mathematical verification of null-space rank-1 weight surgery")
    ver_p.add_argument("--seed", type=int, default=42, help="Random seed")

    ckpt_p = sub.add_parser("inspect-checkpoint", help="Inspect a .safetensors checkpoint header without loading weights")
    ckpt_p.add_argument("path", help="Path to .safetensors file")

    rec_p = sub.add_parser("run-recipe", help="Execute a declarative .forge.json weight surgery recipe")
    rec_p.add_argument("recipe", help="Path to .forge.json recipe file")
    rec_p.add_argument("--checkpoint", default=None, help="Optional input .safetensors file")
    rec_p.add_argument("--out", default=None, help="Optional output .safetensors file")

    bench_p = sub.add_parser("benchmark-engine", help="Run C++ SIMD vs NumPy rank-1 surgery microbenchmark")
    bench_p.add_argument("--dim", type=int, default=512, help="Square matrix dimension")
    bench_p.add_argument("--iters", type=int, default=50, help="Benchmark iterations")

    hud_exp = sub.add_parser("export-hud", help="Export a standalone interactive HTML Mechanistic Telemetry HUD")
    hud_exp.add_argument("--out", default="circuitforge_hud.html", help="Output HTML file path")
    hud_exp.add_argument(
        "--profile",
        default="Qwen3.6-35B-A3B",
        choices=sorted(BUILTIN_PROFILES.keys()),
        help="Model architecture profile",
    )
    hud_exp.add_argument("--alpha", type=float, default=1.8, help="Initial steering multiplier alpha")
    hud_exp.add_argument("--schedule", default="bell", choices=["bell", "uniform"], help="Layer distribution schedule")

    hud_srv = sub.add_parser("serve-hud", help="Launch live interactive Mechanistic Telemetry HUD web server")
    hud_srv.add_argument("--host", default="127.0.0.1", help="Bind host")
    hud_srv.add_argument("--port", type=int, default=8765, help="Bind port")
    hud_srv.add_argument(
        "--profile",
        default="Qwen3.6-35B-A3B",
        choices=sorted(BUILTIN_PROFILES.keys()),
        help="Model architecture profile",
    )

    sent_p = sub.add_parser(
        "verify-sentinel",
        help="Evaluate MoE Anomaly Router & Spike Sentinel (100% interception, 0% false positive)",
    )
    sent_p.add_argument("--seed", type=int, default=42, help="Random seed")

    eng_p = sub.add_parser(
        "verify-engram",
        help="Evaluate Discrete Engram Memory Graph & Weight-Space Parametric Manifold (CFM-06)",
    )
    eng_p.add_argument("--seed", type=int, default=42, help="Random seed")

    comp_p = sub.add_parser(
        "verify-compiler",
        help="Evaluate Tracr/RASP-Style Symbolic-to-Weight Micro-Circuit Compiler (CFM-07)",
    )

    gguf_p = sub.add_parser(
        "inspect-gguf",
        help="Inspect a .gguf v2/v3 checkpoint header and tensor descriptors without loading weights (CFM-08)",
    )
    gguf_p.add_argument("path", help="Path to .gguf file")

    card_p = sub.add_parser(
        "export-card",
        help="Export a Hugging Face Model Card or complete Forged Checkpoint Bundle (CFM-08)",
    )
    card_p.add_argument("--out", default="README_MODEL_CARD.md", help="Output Markdown path or bundle directory")
    card_p.add_argument("--model-id", default="Qwen3.6-35B-A3B-CircuitForged", help="Model repository name")
    card_p.add_argument(
        "--profile",
        default="Qwen3.6-35B-A3B",
        choices=sorted(BUILTIN_PROFILES.keys()),
        help="Base model architecture profile",
    )
    card_p.add_argument("--bundle", action="store_true", help="Export full bundle (.safetensors, .gguf, recipe, README.md)")

    disc_p = sub.add_parser(
        "discover-circuit",
        help="Run Automated Circuit Discovery & Causal Attribution Sweeper across candidate layers (CFM-12)",
    )
    disc_p.add_argument(
        "--profile",
        default="Qwen3.6-35B-A3B",
        choices=sorted(BUILTIN_PROFILES.keys()),
        help="Model architecture profile",
    )
    disc_p.add_argument("--checkpoint", default=None, help="Optional .safetensors checkpoint path to sweep")
    disc_p.add_argument("--top-k", type=int, default=4, help="Number of top causal anchor layers to select")
    disc_p.add_argument("--out-recipe", "--output-recipe", dest="out_recipe", default=None, help="Optional output path for auto-generated .forge.json recipe")

    lora_p = sub.add_parser(
        "export-lora",
        help="Export a surgical weight delta as a Hugging Face PEFT LoRA adapter or llama.cpp control vector (CFM-11)",
    )
    lora_p.add_argument("--out", "--output-dir", dest="out", default="lora_adapter", help="Output directory for PEFT LoRA or .gguf path for control vector")
    lora_p.add_argument("--base-checkpoint", default=None, help="Optional base .safetensors checkpoint path")
    lora_p.add_argument("--forged-checkpoint", default=None, help="Optional forged .safetensors checkpoint path")
    lora_p.add_argument("--rank", type=int, default=4, help="LoRA rank r")
    lora_p.add_argument("--control-vector", action="store_true", help="Export llama.cpp --control-vector GGUF instead of PEFT LoRA")

    harv_p = sub.add_parser(
        "harvest-activations",
        help="Harvest multi-layer forward activations from a .safetensors, sharded, or .gguf checkpoint (CFM-13)",
    )
    harv_p.add_argument("checkpoint", help="Path to .safetensors file, sharded checkpoint directory, or .gguf file")
    harv_p.add_argument("--pairs", default=None, help="Optional JSON/JSONL file of contrastive pairs")
    harv_p.add_argument("--layers", default="0,1", help="Comma-separated layer indices to harvest (e.g. '3,7,11')")
    harv_p.add_argument("--component", default="self_attn.o_proj.weight", help="Target layer component suffix")
    harv_p.add_argument("--out", required=True, help="Output .safetensors path for harvested activations")

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
    if args.cmd == "inspect-checkpoint":
        return cmd_inspect_checkpoint(args.path)
    if args.cmd == "run-recipe":
        return cmd_run_recipe(args.recipe, args.checkpoint, args.out)
    if args.cmd == "benchmark-engine":
        res = benchmark_native_vs_numpy(dim=args.dim, iterations=args.iters)
        res["native_engine_loaded"] = has_native_engine()
        print(json.dumps(res, indent=2))
        return 0 if res["meets_50us_target"] else 1
    if args.cmd == "export-hud":
        out_p, state = export_hud_snapshot(
            output_html_path=args.out,
            profile_name=args.profile,
            alpha=args.alpha,
            schedule=args.schedule,
        )
        print(
            json.dumps(
                {
                    "hud_html_path": str(out_p),
                    "profile": state["profile"]["model_name"],
                    "alpha": state["controls"]["global_alpha"],
                    "schedule": state["controls"]["schedule"],
                    "ring_buffer_events_count": state["ring_buffer_events_count"],
                },
                indent=2,
            )
        )
        return 0
    if args.cmd == "serve-hud":
        hud = MechanisticTelemetryHUD(profile_name=args.profile)
        srv = MechanisticHUDServer(hud, host=args.host, port=args.port)
        url = srv.start_background()
        print(f"CircuitForge Mechanistic Telemetry HUD listening on: {url}")
        try:
            while True:
                import time as _t
                _t.sleep(1.0)
        except KeyboardInterrupt:
            srv.stop()
            hud.close()
        return 0
    if args.cmd == "verify-sentinel":
        res = evaluate_sentinel_benchmark(seed=args.seed)
        print(json.dumps(res, indent=2))
        return 0 if res["verified"] else 1
    if args.cmd == "verify-engram":
        res = evaluate_engram_and_manifold_benchmark(seed=args.seed)
        print(json.dumps(res, indent=2))
        return 0 if res["verified"] else 1
    if args.cmd == "verify-compiler":
        res = evaluate_compiler_benchmark()
        print(json.dumps(res, indent=2))
        return 0 if res["verified"] else 1
    if args.cmd == "inspect-gguf":
        info = inspect_gguf_header(args.path)
        tensors_map: dict[str, GGUFTensorInfo] = info["tensors"]  # type: ignore[assignment]
        serializable = dict(info)
        serializable["tensors"] = {
            k: {
                "shape": list(v.shape),
                "ggml_type": v.ggml_type_name,
                "relative_offset": v.relative_offset,
                "is_control_tensor": v.is_control_tensor,
                "is_routed_expert": v.is_routed_expert,
            }
            for k, v in tensors_map.items()
        }
        print(json.dumps(serializable, indent=2))
        return 0
    if args.cmd == "export-card":
        if args.bundle:
            res = export_forged_bundle(
                output_dir=args.out,
                model_id=args.model_id,
                base_profile=args.profile,
            )
            print(json.dumps(res, indent=2))
        else:
            out_md = export_model_card(
                output_md_path=args.out,
                model_id=args.model_id,
                base_profile=args.profile,
            )
            print(json.dumps({"model_card_path": str(out_md), "model_id": args.model_id}, indent=2))
        return 0
    if args.cmd == "discover-circuit":
        wmap = load_wiring_map(profile_name=args.profile)
        if args.checkpoint:
            raw_tensors, _ = load_safetensors_file(args.checkpoint)
            layer_weights: dict[int, np.ndarray] = {}
            for tname, arr in raw_tensors.items():
                if tname.endswith("self_attn.o_proj.weight") and arr.ndim == 2:
                    parts = tname.split(".")
                    for idx, p in enumerate(parts):
                        if p == "layers" and idx + 1 < len(parts) and parts[idx + 1].isdigit():
                            layer_weights[int(parts[idx + 1])] = arr
                            break
            if not layer_weights:
                raise ValueError(f"No self_attn.o_proj.weight tensors found in checkpoint {args.checkpoint}")
        else:
            rng = np.random.default_rng(42)
            candidate_layers = wmap.full_attention_layers[:8] or [0, 1, 2, 3]
            layer_weights = {
                l_idx: (rng.standard_normal((64, 64)).astype(np.float32) * (0.08 if l_idx in wmap.mid_to_late_planning_anchors else 0.02))
                for l_idx in candidate_layers
            }
        res = discover_optimal_surgical_circuit(
            layer_weights=layer_weights,
            top_k_layers=args.top_k,
            profile_name=args.profile,
        )
        recipe_obj = res.pop("auto_recipe")
        res["auto_recipe"] = recipe_obj.to_dict()
        if args.out_recipe:
            rp = Path(args.out_recipe).resolve()
            rp.parent.mkdir(parents=True, exist_ok=True)
            rp.write_text(json.dumps(res["auto_recipe"], indent=2) + "\n", encoding="utf-8")
            res["saved_recipe_path"] = str(rp)
        print(json.dumps(res, indent=2))
        return 0
    if args.cmd == "export-lora":
        rng = np.random.default_rng(42)
        if args.control_vector:
            vecs = {15: rng.standard_normal(64).astype(np.float32), 23: rng.standard_normal(64).astype(np.float32)}
            out_cv = export_gguf_control_vector(args.out, vecs)
            print(json.dumps({"control_vector_gguf": str(out_cv), "layers": sorted(vecs.keys())}, indent=2))
        else:
            if args.base_checkpoint and args.forged_checkpoint:
                base_t, _ = load_safetensors_file(args.base_checkpoint)
                forged_t, _ = load_safetensors_file(args.forged_checkpoint)
                deltas: dict[str, np.ndarray] = {}
                for k in forged_t:
                    if k not in base_t:
                        continue
                    diff = forged_t[k].astype(np.float32) - base_t[k].astype(np.float32)
                    if float(np.linalg.norm(diff)) > 1e-7:
                        if is_routed_expert_tensor(k):
                            raise PermissionError(
                                f"CircuitForge Safety Gate: Routed expert '{k}' was modified between base and forged checkpoints!"
                            )
                        deltas[k] = diff
            elif args.base_checkpoint or args.forged_checkpoint:
                raise ValueError("Both --base-checkpoint and --forged-checkpoint must be provided together.")
            else:
                u = rng.standard_normal((64, 2)).astype(np.float32)
                v = rng.standard_normal((2, 64)).astype(np.float32)
                deltas = {"model.layers.15.self_attn.o_proj.weight": u @ v}
            res = export_surgical_lora(args.out, deltas, rank=args.rank)
            print(json.dumps(res, indent=2))
        return 0
    if args.cmd == "harvest-activations":
        layer_list = [int(x.strip()) for x in str(args.layers).split(",") if x.strip()]
        if args.pairs:
            pairs_list = load_contrastive_pairs(args.pairs)
        else:
            pairs_list = [
                ContrastivePair(
                    id="default_p1",
                    prompt="Lock sprint requirements in state.md before editing source files",
                    positive="edit_file state.md",
                    negative="edit_file app.py",
                ),
                ContrastivePair(
                    id="default_p2",
                    prompt="Verify module topology boundaries in wiring.yaml",
                    positive="sdcs verify --topology",
                    negative="git commit --no-verify",
                ),
            ]
        harvested = harvest_checkpoint_activations(
            checkpoint_path=args.checkpoint,
            pairs=pairs_list,
            layers=layer_list,
            target_component=args.component,
        )
        summary = export_harvested_activations(args.out, harvested)
        print(json.dumps(summary, indent=2))
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
