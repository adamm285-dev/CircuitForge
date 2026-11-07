"""Unit and integration tests for Milestone CFM-02:
- Universal Architecture Adapter Registry (`Qwen2.5-Coder-7B`, `Llama-3.1-8B`, `DeepSeek-V3`, `Mixtral-8x7B`, custom registration)
- Multi-Rank & Distributed Layer Surgery (`apply_nullspace_rank_k_surgery`, `apply_directional_ablation`, `distribute_multilayer_surgery`)
- Streaming Safetensors Binary I/O (`circuitforge/io.py`) including F32/BF16 round-trip and routed-expert protection gates
- Declarative Forge Recipes & Contrastive Harness (`circuitforge/recipes.py`) and CLI commands
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from circuitforge.cli import cmd_inspect_checkpoint, cmd_run_recipe
from circuitforge.io import (
    inspect_safetensors_header,
    load_safetensor_slice,
    load_safetensors_file,
    save_safetensors_file,
    stream_surgical_safetensors_update,
)
from circuitforge.recipes import (
    execute_forge_recipe,
    load_contrastive_pairs,
    load_forge_recipe,
)
from circuitforge.surgery import (
    apply_directional_ablation,
    apply_nullspace_rank_k_surgery,
    distribute_multilayer_surgery,
)
from circuitforge.wiring import (
    is_routed_expert_tensor,
    is_surgical_control_tensor,
    list_profiles,
    load_wiring_map,
    register_profile,
)


def test_universal_architecture_registry_and_custom_profile(tmp_path: Path) -> None:
    """Verify built-in profiles across Dense GQA, Mixtral, and DeepSeek-V3, plus custom registration."""
    profiles = list_profiles()
    for expected in ("Qwen3.6-35B-A3B", "Qwen3.8-2.4T-A95B", "Qwen2.5-Coder-7B", "Llama-3.1-8B", "DeepSeek-V3", "Mixtral-8x7B"):
        assert expected in profiles

    w_dense = load_wiring_map(profile_name="Qwen2.5-Coder-7B")
    assert w_dense.num_layers == 28
    assert w_dense.num_experts == 0
    assert len(w_dense.full_attention_layers) == 28
    assert len(w_dense.linear_attention_layers) == 0

    from circuitforge.surgery import build_surgical_plan
    dense_plan = build_surgical_plan(w_dense, include_shared_experts=True)
    assert "model.layers.7.mlp.down_proj.weight" in dense_plan[0]["safetensors_targets"]

    w_mixtral = load_wiring_map(profile_name="Mixtral-8x7B")
    mix_plan = build_surgical_plan(w_mixtral, include_shared_experts=True)
    assert mix_plan[0]["safetensors_targets"] == ["model.layers.8.self_attn.o_proj.weight"]

    w_ds = load_wiring_map(profile_name="DeepSeek-V3")
    assert w_ds.num_layers == 61
    assert w_ds.num_experts == 256
    assert w_ds.num_experts_per_tok == 8
    ds_plan = build_surgical_plan(w_ds, include_shared_experts=True)
    assert "model.layers.15.mlp.shared_experts.down_proj.weight" in ds_plan[0]["safetensors_targets"]

    # Mixtral routed expert naming check
    assert is_routed_expert_tensor("model.layers.5.block_sparse_moe.experts.3.w2.weight")
    assert is_surgical_control_tensor("model.layers.5.block_sparse_moe.gate.weight")
    assert is_surgical_control_tensor("model.layers.5.mlp.down_proj.weight")

    # Custom profile registration
    register_profile(
        "Custom-Mini-MoE",
        {
            "architectures": ["Qwen3_5MoeForCausalLM"],
            "hidden_size": 512,
            "num_hidden_layers": 8,
            "head_dim": 64,
            "full_attention_interval": 4,
            "num_experts": 16,
            "num_experts_per_tok": 2,
        },
    )
    w_custom = load_wiring_map(profile_name="Custom-Mini-MoE")
    assert w_custom.num_layers == 8
    assert w_custom.full_attention_layers == [3, 7]

    # Auto-detect dense config.json from disk
    cfg_path = tmp_path / "config.json"
    cfg_path.write_text(
        json.dumps(
            {
                "architectures": ["LlamaForCausalLM"],
                "hidden_size": 1024,
                "num_hidden_layers": 12,
                "head_dim": 64,
            }
        ),
        encoding="utf-8",
    )
    w_disk = load_wiring_map(config_or_dir=cfg_path)
    assert w_disk.architecture == "LlamaForCausalLM"
    assert w_disk.num_layers == 12
    assert w_disk.num_experts == 0
    assert len(w_disk.full_attention_layers) == 12


def test_nullspace_rank_k_and_directional_ablation_and_multilayer() -> None:
    """Verify Rank-k MEMIT surgery zero leakage, directional ablation, and bell-curve multilayer distribution."""
    rng = np.random.default_rng(101)
    d_out, d_in = 48, 48
    W = (rng.standard_normal((d_out, d_in)) * 0.05).astype(np.float32)

    K_preserved = rng.standard_normal((12, d_in)).astype(np.float32)
    K_triggers = rng.standard_normal((4, d_in)).astype(np.float32)
    V_steers = rng.standard_normal((4, d_out)).astype(np.float32)

    W_rk, rep_rk = apply_nullspace_rank_k_surgery(
        weight=W,
        trigger_keys=K_triggers,
        steering_vectors=V_steers,
        preserved_keys=K_preserved,
        alpha=2.0,
        tensor_name="model.layers.11.self_attn.o_proj.weight",
        layer_idx=11,
    )
    assert rep_rk.nullspace_leakage_max < 1e-10
    assert rep_rk.target_alignment_after > rep_rk.target_alignment_before + 1.0
    np.testing.assert_allclose(W @ K_preserved.T, W_rk @ K_preserved.T, atol=1e-6)

    # Routed expert gate on rank-k
    with pytest.raises(PermissionError):
        apply_nullspace_rank_k_surgery(
            W, K_triggers, V_steers, tensor_name="model.layers.11.mlp.experts.7.down_proj.weight"
        )

    # Directional ablation
    r_dir = rng.standard_normal(d_out).astype(np.float32)
    W_abl, rep_abl = apply_directional_ablation(
        weight=W,
        refusal_or_target_dir=r_dir,
        strength=1.0,
        tensor_name="model.layers.11.self_attn.o_proj.weight",
        layer_idx=11,
    )
    assert rep_abl.target_alignment_before > 0.01
    assert rep_abl.target_alignment_after < 1e-6

    # Multilayer bell schedule
    layers = [11, 15, 19, 23]
    layer_W = {l: W.copy() for l in layers}
    layer_k = {l: K_triggers[0] for l in layers}
    layer_v = {l: V_steers[0] for l in layers}
    layer_K0 = {l: K_preserved for l in layers}

    edited_map, reports = distribute_multilayer_surgery(
        layer_weights=layer_W,
        trigger_keys=layer_k,
        steering_vectors=layer_v,
        preserved_keys=layer_K0,
        total_alpha=4.0,
        schedule="bell",
    )
    assert len(edited_map) == 4
    alphas = [r.alpha for r in reports]
    assert np.isclose(sum(alphas), 4.0, atol=1e-6)
    # Middle layers (15, 19) receive higher alpha than boundary layers (11, 23)
    assert alphas[1] > alphas[0]
    assert alphas[2] > alphas[3]
    for r in reports:
        assert r.nullspace_leakage_max < 1e-10


def test_safetensors_streaming_io_and_bf16_roundtrip(tmp_path: Path) -> None:
    """Verify pure-NumPy .safetensors binary serialization, BF16 handling, and streaming surgical edits."""
    rng = np.random.default_rng(55)
    t_ctrl = rng.standard_normal((16, 16)).astype(np.float32)
    t_exp = rng.standard_normal((16, 16)).astype(np.float32)
    t_bf16 = np.array([[1.5, -2.25], [0.125, 4.0]], dtype=np.float32)

    st_path = tmp_path / "shard_01.safetensors"
    save_safetensors_file(
        st_path,
        tensors={
            "model.layers.3.self_attn.o_proj.weight": t_ctrl,
            "model.layers.3.mlp.experts.0.down_proj.weight": t_exp,
            "model.layers.3.mlp.shared_expert.down_proj.weight": t_bf16,
        },
        metadata={"format": "pt"},
        preserve_dtypes={"model.layers.3.mlp.shared_expert.down_proj.weight": "BF16"},
    )

    entries, meta, _ = inspect_safetensors_header(st_path)
    assert meta["format"] == "pt"
    assert entries["model.layers.3.mlp.shared_expert.down_proj.weight"].dtype == "BF16"

    loaded_bf16 = load_safetensor_slice(
        st_path, "model.layers.3.mlp.shared_expert.down_proj.weight"
    )
    np.testing.assert_allclose(loaded_bf16, t_bf16, atol=1e-2)

    # Streaming surgical update on control tensor while keeping routed expert untouched
    out_path = tmp_path / "shard_01_forged.safetensors"
    updated = stream_surgical_safetensors_update(
        input_path=st_path,
        output_path=out_path,
        surgical_edits={
            "model.layers.3.self_attn.o_proj.weight": lambda w: w + 1.0,
        },
    )
    assert updated == ["model.layers.3.self_attn.o_proj.weight"]

    forged_tensors, forged_meta = load_safetensors_file(out_path)
    assert "model.layers.3.self_attn.o_proj.weight" in forged_meta["circuitforge_edited_tensors"]
    np.testing.assert_allclose(
        forged_tensors["model.layers.3.self_attn.o_proj.weight"], t_ctrl + 1.0, atol=1e-6
    )
    np.testing.assert_array_equal(
        forged_tensors["model.layers.3.mlp.experts.0.down_proj.weight"], t_exp
    )

    # Verify routed expert safety gate in streaming update
    with pytest.raises(PermissionError):
        stream_surgical_safetensors_update(
            st_path,
            out_path,
            {"model.layers.3.mlp.experts.0.down_proj.weight": lambda w: w * 0.0},
        )


def test_declarative_forge_recipe_and_cli(tmp_path: Path) -> None:
    """Verify loading contrastive pairs, executing a declarative .forge.json recipe, and CLI commands."""
    pairs_file = tmp_path / "pairs.jsonl"
    pairs_data = [
        {
            "id": "p1",
            "prompt": "Task: update auth.py",
            "positive": "First lock state.md before editing auth.py",
            "negative": "Immediately overwrite auth.py without plan",
        },
        {
            "id": "p2",
            "prompt": "Task: fix bug in router.py",
            "positive": "Record bug fix objective in state.md",
            "negative": "Run random commands",
        },
    ]
    pairs_file.write_text("\n".join(json.dumps(r) for r in pairs_data), encoding="utf-8")
    loaded_pairs = load_contrastive_pairs(pairs_file)
    assert len(loaded_pairs) == 2

    ckpt_in = tmp_path / "base.safetensors"
    rng = np.random.default_rng(99)
    save_safetensors_file(
        ckpt_in,
        {
            "model.layers.11.self_attn.o_proj.weight": (rng.standard_normal((32, 32)) * 0.02).astype(np.float32),
            "model.layers.15.self_attn.o_proj.weight": (rng.standard_normal((32, 32)) * 0.02).astype(np.float32),
            "model.layers.11.mlp.experts.0.down_proj.weight": np.ones((16, 16), dtype=np.float32),
        },
    )

    recipe_file = tmp_path / "test_recipe.forge.json"
    recipe_payload = {
        "recipe_name": "disciplined_coder_v1",
        "profile": "Qwen3.6-35B-A3B",
        "description": "Test rank-2 null-space surgery + ablation recipe",
        "preserve_nullspace": True,
        "nullspace_num_keys": 6,
        "seed": 42,
        "steps": [
            {
                "name": "state_lock_steering",
                "method": "nullspace_rank_k",
                "target_component": "self_attn.o_proj.weight",
                "layers": [11, 15],
                "alpha": 1.8,
                "rank": 2,
                "schedule": "bell",
                "contrastive_file": "pairs.jsonl",
            }
        ],
    }
    recipe_file.write_text(json.dumps(recipe_payload, indent=2), encoding="utf-8")

    ckpt_out = tmp_path / "forged.safetensors"
    assert cmd_inspect_checkpoint(str(ckpt_in)) == 0
    assert cmd_run_recipe(str(recipe_file), str(ckpt_in), str(ckpt_out)) == 0
    assert ckpt_out.is_file()

    forged_tensors, forged_meta = load_safetensors_file(ckpt_out)
    assert forged_meta["circuitforge_recipe"] == "disciplined_coder_v1"
    # Routed expert tensor remains 100% untouched
    np.testing.assert_array_equal(
        forged_tensors["model.layers.11.mlp.experts.0.down_proj.weight"],
        np.ones((16, 16), dtype=np.float32),
    )
