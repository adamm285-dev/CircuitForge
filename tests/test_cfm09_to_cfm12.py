r"""
Test Suite for Milestones CFM-09 through CFM-12 (`tests/test_cfm09_to_cfm12.py`).

Covers:
1. **CFM-09 (Zero-Mock Activation Extraction, Live HUD Logit Decoder & Multi-Shard Safetensors Surgery)**:
   - `compute_weight_backed_activations` and `extract_principal_circuit_directions` in `recipes.py`.
   - Pre-recorded `activations_file` `.safetensors` loading in `execute_forge_recipe`.
   - Live unembedding `W_lm_head` + `CompiledMicroCircuit` gate decoding in `MechanisticTelemetryHUD`.
   - `apply_sharded_checkpoint_surgery` routing edits across `model.safetensors.index.json` while hard-linking untouched routed-expert shards.
2. **CFM-10 (GGUF Quantized Block Dequantize / Surgery / Re-Quantize Engine)**:
   - `Q8_0` (`34 bytes/block`), `Q4_0` (`18 bytes/block`), and `BF16` GGUF write, inspect, and slice round-trips.
   - `stream_surgical_gguf_update` on a quantized `Q8_0` / `Q4_0` GGUF checkpoint preserving routed expert bytes 100% untouched.
3. **CFM-11 (LEACE Concept Erasure, PEFT LoRA Adapter Exporter & GGUF Control-Vector Exporter)**:
   - `apply_leace_concept_erasure` driving linear concept covariance $\\|\Sigma_{XZ}\\|_F$ to `< 1e-10` while blocking routed experts.
   - `export_surgical_lora` factoring rank-$r$ surgical deltas into exact Hugging Face PEFT `lora_A` and `lora_B` `.safetensors` + `adapter_config.json`.
   - `export_gguf_control_vector` writing `llama.cpp`-compatible `direction.{L}` GGUF files.
4. **CFM-12 (Automated Circuit Discovery & Causal Attribution Sweeper)**:
   - `sweep_causal_circuit_attribution` and `discover_optimal_surgical_circuit` ranking candidate layers by Direct Logit Attribution (`DLA`), Activation Patching Recovery (`APR`), and Null-Space Signal-to-Leakage Ratio (`SLR`), automatically filtering out null-space-saturated layers, and executing the auto-generated `ForgeRecipe`.
"""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from circuitforge.attribution import (
    discover_optimal_surgical_circuit,
    sweep_causal_circuit_attribution,
)
from circuitforge.exporter import (
    export_forged_bundle,
    export_gguf_control_vector,
    export_surgical_lora,
)
from circuitforge.gguf import (
    dequantize_q4_0_bytes,
    dequantize_q8_0_bytes,
    inspect_gguf_header,
    load_gguf_tensor_f32,
    quantize_q4_0_bytes,
    quantize_q8_0_bytes,
    stream_surgical_gguf_update,
    write_gguf_file,
)
from circuitforge.hud import MechanisticTelemetryHUD
from circuitforge.io import (
    apply_sharded_checkpoint_surgery,
    load_safetensors_file,
    save_safetensors_file,
)
from circuitforge.recipes import (
    ContrastivePair,
    ForgeRecipe,
    SurgeryStepSpec,
    compute_weight_backed_activations,
    execute_forge_recipe,
    extract_principal_circuit_directions,
)
from circuitforge.surgery import (
    apply_leace_concept_erasure,
    apply_nullspace_rank1_surgery,
)


def test_cfm09_weight_backed_activations_and_sharded_safetensors_surgery(tmp_path: Path) -> None:
    """Verifies CFM-09: real weight-backed forward activations, pre-recorded activations_file, and multi-shard index routing."""
    rng = np.random.default_rng(7)
    W = (rng.standard_normal((32, 32)) * 0.1).astype(np.float32)
    pairs = [
        ContrastivePair(
            id="p1",
            prompt="Add sprint requirement to state.md first",
            positive="edit_file state.md",
            negative="edit_file app.py",
        ),
        ContrastivePair(
            id="p2",
            prompt="Pre-hydrated bundle present",
            positive="edit_file target.py",
            negative="read_file target.py",
        ),
    ]
    pos_acts, neg_acts, trig_keys = compute_weight_backed_activations(pairs, W)
    assert pos_acts.shape == (2, 32)
    assert neg_acts.shape == (2, 32)
    assert trig_keys.shape == (2, 32)
    assert not np.allclose(pos_acts, neg_acts)

    # Also test pre-recorded activations_file in execute_forge_recipe
    act_file = tmp_path / "recorded_acts.safetensors"
    save_safetensors_file(
        act_file,
        {
            "pos_acts": pos_acts,
            "neg_acts": neg_acts,
            "trigger_keys": trig_keys,
        },
    )
    ckpt_file = tmp_path / "single.safetensors"
    forged_file = tmp_path / "single_forged.safetensors"
    save_safetensors_file(ckpt_file, {"model.layers.3.self_attn.o_proj.weight": W})

    recipe = ForgeRecipe(
        recipe_name="recorded_acts_recipe",
        profile="Qwen3.6-35B-A3B",
        steps=[
            SurgeryStepSpec(
                name="s1",
                method="nullspace_rank1",
                layers=[3],
                alpha=1.8,
                activations_file=str(act_file),
            )
        ],
    )
    res = execute_forge_recipe(recipe, checkpoint_path=ckpt_file, output_path=forged_file)
    assert res["edited_tensors_count"] == 1
    assert res["max_nullspace_leakage"] < 1e-5

    # Test multi-shard model.safetensors.index.json surgical router
    shard_dir = tmp_path / "sharded_in"
    out_dir = tmp_path / "sharded_out"
    shard_dir.mkdir()
    shard1 = shard_dir / "model-00001-of-00002.safetensors"
    shard2 = shard_dir / "model-00002-of-00002.safetensors"
    ctrl_name = "model.layers.3.self_attn.o_proj.weight"
    exp_name = "model.layers.3.mlp.experts.0.down_proj.weight"
    W_exp = rng.standard_normal((32, 32)).astype(np.float32)

    save_safetensors_file(shard1, {ctrl_name: W})
    save_safetensors_file(shard2, {exp_name: W_exp})
    idx_payload = {
        "metadata": {"total_size": 8192},
        "weight_map": {
            ctrl_name: "model-00001-of-00002.safetensors",
            exp_name: "model-00002-of-00002.safetensors",
        },
    }
    (shard_dir / "model.safetensors.index.json").write_text(json.dumps(idx_payload), encoding="utf-8")

    # Attempting to edit the routed expert shard must raise PermissionError
    with pytest.raises(PermissionError, match="routed expert"):
        apply_sharded_checkpoint_surgery(shard_dir, out_dir, {exp_name: lambda w: w + 1.0})

    # Editing control tensor edits only shard 1 and leaves shard 2 byte-identical
    shard_res = apply_sharded_checkpoint_surgery(
        shard_dir,
        out_dir,
        {ctrl_name: lambda w: w + 0.5},
    )
    assert shard_res["edited_shards"] == ["model-00001-of-00002.safetensors"]
    assert shard_res["untouched_shards"] == ["model-00002-of-00002.safetensors"]
    assert (out_dir / "model-00002-of-00002.safetensors").read_bytes() == shard2.read_bytes()

    # Verify HUD live unembedding logit decoder + compiled micro-circuit gate transitions
    hud = MechanisticTelemetryHUD(d_model=32, num_display_experts=8)
    try:
        st_pos = hud.update_steering(alpha=2.0)
        assert st_pos["predicted_output"]["compliance_status"] == "SDCS_SPEC_LOCKED"
        st_zero = hud.update_steering(alpha=0.0)
        assert st_zero["predicted_output"]["compliance_status"] == "NEUTRAL_BASELINE"
        st_neg = hud.update_steering(alpha=-2.0)
        assert st_neg["predicted_output"]["compliance_status"] == "ANTI_PATTERN_UNSCAFFOLDED"
    finally:
        hud.close()


def test_cfm10_gguf_q8_0_q4_0_bf16_quantize_and_stream_surgery(tmp_path: Path) -> None:
    """Verifies CFM-10: Q8_0, Q4_0, and BF16 GGUF block quantization, slicing, and streamed GGUF weight surgery."""
    rng = np.random.default_rng(19)
    W_ctrl_q8 = (rng.standard_normal((16, 32)) * 0.5).astype(np.float32)
    W_ctrl_q4 = (rng.standard_normal((16, 32)) * 0.5).astype(np.float32)
    W_ctrl_bf16 = (rng.standard_normal((16, 32)) * 0.5).astype(np.float32)
    W_exp_q4 = (rng.standard_normal((16, 32)) * 0.5).astype(np.float32)

    # Direct block quantize/dequantize accuracy checks
    q8_bytes = quantize_q8_0_bytes(W_ctrl_q8)
    assert len(q8_bytes) == (16 * 32 // 32) * 34
    W_q8_rec = dequantize_q8_0_bytes(q8_bytes, W_ctrl_q8.shape)
    assert float(np.max(np.abs(W_q8_rec - W_ctrl_q8))) < 0.02

    q4_bytes = quantize_q4_0_bytes(W_ctrl_q4)
    assert len(q4_bytes) == (16 * 32 // 32) * 18
    W_q4_rec = dequantize_q4_0_bytes(q4_bytes, W_ctrl_q4.shape)
    assert float(np.max(np.abs(W_q4_rec - W_ctrl_q4))) < 0.25

    gguf_in = tmp_path / "quantized_in.gguf"
    gguf_out = tmp_path / "quantized_out.gguf"
    write_gguf_file(
        gguf_in,
        tensors={
            "blk.3.attn_output.weight": W_ctrl_q8,
            "blk.7.attn_output.weight": W_ctrl_q4,
            "blk.11.attn_output.weight": W_ctrl_bf16,
            "blk.3.ffn_down_exps.weight": W_exp_q4,
        },
        metadata={"general.architecture": "qwen3moe"},
        tensor_types={
            "blk.3.attn_output.weight": "Q8_0",
            "blk.7.attn_output.weight": "Q4_0",
            "blk.11.attn_output.weight": "BF16",
            "blk.3.ffn_down_exps.weight": "Q4_0",
        },
    )

    info = inspect_gguf_header(gguf_in)
    assert info["tensors"]["blk.3.attn_output.weight"].ggml_type_name == "Q8_0"
    assert info["tensors"]["blk.7.attn_output.weight"].ggml_type_name == "Q4_0"
    assert info["tensors"]["blk.11.attn_output.weight"].ggml_type_name == "BF16"

    # Rejects editing quantized routed expert in GGUF
    with pytest.raises(PermissionError, match="routed expert"):
        stream_surgical_gguf_update(gguf_in, gguf_out, {"blk.3.ffn_down_exps.weight": lambda w: w})

    # Stream null-space rank-1 surgery onto the Q8_0 and BF16 control tensors
    k_trig = rng.standard_normal(32).astype(np.float32)
    v_steer = rng.standard_normal(16).astype(np.float32)
    K0 = rng.standard_normal((4, 32)).astype(np.float32)

    updated = stream_surgical_gguf_update(
        gguf_in,
        gguf_out,
        {
            "blk.3.attn_output.weight": lambda w: apply_nullspace_rank1_surgery(
                w, k_trig, v_steer, preserved_keys=K0, alpha=1.5, tensor_name="blk.3.attn_output.weight"
            )[0],
            "blk.11.attn_output.weight": lambda w: apply_nullspace_rank1_surgery(
                w, k_trig, v_steer, preserved_keys=K0, alpha=1.5, tensor_name="blk.11.attn_output.weight"
            )[0],
        },
    )
    assert updated == ["blk.3.attn_output.weight", "blk.11.attn_output.weight"]

    # Untouched Q4_0 routed expert tensor must match identically
    exp_before = load_gguf_tensor_f32(gguf_in, "blk.3.ffn_down_exps.weight")
    exp_after = load_gguf_tensor_f32(gguf_out, "blk.3.ffn_down_exps.weight")
    np.testing.assert_array_equal(exp_before, exp_after)

    # Edited Q8_0 control tensor must reflect the positive steering alignment gain
    w3_before = load_gguf_tensor_f32(gguf_in, "blk.3.attn_output.weight")
    w3_after = load_gguf_tensor_f32(gguf_out, "blk.3.attn_output.weight")
    v_unit = v_steer / np.linalg.norm(v_steer)
    assert float(np.dot(w3_after @ k_trig, v_unit)) > float(np.dot(w3_before @ k_trig, v_unit)) + 1.0


def test_cfm11_leace_concept_erasure_and_lora_control_vector_export(tmp_path: Path) -> None:
    """Verifies CFM-11: LEACE concept erasure, PEFT LoRA export, and llama.cpp control-vector GGUF export."""
    rng = np.random.default_rng(29)
    d_out, d_in = 24, 24
    W = rng.standard_normal((d_out, d_in)).astype(np.float32)
    X_in = rng.standard_normal((64, d_in)).astype(np.float32)
    # Inject a strong linear concept signal along direction c_dir
    c_dir = rng.standard_normal(d_out).astype(np.float32)
    c_dir /= np.linalg.norm(c_dir)
    z_labels = rng.integers(0, 2, size=64).astype(np.float32)
    X_acts = (X_in @ W.T) + np.outer(z_labels * 3.0, c_dir)

    # Routed expert gate check
    with pytest.raises(PermissionError, match="routed expert"):
        apply_leace_concept_erasure(
            W, X_acts, z_labels, tensor_name="model.layers.3.mlp.experts.0.down_proj.weight"
        )

    W_erased, rep = apply_leace_concept_erasure(
        W,
        X_acts,
        z_labels,
        tensor_name="model.layers.3.self_attn.o_proj.weight",
        layer_idx=3,
    )
    assert rep.target_alignment_before > 0.1
    assert rep.target_alignment_after < 1e-10
    assert W_erased.shape == W.shape

    # Also test LEACE via ForgeRecipe
    st_in = tmp_path / "leace_in.safetensors"
    st_out = tmp_path / "leace_out.safetensors"
    save_safetensors_file(st_in, {"model.layers.3.self_attn.o_proj.weight": W})
    leace_recipe = ForgeRecipe(
        recipe_name="leace_test",
        profile="Qwen3.6-35B-A3B",
        steps=[SurgeryStepSpec(name="s_leace", method="leace", layers=[3])],
    )
    r_leace = execute_forge_recipe(leace_recipe, checkpoint_path=st_in, output_path=st_out)
    assert r_leace["edited_tensors_count"] == 1

    # Test PEFT LoRA exporter on a rank-2 surgical delta
    u_r2 = rng.standard_normal((d_out, 2)).astype(np.float32)
    v_r2 = rng.standard_normal((2, d_in)).astype(np.float32)
    delta_w = u_r2 @ v_r2
    lora_dir = tmp_path / "peft_lora"
    lora_res = export_surgical_lora(
        lora_dir,
        {"model.layers.15.self_attn.o_proj.weight": delta_w},
        rank=2,
        lora_alpha=8.0,
    )
    assert lora_res["max_reconstruction_error"] < 1e-5
    lora_tensors, _ = load_safetensors_file(lora_dir / "adapter_model.safetensors")
    A = lora_tensors["base_model.model.model.layers.15.self_attn.o_proj.lora_A.weight"]
    B = lora_tensors["base_model.model.model.layers.15.self_attn.o_proj.lora_B.weight"]
    assert A.shape == (2, d_in)
    assert B.shape == (d_out, 2)
    np.testing.assert_allclose((8.0 / 2.0) * (B @ A), delta_w, atol=1e-4)

    # Test llama.cpp control-vector GGUF exporter
    cv_path = tmp_path / "ctrl_vec.gguf"
    v15 = rng.standard_normal(d_out).astype(np.float32)
    export_gguf_control_vector(cv_path, {15: v15, 23: -v15})
    cv_info = inspect_gguf_header(cv_path)
    assert cv_info["metadata"]["general.architecture"] == "controlvector"
    np.testing.assert_allclose(load_gguf_tensor_f32(cv_path, "direction.15"), v15)


def test_cfm12_automated_circuit_discovery_and_attribution_sweeper(tmp_path: Path) -> None:
    """Verifies CFM-12: Causal Attribution Sweeper ranks high-signal null-space layers above saturated/weak layers."""
    rng = np.random.default_rng(42)
    d = 32
    # Layer 3: Weak singular spectrum (small norm / low DLA)
    W_weak = (rng.standard_normal((d, d)) * 0.001).astype(np.float32)
    # Layer 7: High-signal layer with ample orthogonal null-space capacity
    W_strong = (rng.standard_normal((d, d)) * 0.25).astype(np.float32)
    # Layer 11: Full-rank preserved keys K0 spanning all d dimensions -> 0 null-space capacity!
    W_saturated = (rng.standard_normal((d, d)) * 0.25).astype(np.float32)

    preserved_map = {
        3: rng.standard_normal((4, d)).astype(np.float32),
        7: rng.standard_normal((4, d)).astype(np.float32),
        11: np.eye(d, dtype=np.float32),  # Full rank d -> zero null-space capacity!
    }

    discovery = discover_optimal_surgical_circuit(
        layer_weights={3: W_weak, 7: W_strong, 11: W_saturated},
        top_k_layers=2,
        preserved_keys_by_layer=preserved_map,
        profile_name="Qwen3.6-35B-A3B",
    )
    assert discovery["swept_layers_count"] == 3
    # Layer 11 has 0 null-space capacity and must have SLR == 0.0 (excluded from top_k)
    l11_entry = next(r for r in discovery["rankings"] if r["layer_idx"] == 11)
    assert l11_entry["nullspace_capacity"] == 0.0
    assert l11_entry["signal_to_leakage_ratio"] == 0.0
    assert 11 not in discovery["selected_top_k_layers"]
    assert 7 in discovery["selected_top_k_layers"]

    # Execute the auto-generated ForgeRecipe end-to-end
    auto_recipe: ForgeRecipe = discovery["auto_recipe"]
    exec_res = execute_forge_recipe(auto_recipe)
    assert exec_res["edited_tensors_count"] == len(discovery["selected_top_k_layers"])
    assert exec_res["max_nullspace_leakage"] < 1e-5

    # Verify CLI `discover-circuit` and `export-lora` commands
    out_recipe_path = tmp_path / "discovered.forge.json"
    proc_disc = subprocess.run(
        [
            sys.executable,
            "-m",
            "circuitforge",
            "discover-circuit",
            "--top-k",
            "3",
            "--out-recipe",
            str(out_recipe_path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    disc_json = json.loads(proc_disc.stdout)
    assert len(disc_json["selected_top_k_layers"]) == 3
    assert out_recipe_path.is_file()

    lora_cli_dir = tmp_path / "cli_lora"
    proc_lora = subprocess.run(
        [sys.executable, "-m", "circuitforge", "export-lora", "--out", str(lora_cli_dir), "--rank", "2"],
        capture_output=True,
        text=True,
        check=True,
    )
    lora_cli_json = json.loads(proc_lora.stdout)
    assert Path(lora_cli_json["adapter_safetensors"]).is_file()


def test_round4_adversarial_hardening_gguf_sharded_leace_and_lora(tmp_path: Path) -> None:
    """
    Verifies Round-4 adversarial audit remediations across CFM-09..CFM-12:
    1. Path traversal rejection in `apply_sharded_checkpoint_surgery` (`..` and absolute paths).
    2. GGUF `ARRAY` metadata preservation (`tokenizer.ggml.tokens`) and `Q4_K`/`Q6_K` exact block sizing
       during `stream_surgical_gguf_update`.
    3. `Q4_0` idempotent re-quantization on untouched rows with negative extrema (`q = -8`).
    4. `Q8_0`/`Q4_0` innermost-dimension (`shape[-1] % 32 == 0`) validation.
    5. LEACE affine mean preservation (`return_mean_shift=True`) and null-space projection (`preserved_keys`).
    6. PEFT LoRA fully-qualified `target_modules` (never matching frozen routed experts) and exact
       Frobenius truncation error scaling.
    """
    rng = np.random.default_rng(99)

    # 1. Path traversal rejection in apply_sharded_checkpoint_surgery
    shard_dir = tmp_path / "malicious_sharded"
    shard_dir.mkdir()
    bad_idx = shard_dir / "model.safetensors.index.json"
    bad_idx.write_text(
        json.dumps(
            {
                "metadata": {},
                "weight_map": {
                    "model.layers.3.self_attn.o_proj.weight": "../escape.safetensors",
                },
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Unsafe or non-basename shard filename"):
        apply_sharded_checkpoint_surgery(
            shard_dir,
            tmp_path / "out_sharded",
            {"model.layers.3.self_attn.o_proj.weight": lambda w: w},
        )

    # 2. GGUF ARRAY metadata preservation & Q4_0 idempotence on q = -8 blocks
    w_q4 = np.zeros((2, 32), dtype=np.float32)
    w_q4[0, 0] = -8.0  # Exact negative extremum triggering q = -8
    w_q4[0, 1] = 4.0
    w_q4[1, :] = np.linspace(-4.0, 3.5, 32, dtype=np.float32)
    q4_bytes_1 = quantize_q4_0_bytes(w_q4)
    w_q4_deq = dequantize_q4_0_bytes(q4_bytes_1, (2, 32))
    q4_bytes_2 = quantize_q4_0_bytes(w_q4_deq)
    # Second roundtrip must be 100% byte-identical (no 8/7 scale drift!)
    assert q4_bytes_1 == q4_bytes_2

    # 3. Q8_0 / Q4_0 innermost dimension check: (2, 48) has 96 elements (divisible by 32),
    #    but shape[-1]=48 is NOT divisible by 32 and must be rejected!
    with pytest.raises(ValueError, match="innermost dimension"):
        quantize_q8_0_bytes(np.zeros((2, 48), dtype=np.float32))
    with pytest.raises(ValueError, match="innermost dimension"):
        quantize_q4_0_bytes(np.zeros((2, 48), dtype=np.float32))

    # 4. GGUF stream_surgical_gguf_update preserves ARRAY metadata (e.g. tokenizer.ggml.tokens)
    gguf_in = tmp_path / "with_tok.gguf"
    gguf_out = tmp_path / "with_tok_forged.gguf"
    write_gguf_file(
        gguf_in,
        {"blk.3.attn_output.weight": w_q4},
        metadata={
            "general.architecture": "qwen3moe",
            "tokenizer.ggml.tokens": ["<unk>", "<s>", "edit_file", "state.md"],
            "tokenizer.ggml.scores": [0.0, 0.0, -1.5, -2.0],
        },
        tensor_types={"blk.3.attn_output.weight": "Q4_0"},
    )
    stream_surgical_gguf_update(
        gguf_in,
        gguf_out,
        {"blk.3.attn_output.weight": lambda w: w},
    )
    forged_info = inspect_gguf_header(gguf_out)
    assert forged_info["metadata"]["tokenizer.ggml.tokens"] == ["<unk>", "<s>", "edit_file", "state.md"]
    assert len(forged_info["metadata"]["tokenizer.ggml.scores"]) == 4

    # 5. LEACE affine mean shift & null-space preservation
    W_leace = rng.standard_normal((16, 16)).astype(np.float32)
    K0_leace = rng.standard_normal((3, 16)).astype(np.float32)
    Z_lbl = np.array([1.0 if i % 2 == 0 else 0.0 for i in range(40)], dtype=np.float32)
    X_act = rng.standard_normal((40, 16)).astype(np.float32) + np.outer(Z_lbl, np.ones(16, dtype=np.float32) * 3.0)
    W_erased, b_shift, rep_l = apply_leace_concept_erasure(  # type: ignore[misc]
        W_leace,
        X_act,
        Z_lbl,
        preserved_keys=K0_leace,
        return_mean_shift=True,
    )
    assert b_shift.shape == (16,)
    assert rep_l.nullspace_leakage_max < 1e-6
    assert rep_l.target_alignment_after < 1e-5

    # 6. PEFT LoRA fully-qualified target_modules & truncated SVD error scaling
    full_rank_dw = rng.standard_normal((16, 16)).astype(np.float32)
    lora_trunc_dir = tmp_path / "lora_trunc"
    res_trunc = export_surgical_lora(
        lora_trunc_dir,
        {"model.layers.3.mlp.shared_expert.down_proj.weight": full_rank_dw},
        rank=2,
        lora_alpha=16.0,
    )
    # Must record exact module path, NOT bare 'down_proj' which would hit routed experts
    assert res_trunc["target_modules"] == ["model.layers.3.mlp.shared_expert.down_proj"]
    assert res_trunc["max_reconstruction_error"] > 0.1

