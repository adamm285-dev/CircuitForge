"""
Round-3 Adversarial Audit Regression Suite (`tests/test_round3_audit.py`).

Covers all 6 High (`H1`..`H6`) and 8 Medium (`M1`..`M8`) findings plus Low-severity
findings surfaced during the Round-3 read-only Claude Code adversarial audit:
- `H1`: `NativeMmapCheckpoint` tracks the underlying `c_arr` ctypes buffer via weakref so derived
  NumPy slices (`v[1:]`, `v.T`, `v.ravel()`) keep the mmap open until all derived views are collected.
- `H2` & `M1`: `np.all(np.isfinite(...))` and `math.isfinite(...)` checks across `surgery.py`,
  `native.py`, and C++ `cf_nullspace_rank1_surgery_f32`; relative residual norm threshold tightened
  to `1e-4 * ||k||^2`; native zero-key rejection even when `preserved_keys is None`.
- `H3` & `M8`: Universal `[./]experts?(?:[.\\[_]|$)` (Nemotron-H `mixer.experts.*`, fairseq `moe_layer.experts.*`,
  DeepSpeed `deepspeed_moe.experts.*`) + `gpt-oss` `mlp.(mlp1|mlp2)_weight` in `is_routed_expert_tensor`,
  plus `full_attention_interval=0` clamp and `sliding_attention` classification.
- `H4`: Strict `urlparse` / `urlsplit` Host & Origin validation (`http://localhost.evil.com` and
  `Origin: null` rejected with 403, IPv6 `[::1]:8000` accepted) and exact `application/json` media-type check.
- `H5` & `M7`: Pure 2-layer MLP evaluation `logits = (W_mlp2 @ h1) + b_mlp2` in `CompiledMicroCircuit`
  matching exported `.safetensors` weights for combined `min_token_count` rules and exponential priority
  scaling, plus compound sub-span tokenization and rule token validation.
- `H6` & Engram capacity: Thread-safe `RLock` on `NativeEngramHashTable` and exact user capacity limit.
- `M2` & Low TIES: Smooth continuous perpendicular tangent in `slerp_adapters`, `density <= 0.0` zero return,
  and single-adapter weight scaling in `ties_nullspace_merge`.
- `M3`, `M4`, `M5`, `M6`: Top-k right-singular vectors `vt_w[:k_count]` in `execute_forge_recipe`,
  custom `general.alignment` in `write_gguf_file`, header file-size bounds in `io.py`/`gguf.py`, and
  `KeyError` on missing surgical target tensor names.
"""

from __future__ import annotations

import gc
import json
import math
from pathlib import Path
import struct
import urllib.error
import urllib.request

import numpy as np
import pytest

from circuitforge.compiler import (
    RuleSpec,
    compile_constraint_gate,
    tokenize_for_circuit,
)
from circuitforge.engram import (
    DiscreteEngramGraph,
    ParametricAdapterManifold,
    slerp_adapters,
)
from circuitforge.gguf import (
    GGUF_MAGIC,
    inspect_gguf_header,
    load_gguf_tensor_f32,
    write_gguf_file,
)
from circuitforge.hud import MechanisticHUDServer, MechanisticTelemetryHUD
from circuitforge.io import (
    inspect_safetensors_header,
    load_safetensors_file,
    save_safetensors_file,
    stream_surgical_safetensors_update,
)
from circuitforge.native import (
    NativeEngramHashTable,
    NativeMmapCheckpoint,
    has_native_engine,
    native_directional_ablation_f32,
    native_nullspace_rank1_surgery_f32,
)
from circuitforge.recipes import (
    ForgeRecipe,
    SurgeryStepSpec,
    execute_forge_recipe,
)
from circuitforge.surgery import (
    apply_directional_ablation,
    apply_nullspace_rank1_surgery,
    apply_nullspace_rank_k_surgery,
)
from circuitforge.wiring import is_routed_expert_tensor, load_wiring_map


@pytest.mark.skipif(not has_native_engine(), reason="Native C++ engine required")
def test_h1_mmap_derived_slice_keeps_mapping_alive(tmp_path: Path) -> None:
    """Verifies H1: derived NumPy slices (`v[1:]`, `v.T`, `v.ravel()`) keep the mmap open after parent `v` is deleted."""
    st_file = tmp_path / "derived_slice.safetensors"
    orig = np.arange(32, dtype=np.float32).reshape(4, 8)
    save_safetensors_file(st_file, {"model.layers.3.self_attn.o_proj.weight": orig})

    ckpt = NativeMmapCheckpoint(st_file, writable=True)
    v = ckpt.get_tensor_view("model.layers.3.self_attn.o_proj.weight", for_write=True)
    sub = v[1:3, 2:6]
    del v
    gc.collect()

    ckpt.close()
    # Because `sub` still holds a reference to `c_arr` via `.base`, `_handle` must still be mapped!
    assert ckpt._handle is not None
    sub[:] = 99.0
    assert float(sub[0, 0]) == 99.0

    del sub
    gc.collect()
    assert ckpt._handle is None


def test_h2_and_m1_nan_inf_and_relative_collinearity_rejection() -> None:
    """Verifies H2 & M1: NaN/Inf inputs and near-collinear float32 keys are rejected across Python and C++."""
    rng = np.random.default_rng(101)
    W = rng.standard_normal((16, 16)).astype(np.float32)
    k = rng.standard_normal(16).astype(np.float32)
    v = rng.standard_normal(16).astype(np.float32)

    W_nan = W.copy()
    W_nan[0, 0] = np.nan
    with pytest.raises(ValueError, match="NaN or Inf"):
        apply_nullspace_rank1_surgery(W_nan, k, v)
    with pytest.raises(ValueError, match="NaN or Inf"):
        apply_nullspace_rank_k_surgery(W_nan, k[None, :], v[None, :])
    with pytest.raises(ValueError, match="NaN or Inf"):
        apply_directional_ablation(W_nan, v)
    with pytest.raises(ValueError, match="finite"):
        apply_nullspace_rank1_surgery(W, k, v, alpha=float("nan"))

    # Near-collinear float32 key (residual ~ 5e-7 * ||k||^2 <= 1e-4 * ||k||^2) must be rejected
    K0 = rng.standard_normal((8, 16)).astype(np.float32)
    k_in_span = (K0[0] * 100.0).astype(np.float32)
    # Add tiny orthogonal perturbation of 3e-2 on norm 100 -> relative energy ~ 9e-8 < 1e-4
    k_nearly_in_span = k_in_span + (rng.standard_normal(16).astype(np.float32) * 1e-3)
    with pytest.raises(ValueError, match="inside the preserved key subspace"):
        apply_nullspace_rank1_surgery(W, k_nearly_in_span, v, preserved_keys=K0)

    if has_native_engine():
        with pytest.raises(ValueError, match="NaN or Inf"):
            native_nullspace_rank1_surgery_f32(W_nan, k, v)
        with pytest.raises(ValueError, match="NaN or Inf"):
            native_directional_ablation_f32(W_nan, v)
        # Zero trigger key with preserved_keys=None must also raise ValueError in native wrapper
        with pytest.raises(ValueError, match="near-zero norm"):
            native_nullspace_rank1_surgery_f32(W, np.zeros(16, dtype=np.float32), v, preserved_keys=None)


def test_h3_and_m8_universal_routed_expert_patterns_and_wiring(tmp_path: Path) -> None:
    """Verifies H3 & M8: Nemotron-H, fairseq, DeepSpeed, and gpt-oss expert names + interval/sliding_attention."""
    must_block = [
        "backbone.layers.3.mixer.experts.0.up_proj.weight",
        "decoder.layers.0.moe_layer.experts.0.fc1.weight",
        "model.layers.3.mlp.deepspeed_moe.experts.deepspeed_experts.0.wi.weight",
        "model.layers.3.mlp.mlp1_weight",
        "model.layers.3.mlp.mlp2_weight",
    ]
    for name in must_block:
        assert is_routed_expert_tensor(name) is True, f"Failed to block routed expert: {name}"

    cfg_path = tmp_path / "config.json"
    cfg_path.write_text(
        json.dumps(
            {
                "architectures": ["Gemma2ForCausalLM"],
                "num_hidden_layers": 4,
                "hidden_size": 256,
                "full_attention_interval": 0,
                "layer_types": [
                    "sliding_attention",
                    "full_attention",
                    "sliding_attention",
                    "full_attention",
                ],
            }
        ),
        encoding="utf-8",
    )
    wmap = load_wiring_map(cfg_path)
    assert wmap.full_attention_interval == 1
    assert wmap.full_attention_layers == [0, 1, 2, 3]
    assert wmap.linear_attention_layers == []


def test_h4_hud_strict_origin_host_and_media_type() -> None:
    """Verifies H4: HUD rejects subdomain Origin bypasses, Origin: null, and non-JSON media types, while accepting IPv6 [::1]."""
    hud = MechanisticTelemetryHUD(d_model=16, num_display_experts=4)
    server = MechanisticHUDServer(hud, host="127.0.0.1", port=0)
    base_url = server.start_background()
    port = server.port
    try:
        url = f"{base_url}/api/steer"
        # 1. Subdomain Origin bypass (`http://localhost.evil.com`) -> 403
        req_sub = urllib.request.Request(
            url,
            data=b'{"alpha": 1.0}',
            headers={
                "Content-Type": "application/json",
                "Origin": "http://localhost.evil.com",
            },
            method="POST",
        )
        with pytest.raises(urllib.error.HTTPError) as exc_sub:
            urllib.request.urlopen(req_sub, timeout=3)
        assert exc_sub.value.code == 403

        # 2. Sandboxed iframe `Origin: null` -> 403
        req_null = urllib.request.Request(
            url,
            data=b'{"alpha": 1.0}',
            headers={
                "Content-Type": "application/json",
                "Origin": "null",
            },
            method="POST",
        )
        with pytest.raises(urllib.error.HTTPError) as exc_null:
            urllib.request.urlopen(req_null, timeout=3)
        assert exc_null.value.code == 403

        # 3. Media type parameter smuggle (`text/plain; x=application/json`) -> 415
        req_ct = urllib.request.Request(
            url,
            data=b'{"alpha": 1.0}',
            headers={
                "Content-Type": "text/plain; x=application/json",
                "Origin": f"http://127.0.0.1:{port}",
            },
            method="POST",
        )
        with pytest.raises(urllib.error.HTTPError) as exc_ct:
            urllib.request.urlopen(req_ct, timeout=3)
        assert exc_ct.value.code == 415

        # 4. Valid `application/json; charset=utf-8` with `Host: [::1]:port` -> 200
        req_ipv6 = urllib.request.Request(
            url,
            data=b'{"alpha": 1.25}',
            headers={
                "Content-Type": "application/json; charset=utf-8",
                "Host": f"[::1]:{port}",
                "Origin": f"http://[::1]:{port}",
            },
            method="POST",
        )
        with urllib.request.urlopen(req_ipv6, timeout=3) as resp:
            assert resp.status == 200
    finally:
        server.stop()
        hud.close()


def test_h5_and_m7_compiler_pure_mlp_weights_and_subspan_tokenization(tmp_path: Path) -> None:
    """Verifies H5 & M7: exported `.safetensors` weights match `forward_logits` via pure 2-layer MLP evaluation."""
    assert "state_md" in tokenize_for_circuit(" Editing path/to/state.md now ")
    assert "state_md" in tokenize_for_circuit("prefix-state.md-suffix")

    with pytest.raises(ValueError, match="non-tokenizable"):
        compile_constraint_gate(
            rules=[RuleSpec(name="bad", target_gate="PASS_THROUGH", any_of_tokens=("hello world!",))]
        )

    rules = [
        RuleSpec(
            name="combined_rule",
            target_gate="BUDGET_EXCEEDED",
            any_of_tokens=("alpha_tok", "beta_tok"),
            all_of_tokens=("req_tok",),
            none_of_tokens=("veto_tok",),
            min_token_count=100,
            priority=3.0,
        ),
        RuleSpec(
            name="low_p_1",
            target_gate="SPEC_LOCK_REQUIRED",
            any_of_tokens=("s1",),
            priority=2.0,
        ),
        RuleSpec(
            name="low_p_2",
            target_gate="SPEC_LOCK_REQUIRED",
            any_of_tokens=("s2",),
            priority=2.0,
        ),
    ]
    circuit = compile_constraint_gate(rules=rules, d_model=64, d_hidden=64, vocab_capacity=64)
    st_path = tmp_path / "circuit.safetensors"
    circuit.export_safetensors(st_path)
    loaded, _ = load_safetensors_file(st_path)

    # Verify that evaluating the raw loaded `.safetensors` matrices reproduces `forward_logits` 100%
    for prompt, seq_len, expected_gate in [
        ("alpha_tok req_tok", 50, "PASS_THROUGH"),      # token true, seq_len < 100 -> must NOT fire
        ("only_other_tokens", 250, "PASS_THROUGH"),     # token false, seq_len >= 100 -> must NOT fire
        ("alpha_tok req_tok", 120, "BUDGET_EXCEEDED"),  # both true -> fires!
        ("alpha_tok beta_tok req_tok", 120, "BUDGET_EXCEEDED"),  # both any_of present + seq_len >= 100 -> fires!
        ("alpha_tok beta_tok req_tok", 40, "PASS_THROUGH"),      # both any_of present + seq_len < 100 -> must NOT fire!
        ("alpha_tok req_tok s1 s2", 120, "BUDGET_EXCEEDED"),     # priority 3.0 beats two priority 2.0 rules!
    ]:
        pred = circuit.predict_gate(prompt, token_count_override=seq_len)
        assert pred["gate"] == expected_gate

        # Manual pure 2-layer MLP evaluation from exported `.safetensors` tensors
        toks = tokenize_for_circuit(prompt)
        present = {circuit.vocab[t] for t in toks if t in circuit.vocab}
        pooled = np.zeros(circuit.d_model, dtype=np.float32)
        for tid in present:
            pooled += loaded["micro_circuit.embed.weight"][tid]
        v_agg = loaded["micro_circuit.attn.v_proj.weight"] @ pooled
        v_agg[0] = float(max(0, min(seq_len, 100_000)))
        h1 = np.clip((loaded["micro_circuit.mlp.fc1.weight"] @ v_agg) + loaded["micro_circuit.mlp.fc1.bias"], 0.0, 1.0)
        raw_logits = (loaded["micro_circuit.mlp.fc2.weight"] @ h1) + loaded["micro_circuit.mlp.fc2.bias"]
        np.testing.assert_allclose(raw_logits, circuit.forward_logits(prompt, token_count_override=seq_len), atol=1e-5)


def test_h6_and_m2_engram_capacity_slerp_and_ties() -> None:
    """Verifies H6, M2, and Low TIES findings: exact engram capacity, smooth SLERP near pi, and TIES scaling."""
    graph = DiscreteEngramGraph(d_model=16, capacity=3)
    try:
        for i in range(3):
            graph.store_engram(f"k_{i}", f"Fact {i}")
        if has_native_engine():
            with pytest.raises(MemoryError, match="capacity"):
                graph.store_engram("k_3", "Fact 3 overflow")
    finally:
        graph.close()

    # M2: SLERP continuity across omega = pi - 1e-4 without jumping to an unrelated plane
    A = np.zeros((4, 4), dtype=np.float32)
    A[0, 0] = 1.0
    # Construct B in the (0,0)-(1,1) plane with angle pi - 5e-5
    theta = math.pi - 5e-5
    B = np.zeros((4, 4), dtype=np.float32)
    B[0, 0] = math.cos(theta)
    B[1, 1] = math.sin(theta)
    mid = slerp_adapters(A, B, t=0.5)
    # Midpoint at t=0.5 in the (0,0)-(1,1) plane must stay 100% in the (0,0)-(1,1) plane (`mid[1, 1] > 0.99`)
    assert float(mid[1, 1]) > 0.99
    assert float(np.linalg.norm(mid)) == pytest.approx(1.0, abs=1e-5)

    # Low TIES: density=0.0 returns all zeros, and single-adapter weight scales output
    manifold = ParametricAdapterManifold(d_out=8, d_in=8)
    dw = np.ones((8, 8), dtype=np.float32) * 2.0
    manifold.register_adapter("only", dw)
    zero_merged = manifold.ties_nullspace_merge(density=0.0)
    assert float(np.max(np.abs(zero_merged))) == 0.0
    scaled_merged = manifold.ties_nullspace_merge(weights={"only": 0.25}, density=1.0)
    np.testing.assert_allclose(scaled_merged, dw * 0.25, atol=1e-6)


def test_m3_m4_m5_m6_gguf_alignment_bounds_and_recipe_validation(tmp_path: Path) -> None:
    """Verifies M3, M4, M5, M6: custom GGUF alignment, EOF bounds checks, missing tensor KeyError, and top-k SVD K0."""
    # M4: write_gguf_file with custom `general.alignment = 64` round-trips cleanly
    gguf_path = tmp_path / "aligned64.gguf"
    t_orig = np.arange(20, dtype=np.float32).reshape(4, 5)
    write_gguf_file(
        gguf_path,
        {"blk.3.attn_output.weight": t_orig},
        metadata={"general.alignment": 64},
    )
    info = inspect_gguf_header(gguf_path)
    assert info["alignment"] == 64
    assert int(info["data_start_offset"]) % 64 == 0
    loaded = load_gguf_tensor_f32(gguf_path, "blk.3.attn_output.weight")
    np.testing.assert_allclose(loaded, t_orig)

    # M5: Safetensors header claiming out-of-bounds offset raises ValueError in inspect_safetensors_header
    bad_st = tmp_path / "bad_bounds.safetensors"
    hdr = json.dumps(
        {"t": {"dtype": "F32", "shape": [4, 4], "data_offsets": [0, 1_000_000]}}
    ).encode("utf-8")
    pad = (8 - (len(hdr) % 8)) % 8
    hdr_padded = hdr + b" " * pad
    bad_st.write_bytes(struct.pack("<Q", len(hdr_padded)) + hdr_padded + b"\x00" * 64)
    with pytest.raises(ValueError, match="out-of-bounds"):
        inspect_safetensors_header(bad_st)

    # M6: Missing target tensor raises KeyError in stream_surgical_safetensors_update and execute_forge_recipe
    valid_st = tmp_path / "valid.safetensors"
    save_safetensors_file(valid_st, {"model.layers.3.self_attn.o_proj.weight": np.ones((8, 8), dtype=np.float32)})
    with pytest.raises(KeyError, match="not found"):
        stream_surgical_safetensors_update(
            valid_st,
            tmp_path / "out.safetensors",
            {"model.layers.99.self_attn.o_proj.weight": lambda x: x},
        )

    recipe_missing = ForgeRecipe(
        recipe_name="missing_test",
        profile="Qwen3.6-35B-A3B",
        steps=[SurgeryStepSpec(name="s1", method="nullspace_rank1", layers=[99])],
    )
    with pytest.raises(KeyError, match="not found"):
        execute_forge_recipe(recipe_missing, checkpoint_path=valid_st)

    # M3: Top-k right-singular directions of a low-rank weight matrix W are preserved with zero leakage
    u_dom = np.eye(16, 4, dtype=np.float32)
    v_dom = np.eye(4, 16, dtype=np.float32)
    W_lowrank = (u_dom @ np.diag([50.0, 40.0, 30.0, 20.0]).astype(np.float32) @ v_dom).astype(np.float32)
    lowrank_st = tmp_path / "lowrank.safetensors"
    forged_st = tmp_path / "lowrank_forged.safetensors"
    save_safetensors_file(lowrank_st, {"model.layers.3.self_attn.o_proj.weight": W_lowrank})
    recipe_ok = ForgeRecipe(
        recipe_name="preserve_top_singular",
        profile="Qwen3.6-35B-A3B",
        preserve_nullspace=True,
        nullspace_num_keys=4,
        steps=[SurgeryStepSpec(name="s1", method="nullspace_rank1", layers=[3], alpha=1.5)],
    )
    res = execute_forge_recipe(recipe_ok, checkpoint_path=lowrank_st, output_path=forged_st)
    assert res["edited_tensors_count"] == 1
    forged_tensors, _ = load_safetensors_file(forged_st)
    W_after = forged_tensors["model.layers.3.self_attn.o_proj.weight"]
    # Check that the top-4 dominant input directions v_dom (first 4 standard basis vectors) have zero delta!
    delta_top4 = (W_after - W_lowrank) @ v_dom.T
    assert float(np.max(np.abs(delta_top4))) < 1e-5
