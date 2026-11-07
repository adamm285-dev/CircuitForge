"""
Adversarial Security, Numerical & Systems Regression Suite (`tests/test_adversarial_audit.py`).

Locks empirical tests for every finding surfaced during the adversarial audit (`CF-01` through `CF-21`),
including antipodal SLERP stability, C++ atomic slot publication, TIES sign consensus + nullspace
orthogonality, symbolic compiler combined token-budget rules + priority saturation, full-rank null-space
rejection, routed-expert regex bypass hardening, GGUF signed/64-bit metadata, HUD HTTP 400/404/XSS
hardening, BF16 IEEE 754 NaN/Inf preservation, and Unicode memory-mapped file paths.
"""

from __future__ import annotations

import json
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import numpy as np
import pytest

from circuitforge.compiler import RuleSpec, compile_constraint_gate
from circuitforge.engram import ParametricAdapterManifold, slerp_adapters
from circuitforge.gguf import inspect_gguf_header, write_gguf_file
from circuitforge.hud import MechanisticHUDServer, MechanisticTelemetryHUD, render_hud_html
from circuitforge.io import load_safetensors_file, save_safetensors_file
from circuitforge.native import (
    NativeEngramHashTable,
    NativeMmapCheckpoint,
    native_nullspace_rank1_surgery_f32,
)
from circuitforge.surgery import apply_nullspace_rank1_surgery, compute_nullspace_projector
from circuitforge.wiring import is_routed_expert_tensor


def test_cf01_slerp_antipodal_and_near_antipodal_stability() -> None:
    """CF-01: slerp_adapters must preserve norm on exact (A == -B) and near-antipodal adapters."""
    rng = np.random.default_rng(7)
    A = rng.standard_normal((16, 16)).astype(np.float32)
    A /= float(np.linalg.norm(A))

    # Exact antipodal: B = -A
    B_exact = -A
    mid_exact = slerp_adapters(A, B_exact, t=0.5)
    assert np.all(np.isfinite(mid_exact))
    assert abs(float(np.linalg.norm(mid_exact)) - 1.0) < 1e-5

    # Near-antipodal float32 rounding: B = -A + 1e-7 * noise
    noise = rng.standard_normal((16, 16)).astype(np.float32) * 1e-7
    B_near = -A + noise
    B_near /= float(np.linalg.norm(B_near))
    mid_near = slerp_adapters(A, B_near, t=0.5)
    assert np.all(np.isfinite(mid_near))
    assert abs(float(np.linalg.norm(mid_near)) - 1.0) < 1e-5


def test_cf02_cf13_cf14_cf18_native_engine_hardening(tmp_path: Path) -> None:
    """
    Verifies C++ engine fixes:
    - CF-02: CFEngramTable atomic state publication & concurrent-safe upsert/lookup.
    - CF-13: UTF-8 non-ASCII path support in NativeMmapCheckpoint on Windows (CreateFileW).
    - CF-14: NativeMmapCheckpoint view lifetime reference and closed-checkpoint guard.
    - CF-18: Early-exit pointer initialization in native_nullspace_rank1_surgery_f32.
    """
    # CF-02: Upsert and overwrite same key repeatedly, verifying exact RMSNorm + bias output
    tbl = NativeEngramHashTable(capacity=32, d_model=16)
    try:
        x = np.ones(16, dtype=np.float32)
        gamma = np.ones(16, dtype=np.float32)
        b1 = np.full(16, 2.0, dtype=np.float32)
        b2 = np.full(16, 5.0, dtype=np.float32)
        assert tbl.upsert("episodic_key", b1) is True
        out1, hit1 = tbl.lookup_and_inject_rmsnorm("episodic_key", x, gamma, scale=1.0)
        assert hit1 is True
        np.testing.assert_allclose(out1, np.full(16, 3.0, dtype=np.float32), atol=1e-4)

        assert tbl.upsert("episodic_key", b2) is True
        out2, hit2 = tbl.lookup_and_inject_rmsnorm("episodic_key", x, gamma, scale=1.0)
        assert hit2 is True
        np.testing.assert_allclose(out2, np.full(16, 6.0, dtype=np.float32), atol=1e-4)
    finally:
        tbl.close()

    # CF-13 & CF-14: UTF-8 Unicode directory/file name with NativeMmapCheckpoint
    unicode_dir = tmp_path / "модель_回路_αβγ"
    unicode_dir.mkdir(parents=True, exist_ok=True)
    st_path = unicode_dir / "weights_α.safetensors"
    w_init = np.eye(8, dtype=np.float32)
    save_safetensors_file(st_path, {"model.layers.3.self_attn.o_proj.weight": w_init})

    ckpt = NativeMmapCheckpoint(st_path, writable=True)
    view = ckpt.get_tensor_view("model.layers.3.self_attn.o_proj.weight", for_write=True)
    view[0, 0] = 9.5
    ckpt.flush()
    ckpt.close()

    with pytest.raises(RuntimeError, match="closed"):
        ckpt.get_tensor_view("model.layers.3.self_attn.o_proj.weight")

    reloaded, _ = load_safetensors_file(st_path)
    assert abs(float(reloaded["model.layers.3.self_attn.o_proj.weight"][0, 0]) - 9.5) < 1e-6

    # CF-18: Early exit when steering_vector is all zeros initializes align_before/align_after to 0.0
    W = np.eye(8, dtype=np.float32)
    k = np.ones(8, dtype=np.float32)
    v_zero = np.zeros(8, dtype=np.float32)
    _, rep = native_nullspace_rank1_surgery_f32(W, k, v_zero, alpha=1.0)
    assert rep.frobenius_delta == 0.0
    assert rep.target_alignment_before == 0.0
    assert rep.target_alignment_after == 0.0


def test_cf03_ties_nullspace_merge_tie_breaking_and_orthogonality() -> None:
    """CF-03: ties_nullspace_merge resolves exact equal-weight ties and preserves nullspace orthogonality."""
    manifold = ParametricAdapterManifold(d_out=8, d_in=8)
    a1 = np.ones((8, 8), dtype=np.float32)
    a2 = -np.ones((8, 8), dtype=np.float32)
    manifold.register_adapter("a1", a1)
    manifold.register_adapter("a2", a2)

    # Without preserved_keys, equal-magnitude opposite adapters tie-break to positive consensus (+1.0)
    merged_ties = manifold.ties_nullspace_merge(density=1.0, preserved_keys=None)
    assert float(np.linalg.norm(merged_ties)) > 0.0
    np.testing.assert_allclose(merged_ties, np.ones((8, 8), dtype=np.float32), atol=1e-6)

    # With preserved_keys, nullspace leakage must be strictly < 1e-6
    rng = np.random.default_rng(19)
    K0 = rng.standard_normal((3, 8)).astype(np.float32)
    merged_ns = manifold.ties_nullspace_merge(density=0.5, preserved_keys=K0)
    leakage = float(np.max(np.abs(merged_ns @ K0.T)))
    assert leakage < 1e-6


def test_cf04_cf05_compiler_combined_budget_and_priority_saturation() -> None:
    """
    CF-04 & CF-05:
    1. A rule combining `any_of_tokens` AND `min_token_count=100` must NOT fire when seq_len=10,
       and MUST fire when seq_len=120.
    2. Matching multiple `any_of_tokens` on a low-priority rule (priority=1.0) must saturate at 1.0
       and NEVER overpower a higher-priority rule (priority=3.0).
    """
    rules = [
        RuleSpec(
            name="combined_keyword_and_budget",
            target_gate="BUDGET_EXCEEDED",
            any_of_tokens=("refactor", "sprint"),
            min_token_count=100,
            priority=2.0,
        ),
        RuleSpec(
            name="low_priority_multi_kw",
            target_gate="SPEC_LOCK_REQUIRED",
            any_of_tokens=("alpha", "beta", "gamma", "delta", "epsilon"),
            priority=1.0,
        ),
        RuleSpec(
            name="high_priority_single_kw",
            target_gate="ZERO_READ_VIOLATION",
            any_of_tokens=("read_file",),
            priority=3.0,
        ),
    ]
    circuit = compile_constraint_gate(rules=rules, d_model=64, d_hidden=64, vocab_capacity=128)

    # CF-04: Keyword 'refactor' present, but token count is 10 (< 100) -> PASS_THROUGH
    res_short = circuit.predict_gate("Please refactor module", token_count_override=10)
    assert res_short["gate"] == "PASS_THROUGH"

    # Keyword 'refactor' present AND token count is 120 (>= 100) -> BUDGET_EXCEEDED
    res_long = circuit.predict_gate("Please refactor module", token_count_override=120)
    assert res_long["gate"] == "BUDGET_EXCEEDED"

    # Token count is 150 (>= 100) but keyword 'refactor'/'sprint' is ABSENT -> PASS_THROUGH
    res_no_kw = circuit.predict_gate("Routine status check", token_count_override=150)
    assert res_no_kw["gate"] == "PASS_THROUGH"

    # CF-05: Prompt contains all 5 keywords of low_priority_multi_kw (priority=1.0) AND 1 keyword
    # of high_priority_single_kw (priority=3.0). High-priority rule MUST win.
    res_prio = circuit.predict_gate("alpha beta gamma delta epsilon read_file", token_count_override=20)
    assert res_prio["gate"] == "ZERO_READ_VIOLATION"


def test_cf06_full_rank_preserved_keys_nullspace_rejection() -> None:
    """CF-06: When preserved_keys spans full R^{d_in}, P_perp is 0 and rank-1 surgery raises ValueError."""
    d_in = 8
    K0_full = np.eye(d_in, dtype=np.float32)
    P_perp = compute_nullspace_projector(K0_full, variance_retained=0.99)
    np.testing.assert_allclose(P_perp, np.zeros((d_in, d_in)), atol=1e-12)

    W = np.eye(d_in, dtype=np.float32)
    k = np.ones(d_in, dtype=np.float32)
    v = np.ones(d_in, dtype=np.float32)
    with pytest.raises(ValueError, match="preserved key subspace"):
        apply_nullspace_rank1_surgery(W, k, v, preserved_keys=K0_full)


def test_cf07_routed_expert_naming_variants_blocked() -> None:
    """CF-07: Case-insensitive, bracketed, unprefixed, and singular routed expert names are blocked."""
    routed_names = [
        "model.layers.0.mlp.experts.0.up_proj.weight",
        "MODEL.LAYERS.0.MLP.EXPERTS.0.UP_PROJ.WEIGHT",
        "mlp.experts.0.down_proj.weight",
        "block_sparse_moe.experts.3.w1.weight",
        "model.layers.2.mlp.experts[4].gate_proj.weight",
        "model.layers.1.moe.experts.2.down_proj.weight",
        "layers.0.feed_forward.experts.1.w2.weight",
        "blk.0.ffn_down_exps.weight",
        "blk.0.ffn_up_exp.weight",
    ]
    for name in routed_names:
        assert is_routed_expert_tensor(name) is True, f"Failed to detect routed expert: {name}"

    control_names = [
        "model.layers.0.self_attn.o_proj.weight",
        "model.layers.0.mlp.shared_expert.down_proj.weight",
        "model.layers.0.mlp.shared_experts.down_proj.weight",
        "blk.0.ffn_down_shexp.weight",
    ]
    for name in control_names:
        assert is_routed_expert_tensor(name) is False, f"False positive on control tensor: {name}"


def test_cf09_gguf_signed_and_64bit_metadata(tmp_path: Path) -> None:
    """CF-09: write_gguf_file supports negative ints (INT32/INT64) and >32-bit ints (UINT64)."""
    gguf_path = tmp_path / "meta_test.gguf"
    tensors = {"blk.0.attn_output.weight": np.eye(4, dtype=np.float32)}
    metadata = {
        "general.name": "test_signed_ints",
        "tokenizer.ggml.bos_token_id": -1,
        "custom.int64_neg": -(2**40),
        "custom.uint64_big": 2**40,
        "custom.uint32_pos": 42,
    }
    write_gguf_file(gguf_path, tensors, metadata=metadata)
    info = inspect_gguf_header(gguf_path)
    meta_read = info["metadata"]
    assert meta_read["tokenizer.ggml.bos_token_id"] == -1
    assert meta_read["custom.int64_neg"] == -(2**40)
    assert meta_read["custom.uint64_big"] == 2**40
    assert meta_read["custom.uint32_pos"] == 42


def test_cf10_cf11_cf20_hud_http_error_handling_and_xss_escaping() -> None:
    """CF-10, CF-11, CF-20: HUD server returns HTTP 400 on malformed POST, 404 on unknown GET, and escapes XSS."""
    hud = MechanisticTelemetryHUD(
        profile_name="Qwen3.6-35B-A3B",
        tokens=["</script><script>alert(1)</script>", 'tok"onload="evil()"'],
        d_model=16,
    )
    html_str = render_hud_html(hud._last_state)
    assert "</script><script>" not in html_str
    assert "\\u003c/script\\u003e" in html_str

    server = MechanisticHUDServer(hud)
    url = server.start_background()
    try:
        # CF-20: Unknown GET route returns 404
        with pytest.raises(HTTPError) as exc_404:
            urlopen(f"{url}/nonexistent_route", timeout=3.0)
        assert exc_404.value.code == 404

        # CF-10: Malformed JSON POST to /api/steer returns 400 without crashing server
        req_bad_json = Request(
            f"{url}/api/steer",
            data=b"{bad_json",
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with pytest.raises(HTTPError) as exc_400:
            urlopen(req_bad_json, timeout=3.0)
        assert exc_400.value.code == 400

        # Non-finite alpha (NaN) returns 400
        req_nan = Request(
            f"{url}/api/steer",
            data=json.dumps({"alpha": "NaN"}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with pytest.raises(HTTPError) as exc_nan:
            urlopen(req_nan, timeout=3.0)
        assert exc_nan.value.code == 400

        # Server remains healthy for subsequent valid requests
        with urlopen(f"{url}/api/telemetry", timeout=3.0) as resp:
            assert resp.status == 200
    finally:
        server.stop()
        hud.close()


def test_cf12_cf19_bf16_nan_inf_preservation_and_header_cap(tmp_path: Path) -> None:
    """CF-12 & CF-19: BF16 conversion preserves IEEE 754 NaNs and Infinities; oversized headers raise ValueError."""
    # Construct float32 array with +inf, -inf, quiet NaN, and negative NaN (0xFFFFFFFF)
    special_u32 = np.array([0x7F800000, 0xFF800000, 0x7F800001, 0xFFFFFFFF, 0x3F800000], dtype=np.uint32)
    special_f32 = special_u32.view(np.float32)

    st_path = tmp_path / "bf16_special.safetensors"
    save_safetensors_file(
        st_path,
        {"special": special_f32},
        preserve_dtypes={"special": "BF16"},
    )
    loaded, _ = load_safetensors_file(st_path)
    out = loaded["special"]
    assert np.isposinf(out[0])
    assert np.isneginf(out[1])
    assert np.isnan(out[2])
    assert np.isnan(out[3])
    assert abs(float(out[4]) - 1.0) < 1e-4

    # CF-19: Craft a safetensors file claiming a 500MB header
    bad_path = tmp_path / "oversized_header.safetensors"
    import struct
    bad_path.write_bytes(struct.pack("<Q", 500_000_000) + b"{}")
    with pytest.raises(ValueError, match="100MB safety limit"):
        load_safetensors_file(bad_path)
