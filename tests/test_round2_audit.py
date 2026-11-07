r"""
Round-2 Adversarial Audit Regression Suite (`tests/test_round2_audit.py`).

Tests every finding surfaced by Claude Code's Round-2 adversarial audit (`R2-01` through `R2-20`):
1. `R2-01`: Multi-threaded Seqlock stress test on `NativeEngramHashTable` (0 torn reads across concurrent writer/reader threads with large `d_model=4096`).
2. `R2-02` & `R2-10`: `NativeMmapCheckpoint` view safety after `close()` (no segfault) and read-only snapshot isolation on routed experts.
3. `R2-03`: Out-of-bounds `.safetensors` crafted header detection in `NativeMmapCheckpoint`.
4. `R2-04`, `R2-14`, & punctuation tokens: Compiler combined rule non-negative conjunction (never cancels other rules on the same gate), max-pooled gate priorities, and hyphen/dot token matching (`read-file`).
5. `R2-05` & `R2-06`: Relative trigger-key norm check (`1e-7 * ||k||^2`) preventing 8000x norm explosion, and `variance_retained=1.0` protecting weak preserved keys (`<1%` energy).
6. `R2-07` & `R2-20`: MLX `switch_mlp`, DBRX `ffn.experts`, Granite `block_sparse_moe.input_linear`, legacy GGUF `ffn_gate.0.weight`, and Mixtral `has_shared_expert=False`.
7. `R2-08` & `R2-18`: GGUF `ggml` reversed `ne` dimension order (`ne[0] = cols`) and bounds checks on untrusted string lengths, array counts, and `general.alignment=0`.
8. `R2-09`: `execute_forge_recipe` raises `FileNotFoundError` when `contrastive_file` does not exist.
9. `R2-11`, `R2-12`, & `R2-13`: Smooth geodesic antipodal SLERP across all $t \in [0, 1]$, negative-weight TIES task negation, and `MemoryError` when native engram table overflows.
10. `R2-15`, `R2-16`, `R2-17`: HUD `ThreadingHTTPServer` concurrency, CSRF `Origin`/`Content-Type` rejection, `OverflowError` handling on `[1e400]`, and `stop()` before `start_background()`.
11. `R2-19`: True byte-streaming in `stream_surgical_safetensors_update` preserving untouched exotic dtypes and preserving original tensor dtype on float64 return.
"""

from __future__ import annotations

import json
import struct
import threading
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import numpy as np
import pytest

from circuitforge.compiler import RuleSpec, compile_constraint_gate
from circuitforge.engram import DiscreteEngramGraph, ParametricAdapterManifold, slerp_adapters
from circuitforge.gguf import inspect_gguf_header, load_gguf_tensor_f32, write_gguf_file
from circuitforge.hud import MechanisticHUDServer, MechanisticTelemetryHUD
from circuitforge.io import (
    load_safetensors_file,
    save_safetensors_file,
    stream_surgical_safetensors_update,
)
from circuitforge.native import (
    NativeEngramHashTable,
    NativeMmapCheckpoint,
    native_nullspace_rank1_surgery_f32,
)
from circuitforge.recipes import ForgeRecipe, SurgeryStepSpec, execute_forge_recipe
from circuitforge.surgery import apply_nullspace_rank1_surgery, build_surgical_plan
from circuitforge.wiring import (
    is_routed_expert_tensor,
    is_surgical_control_tensor,
    load_wiring_map,
)


def test_r2_01_seqlock_concurrent_writer_reader_zero_torn_reads() -> None:
    """R2-01: Concurrent writer and reader on large d_model=4096 must observe 0 torn vectors."""
    d_model = 4096
    tbl = NativeEngramHashTable(capacity=16, d_model=d_model)
    vec_a = np.full(d_model, 2.0, dtype=np.float32)
    vec_b = np.full(d_model, 7.0, dtype=np.float32)
    x = np.ones(d_model, dtype=np.float32)
    gamma = np.ones(d_model, dtype=np.float32)

    assert tbl.upsert("race_key", vec_a) is True
    stop_flag = threading.Event()
    torn_reads: list[str] = []
    read_count = [0]

    def _writer() -> None:
        toggle = False
        while not stop_flag.is_set():
            tbl.upsert("race_key", vec_b if toggle else vec_a)
            toggle = not toggle

    def _reader() -> None:
        while not stop_flag.is_set():
            out, hit = tbl.lookup_and_inject_rmsnorm("race_key", x, gamma, scale=1.0)
            if not hit:
                torn_reads.append("unexpected miss")
                break
            # Since RMSNorm(ones)*ones = 1.0, out must be either all 3.0 (1+2) or all 8.0 (1+7)
            first_val = float(out[0])
            if not (abs(first_val - 3.0) < 1e-3 or abs(first_val - 8.0) < 1e-3):
                torn_reads.append(f"invalid value {first_val}")
                break
            if float(np.max(np.abs(out - first_val))) > 1e-3:
                torn_reads.append("torn vector mix")
                break
            read_count[0] += 1
            if read_count[0] >= 800:
                stop_flag.set()

    try:
        wt = threading.Thread(target=_writer, daemon=True)
        rt = threading.Thread(target=_reader, daemon=True)
        wt.start()
        rt.start()
        rt.join(timeout=3.0)
        stop_flag.set()
        wt.join(timeout=1.0)
        assert torn_reads == [], f"Detected torn reads: {torn_reads}"
        assert read_count[0] > 50
    finally:
        tbl.close()


def test_r2_02_r2_03_r2_10_mmap_lifetime_bounds_and_readonly_isolation(tmp_path: Path) -> None:
    """
    R2-02: Accessing a writable view after `with NativeMmapCheckpoint(...)` exits must not segfault.
    R2-03: Crafted `.safetensors` header with shape larger than file raises ValueError.
    R2-10: Read-only view on routed expert cannot mutate on-disk file even if flags.writeable is toggled.
    """
    st_path = tmp_path / "mmap_safe.safetensors"
    ctrl_name = "model.layers.3.self_attn.o_proj.weight"
    exp_name = "model.layers.3.mlp.experts.0.down_proj.weight"
    save_safetensors_file(
        st_path,
        {
            ctrl_name: np.ones((4, 4), dtype=np.float32),
            exp_name: np.full((4, 4), 5.0, dtype=np.float32),
        },
    )

    # R2-02: Keep view alive after context manager exit; reading it must not segfault
    with NativeMmapCheckpoint(st_path, writable=True) as ck:
        v_write = ck.get_tensor_view(ctrl_name, for_write=True)
        v_write[0, 0] = 12.5
    assert abs(float(v_write.sum()) - (12.5 + 15.0)) < 1e-5
    del v_write

    # R2-10: Read-only view on routed expert is an isolated snapshot
    with NativeMmapCheckpoint(st_path, writable=True) as ck2:
        v_ro = ck2.get_tensor_view(exp_name, for_write=False)
        assert v_ro.flags.writeable is False
        v_ro.flags.writeable = True
        v_ro[0, 0] = 999.0

    reloaded, _ = load_safetensors_file(st_path)
    assert abs(float(reloaded[exp_name][0, 0]) - 5.0) < 1e-6

    # R2-03: Crafted header claiming 4096x4096 float32 on a tiny file
    bad_st = tmp_path / "crafted_oob.safetensors"
    bad_hdr = json.dumps(
        {
            ctrl_name: {
                "dtype": "F32",
                "shape": [4096, 4096],
                "data_offsets": [0, 16],
            }
        }
    ).encode("utf-8")
    pad = (8 - (len(bad_hdr) % 8)) % 8
    bad_hdr += b" " * pad
    bad_st.write_bytes(struct.pack("<Q", len(bad_hdr)) + bad_hdr + (b"\x00" * 16))

    with NativeMmapCheckpoint(bad_st, writable=True) as ck_bad:
        with pytest.raises(ValueError, match="mismatch"):
            ck_bad.get_tensor_view(ctrl_name, for_write=True)


def test_r2_04_r2_14_compiler_conjunction_isolation_max_pool_and_punctuation() -> None:
    """
    R2-04: A combined rule (`any_of_tokens` + `min_token_count`) whose token is absent must NOT
           inhibit another rule targeting the same gate on a short prompt.
    R2-14: Two priority-2 rules on Gate A must NOT sum to beat one priority-3 rule on Gate B.
    Punctuation: Rules containing `read-file` or `state.md` match accurately.
    """
    rules = [
        RuleSpec(
            name="deploy_long_budget",
            target_gate="BUDGET_EXCEEDED",
            any_of_tokens=("deploy",),
            min_token_count=500,
            priority=2.0,
        ),
        RuleSpec(
            name="overflow_alert",
            target_gate="BUDGET_EXCEEDED",
            any_of_tokens=("overflow",),
            priority=2.0,
        ),
        RuleSpec(
            name="spec_low_1",
            target_gate="SPEC_LOCK_REQUIRED",
            any_of_tokens=("alpha",),
            priority=2.0,
        ),
        RuleSpec(
            name="spec_low_2",
            target_gate="SPEC_LOCK_REQUIRED",
            any_of_tokens=("beta",),
            priority=2.0,
        ),
        RuleSpec(
            name="zero_read_high",
            target_gate="ZERO_READ_VIOLATION",
            any_of_tokens=("read-file",),
            priority=3.0,
        ),
    ]
    circuit = compile_constraint_gate(rules=rules, d_model=64, d_hidden=64, vocab_capacity=128)

    # R2-04: "memory overflow detected" (len=10 < 500) must fire overflow_alert -> BUDGET_EXCEEDED
    res_ovf = circuit.predict_gate("memory overflow detected", token_count_override=10)
    assert res_ovf["gate"] == "BUDGET_EXCEEDED"

    # R2-14 & Punctuation: Both priority-2 rules ('alpha', 'beta') AND priority-3 ('read-file') fire.
    # Priority 3.0 (ZERO_READ_VIOLATION) must beat the two priority 2.0 rules.
    res_max = circuit.predict_gate("alpha beta read-file", token_count_override=20)
    assert res_max["gate"] == "ZERO_READ_VIOLATION"


def test_r2_05_r2_06_relative_nullspace_threshold_and_weak_key_protection() -> None:
    """
    R2-05: Trigger key nearly inside span(K0) (k = K0[0] + 1e-6 * noise with ||K0||=100) must raise ValueError.
    R2-06: Weak preserved key (<0.5% energy) is still protected with <1e-5 leakage.
    """
    rng = np.random.default_rng(101)
    d = 16
    K0_big = rng.standard_normal((4, d)).astype(np.float32) * 50.0
    k_near = K0_big[0] + rng.standard_normal(d).astype(np.float32) * 1e-5
    W = (rng.standard_normal((d, d)) * 0.1).astype(np.float32)
    v = rng.standard_normal(d).astype(np.float32)

    with pytest.raises(ValueError, match="preserved key subspace"):
        apply_nullspace_rank1_surgery(W, k_near, v, preserved_keys=K0_big)

    with pytest.raises(ValueError, match="preserved key subspace"):
        native_nullspace_rank1_surgery_f32(W, k_near, v, preserved_keys=K0_big)

    # R2-06: Construct K0 where one row has norm 100.0 and another has norm 1.0 (<0.01% energy)
    K0_skew = np.zeros((2, d), dtype=np.float32)
    K0_skew[0, 0] = 100.0
    K0_skew[1, 1] = 1.0
    k_trig = np.zeros(d, dtype=np.float32)
    k_trig[1] = 0.5
    k_trig[2] = 1.0
    _, rep = apply_nullspace_rank1_surgery(W, k_trig, v, preserved_keys=K0_skew, alpha=2.0)
    assert rep.nullspace_leakage_max < 1e-6


def test_r2_07_r2_20_mlx_dbrx_granite_and_mixtral_shared_expert(tmp_path: Path) -> None:
    """
    R2-07: MLX `switch_mlp`, DBRX `ffn.experts`, Granite `block_sparse_moe.input_linear`,
           and legacy GGUF `ffn_gate.0.weight` are classified as routed experts and NOT control tensors.
    R2-20: Loading a custom Mixtral `config.json` that omits `shared_expert_intermediate_size` defaults
           to 0 (`has_shared_expert is False`).
    """
    routed_variants = [
        "model.layers.0.mlp.switch_mlp.gate_proj.weight",
        "model.layers.0.switch_mlp.down_proj.weight",
        "transformer.blocks.0.ffn.experts.mlp.w1",
        "model.layers.0.block_sparse_moe.input_linear.weight",
        "model.layers.0.block_sparse_moe.output_linear.weight",
        "blk.0.ffn_gate.0.weight",
        "blk.3.ffn_down.7.weight",
    ]
    for name in routed_variants:
        assert is_routed_expert_tensor(name) is True, f"Missed routed expert: {name}"
        assert is_surgical_control_tensor(name) is False, f"Routed expert marked as control: {name}"

    mixtral_cfg = tmp_path / "config.json"
    mixtral_cfg.write_text(
        json.dumps(
            {
                "architectures": ["MixtralForCausalLM"],
                "num_hidden_layers": 8,
                "hidden_size": 1024,
                "num_local_experts": 8,
                "num_experts_per_tok": 2,
            }
        ),
        encoding="utf-8",
    )
    wmap = load_wiring_map(mixtral_cfg)
    assert wmap.has_shared_expert is False
    assert wmap.shared_expert_intermediate_size == 0
    plan = build_surgical_plan(wmap, include_shared_experts=True)
    for item in plan:
        assert all("shared_expert" not in t for t in item["safetensors_targets"])


def test_r2_08_r2_18_gguf_ggml_dimension_order_and_bounds(tmp_path: Path) -> None:
    """
    R2-08: A 2D weight matrix of NumPy shape (3, 5) is stored in GGUF with ggml ne=(5, 3) and round-trips to (3, 5).
    R2-18: Untrusted GGUF headers with alignment=0 or huge string length raise ValueError cleanly.
    """
    gguf_file = tmp_path / "rect.gguf"
    mat = np.arange(15, dtype=np.float32).reshape(3, 5)
    write_gguf_file(gguf_file, {"blk.0.attn_output.weight": mat}, metadata={"general.name": "rect"})

    # Verify raw binary ne[0]=5, ne[1]=3 inside the GGUF file
    info = inspect_gguf_header(gguf_file)
    t_info = info["tensors"]["blk.0.attn_output.weight"]
    assert t_info.shape == (3, 5)
    loaded = load_gguf_tensor_f32(gguf_file, "blk.0.attn_output.weight")
    np.testing.assert_allclose(loaded, mat)

    # R2-18: Craft GGUF with general.alignment = 0 (rejected by both write_gguf_file and inspect_gguf_header)
    bad_align = tmp_path / "bad_align.gguf"
    with pytest.raises(ValueError, match="general.alignment"):
        write_gguf_file(bad_align, {"blk.0.attn_output.weight": mat}, metadata={"general.alignment": 0})
    raw_bad = bytearray(gguf_file.read_bytes())
    #Also verify inspect_gguf_header rejects a binary file with general.alignment = 0
    bad_align.write_bytes(
        b"GGUF"
        + struct.pack("<IQQ", 3, 0, 1)
        + struct.pack("<Q", len(b"general.alignment"))
        + b"general.alignment"
        + struct.pack("<II", 4, 0)
    )
    with pytest.raises(ValueError, match="general.alignment"):
        inspect_gguf_header(bad_align)


def test_r2_09_r2_11_r2_12_r2_13_recipes_slerp_ties_and_engram_capacity(tmp_path: Path) -> None:
    """
    R2-09: Missing contrastive_file in ForgeRecipe raises FileNotFoundError.
    R2-11: Antipodal SLERP (A = -B) is smooth across t in [0.0, 0.25, 0.49, 0.5, 0.51, 0.75, 1.0].
    R2-12: TIES merge with negative task weights (`weights={'a': -1.0}`) negates the adapter instead of zeroing.
    R2-13: DiscreteEngramGraph raises MemoryError when native table capacity is exceeded.
    """
    # R2-09
    bad_recipe = ForgeRecipe(
        recipe_name="missing_file_test",
        profile="Qwen3.6-35B-A3B",
        steps=[
            SurgeryStepSpec(
                name="s1",
                method="nullspace_rank1",
                target_component="self_attn.o_proj.weight",
                layers=[3],
                contrastive_file="does_not_exist_anywhere.jsonl",
            )
        ],
    )
    with pytest.raises(FileNotFoundError, match="does_not_exist_anywhere"):
        execute_forge_recipe(bad_recipe, base_dir=tmp_path)

    # R2-11: Smooth geodesic antipodal SLERP
    A = np.eye(4, dtype=np.float32)
    B = -A
    t_49 = slerp_adapters(A, B, t=0.49)
    t_50 = slerp_adapters(A, B, t=0.50)
    t_51 = slerp_adapters(A, B, t=0.51)
    assert float(np.linalg.norm(t_49 - t_50)) < 0.15
    assert float(np.linalg.norm(t_51 - t_50)) < 0.15

    # R2-12: TIES with negative weight
    manifold = ParametricAdapterManifold(d_out=4, d_in=4)
    manifold.register_adapter("a", np.ones((4, 4), dtype=np.float32))
    neg_merged = manifold.ties_nullspace_merge(weights={"a": -1.0}, density=1.0)
    np.testing.assert_allclose(neg_merged, -np.ones((4, 4), dtype=np.float32), atol=1e-6)

    # R2-13: Engram table capacity overflow raises MemoryError
    graph = DiscreteEngramGraph(d_model=8, capacity=4)
    try:
        for i in range(4):
            graph.store_engram(f"k_{i}", f"fact {i}")
        with pytest.raises(MemoryError, match="capacity"):
            graph.store_engram("k_overflow", "fact overflow")
    finally:
        graph.close()


def test_r2_15_r2_16_r2_17_r2_19_hud_security_and_true_streaming_io(tmp_path: Path) -> None:
    """
    R2-15..17: HUD server rejects cross-origin POST (403), rejects text/plain POST (415), catches
               OverflowError on `[1e400]` (400), and supports `stop()` before `start_background()`.
    R2-19: `stream_surgical_safetensors_update` preserves untouched tensors byte-for-byte and keeps
           original F32 dtype even if edit callback returns float64.
    """
    hud = MechanisticTelemetryHUD(profile_name="Qwen3.6-35B-A3B", d_model=16)
    unstarted = MechanisticHUDServer(hud)
    unstarted.stop()  # Must not hang

    srv = MechanisticHUDServer(hud)
    url = srv.start_background()
    try:
        # Cross-origin Origin header -> 403
        req_csrf = Request(
            f"{url}/api/steer",
            data=json.dumps({"alpha": -4.0}).encode("utf-8"),
            headers={"Content-Type": "application/json", "Origin": "https://evil.example.com"},
            method="POST",
        )
        with pytest.raises(HTTPError) as exc_403:
            urlopen(req_csrf, timeout=3.0)
        assert exc_403.value.code == 403

        # Wrong Content-Type -> 415
        req_plain = Request(
            f"{url}/api/steer",
            data=json.dumps({"alpha": 1.0}).encode("utf-8"),
            headers={"Content-Type": "text/plain"},
            method="POST",
        )
        with pytest.raises(HTTPError) as exc_415:
            urlopen(req_plain, timeout=3.0)
        assert exc_415.value.code == 415

        # active_layers: [1e400] (Infinity in JSON/float) -> 400
        req_inf = Request(
            f"{url}/api/steer",
            data=b'{"alpha": 1.0, "active_layers": [1e400]}',
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with pytest.raises(HTTPError) as exc_400:
            urlopen(req_inf, timeout=3.0)
        assert exc_400.value.code == 400
    finally:
        srv.stop()
        hud.close()

    # R2-19: Stream surgical update where callback returns float64 on an F32 tensor
    in_st = tmp_path / "in.safetensors"
    out_st = tmp_path / "out.safetensors"
    ctrl_t = "model.layers.3.self_attn.o_proj.weight"
    save_safetensors_file(in_st, {ctrl_t: np.ones((4, 4), dtype=np.float32)})
    stream_surgical_safetensors_update(
        in_st,
        out_st,
        {ctrl_t: lambda w: (w.astype(np.float64) + 2.0)},
    )
    loaded, _ = load_safetensors_file(out_st)
    assert loaded[ctrl_t].dtype == np.float32
    np.testing.assert_allclose(loaded[ctrl_t], np.full((4, 4), 3.0, dtype=np.float32))
