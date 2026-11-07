"""
Unit and integration tests for Milestone CFM-07 (`circuitforge/compiler.py`):
- RASP / Tracr-style symbolic predicate compilation (`RuleSpec`, `compile_constraint_gate`)
- Exact 2-layer MLP/Attention weight synthesis (`50k..100k` parameters) with 0 gradient steps
- Classification across all 4 harness gates (`PASS_THROUGH`, `SPEC_LOCK_REQUIRED`,
  `ZERO_READ_VIOLATION`, `BUDGET_EXCEEDED`), `.safetensors` export, and CLI `verify-compiler`.
"""

from __future__ import annotations

from pathlib import Path

from circuitforge.compiler import (
    RuleSpec,
    compile_constraint_gate,
    evaluate_compiler_benchmark,
)
from circuitforge.io import inspect_safetensors_header, load_safetensors_file


def test_compile_constraint_gate_parameter_band_and_classification() -> None:
    circuit = compile_constraint_gate(d_model=128, d_hidden=128, vocab_capacity=200)
    assert 50_000 <= circuit.num_parameters <= 100_000

    # 1. Baseline pass-through
    r_pass = circuit.predict_gate("Edit billing.py per existing TASK.md", token_count_override=50)
    assert r_pass["gate"] == "PASS_THROUGH"

    # 2. Mid-session requirement without state.md lock -> SPEC_LOCK_REQUIRED
    r_lock = circuit.predict_gate(
        "Mid-session requirement: add parameter max_retries with default 3 and formula x >= 0",
        token_count_override=65,
    )
    assert r_lock["gate"] == "SPEC_LOCK_REQUIRED"

    # 3. Veto token `state_locked` suppresses SPEC_LOCK_REQUIRED -> PASS_THROUGH
    r_veto = circuit.predict_gate(
        "Mid-session requirement: add parameter max_retries already state_locked",
        token_count_override=65,
    )
    assert r_veto["gate"] == "PASS_THROUGH"

    # 4. Zero-read violation when pre_hydrated is present and read_file is called
    r_zr = circuit.predict_gate(
        "Prompt has # [PRE-HYDRATED SOURCE FILES] and agent calls read_file",
        token_count_override=80,
    )
    assert r_zr["gate"] == "ZERO_READ_VIOLATION"

    # 5. Calling read_file WITHOUT pre_hydrated is allowed -> PASS_THROUGH
    r_read_ok = circuit.predict_gate(
        "Agent calls read_file on unhydrated config file",
        token_count_override=80,
    )
    assert r_read_ok["gate"] == "PASS_THROUGH"

    # 6. Token budget exceeded (>= 350 tokens) -> BUDGET_EXCEEDED
    r_budget = circuit.predict_gate("Normal edit_file step", token_count_override=420)
    assert r_budget["gate"] == "BUDGET_EXCEEDED"


def test_custom_symbolic_rules_and_safetensors_export(tmp_path: Path) -> None:
    custom_rules = [
        RuleSpec(
            name="forbid_force_push",
            target_gate="ZERO_READ_VIOLATION",
            all_of_tokens=("git", "push", "force"),
            none_of_tokens=("override_approved",),
            priority=4.0,
        ),
    ]
    circuit = compile_constraint_gate(
        rules=custom_rules,
        d_model=96,
        d_hidden=128,
        vocab_capacity=256,
    )
    assert 50_000 <= circuit.num_parameters <= 100_000

    res_block = circuit.predict_gate("git push --force origin main")
    assert res_block["gate"] == "ZERO_READ_VIOLATION"

    res_partial = circuit.predict_gate("git push origin main")
    assert res_partial["gate"] == "PASS_THROUGH"

    res_approved = circuit.predict_gate("git push --force origin main override_approved")
    assert res_approved["gate"] == "PASS_THROUGH"

    out_st = tmp_path / "micro_circuit.safetensors"
    saved = circuit.export_safetensors(out_st)
    assert saved.exists()

    entries, meta, _ = inspect_safetensors_header(saved)
    assert meta["format"] == "circuitforge_compiled_micro_circuit_v1"
    assert "micro_circuit.mlp.fc1.weight" in entries
    loaded_tensors, _ = load_safetensors_file(saved)
    assert loaded_tensors["micro_circuit.embed.weight"].shape == (256, 96)


def test_evaluate_compiler_benchmark() -> None:
    bench = evaluate_compiler_benchmark()
    assert bench["verified"] is True
    assert bench["accuracy"] == 1.0
    assert bench["within_50k_to_100k_band"] is True
