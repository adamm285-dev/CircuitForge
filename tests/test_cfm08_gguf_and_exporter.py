"""
Unit and integration tests for Milestone CFM-08 (`circuitforge/gguf.py` & `circuitforge/exporter.py`):
- Zero-dependency GGUF v3 binary writer, header inspector, metadata KV parser, and tensor slicer
- Control vs. Frozen Routed Expert classification on GGUF `blk.{L}.*` tensor names
- Hugging Face `README.md` Model Card rendering and full Forged Checkpoint Bundle export.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from circuitforge.exporter import export_forged_bundle, export_model_card
from circuitforge.gguf import inspect_gguf_header, load_gguf_tensor_f32, write_gguf_file


def test_gguf_v3_write_inspect_and_slice_roundtrip(tmp_path: Path) -> None:
    rng = np.random.default_rng(7)
    w_ctrl = rng.standard_normal((16, 16)).astype(np.float32)
    w_exp = rng.standard_normal((16, 16)).astype(np.float16)

    gguf_path = tmp_path / "test_model.gguf"
    write_gguf_file(
        gguf_path,
        tensors={
            "blk.15.attn_output.weight": w_ctrl,
            "blk.15.ffn_down_exps.weight": w_exp,
        },
        metadata={
            "general.architecture": "qwen3_5_moe",
            "general.name": "Qwen3.6-Test",
            "qwen3_5_moe.block_count": 40,
            "circuitforge.steered": True,
        },
    )

    info = inspect_gguf_header(gguf_path)
    assert info["version"] == 3
    assert info["tensor_count"] == 2
    assert info["control_tensor_count"] == 1
    assert info["routed_expert_tensor_count"] == 1
    assert info["metadata"]["general.architecture"] == "qwen3_5_moe"
    assert info["metadata"]["qwen3_5_moe.block_count"] == 40
    assert info["metadata"]["circuitforge.steered"] is True

    # Slice both F32 and F16 tensors back from disk
    sliced_ctrl = load_gguf_tensor_f32(gguf_path, "blk.15.attn_output.weight")
    sliced_exp = load_gguf_tensor_f32(gguf_path, "blk.15.ffn_down_exps.weight")
    np.testing.assert_allclose(sliced_ctrl, w_ctrl, atol=1e-6)
    np.testing.assert_allclose(sliced_exp, w_exp.astype(np.float32), atol=1e-3)


def test_export_model_card_and_forged_bundle(tmp_path: Path) -> None:
    bundle_dir = tmp_path / "forged_bundle"
    res = export_forged_bundle(
        output_dir=bundle_dir,
        model_id="Qwen3.6-35B-A3B-TestForge",
        base_profile="Qwen3.6-35B-A3B",
        alpha=2.2,
        d_model=16,
    )
    assert Path(res["safetensors_path"]).exists()
    assert Path(res["gguf_path"]).exists()
    assert Path(res["recipe_path"]).exists()
    assert Path(res["model_card_path"]).exists()
    assert res["max_nullspace_leakage"] < 1e-6

    card_text = Path(res["model_card_path"]).read_text(encoding="utf-8")
    assert "base_model: Qwen3.6-35B-A3B" in card_text
    assert "library_name: circuitforge" in card_text
    assert "## 2. Surgical Intervention Ledger" in card_text

    # Verify the GGUF control tensor matches the forged Safetensors tensor (not the unsteered base)
    from circuitforge.io import load_safetensors_file

    st_tensors, _ = load_safetensors_file(res["safetensors_path"])
    gguf_ctrl = load_gguf_tensor_f32(res["gguf_path"], "blk.11.attn_output.weight")
    np.testing.assert_allclose(gguf_ctrl, st_tensors["model.layers.11.self_attn.o_proj.weight"], atol=1e-6)


def test_cli_verify_compiler_inspect_gguf_and_export_card(tmp_path: Path) -> None:
    # 1. verify-compiler CLI
    proc_comp = subprocess.run(
        [sys.executable, "-m", "circuitforge", "verify-compiler"],
        capture_output=True,
        text=True,
        check=True,
    )
    comp_json = json.loads(proc_comp.stdout)
    assert comp_json["verified"] is True
    assert comp_json["accuracy"] == 1.0

    # 2. export-card --bundle CLI
    bdir = tmp_path / "cli_bundle"
    proc_card = subprocess.run(
        [
            sys.executable,
            "-m",
            "circuitforge",
            "export-card",
            "--bundle",
            "--out",
            str(bdir),
            "--model-id",
            "Qwen2.5-Coder-7B-Forged",
            "--profile",
            "Qwen2.5-Coder-7B",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    card_json = json.loads(proc_card.stdout)
    gguf_file = card_json["gguf_path"]

    # 3. inspect-gguf CLI
    proc_gguf = subprocess.run(
        [sys.executable, "-m", "circuitforge", "inspect-gguf", gguf_file],
        capture_output=True,
        text=True,
        check=True,
    )
    gguf_json = json.loads(proc_gguf.stdout)
    assert gguf_json["version"] == 3
    assert gguf_json["control_tensor_count"] >= 1
    assert gguf_json["routed_expert_tensor_count"] == 1
