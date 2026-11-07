"""
Test Suite for Milestones CFM-13 and CFM-14 (`tests/test_cfm13_and_cfm14.py`).

Covers:
1. **CFM-14 (`Q4_K` and `Q6_K` 256-Element Super-Block GGUF Quantize / Dequantize & Surgery)**:
   - `quantize_q4_k_bytes` / `dequantize_q4_k_bytes` (`144 bytes/block` for 256 elements) fidelity and idempotence.
   - `quantize_q6_k_bytes` / `dequantize_q6_k_bytes` (`210 bytes/block` for 256 elements) high-precision round-trip.
   - `write_gguf_file`, `load_gguf_tensor_f32`, and `stream_surgical_gguf_update` directly editing a `Q4_K_M` / `Q6_K` GGUF checkpoint while keeping quantized routed experts 100% byte-identical.
2. **CFM-13 (Multi-Layer Checkpoint & Live PyTorch Hook Activation Harvester)**:
   - `harvest_checkpoint_activations` across `.safetensors`, sharded directories, and `.gguf` (`Q4_K`/`Q6_K`) checkpoints, verifying sequential multi-layer residual propagation from layer `0..L`.
   - `harvest_pytorch_module_activations` attaching live `register_forward_hook` callbacks, extracting last-token pre-projection keys and post-projection outputs, and cleaning up hooks on exit.
   - `export_harvested_activations` writing `.safetensors` activation bundles that plug directly into `execute_forge_recipe` and `discover_optimal_surgical_circuit` (and via `circuitforge harvest-activations` CLI).
"""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Callable

import numpy as np
import pytest

from circuitforge.attribution import discover_optimal_surgical_circuit
from circuitforge.gguf import (
    dequantize_q4_k_bytes,
    dequantize_q6_k_bytes,
    inspect_gguf_header,
    load_gguf_tensor_f32,
    quantize_q4_k_bytes,
    quantize_q6_k_bytes,
    stream_surgical_gguf_update,
    write_gguf_file,
)
from circuitforge.harvester import (
    export_harvested_activations,
    harvest_checkpoint_activations,
    harvest_pytorch_module_activations,
)
from circuitforge.io import load_safetensors_file, save_safetensors_file
from circuitforge.recipes import (
    ContrastivePair,
    ForgeRecipe,
    SurgeryStepSpec,
    execute_forge_recipe,
)
from circuitforge.surgery import apply_nullspace_rank1_surgery


def test_cfm14_q4_k_and_q6_k_superblock_quantize_and_gguf_surgery(tmp_path: Path) -> None:
    """
    Verifies CFM-14:
    - 256-element super-block `Q4_K` (`144 bytes/block`) and `Q6_K` (`210 bytes/block`) roundtrips.
    - Innermost dimension (`shape[-1] % 256 == 0`) validation.
    - Direct `stream_surgical_gguf_update` on a `Q4_K` and `Q6_K` GGUF checkpoint.
    """
    rng = np.random.default_rng(2026)
    w_orig = (rng.standard_normal((4, 256)) * 1.5).astype(np.float32)

    # 1. Q4_K round-trip & idempotence
    q4k_bytes = quantize_q4_k_bytes(w_orig)
    assert len(q4k_bytes) == 4 * 144
    w_q4k = dequantize_q4_k_bytes(q4k_bytes, (4, 256))
    assert w_q4k.shape == (4, 256)
    cos_q4k = float(np.sum(w_orig * w_q4k) / (np.linalg.norm(w_orig) * np.linalg.norm(w_q4k)))
    assert cos_q4k > 0.985
    # Re-quantizing an already quantized Q4_K tensor must be 100% byte-identical
    q4k_bytes_2 = quantize_q4_k_bytes(w_q4k)
    assert q4k_bytes == q4k_bytes_2

    # 2. Q6_K round-trip & high precision
    q6k_bytes = quantize_q6_k_bytes(w_orig)
    assert len(q6k_bytes) == 4 * 210
    w_q6k = dequantize_q6_k_bytes(q6k_bytes, (4, 256))
    assert w_q6k.shape == (4, 256)
    cos_q6k = float(np.sum(w_orig * w_q6k) / (np.linalg.norm(w_orig) * np.linalg.norm(w_q6k)))
    assert cos_q6k > 0.998
    q6k_bytes_2 = quantize_q6_k_bytes(w_q6k)
    assert q6k_bytes == q6k_bytes_2

    # 3. Innermost dimension validation (shape[-1] must be divisible by 256)
    with pytest.raises(ValueError, match="256"):
        quantize_q4_k_bytes(np.zeros((2, 128), dtype=np.float32))
    with pytest.raises(ValueError, match="256"):
        quantize_q6_k_bytes(np.zeros((2, 128), dtype=np.float32))

    # 4. End-to-end GGUF write, slice, and streaming surgical update on Q4_K and Q6_K tensors
    gguf_in = tmp_path / "model_kquant.gguf"
    gguf_out = tmp_path / "model_kquant_forged.gguf"
    w_attn_q4k = (rng.standard_normal((8, 256)) * 0.5).astype(np.float32)
    w_shexp_q6k = (rng.standard_normal((8, 256)) * 0.5).astype(np.float32)
    w_routed_q4k = (rng.standard_normal((8, 256)) * 0.5).astype(np.float32)

    write_gguf_file(
        gguf_in,
        {
            "blk.3.attn_output.weight": w_attn_q4k,
            "blk.3.ffn_down_shexp.weight": w_shexp_q6k,
            "blk.3.ffn_down_exps.weight": w_routed_q4k,
        },
        metadata={"general.architecture": "qwen3moe"},
        tensor_types={
            "blk.3.attn_output.weight": "Q4_K",
            "blk.3.ffn_down_shexp.weight": "Q6_K",
            "blk.3.ffn_down_exps.weight": "Q4_K",
        },
    )

    k_trig = np.zeros(256, dtype=np.float32)
    k_trig[0] = 1.0
    v_steer = np.ones(8, dtype=np.float32)

    edited = stream_surgical_gguf_update(
        gguf_in,
        gguf_out,
        {
            "blk.3.attn_output.weight": lambda w: apply_nullspace_rank1_surgery(
                w, k_trig, v_steer, alpha=2.5, tensor_name="blk.3.attn_output.weight"
            )[0],
            "blk.3.ffn_down_shexp.weight": lambda w: apply_nullspace_rank1_surgery(
                w, k_trig, v_steer, alpha=2.0, tensor_name="blk.3.ffn_down_shexp.weight"
            )[0],
        },
    )
    assert edited == ["blk.3.attn_output.weight", "blk.3.ffn_down_shexp.weight"]

    info_out = inspect_gguf_header(gguf_out)
    assert info_out["tensors"]["blk.3.attn_output.weight"].ggml_type_name == "Q4_K"
    assert info_out["tensors"]["blk.3.ffn_down_shexp.weight"].ggml_type_name == "Q6_K"

    # Untouched routed expert Q4_K tensor must remain 100% identical
    routed_before = load_gguf_tensor_f32(gguf_in, "blk.3.ffn_down_exps.weight")
    routed_after = load_gguf_tensor_f32(gguf_out, "blk.3.ffn_down_exps.weight")
    np.testing.assert_array_equal(routed_before, routed_after)

    # Edited Q4_K and Q6_K control tensors must reflect the surgical shift along k_trig
    attn_before = load_gguf_tensor_f32(gguf_in, "blk.3.attn_output.weight")
    attn_after = load_gguf_tensor_f32(gguf_out, "blk.3.attn_output.weight")
    assert float(np.linalg.norm((attn_after - attn_before) @ k_trig)) > 1.5


def test_cfm13_harvester_checkpoint_pytorch_hooks_and_recipe_integration(tmp_path: Path) -> None:
    """
    Verifies CFM-13:
    1. Multi-layer sequential residual forward harvesting from `.safetensors` and `.gguf` checkpoints.
    2. Live `harvest_pytorch_module_activations` hook registration, activation capture, and hook removal.
    3. End-to-end integration of `export_harvested_activations` into `execute_forge_recipe`,
       `discover_optimal_surgical_circuit`, and the `circuitforge harvest-activations` CLI.
    """
    rng = np.random.default_rng(77)
    d_model = 32
    ckpt_path = tmp_path / "multilayer_model.safetensors"
    tensors = {
        "model.embed_tokens.weight": (rng.standard_normal((128, d_model)) * 0.1).astype(np.float32),
        "model.layers.0.input_layernorm.weight": np.ones(d_model, dtype=np.float32),
        "model.layers.0.self_attn.v_proj.weight": (rng.standard_normal((d_model, d_model)) * 0.1).astype(np.float32),
        "model.layers.0.self_attn.o_proj.weight": (rng.standard_normal((d_model, d_model)) * 0.1).astype(np.float32),
        "model.layers.1.input_layernorm.weight": np.ones(d_model, dtype=np.float32),
        "model.layers.1.self_attn.v_proj.weight": (rng.standard_normal((d_model, d_model)) * 0.1).astype(np.float32),
        "model.layers.1.self_attn.o_proj.weight": (rng.standard_normal((d_model, d_model)) * 0.15).astype(np.float32),
    }
    save_safetensors_file(ckpt_path, tensors)

    pairs = [
        ContrastivePair(
            id="p1",
            prompt="Record new sprint requirements in state.md first",
            positive="edit_file state.md",
            negative="edit_file main.py",
        ),
        ContrastivePair(
            id="p2",
            prompt="Check topology boundaries before adding imports",
            positive="sdcs verify --topology",
            negative="git commit --no-verify",
        ),
        ContrastivePair(
            id="p3",
            prompt="Keep state.md under token budget",
            positive="sdcs verify --state",
            negative="append 2000 tokens to state.md",
        ),
    ]

    harvested = harvest_checkpoint_activations(
        checkpoint_path=ckpt_path,
        pairs=pairs,
        layers=[0, 1],
        target_component="self_attn.o_proj.weight",
    )
    assert set(harvested.keys()) == {0, 1}
    assert harvested[1].pos_acts.shape == (3, d_model)
    assert harvested[1].trigger_keys.shape == (3, d_model)
    assert harvested[1].preserved_keys.shape == (4, d_model)

    # Export to .safetensors and feed directly into execute_forge_recipe
    act_file = tmp_path / "harvested_acts.safetensors"
    exp_summary = export_harvested_activations(act_file, harvested)
    assert exp_summary["layers"] == [0, 1]
    assert act_file.is_file()

    recipe = ForgeRecipe(
        recipe_name="harvested_nullspace_surgery",
        profile="Qwen3.6-35B-A3B",
        description="Surgery driven by multi-layer harvested activations",
        preserve_nullspace=True,
        nullspace_num_keys=2,
        seed=42,
        steps=[
            SurgeryStepSpec(
                name="steer_with_harvested",
                method="nullspace_rank1",
                target_component="self_attn.o_proj.weight",
                layers=[0, 1],
                alpha=1.8,
                activations_file=str(act_file),
            )
        ],
    )
    forged_ckpt = tmp_path / "forged_from_harvest.safetensors"
    res_exec = execute_forge_recipe(recipe, checkpoint_path=ckpt_path, output_path=forged_ckpt)
    assert res_exec["edited_tensors_count"] == 2
    assert res_exec["max_nullspace_leakage"] < 1e-5

    # Feed harvested preserved_keys into CFM-12 discover_optimal_surgical_circuit
    disc = discover_optimal_surgical_circuit(
        layer_weights={
            0: tensors["model.layers.0.self_attn.o_proj.weight"],
            1: tensors["model.layers.1.self_attn.o_proj.weight"],
        },
        pairs=pairs,
        preserved_keys_by_layer={
            0: harvested[0].preserved_keys[:2],
            1: harvested[1].preserved_keys[:2],
        },
        top_k_layers=1,
    )
    assert len(disc["selected_top_k_layers"]) == 1

    # 2. Test live PyTorch-compatible forward hook harvesting (with hook cleanup verification)
    class _MockHookHandle:
        def __init__(self, hooks_list: list[Callable[..., None]], fn: Callable[..., None]) -> None:
            self._list = hooks_list
            self._fn = fn

        def remove(self) -> None:
            if self._fn in self._list:
                self._list.remove(self._fn)

    class _MockLinearModule:
        def __init__(self, weight: np.ndarray) -> None:
            self.weight = weight
            self.hooks: list[Callable[..., None]] = []

        def register_forward_hook(self, fn: Callable[..., None]) -> _MockHookHandle:
            self.hooks.append(fn)
            return _MockHookHandle(self.hooks, fn)

        def forward(self, x_seq: np.ndarray) -> np.ndarray:
            out = x_seq @ self.weight.T
            for h in list(self.hooks):
                h(self, (x_seq,), out)
            return out

    mod_0 = _MockLinearModule(tensors["model.layers.0.self_attn.o_proj.weight"])
    mod_1 = _MockLinearModule(tensors["model.layers.1.self_attn.o_proj.weight"])

    def _run_live_model(_m: Any, text: str) -> np.ndarray:
        # Simulate sequence of 3 token hidden states (seq_len=3, d_model=32)
        seed_val = sum(ord(c) for c in text)
        local_rng = np.random.default_rng(seed_val)
        x = local_rng.standard_normal((3, d_model)).astype(np.float32)
        h0 = mod_0.forward(x)
        return mod_1.forward(x + 0.2 * h0)

    pt_harvested = harvest_pytorch_module_activations(
        model=None,
        forward_runner=_run_live_model,
        module_map={0: mod_0, 1: mod_1},
        pairs=pairs,
    )
    assert len(mod_0.hooks) == 0 and len(mod_1.hooks) == 0  # Hooks cleanly removed in finally block
    assert pt_harvested[0].pos_acts.shape == (3, d_model)
    assert pt_harvested[1].trigger_keys.shape == (3, d_model)

    # 3. Verify `circuitforge harvest-activations` CLI command
    cli_act_out = tmp_path / "cli_harvested.safetensors"
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "circuitforge",
            "harvest-activations",
            str(ckpt_path),
            "--layers",
            "0,1",
            "--out",
            str(cli_act_out),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    cli_json = json.loads(proc.stdout)
    assert cli_json["layers"] == [0, 1]
    loaded_cli_acts, loaded_meta = load_safetensors_file(cli_act_out)
    assert "layer_0.pos_acts" in loaded_cli_acts
    assert "layer_1.preserved_keys" in loaded_cli_acts
    assert loaded_meta["format"] == "circuitforge.activations.v1"
