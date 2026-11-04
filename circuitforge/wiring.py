"""
CircuitForge Wiring & Topology Analyzer for Hybrid MoE Transformers (`Qwen3_5Moe`).

Maps the internal circuit hierarchy of both:
1. `Qwen3.6-35B-A3B` (40 layers, hidden_size=2048, 256 routed experts, top_k=8)
2. `Qwen3.8-2.4T-A95B` (92 layers, hidden_size=8192, 512 routed experts, top_k=10)

Identifies and separates:
- **Full-Attention Planning Anchors** (every 4th layer: `layers 3, 7, 11, ...` with `head_dim=256`, `attn_output_gate=True`)
- **Gated DeltaNet Linear-Attention Layers** (3 out of every 4 layers: `conv_kernel=4`, `16 key heads`, `32/128 value heads`)
- **Shared Experts & Routers** (`mlp.shared_expert`, `mlp.shared_expert_gate`, `mlp.gate`)
- **Protected Routed Experts** (`mlp.experts.0..N`, which hold memorized code syntax and must remain frozen during behavioral surgery)
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class LayerCircuitSpec:
    """Circuit classification for a single transformer layer."""

    layer_idx: int
    attention_type: str  # "full_attention" or "linear_attention"
    is_planning_anchor: bool
    num_routed_experts: int
    top_k: int
    control_tensors: list[str] = field(default_factory=list)
    control_shards: list[str] = field(default_factory=list)


@dataclass
class ModelWiringMap:
    """Complete architectural circuit map and shard isolation plan."""

    model_name: str
    architecture: str
    num_layers: int
    hidden_size: int
    head_dim: int
    full_attention_interval: int
    attn_output_gate: bool
    num_experts: int
    num_experts_per_tok: int
    moe_intermediate_size: int
    shared_expert_intermediate_size: int
    linear_num_key_heads: int
    linear_num_value_heads: int
    linear_key_head_dim: int
    linear_value_head_dim: int
    full_attention_layers: list[int]
    linear_attention_layers: list[int]
    layers: list[LayerCircuitSpec]
    total_shards: int = 0
    control_only_shards: list[str] = field(default_factory=list)
    routed_expert_shards: list[str] = field(default_factory=list)

    @property
    def mid_to_late_planning_anchors(self) -> list[int]:
        """
        Returns the middle-to-late Full-Attention anchor layers (25% to 85% depth)
        where behavioral planning and tool-use routing vectors are concentrated.
        """
        lo = int(self.num_layers * 0.25)
        hi = int(self.num_layers * 0.85)
        anchors = [idx for idx in self.full_attention_layers if lo <= idx <= hi]
        return anchors if anchors else list(self.full_attention_layers)

    def summary_dict(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "architecture": self.architecture,
            "num_layers": self.num_layers,
            "hidden_size": self.hidden_size,
            "head_dim": self.head_dim,
            "full_attention_count": len(self.full_attention_layers),
            "linear_attention_count": len(self.linear_attention_layers),
            "full_attention_layers": self.full_attention_layers,
            "mid_to_late_planning_anchors": self.mid_to_late_planning_anchors,
            "num_experts": self.num_experts,
            "top_k": self.num_experts_per_tok,
            "total_shards": self.total_shards,
            "control_shard_count": len(self.control_only_shards),
            "routed_expert_shard_count": len(self.routed_expert_shards),
        }


# Built-in reference profiles for Qwen3.6-35B-A3B and Qwen3.8-2.4T-A95B
BUILTIN_PROFILES: dict[str, dict[str, Any]] = {
    "Qwen3.6-35B-A3B": {
        "architectures": ["Qwen3_5MoeForConditionalGeneration"],
        "hidden_size": 2048,
        "num_hidden_layers": 40,
        "num_attention_heads": 16,
        "num_key_value_heads": 2,
        "head_dim": 256,
        "full_attention_interval": 4,
        "attn_output_gate": True,
        "num_experts": 256,
        "num_experts_per_tok": 8,
        "moe_intermediate_size": 512,
        "shared_expert_intermediate_size": 512,
        "linear_num_key_heads": 16,
        "linear_num_value_heads": 32,
        "linear_key_head_dim": 128,
        "linear_value_head_dim": 128,
        "layer_types": [
            "full_attention" if (i + 1) % 4 == 0 else "linear_attention"
            for i in range(40)
        ],
    },
    "Qwen3.8-2.4T-A95B": {
        "architectures": ["Qwen3_5MoeForCausalLM"],
        "hidden_size": 8192,
        "num_hidden_layers": 92,
        "num_attention_heads": 64,
        "num_key_value_heads": 4,
        "head_dim": 256,
        "full_attention_interval": 4,
        "attn_output_gate": True,
        "num_experts": 512,
        "num_experts_per_tok": 10,
        "moe_intermediate_size": 2048,
        "shared_expert_intermediate_size": 2048,
        "linear_num_key_heads": 16,
        "linear_num_value_heads": 128,
        "linear_key_head_dim": 128,
        "linear_value_head_dim": 128,
        "layer_types": [
            "full_attention" if (i + 1) % 4 == 0 else "linear_attention"
            for i in range(92)
        ],
    },
}


_LAYER_RE = re.compile(r"(?:model\.layers|blk)\.(\d+)\.")


def is_routed_expert_tensor(tensor_name: str) -> bool:
    """
    Returns True if the tensor belongs to the banked routed MoE experts
    (which hold parametric code syntax and should remain frozen during circuit surgery).
    Supports both HuggingFace Safetensors (`model.layers.L.mlp.experts.*`) and
    GGUF (`blk.L.ffn_{gate,up,down}_exps.weight`) naming conventions.
    """
    if ".mlp.experts." in tensor_name:
        return True
    if "_exps." in tensor_name and "_shexp" not in tensor_name:
        return True
    return False


def is_surgical_control_tensor(tensor_name: str) -> bool:
    """
    Returns True if the tensor belongs to a surgical control circuit:
    - Full-Attention projections (`self_attn.{q,k,v,o}_proj`, `attn_q/k/v/output`)
    - Linear-Attention projections (`linear_attn.{in_proj,out_proj}`, `ssm_*`)
    - Shared Expert projections (`mlp.shared_expert.*`, `ffn_*_shexp`)
    - MoE Router Gates (`mlp.gate.weight`, `ffn_gate_inp.weight`)
    """
    if is_routed_expert_tensor(tensor_name):
        return False
    control_markers = (
        "self_attn.",
        "linear_attn.",
        "mlp.shared_expert",
        "mlp.gate.",
        "attn_q",
        "attn_k",
        "attn_v",
        "attn_output",
        "ffn_gate_shexp",
        "ffn_up_shexp",
        "ffn_down_shexp",
        "ffn_gate_inp",
    )
    return any(m in tensor_name for m in control_markers)


def load_wiring_map(
    config_or_dir: str | Path | None = None,
    profile_name: str = "Qwen3.6-35B-A3B",
) -> ModelWiringMap:
    """
    Loads a `ModelWiringMap` from a model directory (reading `config.json` and optional
    `model.safetensors.index.json`) or falls back to a built-in profile.
    """
    raw_cfg: dict[str, Any]
    index_map: dict[str, str] = {}
    model_label = profile_name

    if config_or_dir is not None:
        p = Path(config_or_dir)
        cfg_file = p / "config.json" if p.is_dir() else p
        if cfg_file.is_file():
            loaded = json.loads(cfg_file.read_text(encoding="utf-8"))
            arch_list = loaded.get("architectures", [])
            raw_cfg = loaded.get("text_config", loaded)
            if arch_list and "architectures" not in raw_cfg:
                raw_cfg["architectures"] = arch_list
            model_label = p.name if p.is_dir() else p.parent.name
            idx_file = cfg_file.parent / "model.safetensors.index.json"
            if idx_file.is_file():
                idx_data = json.loads(idx_file.read_text(encoding="utf-8"))
                index_map = idx_data.get("weight_map", {})
        else:
            raw_cfg = dict(BUILTIN_PROFILES[profile_name])
    else:
        raw_cfg = dict(BUILTIN_PROFILES[profile_name])

    num_layers = int(raw_cfg.get("num_hidden_layers", 40))
    interval = int(raw_cfg.get("full_attention_interval", 4))
    layer_types: list[str] = raw_cfg.get("layer_types") or [
        "full_attention" if (i + 1) % interval == 0 else "linear_attention"
        for i in range(num_layers)
    ]
    num_experts = int(raw_cfg.get("num_experts", 256))
    top_k = int(raw_cfg.get("num_experts_per_tok", 8))

    per_layer_control_tensors: dict[int, list[str]] = {i: [] for i in range(num_layers)}
    per_layer_control_shards: dict[int, set[str]] = {i: set() for i in range(num_layers)}
    control_shards_all: set[str] = set()
    expert_shards_all: set[str] = set()
    all_shards: set[str] = set()

    for tname, shard in index_map.items():
        all_shards.add(shard)
        if is_routed_expert_tensor(tname):
            expert_shards_all.add(shard)
        else:
            control_shards_all.add(shard)
            m = _LAYER_RE.search(tname)
            if m:
                l_idx = int(m.group(1))
                if 0 <= l_idx < num_layers:
                    per_layer_control_tensors[l_idx].append(tname)
                    per_layer_control_shards[l_idx].add(shard)

    full_layers: list[int] = []
    linear_layers: list[int] = []
    layer_specs: list[LayerCircuitSpec] = []

    for i in range(num_layers):
        ltype = layer_types[i] if i < len(layer_types) else "linear_attention"
        is_full = ltype == "full_attention"
        if is_full:
            full_layers.append(i)
        else:
            linear_layers.append(i)
        layer_specs.append(
            LayerCircuitSpec(
                layer_idx=i,
                attention_type=ltype,
                is_planning_anchor=is_full,
                num_routed_experts=num_experts,
                top_k=top_k,
                control_tensors=sorted(per_layer_control_tensors[i]),
                control_shards=sorted(per_layer_control_shards[i]),
            )
        )

    arch_names = raw_cfg.get("architectures", ["Qwen3_5Moe"])
    arch_str = arch_names[0] if arch_names else "Qwen3_5Moe"

    return ModelWiringMap(
        model_name=model_label,
        architecture=arch_str,
        num_layers=num_layers,
        hidden_size=int(raw_cfg.get("hidden_size", 2048)),
        head_dim=int(raw_cfg.get("head_dim", 256)),
        full_attention_interval=interval,
        attn_output_gate=bool(raw_cfg.get("attn_output_gate", True)),
        num_experts=num_experts,
        num_experts_per_tok=top_k,
        moe_intermediate_size=int(raw_cfg.get("moe_intermediate_size", 512)),
        shared_expert_intermediate_size=int(raw_cfg.get("shared_expert_intermediate_size", 512)),
        linear_num_key_heads=int(raw_cfg.get("linear_num_key_heads", 16)),
        linear_num_value_heads=int(raw_cfg.get("linear_num_value_heads", 32)),
        linear_key_head_dim=int(raw_cfg.get("linear_key_head_dim", 128)),
        linear_value_head_dim=int(raw_cfg.get("linear_value_head_dim", 128)),
        full_attention_layers=full_layers,
        linear_attention_layers=linear_layers,
        layers=layer_specs,
        total_shards=len(all_shards),
        control_only_shards=sorted(control_shards_all),
        routed_expert_shards=sorted(expert_shards_all - control_shards_all),
    )
