"""
CircuitForge Declarative Recipe Engine & Contrastive Dataset Harness (`circuitforge/recipes.py`).

Enables open-source users to define repeatable weight surgeries in declarative JSON (`.forge.json`)
or YAML-subset files, load contrastive prompt/response pairs from `.jsonl` / `.json`, and execute
verified null-space or directional ablation surgeries across `.safetensors` checkpoints.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from circuitforge.io import (
    inspect_safetensors_header,
    load_safetensor_slice,
    save_safetensors_file,
)
from circuitforge.surgery import (
    SurgicalEditReport,
    apply_directional_ablation,
    apply_nullspace_rank1_surgery,
    apply_nullspace_rank_k_surgery,
    compute_contrastive_direction,
)
from circuitforge.wiring import is_routed_expert_tensor, load_wiring_map


@dataclass(frozen=True)
class ContrastivePair:
    """A single positive (+X) vs. negative (-X) behavioral scenario pair."""

    id: str
    prompt: str
    positive: str
    negative: str
    category: str = "general"


@dataclass
class SurgeryStepSpec:
    """Specification for a single surgical operation inside a ForgeRecipe."""

    name: str
    method: str  # "nullspace_rank1" | "nullspace_rank_k" | "directional_ablation"
    target_component: str = "self_attn.o_proj.weight"
    layers: list[int] = field(default_factory=list)
    alpha: float = 1.5
    rank: int = 1
    schedule: str = "uniform"  # "uniform" | "bell"
    contrastive_file: str | None = None


@dataclass
class ForgeRecipe:
    """Declarative specification for a CircuitForge model weight surgery."""

    recipe_name: str
    profile: str
    steps: list[SurgeryStepSpec]
    description: str = ""
    preserve_nullspace: bool = True
    nullspace_num_keys: int = 16
    seed: int = 42

    def to_dict(self) -> dict[str, Any]:
        return {
            "recipe_name": self.recipe_name,
            "profile": self.profile,
            "description": self.description,
            "preserve_nullspace": self.preserve_nullspace,
            "nullspace_num_keys": self.nullspace_num_keys,
            "seed": self.seed,
            "steps": [
                {
                    "name": s.name,
                    "method": s.method,
                    "target_component": s.target_component,
                    "layers": s.layers,
                    "alpha": s.alpha,
                    "rank": s.rank,
                    "schedule": s.schedule,
                    "contrastive_file": s.contrastive_file,
                }
                for s in self.steps
            ],
        }


def load_contrastive_pairs(path: str | Path) -> list[ContrastivePair]:
    """
    Loads contrastive behavioral pairs from a `.jsonl` or `.json` file.
    Each entry must have `prompt`, `positive`, and `negative` fields.
    """
    p = Path(path)
    raw_text = p.read_text(encoding="utf-8").strip()
    if not raw_text:
        return []

    records: list[dict[str, Any]]
    if p.suffix.lower() == ".jsonl" or raw_text.startswith("{"):
        if raw_text.startswith("["):
            records = json.loads(raw_text)
        else:
            records = [
                json.loads(line)
                for line in raw_text.splitlines()
                if line.strip() and not line.strip().startswith("#")
            ]
    else:
        records = json.loads(raw_text)

    pairs: list[ContrastivePair] = []
    for idx, item in enumerate(records):
        for req_key in ("prompt", "positive", "negative"):
            if req_key not in item:
                raise ValueError(f"Contrastive record #{idx} in {p} missing required field '{req_key}'")
        pairs.append(
            ContrastivePair(
                id=str(item.get("id", f"pair_{idx:03d}")),
                prompt=str(item["prompt"]),
                positive=str(item["positive"]),
                negative=str(item["negative"]),
                category=str(item.get("category", "general")),
            )
        )
    return pairs


def embed_contrastive_pairs_deterministic(
    pairs: Sequence[ContrastivePair] | list[ContrastivePair],
    d_out: int,
    d_in: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Deterministically projects text contrastive pairs into synthetic `(pos_acts, neg_acts, trigger_keys)`
    matrices of shapes `(N, d_out)`, `(N, d_out)`, and `(N, d_in)` using SHA-256 seeded orthogonal basis
    projections when running in offline/NumPy verification mode without a live PyTorch forward hook.
    """
    # MOCK(M-001): SHA-256 seeded synthetic activation embeddings when running without a live PyTorch model forward hook.
    if not pairs:
        raise ValueError("Cannot embed empty contrastive pair list.")

    pos_list: list[np.ndarray] = []
    neg_list: list[np.ndarray] = []
    key_list: list[np.ndarray] = []

    for pair in pairs:
        p_seed = int.from_bytes(
            hashlib.sha256(f"{pair.prompt}||{pair.positive}".encode("utf-8")).digest()[:8], "little"
        )
        n_seed = int.from_bytes(
            hashlib.sha256(f"{pair.prompt}||{pair.negative}".encode("utf-8")).digest()[:8], "little"
        )
        k_seed = int.from_bytes(
            hashlib.sha256(pair.prompt.encode("utf-8")).digest()[:8], "little"
        )

        pos_rng = np.random.default_rng(p_seed)
        neg_rng = np.random.default_rng(n_seed)
        key_rng = np.random.default_rng(k_seed)

        pos_vec = pos_rng.standard_normal(d_out).astype(np.float32) + 0.75
        neg_vec = neg_rng.standard_normal(d_out).astype(np.float32) - 0.75
        key_vec = key_rng.standard_normal(d_in).astype(np.float32)
        key_vec /= max(float(np.linalg.norm(key_vec)), 1e-8)

        pos_list.append(pos_vec)
        neg_list.append(neg_vec)
        key_list.append(key_vec)

    return np.stack(pos_list), np.stack(neg_list), np.stack(key_list)


def load_forge_recipe(path: str | Path) -> ForgeRecipe:
    """Loads a declarative `ForgeRecipe` from a `.json` file."""
    p = Path(path)
    data = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or "steps" not in data:
        raise ValueError(f"Invalid ForgeRecipe file {p}: missing 'steps' list.")

    profile = str(data.get("profile", "Qwen3.6-35B-A3B"))
    wmap = load_wiring_map(profile_name=profile)

    steps: list[SurgeryStepSpec] = []
    for raw_step in data["steps"]:
        layers = raw_step.get("layers")
        if not layers:
            layers = list(wmap.mid_to_late_planning_anchors)
        steps.append(
            SurgeryStepSpec(
                name=str(raw_step.get("name", "unnamed_step")),
                method=str(raw_step.get("method", "nullspace_rank1")),
                target_component=str(raw_step.get("target_component", "self_attn.o_proj.weight")),
                layers=[int(x) for x in layers],
                alpha=float(raw_step.get("alpha", 1.5)),
                rank=int(raw_step.get("rank", 1)),
                schedule=str(raw_step.get("schedule", "uniform")),
                contrastive_file=raw_step.get("contrastive_file"),
            )
        )

    return ForgeRecipe(
        recipe_name=str(data.get("recipe_name", p.stem)),
        profile=profile,
        description=str(data.get("description", "")),
        preserve_nullspace=bool(data.get("preserve_nullspace", True)),
        nullspace_num_keys=int(data.get("nullspace_num_keys", 16)),
        seed=int(data.get("seed", 42)),
        steps=steps,
    )


def execute_forge_recipe(
    recipe: ForgeRecipe,
    checkpoint_path: str | Path | None = None,
    output_path: str | Path | None = None,
    base_dir: str | Path | None = None,
) -> dict[str, Any]:
    """
    Executes a `ForgeRecipe` against a `.safetensors` checkpoint (or synthesizes a reference
    control checkpoint if `checkpoint_path` is None for dry-run verification), applies all
    surgical steps while strictly enforcing routed-expert freeze gates, and writes the forged
    `.safetensors` file and audit report.
    """
    rng = np.random.default_rng(recipe.seed)
    resolve_root = Path(base_dir) if base_dir is not None else Path.cwd()

    tensors: dict[str, np.ndarray] = {}
    metadata: dict[str, str] = {}
    preserve_dtypes: dict[str, str] = {}

    if checkpoint_path is not None:
        cp = Path(checkpoint_path)
        entries, metadata, _ = inspect_safetensors_header(cp)
        for name, entry in entries.items():
            preserve_dtypes[name] = entry.dtype
            tensors[name] = load_safetensor_slice(cp, name)
    else:
        # Synthesize compact reference weights for the recipe's target layers
        d_model = 64
        for step in recipe.steps:
            for l_idx in step.layers:
                tname = f"model.layers.{l_idx}.{step.target_component}"
                if tname not in tensors:
                    tensors[tname] = (rng.standard_normal((d_model, d_model)) * 0.02).astype(np.float32)

    all_reports: list[SurgicalEditReport] = []

    for step in recipe.steps:
        n_layers = max(len(step.layers), 1)
        if step.schedule == "bell" and n_layers > 1:
            idx = np.arange(n_layers, dtype=np.float64)
            raw = np.sin(np.pi * (idx + 1.0) / (n_layers + 1.0)) ** 2
            alpha_mults = raw / np.mean(raw)
        else:
            alpha_mults = np.ones(n_layers, dtype=np.float64)

        pairs: list[ContrastivePair] = []
        if step.contrastive_file:
            cpath = Path(step.contrastive_file)
            if not cpath.is_absolute():
                cpath = resolve_root / cpath
            if not cpath.is_file():
                raise FileNotFoundError(
                    f"Contrastive pairs file '{step.contrastive_file}' (resolved to '{cpath}') does not exist."
                )
            pairs = load_contrastive_pairs(cpath)

        for i, l_idx in enumerate(step.layers):
            tname = f"model.layers.{l_idx}.{step.target_component}"
            if is_routed_expert_tensor(tname):
                raise PermissionError(
                    f"CircuitForge Safety Gate: Recipe step '{step.name}' attempted to edit routed expert '{tname}'."
                )
            if tname not in tensors:
                raise KeyError(
                    f"Recipe step '{step.name}' target tensor '{tname}' not found in checkpoint."
                )

            W = tensors[tname]
            d_out, d_in = W.shape
            layer_alpha = float(step.alpha * alpha_mults[i])

            if pairs:
                pos_acts, neg_acts, trig_keys = embed_contrastive_pairs_deterministic(pairs, d_out, d_in)
            else:
                # MOCK(M-004): Seeded synthetic contrastive activations and baseline syntax keys when executing a dry-run recipe without a contrastive_file.
                pos_acts = (rng.standard_normal((8, d_out)) + 0.5).astype(np.float32)
                neg_acts = (rng.standard_normal((8, d_out)) - 0.5).astype(np.float32)
                trig_keys = rng.standard_normal((max(step.rank, 4), d_in)).astype(np.float32)

            K0 = None
            if recipe.preserve_nullspace and recipe.nullspace_num_keys > 0:
                k_count = min(recipe.nullspace_num_keys, max(d_in // 2, 1))
                # Extract top-k_count right-singular input directions of W as the baseline syntax subspace K0
                # so null-space surgery preserves the checkpoint's dominant input subspace.
                _, s_w, vt_w = np.linalg.svd(np.asarray(W, dtype=np.float64), full_matrices=False)
                if len(s_w) >= k_count and float(s_w[0]) > 1e-12:
                    K0 = vt_w[:k_count, :].astype(np.float32).copy()
                else:
                    K0 = rng.standard_normal((k_count, d_in)).astype(np.float32)

            if step.method == "nullspace_rank1":
                v_steer = compute_contrastive_direction(pos_acts, neg_acts)
                k_star = trig_keys.mean(axis=0)
                W_new, rep = apply_nullspace_rank1_surgery(
                    weight=W,
                    trigger_key=k_star,
                    steering_vector=v_steer,
                    preserved_keys=K0,
                    alpha=layer_alpha,
                    tensor_name=tname,
                    layer_idx=l_idx,
                )
            elif step.method == "nullspace_rank_k":
                rank = min(step.rank, trig_keys.shape[0], d_in - 1)
                k_batch = trig_keys[:rank]
                # Construct rank distinct steering targets from pair differences
                diffs = pos_acts[:rank] - neg_acts[:rank]
                if diffs.shape[0] < rank:
                    v_base = compute_contrastive_direction(pos_acts, neg_acts)
                    diffs = np.tile(v_base, (rank, 1))
                W_new, rep = apply_nullspace_rank_k_surgery(
                    weight=W,
                    trigger_keys=k_batch,
                    steering_vectors=diffs,
                    preserved_keys=K0,
                    alpha=layer_alpha,
                    tensor_name=tname,
                    layer_idx=l_idx,
                )
            elif step.method == "directional_ablation":
                v_ablate = compute_contrastive_direction(neg_acts, pos_acts)
                W_new, rep = apply_directional_ablation(
                    weight=W,
                    refusal_or_target_dir=v_ablate,
                    strength=min(max(layer_alpha, 0.0), 1.0),
                    tensor_name=tname,
                    layer_idx=l_idx,
                )
            else:
                raise ValueError(f"Unsupported surgical method: {step.method}")

            tensors[tname] = W_new
            all_reports.append(rep)

    if output_path is not None:
        meta_out = dict(metadata)
        meta_out["circuitforge_recipe"] = recipe.recipe_name
        meta_out["circuitforge_profile"] = recipe.profile
        save_safetensors_file(output_path, tensors, metadata=meta_out, preserve_dtypes=preserve_dtypes)

    return {
        "recipe_name": recipe.recipe_name,
        "profile": recipe.profile,
        "edited_tensors_count": len(all_reports),
        "max_nullspace_leakage": max((r.nullspace_leakage_max for r in all_reports), default=0.0),
        "output_path": str(output_path) if output_path else None,
        "reports": [
            {
                "tensor_name": r.tensor_name,
                "layer_idx": r.layer_idx,
                "shape": list(r.shape),
                "alpha": round(r.alpha, 6),
                "frobenius_delta": round(r.frobenius_delta, 6),
                "nullspace_leakage_max": float(f"{r.nullspace_leakage_max:.3e}"),
                "target_alignment_before": round(r.target_alignment_before, 6),
                "target_alignment_after": round(r.target_alignment_after, 6),
            }
            for r in all_reports
        ],
    }
