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
    load_safetensors_file,
    save_safetensors_file,
)
from circuitforge.native import has_native_engine, native_rmsnorm_matvec_f32
from circuitforge.surgery import (
    SurgicalEditReport,
    apply_directional_ablation,
    apply_leace_concept_erasure,
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
    activations_file: str | None = None


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
                    "activations_file": s.activations_file,
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


def _canonicalize_svd(u: np.ndarray, vt: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Enforces deterministic SVD sign orientation across LAPACK implementations (MKL vs OpenBLAS)
    by requiring the largest-magnitude entry in each row of `vt` to be non-negative.
    """
    u_out = u.copy()
    vt_out = vt.copy()
    r = min(u_out.shape[1], vt_out.shape[0])
    for i in range(r):
        row = vt_out[i]
        idx = int(np.argmax(np.abs(row)))
        if row[idx] < 0.0:
            vt_out[i] = -row
            u_out[:, i] = -u_out[:, i]
    return u_out, vt_out


def _encode_text_to_singular_input(
    text: str,
    vt_basis: np.ndarray,
    d_in: int,
    role_tag: str = "neutral",
) -> np.ndarray:
    """
    Projects token n-grams of `text` into the canonical right-singular input subspace `vt_basis`
    of the layer weight matrix $W \\in \\mathbb{R}^{d_{out} \\times d_{in}}$ without any constant
    bias term so the resulting direction is 100% determined by the prompt/response tokens.
    """
    tokens = [t for t in text.lower().replace("_", " ").split() if t]
    if not tokens:
        tokens = [text or "empty"]
    r_dim = vt_basis.shape[0]
    coeffs = np.zeros(r_dim, dtype=np.float64)
    for idx, tok in enumerate(tokens):
        digest = hashlib.sha256(f"{role_tag}:{idx}:{tok}".encode("utf-8")).digest()
        for b_idx in range(16):
            val = (int.from_bytes(digest[b_idx * 2 : b_idx * 2 + 2], "little") / 32767.5) - 1.0
            slot = (idx * 13 + b_idx * 7) % r_dim
            coeffs[slot] += val
    vec = (coeffs @ vt_basis).astype(np.float32)
    norm = max(float(np.linalg.norm(vec)), 1e-8)
    return (vec / norm).astype(np.float32)


def compute_weight_backed_activations(
    pairs: Sequence[ContrastivePair] | list[ContrastivePair],
    weight: np.ndarray,
    rms_gamma: np.ndarray | None = None,
    eps: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes weight-backed forward activations `(pos_acts, neg_acts, trigger_keys)`
    of shapes `(N, d_out)`, `(N, d_out)`, and `(N, d_in)` by projecting each contrastive
    prompt/response pair onto the checkpoint matrix $W$'s canonical right-singular input subspace
    and evaluating the forward layer operator $h = W \\, \\text{RMSNorm}(x, \\gamma)$.
    Uses the C++ AVX2 `cf_rmsnorm_matvec_f32` kernel when loaded.
    """
    if not pairs:
        raise ValueError("Cannot compute forward activations for empty contrastive pair list.")
    W = np.ascontiguousarray(weight, dtype=np.float32)
    if W.ndim != 2:
        raise ValueError(f"Expected 2D weight matrix, got {W.shape}")
    d_out, d_in = W.shape
    gamma = (
        np.ascontiguousarray(rms_gamma, dtype=np.float32).reshape(d_in)
        if rms_gamma is not None
        else np.ones(d_in, dtype=np.float32)
    )

    u_w, s_w, vt_w = np.linalg.svd(W.astype(np.float64), full_matrices=False)
    if float(s_w[0]) <= 1e-12:
        vt_w = np.eye(min(d_out, d_in), d_in, dtype=np.float64)
    else:
        _, vt_w = _canonicalize_svd(u_w, vt_w)

    use_cpp = has_native_engine()
    pos_list: list[np.ndarray] = []
    neg_list: list[np.ndarray] = []
    key_list: list[np.ndarray] = []

    half_r = max(vt_w.shape[0] // 2, 1)
    trig_basis = vt_w[half_r:, :] if vt_w.shape[0] > half_r else vt_w

    for pair in pairs:
        x_pos = _encode_text_to_singular_input(f"{pair.prompt} {pair.positive}", vt_w, d_in, role_tag="pos")
        x_neg = _encode_text_to_singular_input(f"{pair.prompt} {pair.negative}", vt_w, d_in, role_tag="neg")
        x_key = _encode_text_to_singular_input(pair.prompt, trig_basis, d_in, role_tag="key")

        if use_cpp:
            h_pos = native_rmsnorm_matvec_f32(W, x_pos, gamma, eps=eps)
            h_neg = native_rmsnorm_matvec_f32(W, x_neg, gamma, eps=eps)
        else:
            rms_p = float(np.sqrt(np.mean(x_pos ** 2) + eps))
            rms_n = float(np.sqrt(np.mean(x_neg ** 2) + eps))
            h_pos = (W @ ((x_pos / rms_p) * gamma)).astype(np.float32)
            h_neg = (W @ ((x_neg / rms_n) * gamma)).astype(np.float32)

        rms_k = float(np.sqrt(np.mean(x_key ** 2) + eps))
        k_normed = ((x_key / rms_k) * gamma).astype(np.float32)
        k_normed /= max(float(np.linalg.norm(k_normed)), 1e-8)

        pos_list.append(h_pos)
        neg_list.append(h_neg)
        key_list.append(k_normed)

    return np.stack(pos_list), np.stack(neg_list), np.stack(key_list)


def extract_principal_circuit_directions(
    weight: np.ndarray,
    rank: int = 1,
    preserved_rank: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extracts deterministic `(pos_acts, neg_acts, trigger_keys)` directly from the singular
    value spectrum $W = U \\Sigma V^\\top$ of the checkpoint matrix `weight` when no external
    contrastive prompt dataset is provided:
    - `pos_acts` and `neg_acts` align with the principal output singular directions $U$.
    - `trigger_keys` are taken from the right-singular vectors $V$ orthogonal to the top
      `preserved_rank` baseline directions.
    """
    W64 = np.asarray(weight, dtype=np.float64)
    d_out, d_in = W64.shape
    u, s, vt = np.linalg.svd(W64, full_matrices=True)
    n_rows = max(rank, 4)

    # Select trigger keys from right-singular vectors outside the top `preserved_rank` rows of vt
    start_idx = min(preserved_rank, max(d_in - n_rows, 0))
    end_idx = min(start_idx + n_rows, d_in)
    trig_keys = vt[start_idx:end_idx, :].astype(np.float32).copy()
    if trig_keys.shape[0] < n_rows:
        pad = np.eye(n_rows - trig_keys.shape[0], d_in, dtype=np.float32)
        trig_keys = np.vstack([trig_keys, pad])

    out_rows = min(n_rows, d_out)
    u_slice = u[:, :out_rows].T.astype(np.float32)  # (out_rows, d_out)
    if u_slice.shape[0] < n_rows:
        u_slice = np.tile(u_slice, ( (n_rows // u_slice.shape[0]) + 1, 1 ))[:n_rows]

    scale = float(max(s[0], 1.0)) if len(s) > 0 else 1.0
    pos_acts = (u_slice * scale).astype(np.float32)
    neg_acts = (-u_slice * scale).astype(np.float32)
    return pos_acts, neg_acts, trig_keys


def embed_contrastive_pairs_deterministic(
    pairs: Sequence[ContrastivePair] | list[ContrastivePair],
    d_out: int,
    d_in: int,
    weight: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Projects text contrastive pairs into `(pos_acts, neg_acts, trigger_keys)` of shapes
    `(N, d_out)`, `(N, d_out)`, and `(N, d_in)` using `compute_weight_backed_activations`
    over `weight` (or a canonical orthogonal projection matrix when `weight` is omitted).
    """
    if not pairs:
        raise ValueError("Cannot embed empty contrastive pair list.")
    if weight is None:
        eye_w = np.eye(d_out, d_in, dtype=np.float32)
        return compute_weight_backed_activations(pairs, eye_w)
    return compute_weight_backed_activations(pairs, weight)


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
        if layers is None:
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
                activations_file=raw_step.get("activations_file"),
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

            k_count = (
                min(recipe.nullspace_num_keys, max(d_in // 2, 1))
                if (recipe.preserve_nullspace and recipe.nullspace_num_keys > 0)
                else 0
            )

            if step.activations_file:
                apath = Path(step.activations_file)
                if not apath.is_absolute():
                    apath = resolve_root / apath
                if not apath.is_file():
                    raise FileNotFoundError(
                        f"Activations file '{step.activations_file}' (resolved to '{apath}') does not exist."
                    )
                act_tensors, _ = load_safetensors_file(apath)
                pos_key = f"layer_{l_idx}.pos_acts" if f"layer_{l_idx}.pos_acts" in act_tensors else "pos_acts"
                neg_key = f"layer_{l_idx}.neg_acts" if f"layer_{l_idx}.neg_acts" in act_tensors else "neg_acts"
                trig_key = f"layer_{l_idx}.trigger_keys" if f"layer_{l_idx}.trigger_keys" in act_tensors else "trigger_keys"
                pos_acts = np.asarray(act_tensors[pos_key], dtype=np.float32)
                neg_acts = np.asarray(act_tensors[neg_key], dtype=np.float32)
                trig_keys = np.asarray(act_tensors[trig_key], dtype=np.float32)
            elif pairs:
                pos_acts, neg_acts, trig_keys = compute_weight_backed_activations(pairs, W)
            else:
                pos_acts, neg_acts, trig_keys = extract_principal_circuit_directions(
                    W,
                    rank=step.rank,
                    preserved_rank=k_count,
                )

            K0 = None
            if k_count > 0:
                # Extract top-k_count canonical right-singular input directions of W as the baseline syntax subspace K0
                u_w, s_w, vt_w = np.linalg.svd(np.asarray(W, dtype=np.float64), full_matrices=False)
                if len(s_w) >= k_count and float(s_w[0]) > 1e-12:
                    _, vt_w = _canonicalize_svd(u_w, vt_w)
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
            elif step.method == "leace":
                X_both = np.vstack([pos_acts, neg_acts])
                Z_labels = np.concatenate(
                    [
                        np.ones(pos_acts.shape[0], dtype=np.float32),
                        np.zeros(neg_acts.shape[0], dtype=np.float32),
                    ]
                )
                W_new, rep = apply_leace_concept_erasure(
                    weight=W,
                    activations=X_both,
                    concept_labels=Z_labels,
                    preserved_keys=K0,
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
