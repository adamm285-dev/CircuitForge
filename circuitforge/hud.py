"""
CircuitForge Real-Time Mechanistic Telemetry HUD & Visual Debugger (`circuitforge/hud.py`).

Implements Milestone CFM-04:
1. **Color-Coded Activation Heatmaps**:
   Real-time matrix of model layers versus tokens displaying activation magnitudes
   (cool blues for baseline, yellow/orange for elevated, bright crimson for magnitude spikes).
2. **Live MoE Routing & Entropy Equalizer**:
   Visual Sankey / load distribution bars per routed expert + shared expert, with a live
   Shannon entropy gauge ($\\mathcal{H}_{\\text{router}}$) powered by the C++ SIMD engine.
3. **Interactive Click-to-Steer Controls**:
   Web-based sliders (`alpha`, `schedule`, `target_layers`) that dynamically adjust steering
   vector multipliers via `POST /api/steer` and immediately recompute both internal layer
   activation heatmaps and cross-layer alignment angles.
4. **Cross-Layer Representation Alignment Inspector**:
   Computes cosine similarity ($\\cos \\theta$) and subspace angles ($\\theta^\\circ$) between
   layer representations and the target contrastive steering vector across transformer blocks.
"""

from __future__ import annotations

import json
import math
import threading
import time
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse, urlsplit

import numpy as np

from circuitforge.native import (
    NativeTelemetryRingBuffer,
    TelemetryEvent,
    has_native_engine,
    native_nullspace_rank1_surgery_f32,
    native_router_entropy_f32,
)
from circuitforge.surgery import apply_nullspace_rank1_surgery
from circuitforge.wiring import ModelWiringMap, load_wiring_map


def classify_activation_color(magnitude: float, baseline: float = 1.0) -> str:
    """
    Classifies activation magnitude into HUD color bands:
    - 'cool_blue': <= 1.35x baseline
    - 'elevated_amber': 1.35x .. 2.25x baseline
    - 'spike_crimson': > 2.25x baseline
    """
    ratio = magnitude / max(baseline, 1e-8)
    if ratio <= 1.35:
        return "cool_blue"
    if ratio <= 2.25:
        return "elevated_amber"
    return "spike_crimson"


def compute_cross_layer_alignment(
    layer_activations: np.ndarray,
    target_vector: np.ndarray,
) -> list[dict[str, float]]:
    """
    Computes vector dot-product, cosine similarity, and subspace angle (degrees)
    between each layer's representation vector (`shape: (num_layers, d_model)`) and
    `target_vector` (`shape: (d_model,)`).
    """
    acts = np.asarray(layer_activations, dtype=np.float64)
    vec = np.asarray(target_vector, dtype=np.float64).reshape(-1)
    if acts.ndim != 2 or acts.shape[1] != vec.shape[0]:
        raise ValueError(f"Shape mismatch: acts {acts.shape} vs target_vector {vec.shape}")

    v_norm = max(float(np.linalg.norm(vec)), 1e-12)
    v_unit = vec / v_norm

    results: list[dict[str, float]] = []
    for l_idx in range(acts.shape[0]):
        h = acts[l_idx]
        h_norm = max(float(np.linalg.norm(h)), 1e-12)
        dot_val = float(np.dot(h, v_unit))
        cos_sim = float(np.clip(dot_val / h_norm, -1.0, 1.0))
        angle_deg = float(math.degrees(math.acos(cos_sim)))
        results.append(
            {
                "layer_idx": int(l_idx),
                "l2_norm": round(h_norm, 4),
                "dot_product": round(dot_val, 4),
                "cosine_similarity": round(cos_sim, 4),
                "subspace_angle_deg": round(angle_deg, 2),
            }
        )
    return results


@dataclass
class MechanisticTelemetryHUD:
    """
    Stateful mechanistic telemetry simulator, C++ ring-buffer consumer, and interactive
    click-to-steer engine for the CircuitForge Visual Debugger.
    """

    profile_name: str = "Qwen3.6-35B-A3B"
    d_model: int = 64
    num_display_experts: int = 16
    seed: int = 42
    wiring_map: ModelWiringMap = field(init=False)
    tokens: list[str] = field(default_factory=lambda: [
        "<|im_start|>",
        "user",
        "Update",
        "billing.py",
        "<|im_end|>",
        "<|im_start|>",
        "assistant",
        "<tool_call>",
        "edit_file",
        "state.md",
        "lock_spec",
        "</tool_call>",
    ])

    def __post_init__(self) -> None:
        self.wiring_map = load_wiring_map(profile_name=self.profile_name)
        self._rng = np.random.default_rng(self.seed)
        self.global_alpha: float = 1.5
        self.schedule: str = "bell"
        self.active_anchor_layers: list[int] = list(self.wiring_map.mid_to_late_planning_anchors)

        # Initialize synthetic circuit weights & vectors for interactive live steering
        n_layers = self.wiring_map.num_layers
        self._base_weights: dict[int, np.ndarray] = {
            l: (self._rng.standard_normal((self.d_model, self.d_model)) * 0.08).astype(np.float32)
            for l in range(n_layers)
        }
        self._preserved_keys = self._rng.standard_normal((8, self.d_model)).astype(np.float32)
        self._trigger_key = self._rng.standard_normal(self.d_model).astype(np.float32)
        self._trigger_key /= max(float(np.linalg.norm(self._trigger_key)), 1e-8)

        raw_steer = self._rng.standard_normal(self.d_model).astype(np.float32)
        self._steering_vec = (raw_steer / max(float(np.linalg.norm(raw_steer)), 1e-8)).astype(np.float32)

        self._token_keys = self._rng.standard_normal((len(self.tokens), self.d_model)).astype(np.float32)
        for t_idx in range(len(self.tokens)):
            self._token_keys[t_idx] /= max(float(np.linalg.norm(self._token_keys[t_idx])), 1e-8)
        # Make decision tokens (tokens 7..10: <tool_call> edit_file state.md) align strongly with trigger_key
        for t_idx in (7, 8, 9, 10):
            if t_idx < len(self.tokens):
                mixed = 0.88 * self._trigger_key + 0.12 * self._token_keys[t_idx]
                self._token_keys[t_idx] = mixed / max(float(np.linalg.norm(mixed)), 1e-8)

        self._ringbuf: NativeTelemetryRingBuffer | None = (
            NativeTelemetryRingBuffer(capacity=256) if has_native_engine() else None
        )
        self._lock = threading.RLock()
        self._step_counter = 0
        self._last_state: dict[str, Any] = {}
        self.recompute_telemetry()

    def close(self) -> None:
        with self._lock:
            if self._ringbuf is not None:
                self._ringbuf.close()
                self._ringbuf = None

    def update_steering(
        self,
        alpha: float,
        schedule: str | None = None,
        active_layers: list[int] | None = None,
    ) -> dict[str, Any]:
        """
        Updates the live steering multiplier (`alpha`), schedule (`uniform` or `bell`),
        or target anchor layers, and immediately recomputes all mechanistic HUD panels.
        """
        with self._lock:
            if not math.isfinite(float(alpha)):
                raise ValueError("alpha must be finite")
            self.global_alpha = float(np.clip(alpha, -5.0, 10.0))
            if schedule in ("uniform", "bell"):
                self.schedule = schedule
            if active_layers is not None:
                valid: list[int] = []
                for l in active_layers:
                    if isinstance(l, bool) or not isinstance(l, (int, float)):
                        raise ValueError(f"Invalid layer index: {l!r}")
                    if isinstance(l, float) and not math.isfinite(l):
                        raise ValueError(f"Non-finite layer index: {l!r}")
                    li = int(l)
                    if 0 <= li < self.wiring_map.num_layers:
                        valid.append(li)
                if valid:
                    self.active_anchor_layers = sorted(set(valid))
            return self.recompute_telemetry()

    def recompute_telemetry(self) -> dict[str, Any]:
        """
        Executes live weight surgery across active anchor layers, streams telemetry events
        through the C++ lock-free ring buffer, and builds the complete HUD state payload.
        """
        with self._lock:
            return self._recompute_telemetry_unlocked()

    def _recompute_telemetry_unlocked(self) -> dict[str, Any]:
        self._step_counter += 1
        n_layers = self.wiring_map.num_layers
        n_tokens = len(self.tokens)

        n_active = max(len(self.active_anchor_layers), 1)
        if self.schedule == "bell" and n_active > 1:
            idx = np.arange(n_active, dtype=np.float64)
            raw = np.sin(np.pi * (idx + 1.0) / (n_active + 1.0)) ** 2
            sched_mults = raw / np.mean(raw)
        else:
            sched_mults = np.ones(n_active, dtype=np.float64)

        layer_alpha_map: dict[int, float] = {
            l_idx: float(self.global_alpha * sched_mults[i])
            for i, l_idx in enumerate(self.active_anchor_layers)
        }

        heatmap_matrix: list[list[float]] = []
        heatmap_colors: list[list[str]] = []
        layer_trigger_reps = np.zeros((n_layers, self.d_model), dtype=np.float32)
        routing_layers: list[dict[str, Any]] = []
        surgery_reports: list[dict[str, Any]] = []

        use_cpp = has_native_engine()

        for l_idx in range(n_layers):
            W_base = self._base_weights[l_idx]
            l_alpha = layer_alpha_map.get(l_idx, 0.0)

            if abs(l_alpha) > 1e-6:
                if use_cpp:
                    W_curr, rep = native_nullspace_rank1_surgery_f32(
                        weight=W_base,
                        trigger_key=self._trigger_key,
                        steering_vector=self._steering_vec,
                        preserved_keys=self._preserved_keys,
                        alpha=l_alpha,
                        tensor_name=f"model.layers.{l_idx}.self_attn.o_proj.weight",
                        layer_idx=l_idx,
                    )
                else:
                    W_curr, rep = apply_nullspace_rank1_surgery(
                        weight=W_base,
                        trigger_key=self._trigger_key,
                        steering_vector=self._steering_vec,
                        preserved_keys=self._preserved_keys,
                        alpha=l_alpha,
                        tensor_name=f"model.layers.{l_idx}.self_attn.o_proj.weight",
                        layer_idx=l_idx,
                    )
                surgery_reports.append(
                    {
                        "layer_idx": l_idx,
                        "alpha": round(l_alpha, 4),
                        "frobenius_delta": round(rep.frobenius_delta, 4),
                        "nullspace_leakage": float(f"{rep.nullspace_leakage_max:.3e}"),
                        "alignment_after": round(rep.target_alignment_after, 4),
                    }
                )
            else:
                W_curr = W_base

            # Compute token activations (n_tokens, d_out)
            acts_t = (W_curr @ self._token_keys.T).T
            norms_t = np.linalg.norm(acts_t, axis=1)
            base_acts_t = (W_base @ self._token_keys.T).T
            baseline_norm = float(np.mean(np.linalg.norm(base_acts_t, axis=1)))

            row_mags = [round(float(m), 4) for m in norms_t]
            row_cols = [classify_activation_color(float(m), baseline_norm) for m in norms_t]
            heatmap_matrix.append(row_mags)
            heatmap_colors.append(row_cols)

            # Representation on trigger key for cross-layer alignment inspector
            rep_vec = W_curr @ self._trigger_key
            layer_trigger_reps[l_idx] = rep_vec

            # Router logits & Shannon entropy across display experts
            router_logits = rep_vec[: self.num_display_experts].copy()
            if abs(l_alpha) > 1e-6:
                router_logits[0] += float(0.65 * l_alpha)
                router_logits[1] += float(0.40 * l_alpha)

            if use_cpp:
                ent, top_id, top_prob = native_router_entropy_f32(router_logits)
            else:
                shifted = router_logits - np.max(router_logits)
                exps = np.exp(shifted)
                probs = exps / np.sum(exps)
                ent = float(-np.sum(probs * np.log(np.maximum(probs, 1e-12))))
                top_id = int(np.argmax(probs))
                top_prob = float(probs[top_id])

            shifted = router_logits - np.max(router_logits)
            exps = np.exp(shifted)
            probs = exps / np.sum(exps)

            max_possible_ent = float(math.log(self.num_display_experts))
            is_spike = any(c == "spike_crimson" for c in row_cols)

            if self._ringbuf is not None:
                self._ringbuf.push(
                    TelemetryEvent(
                        timestamp_ns=time.time_ns(),
                        step_id=self._step_counter,
                        layer_idx=l_idx,
                        activation_l2_norm=float(np.max(norms_t)),
                        router_entropy=ent,
                        steering_alpha=l_alpha,
                        top_expert_id=top_id,
                        top_expert_weight=top_prob,
                        anomaly_flag=1 if is_spike else 0,
                    )
                )

            if l_idx in self.wiring_map.full_attention_layers:
                routing_layers.append(
                    {
                        "layer_idx": l_idx,
                        "attention_type": self.wiring_map.layers[l_idx].attention_type,
                        "router_entropy": round(ent, 4),
                        "max_entropy": round(max_possible_ent, 4),
                        "entropy_ratio": round(ent / max(max_possible_ent, 1e-8), 4),
                        "top_expert_id": top_id,
                        "top_expert_prob": round(top_prob, 4),
                        "expert_probs": [round(float(p), 4) for p in probs],
                    }
                )

        ring_events = (
            [ev.to_dict() for ev in self._ringbuf.drain_all()]
            if self._ringbuf is not None
            else []
        )
        alignment_profile = compute_cross_layer_alignment(layer_trigger_reps, self._steering_vec)

        if self.global_alpha >= 0.8:
            # MOCK(M-002): Deterministic HUD tool-call preview templates driven by live steering multiplier alpha.
            predicted_action = (
                '<tool_call>{"name": "edit_file", "arguments": {"path": "state.md", '
                '"target": "## Active Sprint", "replacement": "- [ ] Update billing.py"}}</tool_call>'
            )
            compliance_tag = "SDCS_SPEC_LOCKED"
        elif self.global_alpha <= -0.5:
            predicted_action = (
                '<tool_call>{"name": "write_file", "arguments": {"path": "billing.py", '
                '"content": "# Unscaffolded full overwrite..."}}</tool_call>'
            )
            compliance_tag = "ANTI_PATTERN_UNSCAFFOLDED"
        else:
            predicted_action = (
                '<tool_call>{"name": "read_file", "arguments": {"path": "billing.py"}}</tool_call>'
            )
            compliance_tag = "NEUTRAL_BASELINE"

        self._last_state = {
            "step_id": self._step_counter,
            "profile": self.wiring_map.summary_dict(),
            "native_engine_active": use_cpp,
            "controls": {
                "global_alpha": round(self.global_alpha, 3),
                "schedule": self.schedule,
                "active_anchor_layers": self.active_anchor_layers,
            },
            "predicted_output": {
                "compliance_status": compliance_tag,
                "tool_call_preview": predicted_action,
            },
            "heatmap": {
                "tokens": self.tokens,
                "num_layers": n_layers,
                "magnitudes": heatmap_matrix,
                "color_bands": heatmap_colors,
            },
            "routing_equalizer": {
                "num_display_experts": self.num_display_experts,
                "total_routed_experts": self.wiring_map.num_experts,
                "anchor_routing": routing_layers,
            },
            "alignment_inspector": alignment_profile,
            "surgery_reports": surgery_reports,
            "ring_buffer_events_count": len(ring_events),
        }
        return self._last_state


def render_hud_html(initial_state: dict[str, Any] | None = None) -> str:
    """
    Renders the standalone single-page HTML5 / SVG / Canvas Mechanistic Telemetry HUD
    with embedded initial state (works both offline as a static `.html` file and connected
    live to `MechanisticHUDServer` via `/api/telemetry` & `/api/steer`).
    """
    embedded_json = (
        json.dumps(initial_state or {})
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("&", "\\u0026")
    )
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<title>CircuitForge :: Real-Time Mechanistic Telemetry HUD</title>
<style>
  :root {{
    --bg-dark: #070b14;
    --panel-bg: #0f172a;
    --panel-border: #1e293b;
    --text-main: #e2e8f0;
    --text-muted: #94a3b8;
    --cool-blue: #0284c7;
    --elevated-amber: #f59e0b;
    --spike-crimson: #ef4444;
    --accent-cyan: #38bdf8;
    --accent-emerald: #10b981;
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    background: var(--bg-dark);
    color: var(--text-main);
    font-family: 'Segoe UI', system-ui, -apple-system, monospace;
    padding: 18px;
    line-height: 1.45;
  }}
  header {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 14px 20px;
    background: var(--panel-bg);
    border: 1px solid var(--panel-border);
    border-radius: 8px;
    margin-bottom: 16px;
  }}
  .brand {{ font-size: 1.25rem; font-weight: 700; color: var(--accent-cyan); letter-spacing: 0.5px; }}
  .badge {{
    display: inline-block;
    padding: 4px 10px;
    border-radius: 999px;
    font-size: 0.78rem;
    font-weight: 600;
    background: #1e293b;
    color: var(--accent-emerald);
    border: 1px solid var(--accent-emerald);
  }}
  .grid-top {{
    display: grid;
    grid-template-columns: 360px 1fr;
    gap: 16px;
    margin-bottom: 16px;
  }}
  .grid-bottom {{
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 16px;
  }}
  .panel {{
    background: var(--panel-bg);
    border: 1px solid var(--panel-border);
    border-radius: 8px;
    padding: 16px;
  }}
  .panel h2 {{
    font-size: 0.95rem;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    color: var(--accent-cyan);
    margin-bottom: 12px;
    border-bottom: 1px solid var(--panel-border);
    padding-bottom: 6px;
  }}
  .control-group {{ margin-bottom: 14px; }}
  .control-group label {{ display: block; font-size: 0.85rem; color: var(--text-muted); margin-bottom: 4px; }}
  input[type=range] {{ width: 100%; accent-color: var(--accent-cyan); cursor: pointer; }}
  select, button {{
    background: #1e293b;
    color: var(--text-main);
    border: 1px solid #334155;
    padding: 7px 12px;
    border-radius: 5px;
    font-size: 0.85rem;
    cursor: pointer;
  }}
  button:hover {{ border-color: var(--accent-cyan); }}
  .output-box {{
    background: #020617;
    border: 1px solid #1e293b;
    border-radius: 6px;
    padding: 10px;
    font-family: monospace;
    font-size: 0.8rem;
    color: #38bdf8;
    margin-top: 8px;
    word-break: break-all;
  }}
  .heatmap-wrap {{ overflow-x: auto; max-height: 360px; overflow-y: auto; }}
  table.heatmap {{ border-collapse: collapse; width: 100%; font-size: 0.72rem; font-family: monospace; }}
  table.heatmap th, table.heatmap td {{
    border: 1px solid #0f172a;
    padding: 3px 5px;
    text-align: center;
  }}
  table.heatmap th {{ background: #1e293b; color: var(--text-muted); position: sticky; top: 0; }}
  .cell-cool_blue {{ background: rgba(2, 132, 199, 0.35); color: #bae6fd; }}
  .cell-elevated_amber {{ background: rgba(245, 158, 11, 0.45); color: #fef3c7; font-weight: 600; }}
  .cell-spike_crimson {{ background: rgba(239, 68, 68, 0.75); color: #ffffff; font-weight: 700; }}
  .bar-row {{ display: flex; align-items: center; gap: 8px; margin-bottom: 6px; font-size: 0.78rem; font-family: monospace; }}
  .bar-track {{ flex: 1; background: #020617; height: 12px; border-radius: 4px; overflow: hidden; }}
  .bar-fill {{ height: 100%; background: linear-gradient(90deg, #0284c7, #38bdf8); }}
  .bar-fill-align {{ height: 100%; background: linear-gradient(90deg, #10b981, #34d399); }}
  .legend {{ display: flex; gap: 14px; font-size: 0.75rem; margin-bottom: 8px; color: var(--text-muted); }}
  .dot {{ display: inline-block; width: 10px; height: 10px; border-radius: 2px; margin-right: 4px; }}
</style>
</head>
<body>
<header>
  <div>
    <div class="brand">CIRCUITFORGE // MECHANISTIC TELEMETRY HUD</div>
    <div style="font-size:0.8rem; color:var(--text-muted);" id="subtitle">Loading architecture profile...</div>
  </div>
  <div style="display:flex; gap:10px; align-items:center;">
    <span class="badge" id="engine-badge">C++ SIMD RINGBUF</span>
    <span class="badge" id="step-badge">STEP 0</span>
  </div>
</header>

<div class="grid-top">
  <!-- Interactive Click-to-Steer Controls -->
  <div class="panel">
    <h2>1. Interactive Click-to-Steer Controls</h2>
    <div class="control-group">
      <label>Steering Multiplier (&alpha;): <strong id="alpha-val" style="color:var(--accent-cyan);">1.50</strong></label>
      <input type="range" id="alpha-slider" min="-2.0" max="5.0" step="0.1" value="1.5" />
    </div>
    <div class="control-group">
      <label>Multi-Layer Distribution Schedule:</label>
      <select id="schedule-select">
        <option value="bell">Bell Curve (Sine-Squared Peak)</option>
        <option value="uniform">Uniform Across Anchors</option>
      </select>
    </div>
    <div class="control-group">
      <button id="apply-btn">Apply Null-Space Steering</button>
      <button id="reset-btn">Zero (&alpha;=0)</button>
    </div>
    <div style="margin-top:12px;">
      <div style="font-size:0.8rem; color:var(--text-muted);">Live Behavioral Reflex Output:</div>
      <div id="compliance-badge" style="font-size:0.82rem; font-weight:700; color:var(--accent-emerald); margin-top:4px;"></div>
      <div class="output-box" id="output-preview"></div>
    </div>
  </div>

  <!-- Color-Coded Activation Heatmap -->
  <div class="panel">
    <h2>2. Color-Coded Layer &times; Token Activation Heatmap</h2>
    <div class="legend">
      <span><i class="dot" style="background:#0284c7;"></i>Cool Blue (Baseline &le;1.35x)</span>
      <span><i class="dot" style="background:#f59e0b;"></i>Elevated Amber (1.35x..2.25x)</span>
      <span><i class="dot" style="background:#ef4444;"></i>Spike Crimson (&gt;2.25x Steered Anchor)</span>
    </div>
    <div class="heatmap-wrap">
      <table class="heatmap" id="heatmap-table"></table>
    </div>
  </div>
</div>

<div class="grid-bottom">
  <!-- Live MoE Routing & Entropy Equalizer -->
  <div class="panel">
    <h2>3. Live MoE Routing &amp; Shannon Entropy Equalizer (H_router)</h2>
    <div id="routing-container"></div>
  </div>

  <!-- Cross-Layer Representation Alignment Inspector -->
  <div class="panel">
    <h2>4. Cross-Layer Representation Alignment Inspector (cos &theta; &amp; Subspace Angle)</h2>
    <div id="alignment-container" style="max-height:300px; overflow-y:auto;"></div>
  </div>
</div>

<script>
let hudState = {embedded_json};

function renderHUD(state) {{
  if (!state || !state.profile) return;
  document.getElementById('subtitle').textContent =
    `Model: ${{state.profile.model_name}} (${{state.profile.architecture}}) | Layers: ${{state.profile.num_layers}} | Frozen Routed Experts: ${{state.profile.num_experts}}`;
  document.getElementById('engine-badge').textContent =
    state.native_engine_active ? 'C++ AVX2 SIMD ACTIVE' : 'NUMPY FALLBACK';
  document.getElementById('step-badge').textContent = `STEP ${{state.step_id}}`;

  document.getElementById('alpha-val').textContent = Number(state.controls.global_alpha).toFixed(2);
  document.getElementById('alpha-slider').value = state.controls.global_alpha;
  document.getElementById('schedule-select').value = state.controls.schedule;

  document.getElementById('compliance-badge').textContent =
    `Status: ${{state.predicted_output.compliance_status}}`;
  document.getElementById('output-preview').textContent =
    state.predicted_output.tool_call_preview;

  // Render Heatmap
  const escHtml = (s) => String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
  const tbl = document.getElementById('heatmap-table');
  const tokens = state.heatmap.tokens;
  let html = '<thead><tr><th>Layer</th>' + tokens.map(t => `<th>${{escHtml(t)}}</th>`).join('') + '</tr></thead><tbody>';
  for (let l = 0; l < state.heatmap.num_layers; l++) {{
    const mags = state.heatmap.magnitudes[l];
    const cols = state.heatmap.color_bands[l];
    html += `<tr><td><strong>L${{l}}</strong></td>`;
    for (let t = 0; t < tokens.length; t++) {{
      html += `<td class="cell-${{cols[t]}}" title="L${{l}} ${{escHtml(tokens[t])}}: ${{mags[t]}}">${{mags[t].toFixed(2)}}</td>`;
    }}
    html += '</tr>';
  }}
  html += '</tbody>';
  tbl.innerHTML = html;

  // Render MoE Routing & Entropy Equalizer
  const rCont = document.getElementById('routing-container');
  let rHtml = '';
  for (const r of state.routing_equalizer.anchor_routing) {{
    const pct = Math.min(100, Math.round(r.entropy_ratio * 100));
    rHtml += `<div class="bar-row">
      <span style="width:75px;">Anchor L${{r.layer_idx}}</span>
      <span style="width:110px; color:var(--text-muted);">H=${{r.router_entropy.toFixed(2)}} / ${{r.max_entropy.toFixed(2)}}</span>
      <div class="bar-track"><div class="bar-fill" style="width:${{pct}}%;"></div></div>
      <span style="width:115px; text-align:right;">Top Exp #${{r.top_expert_id}} (${{(r.top_expert_prob*100).toFixed(1)}}%)</span>
    </div>`;
  }}
  rCont.innerHTML = rHtml;

  // Render Cross-Layer Alignment Inspector
  const aCont = document.getElementById('alignment-container');
  let aHtml = '';
  for (const a of state.alignment_inspector) {{
    const normCos = Math.max(0, Math.min(100, Math.round(((a.cosine_similarity + 1) / 2) * 100)));
    aHtml += `<div class="bar-row">
      <span style="width:55px;">L${{a.layer_idx}}</span>
      <span style="width:95px;">cos &theta; = ${{a.cosine_similarity.toFixed(3)}}</span>
      <div class="bar-track"><div class="bar-fill-align" style="width:${{normCos}}%;"></div></div>
      <span style="width:90px; text-align:right;">&theta; = ${{a.subspace_angle_deg.toFixed(1)}}&deg;</span>
    </div>`;
  }}
  aCont.innerHTML = aHtml;
}}

async function pushSteer(alphaVal, schedVal) {{
  try {{
    const resp = await fetch('/api/steer', {{
      method: 'POST',
      headers: {{ 'Content-Type': 'application/json' }},
      body: JSON.stringify({{ alpha: parseFloat(alphaVal), schedule: schedVal }})
    }});
    if (resp.ok) {{
      hudState = await resp.json();
      renderHUD(hudState);
      return;
    }}
  }} catch (e) {{
    // Offline static preview fallback: approximate shift locally
    hudState.controls.global_alpha = parseFloat(alphaVal);
    hudState.controls.schedule = schedVal;
    renderHUD(hudState);
  }}
}}

document.getElementById('alpha-slider').addEventListener('input', (e) => {{
  document.getElementById('alpha-val').textContent = parseFloat(e.target.value).toFixed(2);
}});
document.getElementById('apply-btn').addEventListener('click', () => {{
  pushSteer(document.getElementById('alpha-slider').value, document.getElementById('schedule-select').value);
}});
document.getElementById('reset-btn').addEventListener('click', () => {{
  document.getElementById('alpha-slider').value = 0.0;
  pushSteer(0.0, document.getElementById('schedule-select').value);
}});

renderHUD(hudState);
</script>
</body>
</html>
"""


class MechanisticHUDServer:
    """
    Lightweight local HTTP server hosting the interactive Mechanistic Telemetry HUD
    (`GET /`, `GET /api/telemetry`, `POST /api/steer`).
    """

    def __init__(self, hud: MechanisticTelemetryHUD, host: str = "127.0.0.1", port: int = 0) -> None:
        self.hud = hud
        self.host = host

        hud_ref = self.hud

        class _Handler(BaseHTTPRequestHandler):
            timeout = 3.0

            def log_message(self, format: str, *args: Any) -> None:
                return  # Silence noisy console logs during tests

            def _check_local_host_and_origin(self, require_json: bool = False) -> bool:
                host_hdr = (self.headers.get("Host") or "").strip()
                if host_hdr:
                    try:
                        parsed_host = urlsplit(f"//{host_hdr}")
                        host_only = (parsed_host.hostname or "").lower()
                    except ValueError:
                        host_only = ""
                    if host_only not in ("127.0.0.1", "localhost", "::1"):
                        self.send_response(403)
                        self.end_headers()
                        return False
                origin_hdr = (self.headers.get("Origin") or "").strip()
                if origin_hdr:
                    if origin_hdr.lower() == "null":
                        self.send_response(403)
                        self.end_headers()
                        return False
                    try:
                        parsed_origin = urlparse(origin_hdr)
                        origin_scheme = (parsed_origin.scheme or "").lower()
                        origin_host = (parsed_origin.hostname or "").lower()
                    except ValueError:
                        origin_scheme = ""
                        origin_host = ""
                    if origin_scheme != "http" or origin_host not in ("127.0.0.1", "localhost", "::1"):
                        self.send_response(403)
                        self.end_headers()
                        return False
                if require_json:
                    ctype = (self.headers.get("Content-Type") or "").strip()
                    media_type = ctype.split(";", 1)[0].strip().lower()
                    if media_type != "application/json":
                        err = json.dumps({"error": "Content-Type must be application/json"}).encode("utf-8")
                        self.send_response(415)
                        self.send_header("Content-Type", "application/json")
                        self.send_header("Content-Length", str(len(err)))
                        self.end_headers()
                        self.wfile.write(err)
                        return False
                return True

            def do_GET(self) -> None:
                if not self._check_local_host_and_origin(require_json=False):
                    return
                if self.path == "/api/telemetry":
                    with hud_ref._lock:
                        payload = json.dumps(hud_ref._last_state).encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(payload)))
                    self.end_headers()
                    self.wfile.write(payload)
                elif self.path in ("/", "/index.html"):
                    with hud_ref._lock:
                        html_bytes = render_hud_html(hud_ref._last_state).encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Content-Length", str(len(html_bytes)))
                    self.end_headers()
                    self.wfile.write(html_bytes)
                else:
                    self.send_response(404)
                    self.end_headers()

            def do_POST(self) -> None:
                if self.path == "/api/steer":
                    if not self._check_local_host_and_origin(require_json=True):
                        return
                    try:
                        length = int(self.headers.get("Content-Length", "0"))
                        if length < 0 or length > 1_000_000:
                            raise ValueError("Invalid Content-Length")
                        raw = self.rfile.read(length) if length > 0 else b"{}"
                        body = json.loads(raw.decode("utf-8"))
                        if not isinstance(body, dict):
                            raise ValueError("Expected JSON object")
                        alpha_val = float(body.get("alpha", hud_ref.global_alpha))
                        if not math.isfinite(alpha_val):
                            raise ValueError("alpha must be finite")
                        new_state = hud_ref.update_steering(
                            alpha=alpha_val,
                            schedule=body.get("schedule"),
                            active_layers=body.get("active_layers"),
                        )
                    except (
                        ValueError,
                        KeyError,
                        TypeError,
                        OverflowError,
                        json.JSONDecodeError,
                        UnicodeDecodeError,
                    ) as exc:
                        err_payload = json.dumps({"error": str(exc)}).encode("utf-8")
                        self.send_response(400)
                        self.send_header("Content-Type", "application/json")
                        self.send_header("Content-Length", str(len(err_payload)))
                        self.end_headers()
                        self.wfile.write(err_payload)
                        return

                    resp = json.dumps(new_state).encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(resp)))
                    self.end_headers()
                    self.wfile.write(resp)
                else:
                    self.send_response(404)
                    self.end_headers()

        self._httpd = ThreadingHTTPServer((self.host, port), _Handler)
        self.port = int(self._httpd.server_port)
        self._thread: threading.Thread | None = None

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}"

    def start_background(self) -> str:
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)
        self._thread.start()
        return self.url

    def stop(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            self._httpd.shutdown()
            self._thread.join(timeout=2.0)
            self._thread = None
        self._httpd.server_close()


def export_hud_snapshot(
    output_html_path: str | Path,
    profile_name: str = "Qwen3.6-35B-A3B",
    alpha: float = 1.8,
    schedule: str = "bell",
) -> tuple[Path, dict[str, Any]]:
    """
    Generates a complete self-contained interactive HTML snapshot of the Mechanistic Telemetry HUD
    at `output_html_path` and returns `(resolved_path, telemetry_state)`.
    """
    hud = MechanisticTelemetryHUD(profile_name=profile_name)
    state = hud.update_steering(alpha=alpha, schedule=schedule)
    html_str = render_hud_html(state)
    out = Path(output_html_path).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html_str, encoding="utf-8")
    hud.close()
    return out, state
