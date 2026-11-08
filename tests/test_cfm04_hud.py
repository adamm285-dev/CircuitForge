r"""Unit and integration tests for Milestone CFM-04:
- Real-Time Mechanistic Telemetry HUD & Visual Debugger (`circuitforge/hud.py`)
- Color-Coded Activation Heatmaps (`cool_blue`, `elevated_amber`, `spike_crimson`)
- Live MoE Routing & Shannon Entropy Equalizer ($\mathcal{H}_{\text{router}}$)
- Interactive Click-to-Steer Controls via `MechanisticHUDServer` (`GET /api/telemetry`, `POST /api/steer`)
- Cross-Layer Representation Alignment Inspector (cosine similarity and subspace angles)
"""

from __future__ import annotations

import json
import urllib.request
from pathlib import Path

import numpy as np

from circuitforge.hud import (
    MechanisticHUDServer,
    MechanisticTelemetryHUD,
    classify_activation_color,
    compute_cross_layer_alignment,
    export_hud_snapshot,
)


def test_activation_color_classification_and_alignment_inspector() -> None:
    """Verify color band classification and cross-layer cosine similarity & subspace angle computation."""
    assert classify_activation_color(1.0, baseline=1.0) == "cool_blue"
    assert classify_activation_color(1.8, baseline=1.0) == "elevated_amber"
    assert classify_activation_color(3.0, baseline=1.0) == "spike_crimson"

    target_v = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    layer_acts = np.array(
        [
            [2.0, 0.0, 0.0, 0.0],   # Parallel -> cos=1.0, angle=0 deg
            [0.0, 3.0, 0.0, 0.0],   # Orthogonal -> cos=0.0, angle=90 deg
            [-1.5, 0.0, 0.0, 0.0],  # Anti-parallel -> cos=-1.0, angle=180 deg
        ],
        dtype=np.float32,
    )
    aligned = compute_cross_layer_alignment(layer_acts, target_v)
    assert len(aligned) == 3
    assert np.isclose(aligned[0]["cosine_similarity"], 1.0, atol=1e-4)
    assert np.isclose(aligned[0]["subspace_angle_deg"], 0.0, atol=1e-2)
    assert np.isclose(aligned[1]["cosine_similarity"], 0.0, atol=1e-4)
    assert np.isclose(aligned[1]["subspace_angle_deg"], 90.0, atol=1e-2)
    assert np.isclose(aligned[2]["cosine_similarity"], -1.0, atol=1e-4)
    assert np.isclose(aligned[2]["subspace_angle_deg"], 180.0, atol=1e-2)


def test_mechanistic_telemetry_hud_state_and_click_to_steer() -> None:
    """Verify live HUD state generation, C++ ring buffer event emission, and interactive steering shifts."""
    hud = MechanisticTelemetryHUD(profile_name="Qwen3.6-35B-A3B", d_model=64)
    try:
        state_zero = hud.update_steering(alpha=0.0, schedule="uniform")
        assert state_zero["predicted_output"]["compliance_status"] == "NEUTRAL_BASELINE"
        assert state_zero["heatmap"]["num_layers"] == 40
        assert len(state_zero["routing_equalizer"]["anchor_routing"]) == 10

        # Increase alpha to trigger positive SDCS spec-lock reflex and activation spikes on anchor layers
        state_steered = hud.update_steering(alpha=3.5, schedule="bell")
        assert state_steered["predicted_output"]["compliance_status"] == "SDCS_SPEC_LOCKED"
        assert state_steered["ring_buffer_events_count"] == 40

        # Anchor layer 19 (middle of 40 layers) should have higher alignment & spike_crimson cells on decision tokens
        align_zero_19 = state_zero["alignment_inspector"][19]["cosine_similarity"]
        align_steer_19 = state_steered["alignment_inspector"][19]["cosine_similarity"]
        assert align_steer_19 > align_zero_19 + 0.3
        assert state_steered["alignment_inspector"][19]["subspace_angle_deg"] < state_zero["alignment_inspector"][19]["subspace_angle_deg"]
        assert "spike_crimson" in state_steered["heatmap"]["color_bands"][19]
    finally:
        hud.close()


def test_hud_http_server_and_html_snapshot_export(tmp_path: Path) -> None:
    """Verify live HTTP server endpoints (`GET /`, `GET /api/telemetry`, `POST /api/steer`) and HTML export."""
    html_out = tmp_path / "hud_snapshot.html"
    out_path, snap_state = export_hud_snapshot(
        output_html_path=html_out,
        profile_name="Qwen3.6-35B-A3B",
        alpha=2.0,
        schedule="bell",
    )
    assert out_path.is_file()
    html_text = out_path.read_text(encoding="utf-8")
    assert "CIRCUITFORGE // MECHANISTIC TELEMETRY HUD" in html_text
    assert snap_state["controls"]["global_alpha"] == 2.0

    hud = MechanisticTelemetryHUD(profile_name="Qwen3.6-35B-A3B", d_model=32)
    srv = MechanisticHUDServer(hud, host="127.0.0.1", port=0)
    base_url = srv.start_background()
    try:
        # 1. GET / HTML dashboard
        with urllib.request.urlopen(f"{base_url}/") as resp:
            assert resp.status == 200
            body_html = resp.read().decode("utf-8")
            assert "Interactive Click-to-Steer Controls" in body_html

        # 2. GET /api/telemetry
        with urllib.request.urlopen(f"{base_url}/api/telemetry") as resp:
            assert resp.status == 200
            telem = json.loads(resp.read().decode("utf-8"))
            assert telem["profile"]["model_name"] == "Qwen3.6-35B-A3B"

        # 3. POST /api/steer (Interactive Click-to-Steer)
        req_data = json.dumps({"alpha": -1.2, "schedule": "uniform"}).encode("utf-8")
        req = urllib.request.Request(
            f"{base_url}/api/steer",
            data=req_data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req) as resp:
            assert resp.status == 200
            updated = json.loads(resp.read().decode("utf-8"))
            assert updated["controls"]["global_alpha"] == -1.2
            assert updated["controls"]["schedule"] == "uniform"
            assert updated["predicted_output"]["compliance_status"] == "ANTI_PATTERN_UNSCAFFOLDED"

        # 4. POST /api/simulate-spike (Zero-Log Visual Spike Radar & Sentinel Circuit-Breaker)
        req_spike = urllib.request.Request(
            f"{base_url}/api/simulate-spike",
            data=json.dumps({"attack_type": "orthogonal_hijack", "layer_idx": 19}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req_spike) as resp:
            assert resp.status == 200
            spike_state = json.loads(resp.read().decode("utf-8"))
            assert spike_state["spike_radar"]["status"] == "SPIKE_NEUTRALIZED_BY_SENTINEL"
            assert spike_state["spike_radar"]["sentinel_interceptions"] >= 1
            assert len(spike_state["spike_radar"]["events"]) == 1

        # 5. POST /api/run-hypothesis (Interactive Hypothesis A & B Lab)
        req_hyp = urllib.request.Request(
            f"{base_url}/api/run-hypothesis",
            data=json.dumps({"hypothesis_id": "hypothesis_b"}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req_hyp) as resp:
            assert resp.status == 200
            hyp_state = json.loads(resp.read().decode("utf-8"))
            assert hyp_state["hypothesis_lab"]["active_hypothesis"] == "hypothesis_b"
            assert "CONFIRMED" in hyp_state["hypothesis_lab"]["verdict"]
    finally:
        srv.stop()
        hud.close()
