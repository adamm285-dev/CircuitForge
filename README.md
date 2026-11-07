# CircuitForge

**Weight-Space Circuit Surgery & Verifiable RL (`RLVR`) for Hybrid MoE Coding Agents**

**Author:** Adam Murphy ([adamm285-dev](https://github.com/adamm285-dev))

---

## Overview

Standard reinforcement learning (`PPO` / `GRPO`) and naive fine-tuning treat Mixture-of-Experts (MoE) language models as uniform black boxes, updating weights across all layers and degrading the fragile parametric code syntax stored inside banked routed experts.

**CircuitForge** separates **parametric syntax memory** from **executive planning circuits** in hybrid linear/full-attention MoE architectures (`Qwen3_5Moe`, such as **`Qwen3.6-35B-A3B`**):

1. **Hybrid Circuit Topology & Layer Mapping (`circuitforge/wiring.py`):**
   - Maps the `3x Linear Attention (Gated DeltaNet) -> 1x Full Attention (head_dim=256, attn_output_gate=True)` repeating structure across **40 layers** (`Qwen3.6-35B-A3B`, 10 Full-Attention planning anchors, 256 routed experts, `top_k=8`).
   - Strictly separates surgical control tensors (Full-Attention `q/k/v/o_proj`, Linear-Attention `in/out_proj`, MoE routers `mlp.gate`, and `shared_expert`) from the 256 banked routed experts (`mlp.experts.0..255`).
2. **Null-Space Constrained Weight-Space Surgery (`circuitforge/surgery.py`):**
   - Extracts unit-norm contrastive steering vectors from positive (`+X`) vs. negative (`-X`) hidden-state activations:
     $$v_l = \frac{\mathbb{E}[h_l^+] - \mathbb{E}[h_l^-]}{\|\mathbb{E}[h_l^+] - \mathbb{E}[h_l^-]\|_2}$$
   - Applies **Null-Space Constrained Rank-1 Updates** (`ROME` / `MEMIT` style) to the Full-Attention output projections (`self_attn.o_proj` / `blk.L.attn_output`) and Shared Experts (`mlp.shared_expert.down_proj` / `blk.L.ffn_down_shexp`) while strictly freezing all banked routed experts:
     $$\Delta W = \alpha \cdot v_l \otimes \frac{P_\perp k^*}{\|P_\perp k^*\|_2^2}, \quad P_\perp = I - U_0 U_0^\top$$
   - Guarantees $\Delta W \cdot K_0 = 0$ on preserved baseline syntax keys $K_0$, preventing catastrophic forgetting.
3. **Streaming Safetensors I/O & Declarative Forge Recipes (`circuitforge/io.py`, `circuitforge/recipes.py`):**
   - Pure-NumPy zero-dependency `.safetensors` binary reader, header inspector, slice loader (`F32`, `F64`, `F16`, `BF16`), and streaming surgical updater.
   - Declarative `.forge.json` recipe engine (`circuitforge run-recipe`) and contrastive JSONL harness supporting Rank-1, Rank-$k$ (`MEMIT`-style), directional ablation, and bell-curve multi-layer distribution across `Qwen3_5Moe`, `Qwen2.5-Coder`, `Llama-3.1`, `DeepSeek-V3`, and `Mixtral`.
4. **Native C++17 / AVX2 Zero-Copy Tensor Engine (`src/engine/`, `circuitforge/native.py`):**
   - Cross-platform zero-copy memory-mapped checkpoint I/O (`MapViewOfFile` / `mmap`) with in-place Rank-1 Null-Space Surgery, Directional Feature Ablation, fused RMSNorm+MatVec, and Router Shannon Entropy AVX2 SIMD kernels (**~22 μs** per `512x512` surgical update).
   - Lock-free single-producer single-consumer (SPSC) telemetry ring buffer (`CFTelemetryRingBuffer`) for real-time mechanistic activation streaming.
5. **Real-Time Mechanistic Telemetry HUD & Visual Debugger (`circuitforge/hud.py`):**
   - Interactive single-page HTML5/SVG visual control room and local HTTP server (`export-hud`, `serve-hud`) featuring color-coded Layer x Token activation heatmaps (`cool_blue`, `elevated_amber`, `spike_crimson`), live MoE routing & Shannon entropy equalizer ($\mathcal{H}_{\text{router}}$), interactive Click-to-Steer controls (`POST /api/steer`), and a cross-layer representation alignment inspector ($\cos \theta$ & subspace angle $\theta^\circ$).
6. **MoE Anomaly Router & Activation Spike Sentinel (`circuitforge/sentinel.py`):**
   - Automated circuit-breaker operating inside MoE routing layers (`MoEAnomalySentinel`) that monitors router gate entropy and orthogonal hidden-state magnitude spikes, dispatches adversarial tokens to a dedicated auditor expert (`auditor_expert_id = -1`), and neutralizes rogue steering directions via safe-subspace orthogonal projection ($P_{\text{safe}} = I - P_\perp$).
7. **Discrete Engram Memory Graph & Weight-Space Latent Synthesis (`circuitforge/engram.py`):**
   - C++ SIMD-accelerated concurrent hash table (`CFEngramTable` / `DiscreteEngramGraph`) storing discrete episodic invariants in host RAM and injecting additive biases directly into intermediate layer RMSNorm activations at zero KV-cache token cost, paired with `ParametricAdapterManifold` for geodesic SLERP (`slerp_adapters`), TIES + null-space adapter merging, and principal manifold coordinate sampling.
8. **Tracr / RASP-Style Symbolic-to-Weight Micro-Circuit Compiler (`circuitforge/compiler.py`):**
   - Deterministically compiles symbolic RASP-style token selector predicates, boolean conjunctions/vetoes (`RuleSpec`), and numeric token-budget thresholds directly into an exact 2-layer `91,780`-parameter gate network (`CompiledMicroCircuit`) with **0 gradient steps** (`python -m circuitforge verify-compiler`).
9. **Zero-Dependency GGUF v3 Quantized Surgery (`Q8_0`, `Q4_0`, `BF16`) & Exporters (`circuitforge/gguf.py`, `circuitforge/exporter.py`):**
   - Parses, slices, and writes 32-byte aligned GGUF v3 binary files (`inspect-gguf`) across `F32`, `F16`, `BF16`, `Q8_0` (`34 B/block`), and `Q4_0` (`18 B/block`).
   - Streams surgical updates directly onto quantized GGUF checkpoints (`stream_surgical_gguf_update`) and multi-shard Hugging Face `model.safetensors.index.json` directories (`apply_sharded_checkpoint_surgery`) while leaving untouched routed-expert shards byte-identical.
   - Exports surgical weight deltas (`\Delta W`) as Hugging Face PEFT LoRA adapters (`export-lora` -> `adapter_model.safetensors` + `adapter_config.json`), `llama.cpp` `--control-vector` GGUF files (`export_gguf_control_vector`), and Hugging Face Model Card bundles (`export-card --bundle`).
10. **LEACE Concept Erasure & Automated Causal Circuit Discovery (`circuitforge/surgery.py`, `circuitforge/attribution.py`):**
    - Implements **LEACE (Least-Squares Concept Erasure)** (`apply_leace_concept_erasure`) using oblique Mahalanobis whitening projection ($P_{\text{LEACE}} = I - \Sigma_{XX}^{1/2} P_W \Sigma_{XX}^{-1/2}$) to provably zero out linear concept covariance $\Sigma_{XZ} = 0$ on control layers.
    - Includes an **Automated Causal Attribution Sweeper** (`discover-circuit` / `discover_optimal_surgical_circuit`) that sweeps candidate layers for **Direct Logit Attribution (DLA)**, **Activation Patching Recovery (APR)**, and **Null-Space Signal-to-Leakage Ratio (SLR)** to automatically rank layers and emit an optimal `.forge.json` surgical recipe.
11. **Verifiable Multi-Turn Agent RL Environment (`circuitforge/rl_env.py`) & Autograd (`circuitforge/autograd.py`):**
   - Gymnasium / `verifiers`-compatible RLVR environment (`VerifiableCodingEnv`) with deterministic step-level process rewards and pure-NumPy reverse-mode automatic differentiation.

---

## Hybrid MoE Circuit Layout (`Qwen3.6-35B-A3B`)

| Circuit Parameter | `Qwen3.6-35B-A3B` | Surgical Policy |
| :--- | :---: | :--- |
| **Architecture Family** | `Qwen3_5Moe` | Hybrid Linear + Full Attention MoE |
| **Total Layers** | `40` | Layers `0..39` (+ `blk.40` MTP draft layer) |
| **Repeating Attention Cycle** | `3x Linear -> 1x Full` (`interval=4`) | 30 Linear-Attention, 10 Full-Attention |
| **Full-Attention Planning Anchors** | `10` layers (`3, 7, ..., 39`) | **Primary Surgical Targets** (`o_proj` / `attn_output`) |
| **Shared Experts** | `1` per layer (`intermediate=512`) | **Secondary Surgical Targets** (`down_proj`) |
| **Routed Experts** | `256` per layer (`top_k=8`) | **100% Frozen** (Preserves parametric code memory) |

---

## Quickstart & CLI Verification

```bash
# 1. Inspect the 40-layer hybrid Gated-DeltaNet / Full-Attention MoE circuit map
python -m circuitforge inspect-wiring --profile Qwen3.6-35B-A3B

# 2. Generate the surgical target manifest for mid-to-late Full-Attention anchors & Shared Experts
python -m circuitforge plan-surgery --profile Qwen3.6-35B-A3B

# 3. Run the mathematical verification of Null-Space Rank-1 Weight Surgery + Autograd
python -m circuitforge verify-surgery

# 4. Benchmark the Native C++ AVX2 SIMD Tensor Engine vs. NumPy
python -m circuitforge benchmark-engine

# 5. Export a standalone interactive Mechanistic Telemetry HUD HTML snapshot
python -m circuitforge export-hud --output hud_snapshot.html --profile Qwen3.6-35B-A3B

# 6. Verify the MoE Anomaly Router & Activation Spike Sentinel
python -m circuitforge verify-sentinel

# 7. Verify the Discrete Engram Memory Graph & Parametric Adapter Manifold
python -m circuitforge verify-engram

# 8. Automatically sweep candidate layers for Direct Logit Attribution (DLA) & Null-Space SLR
python -m circuitforge discover-circuit --checkpoint model.safetensors --profile Qwen3.6-35B-A3B --top-k 2 --output-recipe auto.forge.json

# 9. Export surgical weight deltas between base and forged checkpoints as a Hugging Face PEFT LoRA adapter
python -m circuitforge export-lora --base-checkpoint base.safetensors --forged-checkpoint forged.safetensors --output-dir ./lora_adapter --rank 4

# 10. Run the full pytest suite (62 tests across CFM-01 through CFM-12 + adversarial audits)
pytest -v
```

## License

Apache 2.0
