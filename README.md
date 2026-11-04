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
3. **Verifiable Multi-Turn Agent RL Environment (`circuitforge/rl_env.py`):**
   - Gymnasium / `verifiers`-compatible RLVR environment (`VerifiableCodingEnv`) with deterministic step-level process rewards and invariant safety gates for `GRPO` post-training.
4. **Clean-Room NumPy Reverse-Mode Autograd (`circuitforge/autograd.py`):**
   - Pure-NumPy automatic differentiation engine (`MatMul`, `RMSNorm`, fused `Softmax+CrossEntropy`, and Multi-Head Causal Self-Attention) for surgical projection verification and gradient checks.

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

# 4. Run the full pytest suite (wiring maps, null-space surgery, RLVR environment, and autograd)
pytest -v
```

## License

Apache 2.0
