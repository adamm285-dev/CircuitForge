"""Finite-difference gradient checks and QAT vs. PTQ verification for `circuitforge.autograd_qat`."""

from __future__ import annotations

from typing import Callable

import numpy as np

from circuitforge.autograd_qat import (
    Tensor,
    causal_self_attention,
    rms_norm,
    run_qat_vs_ptq_experiment,
    softmax_cross_entropy,
)


def numerical_grad(fn: Callable[[], Tensor], param: Tensor, eps: float = 1e-6) -> np.ndarray:
    """Compute central finite-difference gradient `(f(x + eps) - f(x - eps)) / (2 * eps)`."""
    grad_num = np.zeros_like(param.data, dtype=np.float64)
    it = np.nditer(param.data, flags=["multi_index"], op_flags=["readwrite"])
    while not it.finished:
        idx = it.multi_index
        orig = float(param.data[idx])
        param.data[idx] = orig + eps
        f_plus = float(fn().data)
        param.data[idx] = orig - eps
        f_minus = float(fn().data)
        param.data[idx] = orig
        grad_num[idx] = (f_plus - f_minus) / (2.0 * eps)
        it.iternext()
    return grad_num


def test_matmul_broadcast_and_fanout_gradients() -> None:
    """Verify batched matmul, fan-out (`x + x * x`), and bias broadcasting gradients against finite differences."""
    rng = np.random.default_rng(101)
    a = Tensor(rng.standard_normal((2, 4, 3)), requires_grad=True)
    b = Tensor(rng.standard_normal((2, 3, 5)), requires_grad=True)
    bias = Tensor(rng.standard_normal((1, 5)), requires_grad=True)

    def forward() -> Tensor:
        # Includes fan-out (`a` used twice) and shape broadcasting (`(2, 4, 5) + (1, 5)`)
        a_combo = a + (a * 0.5)
        y = (a_combo @ b) + bias
        return y.silu().mean()

    loss = forward()
    loss.backward()

    for param in (a, b, bias):
        g_num = numerical_grad(forward, param)
        np.testing.assert_allclose(param.grad, g_num, rtol=1e-5, atol=1e-6)


def test_rms_norm_and_softmax_ce_gradients() -> None:
    """Verify RMSNorm and fused Softmax + Cross-Entropy analytical gradients against finite differences."""
    rng = np.random.default_rng(202)
    x = Tensor(rng.standard_normal((3, 4, 6)), requires_grad=True)
    gamma = Tensor(rng.uniform(0.5, 1.5, size=(6,)), requires_grad=True)
    w_proj = Tensor(rng.standard_normal((6, 5)), requires_grad=True)
    targets = rng.integers(0, 5, size=(3, 4))

    def forward() -> Tensor:
        normed = rms_norm(x, gamma)
        logits = normed @ w_proj
        return softmax_cross_entropy(logits, targets)

    loss = forward()
    loss.backward()

    for param in (x, gamma, w_proj):
        g_num = numerical_grad(forward, param)
        np.testing.assert_allclose(param.grad, g_num, rtol=1e-5, atol=1e-6)


def test_causal_self_attention_gradients() -> None:
    """Verify Multi-Head Causal Self-Attention analytical backward pass against finite differences."""
    rng = np.random.default_rng(303)
    bsz, seq_len, d_model, n_heads = 2, 4, 6, 2
    x = Tensor(rng.standard_normal((bsz, seq_len, d_model)), requires_grad=True)
    wq = Tensor(rng.standard_normal((d_model, d_model)) * 0.4, requires_grad=True)
    wk = Tensor(rng.standard_normal((d_model, d_model)) * 0.4, requires_grad=True)
    wv = Tensor(rng.standard_normal((d_model, d_model)) * 0.4, requires_grad=True)
    wo = Tensor(rng.standard_normal((d_model, d_model)) * 0.4, requires_grad=True)

    def forward() -> Tensor:
        out = causal_self_attention(x, wq, wk, wv, wo, num_heads=n_heads)
        return (out * out).mean()

    loss = forward()
    loss.backward()

    for param in (x, wq, wk, wv, wo):
        g_num = numerical_grad(forward, param)
        np.testing.assert_allclose(param.grad, g_num, rtol=1e-5, atol=1e-6)


def test_tritforge_qat_recovers_158bit_ptq_drop() -> None:
    """Verify 1.58-bit STE Quantization-Aware Training recovers accuracy lost by zero-shot ternary PTQ."""
    metrics = run_qat_vs_ptq_experiment(seed=7, steps=120)
    assert metrics["qat_158bit_loss"] < metrics["ptq_158bit_loss"]
    assert metrics["qat_158bit_accuracy"] > metrics["ptq_158bit_accuracy"]
    assert metrics["qat_158bit_accuracy"] >= 0.85
