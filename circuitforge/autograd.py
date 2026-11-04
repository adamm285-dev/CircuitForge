"""Clean-room Pure-NumPy Tensor Autograd & Causal Transformer Primitives (`circuitforge.autograd`).

Implements:
1. `Tensor`: Reverse-mode automatic differentiation in pure NumPy with topological DFS backward
   pass, fan-out gradient accumulation (`+=`), and general N-D shape unbroadcasting (`unbroadcast`).
2. Core Differentiable Circuit Operations:
   - Batched Matrix Multiplication (`A @ B`) with `np.swapaxes` transpose gradients
   - Numerically stable fused `softmax_cross_entropy`
   - `RMSNorm` (Root-Mean-Square Layer Normalization, matching Qwen3.5-MoE)
   - Multi-Head Causal Self-Attention (`causal_self_attention`)
"""

from __future__ import annotations

from typing import Callable

import numpy as np


def unbroadcast(grad: np.ndarray, target_shape: tuple[int, ...]) -> np.ndarray:
    """Sum out leading or singleton dimensions added by NumPy forward broadcasting."""
    g = np.asarray(grad, dtype=np.float64)
    while g.ndim > len(target_shape):
        g = g.sum(axis=0)
    for axis, dim in enumerate(target_shape):
        if dim == 1 and g.shape[axis] != 1:
            g = g.sum(axis=axis, keepdims=True)
    return g


class Tensor:
    """N-dimensional reverse-mode autograd tensor in pure NumPy."""

    def __init__(
        self,
        data: np.ndarray | list | float,
        *,
        requires_grad: bool = False,
        _children: tuple[Tensor, ...] = (),
        _op: str = "",
    ) -> None:
        self.data: np.ndarray = np.asarray(data, dtype=np.float64)
        self.requires_grad: bool = requires_grad
        self.grad: np.ndarray = np.zeros_like(self.data, dtype=np.float64)
        self._prev: set[Tensor] = set(_children)
        self._op: str = _op
        self._backward: Callable[[], None] = lambda: None

    @property
    def shape(self) -> tuple[int, ...]:
        return self.data.shape

    @property
    def ndim(self) -> int:
        return self.data.ndim

    def zero_grad(self) -> None:
        self.grad = np.zeros_like(self.data, dtype=np.float64)

    def _ensure_tensor(self, other: Tensor | np.ndarray | float) -> Tensor:
        return other if isinstance(other, Tensor) else Tensor(other, requires_grad=False)

    def __add__(self, other: Tensor | np.ndarray | float) -> Tensor:
        other_t = self._ensure_tensor(other)
        req = self.requires_grad or other_t.requires_grad
        out = Tensor(self.data + other_t.data, requires_grad=req, _children=(self, other_t), _op="+")

        def _backward() -> None:
            if self.requires_grad:
                self.grad += unbroadcast(out.grad, self.shape)
            if other_t.requires_grad:
                other_t.grad += unbroadcast(out.grad, other_t.shape)

        out._backward = _backward
        return out

    def __radd__(self, other: Tensor | np.ndarray | float) -> Tensor:
        return self + other

    def __neg__(self) -> Tensor:
        out = Tensor(-self.data, requires_grad=self.requires_grad, _children=(self,), _op="neg")

        def _backward() -> None:
            if self.requires_grad:
                self.grad -= out.grad

        out._backward = _backward
        return out

    def __sub__(self, other: Tensor | np.ndarray | float) -> Tensor:
        other_t = self._ensure_tensor(other)
        req = self.requires_grad or other_t.requires_grad
        out = Tensor(self.data - other_t.data, requires_grad=req, _children=(self, other_t), _op="-")

        def _backward() -> None:
            if self.requires_grad:
                self.grad += unbroadcast(out.grad, self.shape)
            if other_t.requires_grad:
                other_t.grad -= unbroadcast(out.grad, other_t.shape)

        out._backward = _backward
        return out

    def __rsub__(self, other: Tensor | np.ndarray | float) -> Tensor:
        return self._ensure_tensor(other) - self

    def __mul__(self, other: Tensor | np.ndarray | float) -> Tensor:
        other_t = self._ensure_tensor(other)
        req = self.requires_grad or other_t.requires_grad
        out = Tensor(self.data * other_t.data, requires_grad=req, _children=(self, other_t), _op="*")

        def _backward() -> None:
            if self.requires_grad:
                self.grad += unbroadcast(out.grad * other_t.data, self.shape)
            if other_t.requires_grad:
                other_t.grad += unbroadcast(out.grad * self.data, other_t.shape)

        out._backward = _backward
        return out

    def __rmul__(self, other: Tensor | np.ndarray | float) -> Tensor:
        return self * other

    def __matmul__(self, other: Tensor) -> Tensor:
        other_t = self._ensure_tensor(other)
        req = self.requires_grad or other_t.requires_grad
        out = Tensor(self.data @ other_t.data, requires_grad=req, _children=(self, other_t), _op="@")

        def _backward() -> None:
            if self.requires_grad:
                bt = np.swapaxes(other_t.data, -1, -2)
                self.grad += unbroadcast(out.grad @ bt, self.shape)
            if other_t.requires_grad:
                at = np.swapaxes(self.data, -1, -2)
                other_t.grad += unbroadcast(at @ out.grad, other_t.shape)

        out._backward = _backward
        return out

    def relu(self) -> Tensor:
        mask = self.data > 0.0
        out = Tensor(np.where(mask, self.data, 0.0), requires_grad=self.requires_grad, _children=(self,), _op="relu")

        def _backward() -> None:
            if self.requires_grad:
                self.grad += out.grad * mask

        out._backward = _backward
        return out

    def silu(self) -> Tensor:
        sig = 1.0 / (1.0 + np.exp(-np.clip(self.data, -60.0, 60.0)))
        out = Tensor(self.data * sig, requires_grad=self.requires_grad, _children=(self,), _op="silu")

        def _backward() -> None:
            if self.requires_grad:
                grad_silu = sig * (1.0 + self.data * (1.0 - sig))
                self.grad += out.grad * grad_silu

        out._backward = _backward
        return out

    def sum(self, axis: int | tuple[int, ...] | None = None, keepdims: bool = False) -> Tensor:
        out_data = self.data.sum(axis=axis, keepdims=keepdims)
        out = Tensor(out_data, requires_grad=self.requires_grad, _children=(self,), _op="sum")

        def _backward() -> None:
            if not self.requires_grad:
                return
            g = out.grad
            if not keepdims and axis is not None:
                axes = (axis,) if isinstance(axis, int) else tuple(sorted(axis))
                for ax in axes:
                    g = np.expand_dims(g, axis=ax)
            self.grad += np.broadcast_to(g, self.shape)

        out._backward = _backward
        return out

    def mean(self, axis: int | tuple[int, ...] | None = None, keepdims: bool = False) -> Tensor:
        if axis is None:
            count = float(self.data.size)
        elif isinstance(axis, int):
            count = float(self.shape[axis])
        else:
            count = float(np.prod([self.shape[a] for a in axis]))
        return self.sum(axis=axis, keepdims=keepdims) * (1.0 / count)

    def transpose_last_two(self) -> Tensor:
        out = Tensor(
            np.swapaxes(self.data, -1, -2),
            requires_grad=self.requires_grad,
            _children=(self,),
            _op="swapaxes",
        )

        def _backward() -> None:
            if self.requires_grad:
                self.grad += np.swapaxes(out.grad, -1, -2)

        out._backward = _backward
        return out

    def reshape(self, new_shape: tuple[int, ...]) -> Tensor:
        out = Tensor(
            self.data.reshape(new_shape),
            requires_grad=self.requires_grad,
            _children=(self,),
            _op="reshape",
        )

        def _backward() -> None:
            if self.requires_grad:
                self.grad += out.grad.reshape(self.shape)

        out._backward = _backward
        return out

    def transpose(self, axes: tuple[int, ...]) -> Tensor:
        out = Tensor(
            np.transpose(self.data, axes),
            requires_grad=self.requires_grad,
            _children=(self,),
            _op="transpose",
        )
        inv_axes = tuple(np.argsort(axes))

        def _backward() -> None:
            if self.requires_grad:
                self.grad += np.transpose(out.grad, inv_axes)

        out._backward = _backward
        return out

    def backward(self) -> None:
        """Run reverse-mode automatic differentiation from this scalar or tensor output."""
        topo: list[Tensor] = []
        visited: set[Tensor] = set()

        def build_topo(v: Tensor) -> None:
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)
        self.grad = np.ones_like(self.data, dtype=np.float64)
        for node in reversed(topo):
            node._backward()


def softmax(x: Tensor, axis: int = -1) -> Tensor:
    """Numerically stable softmax with exact vector-Jacobian product backward pass."""
    shifted = x.data - np.max(x.data, axis=axis, keepdims=True)
    exps = np.exp(shifted)
    probs = exps / np.sum(exps, axis=axis, keepdims=True)
    out = Tensor(probs, requires_grad=x.requires_grad, _children=(x,), _op="softmax")

    def _backward() -> None:
        if x.requires_grad:
            dot = np.sum(probs * out.grad, axis=axis, keepdims=True)
            x.grad += probs * (out.grad - dot)

    out._backward = _backward
    return out


def softmax_cross_entropy(logits: Tensor, targets: np.ndarray) -> Tensor:
    """
    Numerically stable fused Softmax + Cross-Entropy loss over 2D `(N, C)` or 3D `(B, T, C)` logits.
    Analytical gradient: `d_logits = (probs - one_hot(targets)) / N`.
    """
    flat_logits = logits.data.reshape(-1, logits.shape[-1])
    flat_targets = np.asarray(targets, dtype=np.int64).reshape(-1)
    n_samples = flat_logits.shape[0]

    shifted = flat_logits - np.max(flat_logits, axis=-1, keepdims=True)
    log_sum_exp = np.log(np.sum(np.exp(shifted), axis=-1, keepdims=True))
    log_probs = shifted - log_sum_exp
    probs = np.exp(log_probs)

    loss_val = -np.mean(log_probs[np.arange(n_samples), flat_targets])
    out = Tensor(loss_val, requires_grad=logits.requires_grad, _children=(logits,), _op="softmax_ce")

    def _backward() -> None:
        if logits.requires_grad:
            d_flat = probs.copy()
            d_flat[np.arange(n_samples), flat_targets] -= 1.0
            d_flat /= float(n_samples)
            logits.grad += (d_flat * out.grad).reshape(logits.shape)

    out._backward = _backward
    return out


def rms_norm(x: Tensor, weight: Tensor, eps: float = 1e-6) -> Tensor:
    """
    Root-Mean-Square Layer Normalization (`RMSNorm`), with exact analytical backward pass:
    `y = (x / rms) * weight`, where `rms = sqrt(mean(x^2, axis=-1) + eps)`.
    """
    ms = np.mean(np.square(x.data), axis=-1, keepdims=True)
    inv_rms = 1.0 / np.sqrt(ms + eps)
    x_hat = x.data * inv_rms
    out_data = x_hat * weight.data
    req = x.requires_grad or weight.requires_grad
    out = Tensor(out_data, requires_grad=req, _children=(x, weight), _op="rms_norm")

    def _backward() -> None:
        dy = out.grad
        if weight.requires_grad:
            weight.grad += unbroadcast(dy * x_hat, weight.shape)
        if x.requires_grad:
            dx_hat = dy * weight.data
            mean_xhat_dxhat = np.mean(x_hat * dx_hat, axis=-1, keepdims=True)
            dx = inv_rms * (dx_hat - x_hat * mean_xhat_dxhat)
            x.grad += dx

    out._backward = _backward
    return out


def causal_self_attention(
    x: Tensor,
    wq: Tensor,
    wk: Tensor,
    wv: Tensor,
    wo: Tensor,
    num_heads: int = 2,
) -> Tensor:
    """
    Multi-Head Causal Self-Attention over input `x` of shape `(B, T, D)`.
    All projections and attention softmax flow through `Tensor` autograd operations.
    """
    bsz, seq_len, d_model = x.shape
    assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
    head_dim = d_model // num_heads

    q = (x @ wq).reshape((bsz, seq_len, num_heads, head_dim)).transpose((0, 2, 1, 3))
    k = (x @ wk).reshape((bsz, seq_len, num_heads, head_dim)).transpose((0, 2, 1, 3))
    v = (x @ wv).reshape((bsz, seq_len, num_heads, head_dim)).transpose((0, 2, 1, 3))

    scale = 1.0 / np.sqrt(float(head_dim))
    scores = (q @ k.transpose_last_two()) * scale

    causal_mask = np.triu(np.full((seq_len, seq_len), -1e9, dtype=np.float64), k=1)
    masked_scores = scores + causal_mask.reshape((1, 1, seq_len, seq_len))

    attn_weights = softmax(masked_scores, axis=-1)
    context = (attn_weights @ v).transpose((0, 2, 1, 3)).reshape((bsz, seq_len, d_model))
    return context @ wo
