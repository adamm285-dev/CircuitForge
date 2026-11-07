"""
CircuitForge Native C++ Zero-Copy Tensor Engine & Telemetry Bridge (`circuitforge/native.py`).

Provides zero-copy `ctypes` bindings to the compiled native C++ shared library
(`circuitforge_engine.dll` / `libcircuitforge_engine.so` built from `src/engine/`):
1. **Zero-Copy Memory-Mapped `.safetensors` Mutation (`NativeMmapCheckpoint`)**:
   Maps `.safetensors` files directly into virtual memory via `MapViewOfFile` / `mmap`
   and exposes zero-copy writable NumPy views directly over disk-backed tensor slices
   while enforcing the Routed Expert Protection Gate.
2. **High-Velocity C++ SIMD Kernels**:
   - `native_rank1_update_f32`
   - `native_nullspace_rank1_surgery_f32`
   - `native_directional_ablation_f32`
   - `native_rmsnorm_matvec_f32`
   - `native_router_entropy_f32`
3. **Lock-Free SPSC Telemetry Ring Buffer (`NativeTelemetryRingBuffer`)**:
   Zero-allocation C++ atomic ring buffer for streaming layer activation norms,
   router entropy metrics, and expert selection distributions.
"""

from __future__ import annotations

import ctypes
import math
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import weakref

import numpy as np

from circuitforge.io import inspect_safetensors_header
from circuitforge.surgery import SurgicalEditReport
from circuitforge.wiring import is_routed_expert_tensor


class _CFTelemetryEventStruct(ctypes.Structure):
    _fields_ = [
        ("timestamp_ns", ctypes.c_uint64),
        ("step_id", ctypes.c_uint32),
        ("layer_idx", ctypes.c_uint32),
        ("activation_l2_norm", ctypes.c_float),
        ("router_entropy", ctypes.c_float),
        ("steering_alpha", ctypes.c_float),
        ("top_expert_id", ctypes.c_uint32),
        ("top_expert_weight", ctypes.c_float),
        ("anomaly_flag", ctypes.c_uint32),
    ]


@dataclass(frozen=True)
class TelemetryEvent:
    """High-speed mechanistic telemetry event emitted per layer/step."""

    timestamp_ns: int
    step_id: int
    layer_idx: int
    activation_l2_norm: float
    router_entropy: float
    steering_alpha: float
    top_expert_id: int
    top_expert_weight: float
    anomaly_flag: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp_ns": self.timestamp_ns,
            "step_id": self.step_id,
            "layer_idx": self.layer_idx,
            "activation_l2_norm": round(self.activation_l2_norm, 6),
            "router_entropy": round(self.router_entropy, 6),
            "steering_alpha": round(self.steering_alpha, 6),
            "top_expert_id": self.top_expert_id,
            "top_expert_weight": round(self.top_expert_weight, 6),
            "anomaly_flag": self.anomaly_flag,
        }


def _load_native_dll() -> ctypes.CDLL | None:
    pkg_dir = Path(__file__).resolve().parent
    candidates = [
        pkg_dir / "circuitforge_engine.dll",
        pkg_dir / "libcircuitforge_engine.so",
        pkg_dir / "libcircuitforge_engine.dylib",
    ]
    for cand in candidates:
        if cand.is_file():
            dll = ctypes.CDLL(str(cand))
            # Ring buffer signatures
            dll.cf_ringbuf_create.argtypes = [ctypes.c_uint32]
            dll.cf_ringbuf_create.restype = ctypes.c_void_p
            dll.cf_ringbuf_destroy.argtypes = [ctypes.c_void_p]
            dll.cf_ringbuf_destroy.restype = None
            dll.cf_ringbuf_push.argtypes = [ctypes.c_void_p, ctypes.POINTER(_CFTelemetryEventStruct)]
            dll.cf_ringbuf_push.restype = ctypes.c_int
            dll.cf_ringbuf_pop.argtypes = [ctypes.c_void_p, ctypes.POINTER(_CFTelemetryEventStruct)]
            dll.cf_ringbuf_pop.restype = ctypes.c_int
            dll.cf_ringbuf_size.argtypes = [ctypes.c_void_p]
            dll.cf_ringbuf_size.restype = ctypes.c_uint32
            dll.cf_ringbuf_capacity.argtypes = [ctypes.c_void_p]
            dll.cf_ringbuf_capacity.restype = ctypes.c_uint32

            # Mmap signatures
            dll.cf_mmap_open.argtypes = [ctypes.c_char_p, ctypes.c_int]
            dll.cf_mmap_open.restype = ctypes.c_void_p
            dll.cf_mmap_data.argtypes = [ctypes.c_void_p]
            dll.cf_mmap_data.restype = ctypes.c_void_p
            dll.cf_mmap_size.argtypes = [ctypes.c_void_p]
            dll.cf_mmap_size.restype = ctypes.c_size_t
            dll.cf_mmap_flush.argtypes = [ctypes.c_void_p]
            dll.cf_mmap_flush.restype = ctypes.c_int
            dll.cf_mmap_close.argtypes = [ctypes.c_void_p]
            dll.cf_mmap_close.restype = None

            # Tensor & Surgery SIMD kernel signatures
            f32_p = ctypes.POINTER(ctypes.c_float)
            u32_p = ctypes.POINTER(ctypes.c_uint32)

            dll.cf_rank1_update_f32.argtypes = [
                f32_p, ctypes.c_uint32, ctypes.c_uint32, f32_p, f32_p, ctypes.c_float
            ]
            dll.cf_rank1_update_f32.restype = ctypes.c_float

            dll.cf_nullspace_rank1_surgery_f32.argtypes = [
                f32_p,
                ctypes.c_uint32,
                ctypes.c_uint32,
                f32_p,
                f32_p,
                f32_p,
                ctypes.c_uint32,
                ctypes.c_float,
                f32_p,
                f32_p,
            ]
            dll.cf_nullspace_rank1_surgery_f32.restype = ctypes.c_float

            dll.cf_directional_ablation_f32.argtypes = [
                f32_p, ctypes.c_uint32, ctypes.c_uint32, f32_p, ctypes.c_float
            ]
            dll.cf_directional_ablation_f32.restype = ctypes.c_float

            dll.cf_rmsnorm_matvec_f32.argtypes = [
                f32_p, f32_p, f32_p, f32_p, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_float
            ]
            dll.cf_rmsnorm_matvec_f32.restype = None

            dll.cf_router_entropy_f32.argtypes = [
                f32_p, ctypes.c_uint32, u32_p, f32_p
            ]
            dll.cf_router_entropy_f32.restype = ctypes.c_float

            # Offloaded Discrete Engram Hash Table signatures (CFM-06)
            dll.cf_engram_create.argtypes = [ctypes.c_uint32, ctypes.c_uint32]
            dll.cf_engram_create.restype = ctypes.c_void_p
            dll.cf_engram_destroy.argtypes = [ctypes.c_void_p]
            dll.cf_engram_destroy.restype = None
            dll.cf_fnv1a_hash64.argtypes = [ctypes.c_char_p, ctypes.c_size_t]
            dll.cf_fnv1a_hash64.restype = ctypes.c_uint64
            dll.cf_engram_Upsert.argtypes = [ctypes.c_void_p, ctypes.c_uint64, f32_p]
            dll.cf_engram_Upsert.restype = ctypes.c_int
            dll.cf_engram_lookup_and_inject_f32.argtypes = [
                ctypes.c_void_p,
                ctypes.c_uint64,
                f32_p,
                f32_p,
                f32_p,
                ctypes.c_float,
                ctypes.c_float,
            ]
            dll.cf_engram_lookup_and_inject_f32.restype = ctypes.c_int
            dll.cf_engram_count.argtypes = [ctypes.c_void_p]
            dll.cf_engram_count.restype = ctypes.c_uint32
            return dll
    return None


_DLL = _load_native_dll()


def has_native_engine() -> bool:
    """Returns True if the compiled C++ SIMD engine shared library is loaded."""
    return _DLL is not None


class NativeTelemetryRingBuffer:
    """
    Zero-allocation lock-free Single-Producer Single-Consumer (SPSC) telemetry ring buffer
    backed by C++ `std::atomic` slots.
    """

    def __init__(self, capacity: int = 1024) -> None:
        if _DLL is None:
            raise RuntimeError("Native C++ engine DLL is not loaded.")
        self._handle = _DLL.cf_ringbuf_create(int(capacity))
        if not self._handle:
            raise MemoryError("Failed to allocate C++ CFTelemetryRingBuffer.")

    def close(self) -> None:
        if getattr(self, "_handle", None) and _DLL is not None:
            _DLL.cf_ringbuf_destroy(self._handle)
            self._handle = None

    def __del__(self) -> None:
        self.close()

    @property
    def capacity(self) -> int:
        return int(_DLL.cf_ringbuf_capacity(self._handle)) if self._handle else 0

    def __len__(self) -> int:
        return int(_DLL.cf_ringbuf_size(self._handle)) if self._handle else 0

    def push(self, event: TelemetryEvent) -> bool:
        c_ev = _CFTelemetryEventStruct(
            timestamp_ns=int(event.timestamp_ns),
            step_id=int(event.step_id),
            layer_idx=int(event.layer_idx),
            activation_l2_norm=float(event.activation_l2_norm),
            router_entropy=float(event.router_entropy),
            steering_alpha=float(event.steering_alpha),
            top_expert_id=int(event.top_expert_id),
            top_expert_weight=float(event.top_expert_weight),
            anomaly_flag=int(event.anomaly_flag),
        )
        return bool(_DLL.cf_ringbuf_push(self._handle, ctypes.byref(c_ev)))

    def pop(self) -> TelemetryEvent | None:
        c_ev = _CFTelemetryEventStruct()
        ok = _DLL.cf_ringbuf_pop(self._handle, ctypes.byref(c_ev))
        if not ok:
            return None
        return TelemetryEvent(
            timestamp_ns=int(c_ev.timestamp_ns),
            step_id=int(c_ev.step_id),
            layer_idx=int(c_ev.layer_idx),
            activation_l2_norm=float(c_ev.activation_l2_norm),
            router_entropy=float(c_ev.router_entropy),
            steering_alpha=float(c_ev.steering_alpha),
            top_expert_id=int(c_ev.top_expert_id),
            top_expert_weight=float(c_ev.top_expert_weight),
            anomaly_flag=int(c_ev.anomaly_flag),
        )

    def drain_all(self) -> list[TelemetryEvent]:
        events: list[TelemetryEvent] = []
        while True:
            ev = self.pop()
            if ev is None:
                break
            events.append(ev)
        return events


class NativeMmapCheckpoint:
    """
    Zero-copy memory-mapped `.safetensors` checkpoint engine powered by C++ `MapViewOfFile` / `mmap`.
    Allows direct in-place weight surgery on disk-mapped pages in microseconds without copying
    tensors into Python heap memory.
    """

    def __init__(self, path: str | Path, writable: bool = True) -> None:
        if _DLL is None:
            raise RuntimeError("Native C++ engine DLL is not loaded.")
        self.path = Path(path).resolve()
        self.writable = writable
        self.entries, self.metadata, self.data_start = inspect_safetensors_header(self.path)
        self._handle = _DLL.cf_mmap_open(str(self.path).encode("utf-8"), 1 if writable else 0)
        if not self._handle:
            raise OSError(f"Failed to memory-map file: {self.path}")
        self._base_ptr = _DLL.cf_mmap_data(self._handle)
        self._size = int(_DLL.cf_mmap_size(self._handle))
        self._live_buf_refs: list[weakref.ref[Any]] = []
        self._closed: bool = False

    def _finalize_unmap(self) -> None:
        if getattr(self, "_handle", None) and _DLL is not None:
            _DLL.cf_mmap_close(self._handle)
            self._handle = None
            self._base_ptr = None

    def close(self) -> None:
        if self.writable and self._handle:
            self.flush()
        self._closed = True
        live = [r for r in self._live_buf_refs if r() is not None]
        self._live_buf_refs = live
        if not live:
            self._finalize_unmap()

    def __del__(self) -> None:
        self._finalize_unmap()

    def flush(self) -> bool:
        if self._closed or not self._handle or _DLL is None:
            return False
        return bool(_DLL.cf_mmap_flush(self._handle))

    def __enter__(self) -> NativeMmapCheckpoint:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if self.writable:
            self.flush()
        self.close()

    def get_tensor_view(self, tensor_name: str, for_write: bool = False) -> np.ndarray:
        """
        Returns a zero-copy NumPy array backed directly by the OS memory-mapped page when `for_write=True`,
        or a safe read-only snapshot when `for_write=False`.
        Strictly enforces the CircuitForge Routed Expert Protection Gate and validates file bounds.
        """
        if self._closed or not self._handle or not self._base_ptr:
            raise RuntimeError("Cannot obtain a tensor view from a closed NativeMmapCheckpoint.")
        if for_write:
            if not self.writable:
                raise PermissionError("Checkpoint was opened read-only.")
            if is_routed_expert_tensor(tensor_name):
                raise PermissionError(
                    f"CircuitForge Safety Gate: Refusing writable zero-copy view on routed expert '{tensor_name}'."
                )
        if tensor_name not in self.entries:
            raise KeyError(f"Tensor '{tensor_name}' not found in {self.path}")

        entry = self.entries[tensor_name]
        if entry.dtype != "F32":
            raise ValueError(
                f"Direct zero-copy F32 view requires dtype='F32', got '{entry.dtype}' for '{tensor_name}'"
            )
        start_off, end_off = entry.data_offsets
        if start_off < 0 or end_off < start_off:
            raise ValueError(f"Invalid data_offsets {entry.data_offsets} for tensor '{tensor_name}'")
        n_elems = int(np.prod(entry.shape)) if entry.shape else 1
        expected_bytes = n_elems * 4
        if (end_off - start_off) != expected_bytes:
            raise ValueError(
                f"Safetensors header mismatch for '{tensor_name}': shape {entry.shape} requires "
                f"{expected_bytes} bytes, but data_offsets span {end_off - start_off} bytes."
            )
        offset = self.data_start + start_off
        if offset + expected_bytes > self._size:
            raise ValueError(
                f"Safetensors tensor '{tensor_name}' extends past end of mapped file "
                f"({offset + expected_bytes} > {self._size})."
            )
        c_arr_type = ctypes.c_float * n_elems
        c_arr = c_arr_type.from_address(int(self._base_ptr) + offset)
        raw_view = np.ctypeslib.as_array(c_arr).reshape(entry.shape)
        if not for_write:
            # Return an isolated read-only copy so callers cannot set `view.flags.writeable = True`
            # to mutate frozen routed experts on disk or trigger use-after-unmap segfaults.
            snap = raw_view.copy()
            snap.flags.writeable = False
            return snap

        # Track the underlying `c_arr` ctypes buffer via weakref (NOT put in a strong list on self),
        # because every derived NumPy slice (`v[1:]`, `v.T`, `v.ravel()`) holds a strong reference to `c_arr`
        # via its `.base` chain. Thus `c_arr` is garbage-collected iff ALL derived views are gone.
        c_arr._mmap_owner = self  # type: ignore[attr-defined]

        def _on_buf_gc(_ref: Any) -> None:
            self._live_buf_refs = [r for r in self._live_buf_refs if r() is not None]
            if self._closed and not self._live_buf_refs:
                self._finalize_unmap()

        self._live_buf_refs.append(weakref.ref(c_arr, _on_buf_gc))
        return raw_view

    def inplace_nullspace_rank1_surgery(
        self,
        tensor_name: str,
        trigger_key: np.ndarray,
        steering_vector: np.ndarray,
        preserved_keys: np.ndarray | None = None,
        alpha: float = 1.0,
        layer_idx: int = 3,
    ) -> SurgicalEditReport:
        """
        Executes C++ AVX2 SIMD Null-Space Rank-1 Surgery directly on the memory-mapped tensor page.
        """
        view = self.get_tensor_view(tensor_name, for_write=True)
        _, report = native_nullspace_rank1_surgery_f32(
            weight=view,
            trigger_key=trigger_key,
            steering_vector=steering_vector,
            preserved_keys=preserved_keys,
            alpha=alpha,
            tensor_name=tensor_name,
            layer_idx=layer_idx,
            in_place=True,
        )
        return report


def native_nullspace_rank1_surgery_f32(
    weight: np.ndarray,
    trigger_key: np.ndarray,
    steering_vector: np.ndarray,
    preserved_keys: np.ndarray | None = None,
    alpha: float = 1.0,
    tensor_name: str = "model.layers.3.self_attn.o_proj.weight",
    layer_idx: int = 3,
    in_place: bool = False,
) -> tuple[np.ndarray, SurgicalEditReport]:
    """
    Calls the C++ AVX2 SIMD `cf_nullspace_rank1_surgery_f32` kernel.
    """
    if is_routed_expert_tensor(tensor_name):
        raise PermissionError(
            f"CircuitForge Safety Gate: Refusing to modify banked routed MoE expert tensor '{tensor_name}'."
        )
    if _DLL is None:
        raise RuntimeError("Native C++ engine DLL is not loaded.")

    if not math.isfinite(float(alpha)):
        raise ValueError("alpha must be finite (non-NaN, non-Inf).")
    if not np.all(np.isfinite(weight)):
        raise ValueError("weight matrix contains NaN or Inf values.")
    if not np.all(np.isfinite(trigger_key)):
        raise ValueError("trigger_key contains NaN or Inf values.")
    if not np.all(np.isfinite(steering_vector)):
        raise ValueError("steering_vector contains NaN or Inf values.")

    if in_place:
        if weight.dtype != np.float32 or not weight.flags.c_contiguous or not weight.flags.writeable:
            raise ValueError(
                "in_place=True requires a writable, C-contiguous float32 weight array."
            )
        W = weight
    else:
        W = np.array(weight, dtype=np.float32, copy=True, order="C")
    d_out, d_in = W.shape
    k = np.ascontiguousarray(trigger_key, dtype=np.float32).reshape(d_in)
    v = np.ascontiguousarray(steering_vector, dtype=np.float32).reshape(d_out)
    k64 = k.astype(np.float64)
    k_orig_sq = float(np.dot(k64, k64))
    if k_orig_sq <= 1e-12:
        raise ValueError("Trigger key lies entirely inside the preserved key subspace or has near-zero norm; cannot isolate null-space.")

    basis_ptr: Any = None
    basis_rank = 0
    basis_arr: np.ndarray | None = None

    if preserved_keys is not None and len(preserved_keys) > 0:
        if not np.all(np.isfinite(preserved_keys)):
            raise ValueError("preserved_keys contains NaN or Inf values.")
        K0 = np.asarray(preserved_keys, dtype=np.float64)
        _, s, vt = np.linalg.svd(K0, full_matrices=False)
        s_max = float(s[0]) if len(s) > 0 else 0.0
        if s_max > 1e-12:
            s_tol = max(1e-12, s_max * 1e-7)
            nonzero_rank = int(np.sum(s > s_tol))
            basis_rank = min(nonzero_rank, d_in)
            if basis_rank >= d_in:
                raise ValueError("Trigger key lies entirely inside the preserved key subspace; cannot isolate null-space.")
            basis64 = vt[:basis_rank, :]
            k_proj64 = k64 - basis64.T @ (basis64 @ k64)
            if float(np.dot(k_proj64, k_proj64)) <= max(1e-12, 1e-4 * k_orig_sq):
                raise ValueError("Trigger key lies entirely inside the preserved key subspace; cannot isolate null-space.")
            basis_arr = np.ascontiguousarray(vt[:basis_rank, :], dtype=np.float32)
            basis_ptr = basis_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

    frob_base = max(float(np.linalg.norm(W, ord="fro")), 1e-12)
    K0_before = (W @ np.asarray(preserved_keys, dtype=np.float32).T) if (preserved_keys is not None and len(preserved_keys) > 0) else None

    align_before = ctypes.c_float(0.0)
    align_after = ctypes.c_float(0.0)

    frob_delta = float(
        _DLL.cf_nullspace_rank1_surgery_f32(
            W.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ctypes.c_uint32(d_out),
            ctypes.c_uint32(d_in),
            k.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            v.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            basis_ptr,
            ctypes.c_uint32(basis_rank),
            ctypes.c_float(alpha),
            ctypes.byref(align_before),
            ctypes.byref(align_after),
        )
    )

    leakage = 0.0
    if K0_before is not None and preserved_keys is not None:
        K0_after = W @ np.asarray(preserved_keys, dtype=np.float32).T
        leakage = float(np.max(np.abs(K0_after - K0_before)))

    report = SurgicalEditReport(
        tensor_name=tensor_name,
        layer_idx=layer_idx,
        shape=(d_out, d_in),
        alpha=float(alpha),
        frobenius_delta=frob_delta,
        relative_norm_delta=frob_delta / frob_base,
        nullspace_leakage_max=leakage,
        target_alignment_before=float(align_before.value),
        target_alignment_after=float(align_after.value),
    )
    return W, report


def native_directional_ablation_f32(
    weight: np.ndarray,
    refusal_or_target_dir: np.ndarray,
    strength: float = 1.0,
    tensor_name: str = "model.layers.3.self_attn.o_proj.weight",
    layer_idx: int = 3,
    in_place: bool = False,
) -> tuple[np.ndarray, SurgicalEditReport]:
    """
    Calls the C++ AVX2 SIMD `cf_directional_ablation_f32` kernel.
    """
    if is_routed_expert_tensor(tensor_name):
        raise PermissionError(
            f"CircuitForge Safety Gate: Refusing to modify banked routed MoE expert tensor '{tensor_name}'."
        )
    if _DLL is None:
        raise RuntimeError("Native C++ engine DLL is not loaded.")

    if not math.isfinite(float(strength)):
        raise ValueError("strength must be finite (non-NaN, non-Inf).")
    if not np.all(np.isfinite(weight)):
        raise ValueError("weight matrix contains NaN or Inf values.")
    if not np.all(np.isfinite(refusal_or_target_dir)):
        raise ValueError("refusal_or_target_dir contains NaN or Inf values.")

    if in_place:
        if weight.dtype != np.float32 or not weight.flags.c_contiguous or not weight.flags.writeable:
            raise ValueError(
                "in_place=True requires a writable, C-contiguous float32 weight array."
            )
        W = weight
    else:
        W = np.array(weight, dtype=np.float32, copy=True, order="C")
    d_out, d_in = W.shape
    r = np.ascontiguousarray(refusal_or_target_dir, dtype=np.float32).reshape(d_out)
    r_norm = float(np.linalg.norm(r))
    if r_norm < 1e-12:
        raise ValueError("Target ablation direction has near-zero norm.")
    r_unit = r / r_norm

    frob_base = max(float(np.linalg.norm(W, ord="fro")), 1e-12)
    align_before = float(np.linalg.norm(r_unit @ W))

    frob_delta = float(
        _DLL.cf_directional_ablation_f32(
            W.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ctypes.c_uint32(d_out),
            ctypes.c_uint32(d_in),
            r.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ctypes.c_float(strength),
        )
    )
    align_after = float(np.linalg.norm(r_unit @ W))

    report = SurgicalEditReport(
        tensor_name=tensor_name,
        layer_idx=layer_idx,
        shape=(d_out, d_in),
        alpha=float(strength),
        frobenius_delta=frob_delta,
        relative_norm_delta=frob_delta / frob_base,
        nullspace_leakage_max=0.0,
        target_alignment_before=align_before,
        target_alignment_after=align_after,
    )
    return W, report


def native_rmsnorm_matvec_f32(
    weight: np.ndarray,
    x: np.ndarray,
    gamma: np.ndarray,
    eps: float = 1e-6,
) -> np.ndarray:
    """
    Calls the C++ AVX2 SIMD `cf_rmsnorm_matvec_f32` kernel:
    computes `W @ rms_norm(x, gamma, eps)` in a single fused call.
    """
    if _DLL is None:
        raise RuntimeError("Native C++ engine DLL is not loaded.")
    W = np.ascontiguousarray(weight, dtype=np.float32)
    if W.ndim != 2:
        raise ValueError(f"Expected 2D weight matrix, got {W.shape}")
    d_out, d_in = W.shape
    x_arr = np.ascontiguousarray(x, dtype=np.float32).reshape(d_in)
    g_arr = np.ascontiguousarray(gamma, dtype=np.float32).reshape(d_in)
    out = np.zeros(d_out, dtype=np.float32)

    _DLL.cf_rmsnorm_matvec_f32(
        W.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        x_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        g_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        ctypes.c_uint32(d_out),
        ctypes.c_uint32(d_in),
        ctypes.c_float(eps),
    )
    return out


def native_router_entropy_f32(logits: np.ndarray) -> tuple[float, int, float]:
    """
    Computes `(entropy, top_expert_id, top_expert_prob)` via the C++ engine.
    """
    if _DLL is None:
        raise RuntimeError("Native C++ engine DLL is not loaded.")
    arr = np.ascontiguousarray(logits, dtype=np.float32).reshape(-1)
    top_id = ctypes.c_uint32(0)
    top_prob = ctypes.c_float(0.0)
    ent = float(
        _DLL.cf_router_entropy_f32(
            arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ctypes.c_uint32(arr.size),
            ctypes.byref(top_id),
            ctypes.byref(top_prob),
        )
    )
    return ent, int(top_id.value), float(top_prob.value)


def benchmark_native_vs_numpy(
    dim: int = 512,
    iterations: int = 50,
    seed: int = 42,
) -> dict[str, Any]:
    """
    Microbenchmark comparing Python/NumPy vs. C++ AVX2 SIMD in-place rank-1 weight mutation latency.
    Target for Milestone CFM-03: <50 microseconds per C++ rank-1 update on 512x512.
    Uses best-of-5 batches to eliminate OS context-switch jitter.
    """
    rng = np.random.default_rng(seed)
    W_np = rng.standard_normal((dim, dim)).astype(np.float32) * 0.02
    W_cpp = W_np.copy()
    u = rng.standard_normal(dim).astype(np.float32)
    v = rng.standard_normal(dim).astype(np.float32)

    # Warmup & time NumPy outer + add
    t0 = time.perf_counter()
    for _ in range(iterations):
        W_np = W_np + np.float32(0.5) * np.outer(u, v)
    t1 = time.perf_counter()
    numpy_us = ((t1 - t0) / iterations) * 1e6

    cpp_us = -1.0
    if _DLL is not None:
        u_ptr = u.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        v_ptr = v.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        w_ptr = W_cpp.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        # Warmup
        for _ in range(5):
            _DLL.cf_rank1_update_f32(w_ptr, dim, dim, u_ptr, v_ptr, ctypes.c_float(0.0))

        batch_times: list[float] = []
        for _ in range(5):
            t2 = time.perf_counter()
            for _ in range(iterations):
                _DLL.cf_rank1_update_f32(w_ptr, dim, dim, u_ptr, v_ptr, ctypes.c_float(0.5))
            t3 = time.perf_counter()
            batch_times.append(((t3 - t2) / iterations) * 1e6)
        cpp_us = min(batch_times)

    return {
        "matrix_shape": [dim, dim],
        "iterations": iterations,
        "numpy_avg_us": round(numpy_us, 2),
        "cpp_simd_avg_us": round(cpp_us, 2),
        "speedup_x": round(numpy_us / max(cpp_us, 0.01), 2) if cpp_us > 0 else 0.0,
        "meets_50us_target": (0.0 < cpp_us < 50.0),
    }


class NativeEngramHashTable:
    """
    Zero-allocation C++ atomic open-addressing hash table (`CFEngramTable`) storing
    discrete episodic bias vectors in host RAM and injecting them into intermediate
    layer RMSNorm activations via AVX2 SIMD without inflating the KV-cache.
    """

    def __init__(self, capacity: int = 1024, d_model: int = 64) -> None:
        if _DLL is None:
            raise RuntimeError("Native C++ engine DLL is not loaded.")
        self.d_model = int(d_model)
        self._lock = threading.RLock()
        self._handle = _DLL.cf_engram_create(int(capacity), int(d_model))
        if not self._handle:
            raise MemoryError("Failed to allocate C++ CFEngramTable.")

    def close(self) -> None:
        with self._lock:
            if getattr(self, "_handle", None) and _DLL is not None:
                _DLL.cf_engram_destroy(self._handle)
                self._handle = None

    def __del__(self) -> None:
        self.close()

    def __len__(self) -> int:
        with self._lock:
            return int(_DLL.cf_engram_count(self._handle)) if self._handle else 0

    @staticmethod
    def hash_key(key: str | bytes) -> int:
        raw = key.encode("utf-8") if isinstance(key, str) else bytes(key)
        if _DLL is not None:
            return int(_DLL.cf_fnv1a_hash64(raw, len(raw)))
        # Pure-Python FNV-1a 64-bit fallback
        h = 14695981039346656037
        for b in raw:
            h ^= b
            h = (h * 1099511628211) & 0xFFFFFFFFFFFFFFFF
        return h if h != 0 else 1

    def upsert(self, key: str | int, bias_vector: np.ndarray) -> bool:
        if isinstance(key, int):
            if key < 0 or key > 0xFFFFFFFFFFFFFFFF:
                raise ValueError(f"Integer engram key must be in [0, 2^64 - 1], got {key}")
            kh = key if key != 0 else 1
        else:
            kh = self.hash_key(key)
        vec = np.ascontiguousarray(bias_vector, dtype=np.float32).reshape(self.d_model)
        with self._lock:
            if not self._handle:
                raise RuntimeError("NativeEngramHashTable is closed.")
            return bool(
                _DLL.cf_engram_Upsert(
                    self._handle,
                    ctypes.c_uint64(kh),
                    vec.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                )
            )

    def lookup_and_inject_rmsnorm(
        self,
        key: str | int,
        x: np.ndarray,
        gamma: np.ndarray,
        scale: float = 1.0,
        eps: float = 1e-6,
    ) -> tuple[np.ndarray, bool]:
        """
        Executes fused C++ SIMD `RMSNorm(x, gamma) + scale * engram_bias[key]`.
        Returns `(output_vector, hit_boolean)`.
        """
        if isinstance(key, int):
            if key < 0 or key > 0xFFFFFFFFFFFFFFFF:
                raise ValueError(f"Integer engram key must be in [0, 2^64 - 1], got {key}")
            kh = key if key != 0 else 1
        else:
            kh = self.hash_key(key)
        x_arr = np.ascontiguousarray(x, dtype=np.float32).reshape(self.d_model)
        g_arr = np.ascontiguousarray(gamma, dtype=np.float32).reshape(self.d_model)
        out = np.zeros(self.d_model, dtype=np.float32)
        with self._lock:
            if not self._handle:
                raise RuntimeError("NativeEngramHashTable is closed.")
            hit = _DLL.cf_engram_lookup_and_inject_f32(
                self._handle,
                ctypes.c_uint64(kh),
                x_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                g_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                ctypes.c_float(scale),
                ctypes.c_float(eps),
            )
        return out, bool(hit)

