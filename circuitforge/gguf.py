"""
CircuitForge Zero-Dependency GGUF v3 Binary Inspector & Writer (`circuitforge/gguf.py`).

Implements the GGUF interoperability layer of Milestone CFM-08:
1. **Zero-Dependency GGUF v3 Binary Header & Metadata Parser (`inspect_gguf_header`)**:
   Reads the `GGUF` magic header, format version (`v2` / `v3`), tensor count, metadata KV
   pairs (`UINT8..UINT64`, `INT8..INT64`, `FLOAT32`, `FLOAT64`, `BOOL`, `STRING`, `ARRAY`),
   and tensor descriptor tables (`name`, `dimensions`, `ggml_type`, `byte_offset`).
2. **Surgical Control vs. Frozen Routed Expert Classification**:
   Automatically classifies GGUF tensor names (`blk.{L}.attn_output.weight`, `blk.{L}.ffn_down_shexp.weight`
   vs. `blk.{L}.ffn_down_exps.weight`) using CircuitForge's Routed Expert Protection Gate.
3. **Deterministic GGUF v3 Binary Writer (`write_gguf_file`, `load_gguf_tensor_f32`)**:
   Writes valid 32-byte aligned GGUF v3 files (`F32` and `F16` tensors + metadata KV pairs)
   and slices individual float tensors directly from disk without external libraries.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO, Callable

import numpy as np

from circuitforge.io import _bf16_bytes_to_f32, _f32_to_bf16_bytes
from circuitforge.wiring import is_routed_expert_tensor, is_surgical_control_tensor

GGUF_MAGIC = b"GGUF"
GGUF_DEFAULT_ALIGNMENT = 32

# Standard GGML tensor data types
GGML_TYPE_NAMES: dict[int, str] = {
    0: "F32",
    1: "F16",
    2: "Q4_0",
    3: "Q4_1",
    6: "Q5_0",
    7: "Q5_1",
    8: "Q8_0",
    9: "Q8_1",
    10: "Q2_K",
    11: "Q3_K",
    12: "Q4_K",
    13: "Q5_K",
    14: "Q6_K",
    15: "Q8_K",
    16: "IQ2_XXS",
    17: "IQ2_XS",
    18: "IQ3_XXS",
    19: "IQ1_S",
    20: "IQ4_NL",
    21: "IQ3_S",
    22: "IQ2_S",
    23: "IQ4_XS",
    24: "I8",
    25: "I16",
    26: "I32",
    27: "I64",
    28: "F64",
    29: "IQ1_M",
    30: "BF16",
    34: "TQ1_0",
    35: "TQ2_0",
    39: "MXFP4",
}

# Exact (block_elements, block_bytes) for standard ggml types
GGML_BLOCK_SPEC: dict[str, tuple[int, int]] = {
    "F32": (1, 4),
    "F16": (1, 2),
    "BF16": (1, 2),
    "F64": (1, 8),
    "I8": (1, 1),
    "I16": (1, 2),
    "I32": (1, 4),
    "I64": (1, 8),
    "Q4_0": (32, 18),
    "Q4_1": (32, 20),
    "Q5_0": (32, 22),
    "Q5_1": (32, 24),
    "Q8_0": (32, 34),
    "Q8_1": (32, 36),
    "Q2_K": (256, 84),
    "Q3_K": (256, 110),
    "Q4_K": (256, 144),
    "Q5_K": (256, 176),
    "Q6_K": (256, 210),
    "Q8_K": (256, 292),
    "IQ2_XXS": (256, 66),
    "IQ2_XS": (256, 74),
    "IQ3_XXS": (256, 98),
    "IQ1_S": (256, 50),
    "IQ4_NL": (32, 18),
    "IQ3_S": (256, 110),
    "IQ2_S": (256, 82),
    "IQ4_XS": (256, 136),
    "IQ1_M": (256, 56),
    "TQ1_0": (256, 54),
    "TQ2_0": (256, 66),
    "MXFP4": (32, 17),
}

NAME_TO_GGML_TYPE: dict[str, int] = {v: k for k, v in GGML_TYPE_NAMES.items()}

# GGUF Metadata Value Type IDs
GGUF_VAL_UINT8 = 0
GGUF_VAL_INT8 = 1
GGUF_VAL_UINT16 = 2
GGUF_VAL_INT16 = 3
GGUF_VAL_UINT32 = 4
GGUF_VAL_INT32 = 5
GGUF_VAL_FLOAT32 = 6
GGUF_VAL_BOOL = 7
GGUF_VAL_STRING = 8
GGUF_VAL_ARRAY = 9
GGUF_VAL_UINT64 = 10
GGUF_VAL_INT64 = 11
GGUF_VAL_FLOAT64 = 12


@dataclass(frozen=True)
class GGUFTensorInfo:
    """Descriptor for a single tensor inside a `.gguf` model file."""

    name: str
    shape: tuple[int, ...]
    ggml_type_id: int
    ggml_type_name: str
    relative_offset: int
    is_control_tensor: bool
    is_routed_expert: bool

    @property
    def num_elements(self) -> int:
        prod = 1
        for d in self.shape:
            prod *= d
        return prod


def _read_exact(f: BinaryIO, n: int) -> bytes:
    if n < 0:
        raise ValueError(f"Invalid negative GGUF read length: {n}")
    buf = f.read(n)
    if len(buf) != n:
        raise ValueError(f"Truncated GGUF file: expected {n} bytes, got {len(buf)}")
    return buf


def _read_gguf_string(f: BinaryIO) -> str:
    (length,) = struct.unpack("<Q", _read_exact(f, 8))
    if length > 10_000_000:
        raise ValueError(f"GGUF string length {length} exceeds 10MB safety limit")
    return _read_exact(f, length).decode("utf-8", errors="replace")


def _write_gguf_string(f: BinaryIO, s: str) -> None:
    raw = s.encode("utf-8")
    f.write(struct.pack("<Q", len(raw)))
    f.write(raw)


def _read_gguf_value_with_elem_type(
    f: BinaryIO,
    val_type: int,
    _depth: int = 0,
) -> tuple[object, int | None]:
    if _depth > 8:
        raise ValueError("GGUF nested array depth exceeds safety limit (8)")
    if val_type == GGUF_VAL_UINT8:
        return struct.unpack("<B", _read_exact(f, 1))[0], None
    if val_type == GGUF_VAL_INT8:
        return struct.unpack("<b", _read_exact(f, 1))[0], None
    if val_type == GGUF_VAL_UINT16:
        return struct.unpack("<H", _read_exact(f, 2))[0], None
    if val_type == GGUF_VAL_INT16:
        return struct.unpack("<h", _read_exact(f, 2))[0], None
    if val_type == GGUF_VAL_UINT32:
        return struct.unpack("<I", _read_exact(f, 4))[0], None
    if val_type == GGUF_VAL_INT32:
        return struct.unpack("<i", _read_exact(f, 4))[0], None
    if val_type == GGUF_VAL_FLOAT32:
        return float(struct.unpack("<f", _read_exact(f, 4))[0]), None
    if val_type == GGUF_VAL_BOOL:
        return bool(struct.unpack("<?", _read_exact(f, 1))[0]), None
    if val_type == GGUF_VAL_STRING:
        return _read_gguf_string(f), None
    if val_type == GGUF_VAL_UINT64:
        return struct.unpack("<Q", _read_exact(f, 8))[0], None
    if val_type == GGUF_VAL_INT64:
        return struct.unpack("<q", _read_exact(f, 8))[0], None
    if val_type == GGUF_VAL_FLOAT64:
        return float(struct.unpack("<d", _read_exact(f, 8))[0]), None
    if val_type == GGUF_VAL_ARRAY:
        elem_type, count = struct.unpack("<IQ", _read_exact(f, 12))
        if count > 1_000_000:
            raise ValueError(f"GGUF metadata array count {count} exceeds 1,000,000 safety limit")
        items = [_read_gguf_value_with_elem_type(f, elem_type, _depth + 1)[0] for _ in range(count)]
        return items, int(elem_type)
    raise ValueError(f"Unsupported GGUF metadata value type ID: {val_type}")


def _read_gguf_value(f: BinaryIO, val_type: int, _depth: int = 0) -> object:
    return _read_gguf_value_with_elem_type(f, val_type, _depth)[0]


def _align_offset(pos: int, alignment: int = GGUF_DEFAULT_ALIGNMENT) -> int:
    if alignment <= 0 or (alignment & (alignment - 1)) != 0:
        raise ValueError(f"GGUF alignment must be a positive power of 2, got {alignment}")
    rem = pos % alignment
    return pos if rem == 0 else pos + (alignment - rem)


def ggml_tensor_byte_size(
    num_elements: int,
    ggml_type_name: str,
    shape: tuple[int, ...] | None = None,
) -> int:
    """Computes the exact byte size of a `ggml` tensor payload for `num_elements` and `ggml_type_name`."""
    spec = GGML_BLOCK_SPEC.get(ggml_type_name)
    if spec is not None:
        block_elems, block_bytes = spec
        if block_elems > 1:
            if shape is not None and len(shape) >= 1 and shape[-1] % block_elems != 0:
                raise ValueError(
                    f"{ggml_type_name} tensor innermost dimension shape[-1]={shape[-1]} "
                    f"(from shape {shape}) must be divisible by block size {block_elems}."
                )
            if num_elements % block_elems != 0:
                raise ValueError(
                    f"{ggml_type_name} tensor num_elements ({num_elements}) must be divisible by {block_elems}."
                )
        return (num_elements // block_elems) * block_bytes
    # Conservative 1-byte minimum for non-standard experimental ggml type IDs
    return max(num_elements // 8, 1)


def quantize_q8_0_bytes(arr: np.ndarray) -> bytes:
    """
    Quantizes a float32 NumPy array (innermost dimension divisible by 32) into standard
    `ggml` `block_q8_0` format (`34 bytes/block`: 2-byte `float16` scale $d$ + 32 `int8` quants).
    """
    np_arr = np.asarray(arr, dtype=np.float32)
    if not np.all(np.isfinite(np_arr)):
        raise ValueError("Cannot quantize tensor containing NaN or Inf to Q8_0.")
    if np_arr.ndim >= 1 and np_arr.shape[-1] % 32 != 0:
        raise ValueError(
            f"Q8_0 quantization requires innermost dimension shape[-1] divisible by 32, got shape {np_arr.shape}"
        )
    flat = np_arr.reshape(-1)
    if flat.size % 32 != 0:
        raise ValueError(f"Q8_0 quantization requires element count divisible by 32, got {flat.size}")
    n_blocks = flat.size // 32
    blocks = flat.reshape(n_blocks, 32)
    abs_max = np.max(np.abs(blocks), axis=1)  # (n_blocks,)
    d_f16 = (abs_max / 127.0).astype(np.float16)
    if not np.all(np.isfinite(d_f16)):
        raise ValueError("Q8_0 block scale overflowed float16 range.")
    d_f32 = d_f16.astype(np.float32)[:, None]
    inv_d = np.where(d_f32 > 0.0, 1.0 / np.maximum(d_f32, 1e-30), 0.0)
    qs = np.clip(np.round(blocks * inv_d), -127, 127).astype(np.int8)

    out_dtype = np.dtype([("d", "<f2"), ("qs", "i1", (32,))])
    structured = np.empty(n_blocks, dtype=out_dtype)
    structured["d"] = d_f16
    structured["qs"] = qs
    return structured.tobytes()


def dequantize_q8_0_bytes(raw_bytes: bytes, shape: tuple[int, ...]) -> np.ndarray:
    """
    Dequantizes `ggml` `block_q8_0` bytes (`34 bytes/block`) into a float32 NumPy array of `shape`.
    """
    if shape and shape[-1] % 32 != 0:
        raise ValueError(f"Q8_0 shape {shape} innermost dimension ({shape[-1]}) must be divisible by 32")
    n_elems = int(np.prod(shape)) if shape else 1
    if n_elems % 32 != 0:
        raise ValueError(f"Q8_0 shape {shape} element count ({n_elems}) must be divisible by 32")
    n_blocks = n_elems // 32
    expected = n_blocks * 34
    if len(raw_bytes) != expected:
        raise ValueError(f"Q8_0 payload length mismatch: expected {expected} bytes, got {len(raw_bytes)}")
    in_dtype = np.dtype([("d", "<f2"), ("qs", "i1", (32,))])
    structured = np.frombuffer(raw_bytes, dtype=in_dtype)
    d_f32 = structured["d"].astype(np.float32)[:, None]
    qs_f32 = structured["qs"].astype(np.float32)
    return (d_f32 * qs_f32).reshape(shape).astype(np.float32)


def quantize_q4_0_bytes(arr: np.ndarray) -> bytes:
    """
    Quantizes a float32 NumPy array (innermost dimension divisible by 32) into standard
    `ggml` `block_q4_0` format (`18 bytes/block`: 2-byte `float16` scale $d$ + 16 packed `uint8`
    nibble bytes where low nibble holds indices `0..15` and high nibble holds indices `16..31`,
    centered at `-8`).
    Matches `ggml`'s `quantize_row_q4_0_ref` convention (`d = max_by_abs / -8.0` or `abs_max / 8.0`
    when the signed extremum is negative, `abs_max / 7.0` when strictly positive) so untouched
    blocks round-trip idempotently.
    """
    np_arr = np.asarray(arr, dtype=np.float32)
    if not np.all(np.isfinite(np_arr)):
        raise ValueError("Cannot quantize tensor containing NaN or Inf to Q4_0.")
    if np_arr.ndim >= 1 and np_arr.shape[-1] % 32 != 0:
        raise ValueError(
            f"Q4_0 quantization requires innermost dimension shape[-1] divisible by 32, got shape {np_arr.shape}"
        )
    flat = np_arr.reshape(-1)
    if flat.size % 32 != 0:
        raise ValueError(f"Q4_0 quantization requires element count divisible by 32, got {flat.size}")
    n_blocks = flat.size // 32
    blocks = flat.reshape(n_blocks, 32)
    min_vals = np.min(blocks, axis=1)
    max_vals = np.max(blocks, axis=1)
    # In [-8, 7] grid, -min_vals/8 and max_vals/7 determine the required positive scale d
    d_req = np.maximum(-min_vals / 8.0, max_vals / 7.0)
    d_f16 = d_req.astype(np.float16)
    if not np.all(np.isfinite(d_f16)):
        raise ValueError("Q4_0 block scale overflowed float16 range.")
    d_f32 = d_f16.astype(np.float32)[:, None]
    inv_d = np.where(d_f32 > 0.0, 1.0 / np.maximum(d_f32, 1e-30), 0.0)
    # Signed integer quants in [-8, 7], shifted by +8 into unsigned nibble [0, 15]
    q_unsigned = np.clip(np.round(blocks * inv_d) + 8.0, 0, 15).astype(np.uint8)
    lo_nibbles = q_unsigned[:, :16]
    hi_nibbles = q_unsigned[:, 16:]
    packed = (lo_nibbles | (hi_nibbles << 4)).astype(np.uint8)

    out_dtype = np.dtype([("d", "<f2"), ("qs", "u1", (16,))])
    structured = np.empty(n_blocks, dtype=out_dtype)
    structured["d"] = d_f16
    structured["qs"] = packed
    return structured.tobytes()


def dequantize_q4_0_bytes(raw_bytes: bytes, shape: tuple[int, ...]) -> np.ndarray:
    """
    Dequantizes `ggml` `block_q4_0` bytes (`18 bytes/block`) into a float32 NumPy array of `shape`.
    """
    if shape and shape[-1] % 32 != 0:
        raise ValueError(f"Q4_0 shape {shape} innermost dimension ({shape[-1]}) must be divisible by 32")
    n_elems = int(np.prod(shape)) if shape else 1
    if n_elems % 32 != 0:
        raise ValueError(f"Q4_0 shape {shape} element count ({n_elems}) must be divisible by 32")
    n_blocks = n_elems // 32
    expected = n_blocks * 18
    if len(raw_bytes) != expected:
        raise ValueError(f"Q4_0 payload length mismatch: expected {expected} bytes, got {len(raw_bytes)}")
    in_dtype = np.dtype([("d", "<f2"), ("qs", "u1", (16,))])
    structured = np.frombuffer(raw_bytes, dtype=in_dtype)
    d_f32 = structured["d"].astype(np.float32)[:, None]
    packed = structured["qs"]
    lo = (packed & 0x0F).astype(np.int8) - 8
    hi = (packed >> 4).astype(np.int8) - 8
    qs_f32 = np.concatenate([lo, hi], axis=1).astype(np.float32)
    return (d_f32 * qs_f32).reshape(shape).astype(np.float32)


def quantize_q4_k_bytes(arr: np.ndarray) -> bytes:
    """
    Quantizes a float32 NumPy array (innermost dimension divisible by 256) into standard
    `ggml` `block_q4_K` super-blocks (`144 bytes/block` for 256 elements):
    - `d` (`float16`, 2 bytes) and `dmin` (`float16`, 2 bytes)
    - `scales` (`12 uint8` bytes packing eight 6-bit sub-block scales `sc[0..7]` and eight
      6-bit sub-block mins `m[0..7]` via `ggml`'s `get_scale_min_k4` layout)
    - `qs` (`128 uint8` bytes: four 64-element chunks, each packing 32 low nibbles and 32 high nibbles).
    Reconstructed value in sub-block `j` (`0..7`): `x = d * sc[j] * q - dmin * m[j]`.
    """
    np_arr = np.asarray(arr, dtype=np.float32)
    if not np.all(np.isfinite(np_arr)):
        raise ValueError("Cannot quantize tensor containing NaN or Inf to Q4_K.")
    if np_arr.ndim >= 1 and np_arr.shape[-1] % 256 != 0:
        raise ValueError(
            f"Q4_K quantization requires innermost dimension shape[-1] divisible by 256, got shape {np_arr.shape}"
        )
    flat = np_arr.reshape(-1)
    if flat.size % 256 != 0:
        raise ValueError(f"Q4_K quantization requires element count divisible by 256, got {flat.size}")
    n_blocks = flat.size // 256
    sub_blocks = flat.reshape(n_blocks, 8, 32)

    # Sub-block non-negative min offset: m_raw = max(-min(x), 0) so x + m_raw >= 0
    min_vals = np.min(sub_blocks, axis=2)  # (n_blocks, 8)
    max_vals = np.max(sub_blocks, axis=2)  # (n_blocks, 8)
    m_raw = np.maximum(-min_vals, 0.0)     # (n_blocks, 8)
    # Using initial m_raw, shifted span is max_vals + m_raw over [0..15]
    s_raw = np.maximum((max_vals + m_raw) / 15.0, 0.0)  # (n_blocks, 8)

    max_s = np.max(s_raw, axis=1)  # (n_blocks,)
    max_m = np.max(m_raw, axis=1)  # (n_blocks,)

    d_f16 = (max_s / 63.0).astype(np.float16)
    dmin_f16 = (max_m / 63.0).astype(np.float16)
    if not np.all(np.isfinite(d_f16)) or not np.all(np.isfinite(dmin_f16)):
        raise ValueError("Q4_K super-block scale overflowed float16 range.")

    d_f32 = d_f16.astype(np.float32)[:, None]       # (n_blocks, 1)
    dmin_f32 = dmin_f16.astype(np.float32)[:, None] # (n_blocks, 1)

    inv_d = np.where(d_f32 > 0.0, 1.0 / np.maximum(d_f32, 1e-30), 0.0)
    inv_dmin = np.where(dmin_f32 > 0.0, 1.0 / np.maximum(dmin_f32, 1e-30), 0.0)

    m_q = np.clip(np.round(m_raw * inv_dmin), 0, 63).astype(np.uint8)  # (n_blocks, 8)
    eff_min = dmin_f32 * m_q.astype(np.float32)                        # (n_blocks, 8)

    # Recompute optimal sub-block scale given quantized eff_min so max_vals + eff_min fits in 15 * d * sc
    s_needed = np.maximum((max_vals + eff_min) / 15.0, 0.0)
    sc_q = np.clip(np.round(s_needed * inv_d), 0, 63).astype(np.uint8)  # (n_blocks, 8)
    eff_scale = (d_f32 * sc_q.astype(np.float32))[:, :, None]           # (n_blocks, 8, 1)
    inv_eff_scale = np.where(eff_scale > 0.0, 1.0 / np.maximum(eff_scale, 1e-30), 0.0)

    q_vals = np.clip(
        np.round((sub_blocks + eff_min[:, :, None]) * inv_eff_scale),
        0,
        15,
    ).astype(np.uint8)  # (n_blocks, 8, 32)

    # Pack 12-byte scales array per ggml get_scale_min_k4 convention:
    # bytes 0..3: (sc[0..3] & 63)
    # bytes 4..7: (m[0..3] & 63)
    # bytes 8..11: (sc[4..7] & 0x0F) | ((m[4..7] & 0x0F) << 4), while top 2 bits of sc[4..7]
    #              go into bits 6..7 of bytes 0..3 and top 2 bits of m[4..7] go into bits 6..7 of bytes 4..7
    scales_packed = np.empty((n_blocks, 12), dtype=np.uint8)
    scales_packed[:, 0:4] = (sc_q[:, 0:4] & 0x3F) | ((sc_q[:, 4:8] >> 4) << 6)
    scales_packed[:, 4:8] = (m_q[:, 0:4] & 0x3F) | ((m_q[:, 4:8] >> 4) << 6)
    scales_packed[:, 8:12] = (sc_q[:, 4:8] & 0x0F) | ((m_q[:, 4:8] & 0x0F) << 4)

    # Pack 128-byte qs array: for chunk c in 0..3, low nibble is sub_block 2*c, high nibble is sub_block 2*c + 1
    q_pairs = q_vals.reshape(n_blocks, 4, 2, 32)
    qs_packed = (q_pairs[:, :, 0, :] | (q_pairs[:, :, 1, :] << 4)).reshape(n_blocks, 128).astype(np.uint8)

    out_dtype = np.dtype([("d", "<f2"), ("dmin", "<f2"), ("scales", "u1", (12,)), ("qs", "u1", (128,))])
    structured = np.empty(n_blocks, dtype=out_dtype)
    structured["d"] = d_f16
    structured["dmin"] = dmin_f16
    structured["scales"] = scales_packed
    structured["qs"] = qs_packed
    return structured.tobytes()


def dequantize_q4_k_bytes(raw_bytes: bytes, shape: tuple[int, ...]) -> np.ndarray:
    """
    Dequantizes `ggml` `block_q4_K` super-blocks (`144 bytes/block` for 256 elements)
    into a float32 NumPy array of `shape`.
    """
    if shape and shape[-1] % 256 != 0:
        raise ValueError(f"Q4_K shape {shape} innermost dimension ({shape[-1]}) must be divisible by 256")
    n_elems = int(np.prod(shape)) if shape else 1
    if n_elems % 256 != 0:
        raise ValueError(f"Q4_K shape {shape} element count ({n_elems}) must be divisible by 256")
    n_blocks = n_elems // 256
    expected = n_blocks * 144
    if len(raw_bytes) != expected:
        raise ValueError(f"Q4_K payload length mismatch: expected {expected} bytes, got {len(raw_bytes)}")

    in_dtype = np.dtype([("d", "<f2"), ("dmin", "<f2"), ("scales", "u1", (12,)), ("qs", "u1", (128,))])
    structured = np.frombuffer(raw_bytes, dtype=in_dtype)
    d_f32 = structured["d"].astype(np.float32)[:, None]       # (n_blocks, 1)
    dmin_f32 = structured["dmin"].astype(np.float32)[:, None] # (n_blocks, 1)
    sc_bytes = structured["scales"]                           # (n_blocks, 12)

    sc_q = np.empty((n_blocks, 8), dtype=np.float32)
    m_q = np.empty((n_blocks, 8), dtype=np.float32)
    sc_q[:, 0:4] = (sc_bytes[:, 0:4] & 0x3F).astype(np.float32)
    m_q[:, 0:4] = (sc_bytes[:, 4:8] & 0x3F).astype(np.float32)
    sc_q[:, 4:8] = ((sc_bytes[:, 8:12] & 0x0F) | ((sc_bytes[:, 0:4] >> 6) << 4)).astype(np.float32)
    m_q[:, 4:8] = ((sc_bytes[:, 8:12] >> 4) | ((sc_bytes[:, 4:8] >> 6) << 4)).astype(np.float32)

    qs_chunks = structured["qs"].reshape(n_blocks, 4, 32)
    lo = (qs_chunks & 0x0F).astype(np.float32)
    hi = (qs_chunks >> 4).astype(np.float32)
    q_sub = np.stack([lo, hi], axis=2).reshape(n_blocks, 8, 32)

    scales_eff = (d_f32 * sc_q)[:, :, None]
    mins_eff = (dmin_f32 * m_q)[:, :, None]
    out = scales_eff * q_sub - mins_eff
    return out.reshape(shape).astype(np.float32)


def quantize_q6_k_bytes(arr: np.ndarray) -> bytes:
    """
    Quantizes a float32 NumPy array (innermost dimension divisible by 256) into standard
    `ggml` `block_q6_K` super-blocks (`210 bytes/block` for 256 elements):
    - `ql` (`128 uint8` bytes: low 4 bits of the 256 6-bit quants)
    - `qh` (`64 uint8` bytes: high 2 bits of the 256 6-bit quants)
    - `scales` (`16 int8` sub-block scales, one per 16-element sub-block)
    - `d` (`float16` super-block scale, 2 bytes).
    Reconstructed value in 16-element sub-block `j` (`0..15`): `x = d * scales[j] * (q - 32)` where `q in [0..63]`.
    """
    np_arr = np.asarray(arr, dtype=np.float32)
    if not np.all(np.isfinite(np_arr)):
        raise ValueError("Cannot quantize tensor containing NaN or Inf to Q6_K.")
    if np_arr.ndim >= 1 and np_arr.shape[-1] % 256 != 0:
        raise ValueError(
            f"Q6_K quantization requires innermost dimension shape[-1] divisible by 256, got shape {np_arr.shape}"
        )
    flat = np_arr.reshape(-1)
    if flat.size % 256 != 0:
        raise ValueError(f"Q6_K quantization requires element count divisible by 256, got {flat.size}")
    n_blocks = flat.size // 256
    sub_blocks = flat.reshape(n_blocks, 16, 16)

    min_vals = np.min(sub_blocks, axis=2)  # (n_blocks, 16)
    max_vals = np.max(sub_blocks, axis=2)  # (n_blocks, 16)
    # Signed 6-bit grid is [-32, 31]
    s_req = np.maximum(-min_vals / 32.0, max_vals / 31.0)  # (n_blocks, 16)
    max_s = np.max(s_req, axis=1)                          # (n_blocks,)

    d_f16 = (max_s / 127.0).astype(np.float16)
    if not np.all(np.isfinite(d_f16)):
        raise ValueError("Q6_K super-block scale overflowed float16 range.")
    d_f32 = d_f16.astype(np.float32)[:, None]  # (n_blocks, 1)
    inv_d = np.where(d_f32 > 0.0, 1.0 / np.maximum(d_f32, 1e-30), 0.0)

    sc_i8 = np.clip(np.round(s_req * inv_d), -127, 127).astype(np.int8)  # (n_blocks, 16)
    eff_scale = (d_f32 * sc_i8.astype(np.float32))[:, :, None]          # (n_blocks, 16, 1)
    inv_eff = np.where(np.abs(eff_scale) > 0.0, 1.0 / np.where(eff_scale != 0.0, eff_scale, 1.0), 0.0)

    q_u6 = np.clip(np.round(sub_blocks * inv_eff) + 32.0, 0, 63).astype(np.uint8)  # (n_blocks, 16, 16)
    # Reshape into two 128-element halves: (n_blocks, 2, 4, 32)
    q_halves = q_u6.reshape(n_blocks, 2, 4, 32)
    q0 = q_halves[:, :, 0, :]
    q1 = q_halves[:, :, 1, :]
    q2 = q_halves[:, :, 2, :]
    q3 = q_halves[:, :, 3, :]

    ql_0 = (q0 & 0x0F) | ((q2 & 0x0F) << 4)  # (n_blocks, 2, 32)
    ql_1 = (q1 & 0x0F) | ((q3 & 0x0F) << 4)  # (n_blocks, 2, 32)
    ql = np.concatenate([ql_0, ql_1], axis=2).reshape(n_blocks, 128).astype(np.uint8)

    qh = (
        ((q0 >> 4) & 0x03)
        | (((q1 >> 4) & 0x03) << 2)
        | (((q2 >> 4) & 0x03) << 4)
        | (((q3 >> 4) & 0x03) << 6)
    ).reshape(n_blocks, 64).astype(np.uint8)

    out_dtype = np.dtype([("ql", "u1", (128,)), ("qh", "u1", (64,)), ("scales", "i1", (16,)), ("d", "<f2")])
    structured = np.empty(n_blocks, dtype=out_dtype)
    structured["ql"] = ql
    structured["qh"] = qh
    structured["scales"] = sc_i8
    structured["d"] = d_f16
    return structured.tobytes()


def dequantize_q6_k_bytes(raw_bytes: bytes, shape: tuple[int, ...]) -> np.ndarray:
    """
    Dequantizes `ggml` `block_q6_K` super-blocks (`210 bytes/block` for 256 elements)
    into a float32 NumPy array of `shape`.
    """
    if shape and shape[-1] % 256 != 0:
        raise ValueError(f"Q6_K shape {shape} innermost dimension ({shape[-1]}) must be divisible by 256")
    n_elems = int(np.prod(shape)) if shape else 1
    if n_elems % 256 != 0:
        raise ValueError(f"Q6_K shape {shape} element count ({n_elems}) must be divisible by 256")
    n_blocks = n_elems // 256
    expected = n_blocks * 210
    if len(raw_bytes) != expected:
        raise ValueError(f"Q6_K payload length mismatch: expected {expected} bytes, got {len(raw_bytes)}")

    in_dtype = np.dtype([("ql", "u1", (128,)), ("qh", "u1", (64,)), ("scales", "i1", (16,)), ("d", "<f2")])
    structured = np.frombuffer(raw_bytes, dtype=in_dtype)
    d_f32 = structured["d"].astype(np.float32)[:, None]       # (n_blocks, 1)
    sc_f32 = structured["scales"].astype(np.float32)          # (n_blocks, 16)
    ql = structured["ql"].reshape(n_blocks, 2, 64)
    qh = structured["qh"].reshape(n_blocks, 2, 32)

    ql_0 = ql[:, :, 0:32]
    ql_1 = ql[:, :, 32:64]

    q0 = ((ql_0 & 0x0F) | (((qh >> 0) & 0x03) << 4)).astype(np.int8) - 32
    q1 = ((ql_1 & 0x0F) | (((qh >> 2) & 0x03) << 4)).astype(np.int8) - 32
    q2 = ((ql_0 >> 4) | (((qh >> 4) & 0x03) << 4)).astype(np.int8) - 32
    q3 = ((ql_1 >> 4) | (((qh >> 6) & 0x03) << 4)).astype(np.int8) - 32

    q_halves = np.stack([q0, q1, q2, q3], axis=2).reshape(n_blocks, 16, 16).astype(np.float32)
    eff_scales = (d_f32 * sc_f32)[:, :, None]
    out = eff_scales * q_halves
    return out.reshape(shape).astype(np.float32)


def inspect_gguf_header(path: str | Path) -> dict[str, object]:
    """
    Parses a `.gguf` binary file header (versions 2 and 3) without loading tensor payloads.
    Returns a dictionary containing `version`, `tensor_count`, `kv_count`, `data_start_offset`,
    `metadata`, `tensors` (mapping name -> `GGUFTensorInfo`), and surgical/expert counts.
    """
    fpath = Path(path)
    file_size = fpath.stat().st_size
    with fpath.open("rb") as f:
        magic = f.read(4)
        if magic != GGUF_MAGIC:
            raise ValueError(f"Invalid GGUF magic bytes in {fpath}: {magic!r}")
        version, tensor_count, kv_count = struct.unpack("<IQQ", _read_exact(f, 20))
        if version not in (2, 3):
            raise ValueError(f"Unsupported GGUF version: {version} (expected 2 or 3)")
        if tensor_count > 100_000 or kv_count > 100_000:
            raise ValueError(f"GGUF counts exceed safety limits: tensors={tensor_count}, kv={kv_count}")

        metadata: dict[str, object] = {}
        metadata_types: dict[str, int] = {}
        metadata_array_elem_types: dict[str, int] = {}
        for _ in range(kv_count):
            key = _read_gguf_string(f)
            (val_type,) = struct.unpack("<I", _read_exact(f, 4))
            metadata_types[key] = val_type
            val_obj, arr_elem_type = _read_gguf_value_with_elem_type(f, val_type)
            metadata[key] = val_obj
            if arr_elem_type is not None:
                metadata_array_elem_types[key] = arr_elem_type

        raw_align = metadata.get("general.alignment", GGUF_DEFAULT_ALIGNMENT)
        alignment = int(raw_align) if isinstance(raw_align, (int, float)) else GGUF_DEFAULT_ALIGNMENT
        if alignment <= 0 or (alignment & (alignment - 1)) != 0 or alignment > 1_048_576:
            raise ValueError(f"Invalid GGUF general.alignment: {raw_align}")

        tensors: dict[str, GGUFTensorInfo] = {}
        for _ in range(tensor_count):
            t_name = _read_gguf_string(f)
            (n_dims,) = struct.unpack("<I", _read_exact(f, 4))
            if n_dims > 8:
                raise ValueError(f"Invalid GGUF tensor n_dims={n_dims} for '{t_name}'")
            # ggml stores innermost dimension first (ne[0] = cols, ne[1] = rows);
            # reverse into standard NumPy/PyTorch row-major shape (d_out, d_in).
            ggml_ne = tuple(struct.unpack(f"<{n_dims}Q", _read_exact(f, 8 * n_dims)))
            dims = tuple(reversed(ggml_ne))
            ggml_type_id, rel_offset = struct.unpack("<IQ", _read_exact(f, 12))
            t_type_name = GGML_TYPE_NAMES.get(ggml_type_id, f"UNKNOWN_{ggml_type_id}")
            tensors[t_name] = GGUFTensorInfo(
                name=t_name,
                shape=dims,
                ggml_type_id=ggml_type_id,
                ggml_type_name=t_type_name,
                relative_offset=rel_offset,
                is_control_tensor=is_surgical_control_tensor(t_name),
                is_routed_expert=is_routed_expert_tensor(t_name),
            )

        data_start = _align_offset(f.tell(), alignment)

    for t_info in tensors.values():
        min_bytes = ggml_tensor_byte_size(t_info.num_elements, t_info.ggml_type_name, t_info.shape)
        if data_start + t_info.relative_offset + min_bytes > file_size:
            raise ValueError(
                f"GGUF tensor '{t_info.name}' extends past EOF "
                f"({data_start + t_info.relative_offset + min_bytes} > {file_size}) in {fpath}"
            )

    control_count = sum(1 for t in tensors.values() if t.is_control_tensor)
    expert_count = sum(1 for t in tensors.values() if t.is_routed_expert)

    return {
        "file": fpath.name,
        "version": version,
        "tensor_count": tensor_count,
        "kv_count": kv_count,
        "alignment": alignment,
        "data_start_offset": data_start,
        "control_tensor_count": control_count,
        "routed_expert_tensor_count": expert_count,
        "metadata": metadata,
        "metadata_types": metadata_types,
        "metadata_array_elem_types": metadata_array_elem_types,
        "tensors": tensors,
    }


def _infer_gguf_val_type(val: Any) -> int:
    if isinstance(val, bool):
        return GGUF_VAL_BOOL
    if isinstance(val, int):
        if val < 0:
            return GGUF_VAL_INT32 if val >= -2147483648 else GGUF_VAL_INT64
        return GGUF_VAL_UINT32 if val <= 4294967295 else GGUF_VAL_UINT64
    if isinstance(val, float):
        return GGUF_VAL_FLOAT32
    if isinstance(val, (list, tuple)):
        return GGUF_VAL_ARRAY
    return GGUF_VAL_STRING


def _infer_array_elem_type(seq: list[Any]) -> int:
    if not seq:
        return GGUF_VAL_STRING
    if all(isinstance(x, bool) for x in seq):
        return GGUF_VAL_BOOL
    if all(isinstance(x, int) and not isinstance(x, bool) for x in seq):
        min_v = min(seq)
        max_v = max(seq)
        if min_v < 0:
            if min_v >= -2147483648 and max_v <= 2147483647:
                return GGUF_VAL_INT32
            return GGUF_VAL_INT64
        if max_v <= 4294967295:
            return GGUF_VAL_UINT32
        return GGUF_VAL_UINT64
    if all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in seq):
        return GGUF_VAL_FLOAT32
    return GGUF_VAL_STRING


def _write_gguf_value_body(
    f: BinaryIO,
    val: Any,
    val_type: int,
    array_elem_type: int | None = None,
) -> None:
    if val_type == GGUF_VAL_UINT8:
        f.write(struct.pack("<B", int(val)))
    elif val_type == GGUF_VAL_INT8:
        f.write(struct.pack("<b", int(val)))
    elif val_type == GGUF_VAL_UINT16:
        f.write(struct.pack("<H", int(val)))
    elif val_type == GGUF_VAL_INT16:
        f.write(struct.pack("<h", int(val)))
    elif val_type == GGUF_VAL_UINT32:
        f.write(struct.pack("<I", int(val)))
    elif val_type == GGUF_VAL_INT32:
        f.write(struct.pack("<i", int(val)))
    elif val_type == GGUF_VAL_UINT64:
        f.write(struct.pack("<Q", int(val)))
    elif val_type == GGUF_VAL_INT64:
        f.write(struct.pack("<q", int(val)))
    elif val_type == GGUF_VAL_FLOAT32:
        f.write(struct.pack("<f", float(val)))
    elif val_type == GGUF_VAL_FLOAT64:
        f.write(struct.pack("<d", float(val)))
    elif val_type == GGUF_VAL_BOOL:
        f.write(struct.pack("<?", bool(val)))
    elif val_type == GGUF_VAL_STRING:
        _write_gguf_string(f, str(val))
    elif val_type == GGUF_VAL_ARRAY:
        seq = list(val)
        elem_type = array_elem_type if array_elem_type is not None else _infer_array_elem_type(seq)
        f.write(struct.pack("<IQ", elem_type, len(seq)))
        for item in seq:
            _write_gguf_value_body(f, item, elem_type)
    else:
        raise ValueError(f"Unsupported GGUF metadata value type ID for write: {val_type}")


def _write_gguf_kv(
    f: BinaryIO,
    key: str,
    val: Any,
    val_type: int | None = None,
    array_elem_type: int | None = None,
) -> None:
    _write_gguf_string(f, str(key))
    vtype = val_type if val_type is not None else _infer_gguf_val_type(val)
    f.write(struct.pack("<I", vtype))
    _write_gguf_value_body(f, val, vtype, array_elem_type)


def _encode_gguf_tensor_payload(arr: np.ndarray, target_type: str | None = None) -> tuple[int, bytes]:
    """Encodes a NumPy array into raw GGUF bytes and returns `(ggml_type_id, raw_bytes)`."""
    np_arr = np.ascontiguousarray(arr)
    t_str = (target_type or "").upper()
    if t_str in ("Q6_K", "Q4_K", "Q8_0", "Q4_0", "BF16", "F16", "F32") or not t_str:
        if not np.all(np.isfinite(np_arr)):
            raise ValueError(f"Cannot encode GGUF tensor containing NaN or Inf as {t_str or 'F32'}.")
    if t_str == "Q6_K":
        return NAME_TO_GGML_TYPE["Q6_K"], quantize_q6_k_bytes(np_arr)
    if t_str == "Q4_K":
        return NAME_TO_GGML_TYPE["Q4_K"], quantize_q4_k_bytes(np_arr)
    if t_str == "Q8_0":
        return NAME_TO_GGML_TYPE["Q8_0"], quantize_q8_0_bytes(np_arr)
    if t_str == "Q4_0":
        return NAME_TO_GGML_TYPE["Q4_0"], quantize_q4_0_bytes(np_arr)
    if t_str == "BF16":
        return NAME_TO_GGML_TYPE["BF16"], _f32_to_bf16_bytes(np_arr)
    if t_str == "F16" or (not t_str and np_arr.dtype == np.float16):
        f16_arr = np_arr.astype(np.float16)
        if not np.all(np.isfinite(f16_arr)):
            raise ValueError("Tensor values overflow float16 range during GGUF F16 encoding.")
        return NAME_TO_GGML_TYPE["F16"], f16_arr.tobytes()
    return NAME_TO_GGML_TYPE["F32"], np_arr.astype(np.float32).tobytes()


def write_gguf_file(
    path: str | Path,
    tensors: dict[str, np.ndarray],
    metadata: dict[str, Any] | None = None,
    tensor_types: dict[str, str] | None = None,
    metadata_types: dict[str, int] | None = None,
    metadata_array_elem_types: dict[str, int] | None = None,
) -> Path:
    """
    Writes a valid GGUF v3 binary file containing `tensors` (`F32`, `F16`, `BF16`, `Q8_0`, or `Q4_0`)
    and `metadata` key-value pairs (including `ARRAY` metadata) with configurable `general.alignment`.
    """
    out_path = Path(path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    meta = dict(metadata or {})
    mtypes = dict(metadata_types or {})
    melem_types = dict(metadata_array_elem_types or {})
    raw_align = meta.get("general.alignment", GGUF_DEFAULT_ALIGNMENT)
    alignment = int(raw_align) if isinstance(raw_align, (int, float)) and not isinstance(raw_align, bool) else GGUF_DEFAULT_ALIGNMENT
    if alignment <= 0 or (alignment & (alignment - 1)) != 0 or alignment > 1_048_576:
        raise ValueError(f"Invalid GGUF general.alignment: {raw_align}")

    # Prepare tensor binary payloads and relative offsets
    tensor_names = list(tensors.keys())
    encoded_blobs: list[tuple[str, tuple[int, ...], int, bytes]] = []
    rel_offset = 0

    for name in tensor_names:
        arr = np.ascontiguousarray(tensors[name])
        req_type = (tensor_types or {}).get(name)
        ggml_type, raw_bytes = _encode_gguf_tensor_payload(arr, req_type)
        encoded_blobs.append((name, arr.shape, ggml_type, raw_bytes))
        rel_offset = _align_offset(rel_offset + len(raw_bytes), alignment)

    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    with tmp_path.open("wb") as f:
        f.write(GGUF_MAGIC)
        f.write(struct.pack("<IQQ", 3, len(encoded_blobs), len(meta)))

        for k, v in meta.items():
            _write_gguf_kv(f, str(k), v, mtypes.get(str(k)), melem_types.get(str(k)))

        curr_rel = 0
        for name, shape, ggml_type, raw_bytes in encoded_blobs:
            _write_gguf_string(f, name)
            f.write(struct.pack("<I", len(shape)))
            # Write ggml dimensions innermost-first (ne[0] = shape[-1], ne[1] = shape[-2], ...)
            for dim in reversed(shape):
                f.write(struct.pack("<Q", int(dim)))
            f.write(struct.pack("<IQ", ggml_type, curr_rel))
            curr_rel = _align_offset(curr_rel + len(raw_bytes), alignment)

        header_end = f.tell()
        data_start = _align_offset(header_end, alignment)
        if data_start > header_end:
            f.write(b"\x00" * (data_start - header_end))

        curr_pos = data_start
        for _, _, _, raw_bytes in encoded_blobs:
            f.write(raw_bytes)
            curr_pos += len(raw_bytes)
            aligned_pos = _align_offset(curr_pos, alignment)
            if aligned_pos > curr_pos:
                f.write(b"\x00" * (aligned_pos - curr_pos))
                curr_pos = aligned_pos

    tmp_path.replace(out_path)
    return out_path


def _decode_gguf_tensor_f32_from_handle(
    f: BinaryIO,
    data_start: int,
    t_info: GGUFTensorInfo,
) -> np.ndarray:
    """Slices and decodes a single tensor from an already-open GGUF binary handle `f`."""
    tname_type = t_info.ggml_type_name
    if tname_type not in ("F32", "F16", "BF16", "Q8_0", "Q4_0", "Q4_K", "Q6_K"):
        raise ValueError(f"Direct float slicing supports F32/F16/BF16/Q8_0/Q4_0/Q4_K/Q6_K, got {tname_type}")
    byte_len = ggml_tensor_byte_size(t_info.num_elements, tname_type, t_info.shape)
    f.seek(data_start + t_info.relative_offset)
    buf = _read_exact(f, byte_len)
    if tname_type == "F32":
        return np.frombuffer(buf, dtype=np.float32).reshape(t_info.shape).copy()
    if tname_type == "F16":
        return np.frombuffer(buf, dtype=np.float16).reshape(t_info.shape).astype(np.float32)
    if tname_type == "BF16":
        return _bf16_bytes_to_f32(buf, t_info.shape)
    if tname_type == "Q8_0":
        return dequantize_q8_0_bytes(buf, t_info.shape)
    if tname_type == "Q4_0":
        return dequantize_q4_0_bytes(buf, t_info.shape)
    if tname_type == "Q4_K":
        return dequantize_q4_k_bytes(buf, t_info.shape)
    return dequantize_q6_k_bytes(buf, t_info.shape)


def load_gguf_tensor_f32(path: str | Path, tensor_name: str) -> np.ndarray:
    """
    Slices and decodes a single `F32`, `F16`, `BF16`, `Q8_0`, or `Q4_0` tensor from a `.gguf`
    file into a float32 NumPy array.
    """
    info = inspect_gguf_header(path)
    tensors_map: dict[str, GGUFTensorInfo] = info["tensors"]  # type: ignore[assignment]
    if tensor_name not in tensors_map:
        raise KeyError(f"Tensor '{tensor_name}' not found in GGUF file {path}")

    t_info = tensors_map[tensor_name]
    data_start = int(info["data_start_offset"])  # type: ignore[arg-type]
    with Path(path).open("rb") as f:
        return _decode_gguf_tensor_f32_from_handle(f, data_start, t_info)


def stream_surgical_gguf_update(
    input_path: str | Path,
    output_path: str | Path,
    surgical_edits: dict[str, Callable[[np.ndarray], np.ndarray]],
) -> list[str]:
    """
    Applies surgical weight updates directly to a `.gguf` v2/v3 file (including `Q8_0`, `Q4_0`,
    `BF16`, `F16`, and `F32` control tensors) using true out-of-core streaming:
    1. Strictly blocks edits to banked routed MoE experts (`blk.{L}.ffn_*_exps.weight`).
    2. Dequantizes ONLY targeted control tensors to `float32`, applies `surgical_edits[name](w_f32)`,
       validates finiteness, and re-quantizes back to the tensor's exact original `ggml_type`.
    3. Preserves all scalar and array metadata (including `tokenizer.ggml.*` arrays) and streams
       untouched tensors (including all quantized routed experts) directly from `fin` to `fout`
       in 4MB chunks without buffering the checkpoint in RAM.
    """
    for tname in surgical_edits:
        if is_routed_expert_tensor(tname):
            raise PermissionError(
                f"CircuitForge Safety Gate: Cannot register surgical edit on GGUF routed expert '{tname}'."
            )

    in_path = Path(input_path).resolve()
    out_path = Path(output_path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    info = inspect_gguf_header(in_path)
    tensors_map: dict[str, GGUFTensorInfo] = info["tensors"]  # type: ignore[assignment]
    metadata: dict[str, Any] = dict(info["metadata"])  # type: ignore[arg-type]
    metadata_types: dict[str, int] = dict(info.get("metadata_types", {}))  # type: ignore[arg-type]
    metadata_array_elem_types: dict[str, int] = dict(info.get("metadata_array_elem_types", {}))  # type: ignore[arg-type]
    alignment = int(info["alignment"])  # type: ignore[arg-type]
    data_start_in = int(info["data_start_offset"])  # type: ignore[arg-type]
    in_file_size = in_path.stat().st_size

    missing = [tname for tname in surgical_edits if tname not in tensors_map]
    if missing:
        raise KeyError(f"Target tensor(s) {missing} not found in GGUF file {in_path}")

    # Build sorted relative-offset table for exact gap-based byte length fallback on unknown types
    sorted_by_offset = sorted(tensors_map.values(), key=lambda t: t.relative_offset)
    next_rel_offset: dict[str, int] = {}
    for idx_t, t_obj in enumerate(sorted_by_offset):
        if idx_t + 1 < len(sorted_by_offset):
            next_rel_offset[t_obj.name] = sorted_by_offset[idx_t + 1].relative_offset
        else:
            next_rel_offset[t_obj.name] = max(in_file_size - data_start_in, t_obj.relative_offset)

    updated_names: list[str] = []
    edited_payloads: dict[str, tuple[int, bytes]] = {}
    tensor_byte_lengths: dict[str, int] = {}

    with in_path.open("rb") as fin:
        for name, t_info in tensors_map.items():
            if name in surgical_edits:
                w_f32 = _decode_gguf_tensor_f32_from_handle(fin, data_start_in, t_info)
                edited_f32 = np.ascontiguousarray(surgical_edits[name](w_f32), dtype=np.float32)
                if edited_f32.shape != t_info.shape:
                    raise ValueError(
                        f"Surgical edit changed GGUF tensor '{name}' shape from {t_info.shape} to {edited_f32.shape}"
                    )
                if not np.all(np.isfinite(edited_f32)):
                    raise ValueError(
                        f"Surgical edit on GGUF tensor '{name}' produced NaN or Inf values."
                    )
                ggml_id, raw_bytes = _encode_gguf_tensor_payload(edited_f32, t_info.ggml_type_name)
                edited_payloads[name] = (ggml_id, raw_bytes)
                tensor_byte_lengths[name] = len(raw_bytes)
                updated_names.append(name)
            else:
                if t_info.ggml_type_name in GGML_BLOCK_SPEC:
                    tensor_byte_lengths[name] = ggml_tensor_byte_size(
                        t_info.num_elements, t_info.ggml_type_name, t_info.shape
                    )
                else:
                    gap_bytes = next_rel_offset[name] - t_info.relative_offset
                    if gap_bytes <= 0:
                        raise ValueError(
                            f"Cannot determine byte size for unknown GGUF tensor type '{t_info.ggml_type_name}' on '{name}'"
                        )
                    tensor_byte_lengths[name] = gap_bytes

        metadata["circuitforge.edited_tensors"] = ",".join(updated_names)
        metadata_types["circuitforge.edited_tensors"] = GGUF_VAL_STRING
        tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")

        with tmp_path.open("wb") as fout:
            fout.write(GGUF_MAGIC)
            fout.write(struct.pack("<IQQ", 3, len(tensors_map), len(metadata)))
            for k, v in metadata.items():
                _write_gguf_kv(
                    fout,
                    str(k),
                    v,
                    metadata_types.get(str(k)),
                    metadata_array_elem_types.get(str(k)),
                )

            curr_rel = 0
            for name, t_info in tensors_map.items():
                b_len = tensor_byte_lengths[name]
                ggml_id = edited_payloads[name][0] if name in edited_payloads else t_info.ggml_type_id
                _write_gguf_string(fout, name)
                fout.write(struct.pack("<I", len(t_info.shape)))
                for dim in reversed(t_info.shape):
                    fout.write(struct.pack("<Q", int(dim)))
                fout.write(struct.pack("<IQ", ggml_id, curr_rel))
                curr_rel = _align_offset(curr_rel + b_len, alignment)

            header_end = fout.tell()
            data_start_out = _align_offset(header_end, alignment)
            if data_start_out > header_end:
                fout.write(b"\x00" * (data_start_out - header_end))

            curr_pos = data_start_out
            for name, t_info in tensors_map.items():
                b_len = tensor_byte_lengths[name]
                if name in edited_payloads:
                    fout.write(edited_payloads[name][1])
                else:
                    fin.seek(data_start_in + t_info.relative_offset)
                    rem = b_len
                    while rem > 0:
                        chunk = fin.read(min(rem, 4 * 1024 * 1024))
                        if not chunk:
                            raise ValueError(
                                f"Unexpected EOF while streaming GGUF tensor '{name}' from {in_path}"
                            )
                        fout.write(chunk)
                        rem -= len(chunk)
                curr_pos += b_len
                aligned_pos = _align_offset(curr_pos, alignment)
                if aligned_pos > curr_pos:
                    fout.write(b"\x00" * (aligned_pos - curr_pos))
                    curr_pos = aligned_pos

    if tmp_path != out_path:
        tmp_path.replace(out_path)
    return updated_names
