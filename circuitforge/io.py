"""
CircuitForge Streaming Safetensors & Checkpoint I/O Engine (`circuitforge/io.py`).

Implements a pure-Python/NumPy zero-dependency `.safetensors` binary reader, header inspector,
slice loader, and writer, so open-weight researchers can inspect, slice, apply surgical deltas,
and save modified `.safetensors` checkpoints without loading entire multi-gigabyte models into
GPU VRAM or requiring external C bindings.

Supported dtypes:
- `F32` (float32), `F64` (float64), `F16` (float16), `BF16` (bfloat16 converted to/from float32)
- `I8`, `I16`, `I32`, `I64`, `U8`
"""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

from circuitforge.wiring import is_routed_expert_tensor

_DTYPE_TO_NUMPY: dict[str, np.dtype[Any]] = {
    "F32": np.dtype("<f4"),
    "F64": np.dtype("<f8"),
    "F16": np.dtype("<f2"),
    "I8": np.dtype("<i1"),
    "I16": np.dtype("<i2"),
    "I32": np.dtype("<i4"),
    "I64": np.dtype("<i8"),
    "U8": np.dtype("<u1"),
}

_NUMPY_TO_DTYPE: dict[str, str] = {
    "float32": "F32",
    "float64": "F64",
    "float16": "F16",
    "int8": "I8",
    "int16": "I16",
    "int32": "I32",
    "int64": "I64",
    "uint8": "U8",
}


@dataclass(frozen=True)
class SafetensorEntry:
    """Metadata for a single tensor inside a `.safetensors` file."""

    name: str
    dtype: str
    shape: tuple[int, ...]
    data_offsets: tuple[int, int]

    @property
    def byte_length(self) -> int:
        return self.data_offsets[1] - self.data_offsets[0]


def inspect_safetensors_header(
    path: str | Path,
) -> tuple[dict[str, SafetensorEntry], dict[str, str], int]:
    """
    Reads only the JSON header of a `.safetensors` file without loading tensor buffers.
    Returns `(entries_dict, metadata_dict, data_start_byte_offset)`.
    """
    p = Path(path)
    file_size = p.stat().st_size
    with p.open("rb") as f:
        header_len_bytes = f.read(8)
        if len(header_len_bytes) < 8:
            raise ValueError(f"Invalid or truncated safetensors file: {p}")
        (header_len,) = struct.unpack("<Q", header_len_bytes)
        if header_len > 100_000_000:
            raise ValueError(f"Safetensors header length {header_len} exceeds 100MB safety limit in: {p}")
        header_raw = f.read(header_len)
        if len(header_raw) < header_len:
            raise ValueError(f"Truncated safetensors header in: {p}")

    raw_obj: dict[str, Any] = json.loads(header_raw.decode("utf-8"))
    metadata: dict[str, str] = {}
    entries: dict[str, SafetensorEntry] = {}
    data_start = 8 + header_len

    for k, val in raw_obj.items():
        if k == "__metadata__":
            if isinstance(val, dict):
                metadata = {str(mk): str(mv) for mk, mv in val.items()}
            continue
        offsets = val["data_offsets"]
        s_off, e_off = int(offsets[0]), int(offsets[1])
        if s_off < 0 or e_off < s_off or (data_start + e_off) > file_size:
            raise ValueError(
                f"Invalid or out-of-bounds safetensors data_offsets {[s_off, e_off]} "
                f"(file_size={file_size}, data_start={data_start}) for tensor '{k}' in {p}"
            )
        entries[k] = SafetensorEntry(
            name=k,
            dtype=str(val["dtype"]),
            shape=tuple(int(dim) for dim in val["shape"]),
            data_offsets=(s_off, e_off),
        )

    return entries, metadata, data_start


def _bf16_bytes_to_f32(raw_bytes: bytes, shape: tuple[int, ...]) -> np.ndarray:
    """Converts little-endian BF16 raw bytes into a float32 NumPy array."""
    u16 = np.frombuffer(raw_bytes, dtype="<u2")
    u32 = u16.astype(np.uint32) << 16
    return u32.view(np.float32).reshape(shape).copy()


def _f32_to_bf16_bytes(arr: np.ndarray) -> bytes:
    """Converts a float32 NumPy array into little-endian BF16 raw bytes (preserving IEEE 754 NaN/Inf)."""
    f32 = np.asarray(arr, dtype=np.float32)
    u32 = f32.view(np.uint32)
    exp_mask = np.uint32(0x7F800000)
    mant_mask = np.uint32(0x007FFFFF)
    is_special = (u32 & exp_mask) == exp_mask
    is_nan = is_special & ((u32 & mant_mask) != 0)
    rounding_bias = np.uint32(0x00007FFF) + ((u32 >> np.uint32(16)) & np.uint32(1))
    rounded_u16 = ((u32 + rounding_bias) >> np.uint32(16)).astype(np.uint16)
    trunc_u16 = (u32 >> np.uint32(16)).astype(np.uint16)
    nan_u16 = (trunc_u16 | np.uint16(0x0040)).astype(np.uint16)
    bf16_u16 = np.where(is_nan, nan_u16, np.where(is_special, trunc_u16, rounded_u16)).astype("<u2")
    return bf16_u16.tobytes()


def load_safetensor_slice(path: str | Path, tensor_name: str) -> np.ndarray:
    """
    Streams a single named tensor from a `.safetensors` file by seeking directly
    to its byte offset without reading other tensors into memory.
    """
    entries, _, data_start = inspect_safetensors_header(path)
    if tensor_name not in entries:
        raise KeyError(f"Tensor '{tensor_name}' not found in {path}")

    entry = entries[tensor_name]
    start, end = entry.data_offsets
    nbytes = end - start
    if nbytes < 0:
        raise ValueError(f"Invalid negative byte length ({nbytes}) for tensor '{tensor_name}' in {path}")

    with Path(path).open("rb") as f:
        f.seek(data_start + start)
        raw = f.read(nbytes)
        if len(raw) != nbytes:
            raise ValueError(f"Truncated safetensors payload for '{tensor_name}' in {path}: expected {nbytes}, got {len(raw)}")

    if entry.dtype == "BF16":
        return _bf16_bytes_to_f32(raw, entry.shape)

    if entry.dtype not in _DTYPE_TO_NUMPY:
        raise ValueError(f"Unsupported safetensors dtype: {entry.dtype}")

    np_dtype = _DTYPE_TO_NUMPY[entry.dtype]
    return np.frombuffer(raw, dtype=np_dtype).reshape(entry.shape).copy()


def load_safetensors_file(
    path: str | Path,
) -> tuple[dict[str, np.ndarray], dict[str, str]]:
    """Loads all tensors and metadata from a `.safetensors` file."""
    entries, metadata, _ = inspect_safetensors_header(path)
    tensors: dict[str, np.ndarray] = {}
    for name in entries:
        tensors[name] = load_safetensor_slice(path, name)
    return tensors, metadata


def save_safetensors_file(
    path: str | Path,
    tensors: dict[str, np.ndarray],
    metadata: dict[str, str] | None = None,
    preserve_dtypes: dict[str, str] | None = None,
) -> Path:
    """
    Serializes a dictionary of NumPy arrays into the standard `.safetensors` binary format
    (8-byte little-endian header size + UTF-8 JSON header padded to 8-byte alignment + raw tensor buffers).
    """
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    header_dict: dict[str, Any] = {}
    if metadata:
        header_dict["__metadata__"] = {str(k): str(v) for k, v in metadata.items()}

    encoded_buffers: list[bytes] = []
    current_offset = 0

    for name, arr in tensors.items():
        np_arr = np.ascontiguousarray(arr)
        target_st_dtype = (preserve_dtypes or {}).get(name)
        if target_st_dtype == "BF16":
            raw_bytes = _f32_to_bf16_bytes(np_arr)
            st_dtype = "BF16"
        elif target_st_dtype and target_st_dtype in _DTYPE_TO_NUMPY:
            st_dtype = target_st_dtype
            le_dtype = _DTYPE_TO_NUMPY[st_dtype]
            raw_bytes = np_arr.astype(le_dtype, copy=False).tobytes()
        else:
            dtype_name = np_arr.dtype.name
            if dtype_name not in _NUMPY_TO_DTYPE:
                np_arr = np_arr.astype(np.float32)
                dtype_name = "float32"
            st_dtype = _NUMPY_TO_DTYPE[dtype_name]
            le_dtype = _DTYPE_TO_NUMPY[st_dtype]
            raw_bytes = np_arr.astype(le_dtype, copy=False).tobytes()

        next_offset = current_offset + len(raw_bytes)
        header_dict[name] = {
            "dtype": st_dtype,
            "shape": list(np_arr.shape),
            "data_offsets": [current_offset, next_offset],
        }
        encoded_buffers.append(raw_bytes)
        current_offset = next_offset

    header_json = json.dumps(header_dict, separators=(",", ":")).encode("utf-8")
    # Pad header with spaces to 8-byte alignment per safetensors specification
    pad_len = (8 - (len(header_json) % 8)) % 8
    header_padded = header_json + (b" " * pad_len)

    with out_path.open("wb") as f:
        f.write(struct.pack("<Q", len(header_padded)))
        f.write(header_padded)
        for buf in encoded_buffers:
            f.write(buf)

    return out_path


def stream_surgical_safetensors_update(
    input_path: str | Path,
    output_path: str | Path,
    surgical_edits: dict[str, Callable[[np.ndarray], np.ndarray]],
) -> list[str]:
    """
    Streams through a `.safetensors` shard, copies untouched tensors directly at the byte level
    (including all frozen routed MoE experts and any exotic/quantized dtypes), applies
    `surgical_edits[tensor_name](weight)` only to targeted control tensors, and writes the
    updated `.safetensors` file.
    """
    for tname in surgical_edits:
        if is_routed_expert_tensor(tname):
            raise PermissionError(
                f"CircuitForge Safety Gate: Cannot register surgical edit on routed expert tensor '{tname}'."
            )

    in_path = Path(input_path)
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    entries, metadata, data_start = inspect_safetensors_header(in_path)
    missing_targets = [tname for tname in surgical_edits if tname not in entries]
    if missing_targets:
        raise KeyError(
            f"Target tensor(s) {missing_targets} not found in safetensors file {in_path}"
        )
    updated_names: list[str] = []
    edited_payloads: dict[str, bytes] = {}
    header_dict: dict[str, Any] = {}

    for name, entry in entries.items():
        if name in surgical_edits:
            arr = load_safetensor_slice(in_path, name)
            edited = np.ascontiguousarray(surgical_edits[name](arr))
            if edited.shape != arr.shape:
                raise ValueError(
                    f"Surgical edit changed shape of '{name}' from {arr.shape} to {edited.shape}"
                )
            if entry.dtype == "BF16":
                raw_bytes = _f32_to_bf16_bytes(edited)
            elif entry.dtype in _DTYPE_TO_NUMPY:
                raw_bytes = edited.astype(_DTYPE_TO_NUMPY[entry.dtype], copy=False).tobytes()
            else:
                raise ValueError(f"Unsupported safetensors dtype for surgical edit: {entry.dtype}")
            edited_payloads[name] = raw_bytes
            updated_names.append(name)

    meta_out = dict(metadata)
    meta_out["circuitforge_edited_tensors"] = ",".join(updated_names)
    header_dict["__metadata__"] = {str(k): str(v) for k, v in meta_out.items()}

    current_offset = 0
    for name, entry in entries.items():
        b_len = len(edited_payloads[name]) if name in edited_payloads else entry.byte_length
        next_offset = current_offset + b_len
        header_dict[name] = {
            "dtype": entry.dtype,
            "shape": list(entry.shape),
            "data_offsets": [current_offset, next_offset],
        }
        current_offset = next_offset

    header_json = json.dumps(header_dict, separators=(",", ":")).encode("utf-8")
    pad_len = (8 - (len(header_json) % 8)) % 8
    header_padded = header_json + (b" " * pad_len)

    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp") if in_path.resolve() == out_path.resolve() else out_path
    with in_path.open("rb") as fin, tmp_path.open("wb") as fout:
        fout.write(struct.pack("<Q", len(header_padded)))
        fout.write(header_padded)
        for name, entry in entries.items():
            if name in edited_payloads:
                fout.write(edited_payloads[name])
            else:
                # Stream untouched tensor bytes directly from disk without decoding
                fin.seek(data_start + entry.data_offsets[0])
                rem = entry.byte_length
                while rem > 0:
                    chunk = fin.read(min(rem, 4 * 1024 * 1024))
                    if not chunk:
                        raise ValueError(f"Unexpected EOF while streaming tensor '{name}' from {in_path}")
                    fout.write(chunk)
                    rem -= len(chunk)

    if tmp_path != out_path:
        tmp_path.replace(out_path)
    return updated_names
