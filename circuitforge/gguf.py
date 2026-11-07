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
from typing import BinaryIO

import numpy as np

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
    24: "I8",
    25: "I16",
    26: "I32",
    27: "I64",
    28: "F64",
    30: "BF16",
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


def _read_gguf_value(f: BinaryIO, val_type: int, _depth: int = 0) -> object:
    if _depth > 8:
        raise ValueError("GGUF nested array depth exceeds safety limit (8)")
    if val_type == GGUF_VAL_UINT8:
        return struct.unpack("<B", _read_exact(f, 1))[0]
    if val_type == GGUF_VAL_INT8:
        return struct.unpack("<b", _read_exact(f, 1))[0]
    if val_type == GGUF_VAL_UINT16:
        return struct.unpack("<H", _read_exact(f, 2))[0]
    if val_type == GGUF_VAL_INT16:
        return struct.unpack("<h", _read_exact(f, 2))[0]
    if val_type == GGUF_VAL_UINT32:
        return struct.unpack("<I", _read_exact(f, 4))[0]
    if val_type == GGUF_VAL_INT32:
        return struct.unpack("<i", _read_exact(f, 4))[0]
    if val_type == GGUF_VAL_FLOAT32:
        return float(struct.unpack("<f", _read_exact(f, 4))[0])
    if val_type == GGUF_VAL_BOOL:
        return bool(struct.unpack("<?", _read_exact(f, 1))[0])
    if val_type == GGUF_VAL_STRING:
        return _read_gguf_string(f)
    if val_type == GGUF_VAL_UINT64:
        return struct.unpack("<Q", _read_exact(f, 8))[0]
    if val_type == GGUF_VAL_INT64:
        return struct.unpack("<q", _read_exact(f, 8))[0]
    if val_type == GGUF_VAL_FLOAT64:
        return float(struct.unpack("<d", _read_exact(f, 8))[0])
    if val_type == GGUF_VAL_ARRAY:
        elem_type, count = struct.unpack("<IQ", _read_exact(f, 12))
        if count > 1_000_000:
            raise ValueError(f"GGUF metadata array count {count} exceeds 1,000,000 safety limit")
        return [_read_gguf_value(f, elem_type, _depth + 1) for _ in range(count)]
    raise ValueError(f"Unsupported GGUF metadata value type ID: {val_type}")


def _align_offset(pos: int, alignment: int = GGUF_DEFAULT_ALIGNMENT) -> int:
    if alignment <= 0 or (alignment & (alignment - 1)) != 0:
        raise ValueError(f"GGUF alignment must be a positive power of 2, got {alignment}")
    rem = pos % alignment
    return pos if rem == 0 else pos + (alignment - rem)


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
        for _ in range(kv_count):
            key = _read_gguf_string(f)
            (val_type,) = struct.unpack("<I", _read_exact(f, 4))
            metadata[key] = _read_gguf_value(f, val_type)

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
        if t_info.ggml_type_name == "F32":
            elem_bytes = 4
        elif t_info.ggml_type_name in ("F16", "BF16", "I16"):
            elem_bytes = 2
        elif t_info.ggml_type_name in ("F64", "I64"):
            elem_bytes = 8
        elif t_info.ggml_type_name == "I32":
            elem_bytes = 4
        else:
            elem_bytes = 1
        min_bytes = t_info.num_elements * elem_bytes // (2 if t_info.ggml_type_name.startswith("Q") else 1)
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
        "tensors": tensors,
    }


def write_gguf_file(
    path: str | Path,
    tensors: dict[str, np.ndarray],
    metadata: dict[str, str | int | float | bool] | None = None,
) -> Path:
    """
    Writes a valid GGUF v3 binary file containing `tensors` (`float32` or `float16`) and
    `metadata` key-value pairs with 32-byte data alignment.
    """
    out_path = Path(path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    meta = dict(metadata or {})
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
        if arr.dtype == np.float16:
            ggml_type = 1  # F16
        else:
            arr = arr.astype(np.float32)
            ggml_type = 0  # F32
        raw_bytes = arr.tobytes()
        encoded_blobs.append((name, arr.shape, ggml_type, raw_bytes))
        rel_offset = _align_offset(rel_offset + len(raw_bytes), alignment)

    with out_path.open("wb") as f:
        f.write(GGUF_MAGIC)
        f.write(struct.pack("<IQQ", 3, len(encoded_blobs), len(meta)))

        for k, v in meta.items():
            _write_gguf_string(f, str(k))
            if isinstance(v, bool):
                f.write(struct.pack("<I?", GGUF_VAL_BOOL, v))
            elif isinstance(v, int):
                iv = int(v)
                if iv < 0:
                    if iv >= -2147483648:
                        f.write(struct.pack("<Ii", GGUF_VAL_INT32, iv))
                    else:
                        f.write(struct.pack("<Iq", GGUF_VAL_INT64, iv))
                elif iv <= 4294967295:
                    f.write(struct.pack("<II", GGUF_VAL_UINT32, iv))
                else:
                    f.write(struct.pack("<IQ", GGUF_VAL_UINT64, iv))
            elif isinstance(v, float):
                f.write(struct.pack("<If", GGUF_VAL_FLOAT32, float(v)))
            else:
                f.write(struct.pack("<I", GGUF_VAL_STRING))
                _write_gguf_string(f, str(v))

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

    return out_path


def load_gguf_tensor_f32(path: str | Path, tensor_name: str) -> np.ndarray:
    """Slices and decodes a single `F32` or `F16` tensor from a `.gguf` file into a float32 NumPy array."""
    info = inspect_gguf_header(path)
    tensors_map: dict[str, GGUFTensorInfo] = info["tensors"]  # type: ignore[assignment]
    if tensor_name not in tensors_map:
        raise KeyError(f"Tensor '{tensor_name}' not found in GGUF file {path}")

    t_info = tensors_map[tensor_name]
    data_start = int(info["data_start_offset"])  # type: ignore[arg-type]
    abs_offset = data_start + t_info.relative_offset

    if t_info.ggml_type_name == "F32":
        dtype = np.float32
        elem_bytes = 4
    elif t_info.ggml_type_name == "F16":
        dtype = np.float16
        elem_bytes = 2
    else:
        raise ValueError(f"Direct float slicing supports F32/F16, got {t_info.ggml_type_name}")

    byte_len = t_info.num_elements * elem_bytes
    with Path(path).open("rb") as f:
        f.seek(abs_offset)
        buf = _read_exact(f, byte_len)

    arr = np.frombuffer(buf, dtype=dtype).reshape(t_info.shape).copy()
    return arr.astype(np.float32)
