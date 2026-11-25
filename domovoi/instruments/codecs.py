"""Canonical little-endian CPU weight formats. None is a GPU decoder."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from functools import lru_cache
import struct

import numpy as np

GROUP = 64
ALIGN = 64
HEADER = struct.Struct("<4sB3xII")
MAGIC = b"DMC1"


class Format(IntEnum):
    BF16 = 1
    INT8 = 2
    INT4 = 3
    SCALAR2 = 4
    TERNARY = 5
    ROTATED2 = 6
    E8 = 7
    RESIDUAL = 8
    LEARNED = 9


_VECTOR = {Format.ROTATED2, Format.E8, Format.RESIDUAL, Format.LEARNED}
_BOOK = {Format.RESIDUAL, Format.LEARNED}
SCALAR_LEVELS = np.array([-3, -1, 1, 3], dtype=np.int8)
TRIT_POWERS = np.array([1, 3, 9, 27, 81], dtype=np.int64)


def _ceil(n, divisor):
    return (n + divisor - 1) // divisor


def _pad(n, unit=ALIGN):
    return _ceil(n, unit) * unit


def _sizes(fmt, n):
    count = _pad(n, 8) if fmt in _VECTOR else n
    if fmt == Format.BF16:
        return 2 * n, 0, 0
    size = {
        Format.INT8: n, Format.INT4: _ceil(n, 2), Format.SCALAR2: _ceil(n, 4),
        Format.TERNARY: _ceil(n, 5), Format.ROTATED2: count // 4,
        Format.E8: count // 4, Format.RESIDUAL: count // 4, Format.LEARNED: count // 4,
    }[fmt]
    return size, 2 * _ceil(n, GROUP), 8192 if fmt in _BOOK else 0


def _dimensions(shape):
    if (not isinstance(shape, tuple) or len(shape) != 2 or
            any(type(dim) is not int or not 0 < dim < 2**32 for dim in shape)):
        raise ValueError("Expected a positive two-dimensional shape.")
    return shape[0] * shape[1]


@dataclass(frozen=True)
class PackedMatrix:
    format: Format
    shape: tuple[int, int]
    payload: bytes
    scales: bytes = b""
    books: bytes = b""

    def __post_init__(self):
        n = _dimensions(self.shape)
        if not isinstance(self.format, Format):
            raise ValueError("Unknown codec format.")
        parts = (self.payload, self.scales, self.books)
        if any(type(part) is not bytes for part in parts) or tuple(map(len, parts)) != _sizes(self.format, n):
            raise ValueError("Incorrect packed segment sizes.")
        scales = np.frombuffer(self.scales, dtype="<f2")
        if not np.isfinite(scales).all() or (scales <= 0).any():
            raise ValueError("Packed scales must be finite and strictly positive.")
        if self.books:
            books = np.frombuffer(self.books, dtype="<f2").reshape(2, 256, 8)
            if not np.isfinite(books).all():
                raise ValueError("Book coordinates must be finite.")
            if self.format == Format.RESIDUAL and self.books != residual_books().astype("<f2").tobytes():
                raise ValueError("Fixed residual books do not match codec v1.")
        if self.format == Format.BF16 and not np.isfinite(_decode_bf16(self.payload)).all():
            raise ValueError("Only finite BF16 values are supported.")
        if self.format == Format.INT8 and (np.frombuffer(self.payload, dtype="i1") == -128).any():
            raise ValueError("Invalid symmetric INT8 code.")
        if self.format == Format.INT4 and (_unpack_bits(self.payload, 4, n) == 15).any():
            raise ValueError("Index 15 is unused in symmetric INT4.")
        if self.format == Format.SCALAR2:
            _unpack_bits(self.payload, 2, n)
        if self.format == Format.TERNARY:
            _unpack_trits(self.payload, n)
        if self.format == Format.E8 and (np.frombuffer(self.payload, dtype="<u2") == 65535).any():
            raise ValueError("E8 index 65535 is reserved.")

    def byte_counts(self) -> dict[str, int]:
        lengths = (HEADER.size, len(self.payload), len(self.scales), len(self.books))
        return dict(index_metadata=lengths[0], packed_weights=lengths[1], scales=lengths[2],
                    codebooks=lengths[3], alignment=sum(_pad(n) - n for n in lengths))

    def to_bytes(self) -> bytes:
        parts = (HEADER.pack(MAGIC, self.format, *self.shape),
                 self.payload, self.scales, self.books)
        return b"".join(part + bytes(_pad(len(part)) - len(part)) for part in parts)

    @classmethod
    def from_bytes(cls, blob: bytes) -> PackedMatrix:
        if type(blob) is not bytes or len(blob) < ALIGN:
            raise ValueError("Truncated codec header.")
        magic, format_id, rows, cols = HEADER.unpack_from(blob)
        if magic != MAGIC or any(blob[5:8]) or any(blob[HEADER.size:ALIGN]):
            raise ValueError("Invalid codec header or header padding.")
        fmt = Format(format_id)
        lens = _sizes(fmt, _dimensions((rows, cols)))
        if len(blob) != ALIGN + sum(map(_pad, lens)):
            raise ValueError("Invalid packed length.")
        cursor, parts = ALIGN, []
        for n in lens:
            if any(blob[cursor + n:cursor + _pad(n)]):
                raise ValueError("Alignment padding must be zero.")
            parts.append(blob[cursor:cursor + n])
            cursor += _pad(n)
        return cls(fmt, (rows, cols), *parts)


def _encode_bf16(values):
    bits = values.astype("<f4", copy=False).view("<u4")
    return ((bits + 0x7fff + ((bits >> 16) & 1)) >> 16).astype("<u2").tobytes()


def _decode_bf16(payload):
    return (np.frombuffer(payload, dtype="<u2").astype("<u4") << 16).view("<f4")


def _pack_bits(indices, bits):
    fields = 8 // bits
    indices = np.pad(np.asarray(indices, dtype=np.uint8), (0, -len(indices) % fields))
    return np.bitwise_or.reduce(indices.reshape(-1, fields) << (bits * np.arange(fields)),
                                axis=1).astype(np.uint8).tobytes()


def _unpack_bits(payload, bits, n):
    shifts = bits * np.arange(8 // bits)
    fields = ((np.frombuffer(payload, dtype=np.uint8)[:, None] >> shifts)
              & ((1 << bits) - 1)).ravel()
    if fields[n:].any():
        raise ValueError("Unused scalar fields must be zero.")
    return fields[:n]


def _pack_trits(indices):
    padded = np.pad(indices, (0, -len(indices) % 5)).reshape(-1, 5)
    return (padded @ TRIT_POWERS).astype(np.uint8).tobytes()


def _unpack_trits(payload, n):
    packed = np.frombuffer(payload, dtype=np.uint8)
    if (packed >= 243).any():
        raise ValueError("A base-3 byte must be less than 243.")
    fields = (packed[:, None] // TRIT_POWERS % 3).ravel()
    if fields[n:].any():
        raise ValueError("Unused trits must be zero.")
    return fields[:n]


def hadamard8():
    """Normalized Sylvester matrix with standard row and column order."""
    matrix = np.ones((1, 1))
    for _ in range(3):
        matrix = np.block([[matrix, matrix], [matrix, -matrix]])
    return matrix / np.sqrt(8)


def residual_books(first=None, second=None):
    """Sign order is bit 0 first. Default bases I and H have no parallel columns."""
    first = np.eye(8) if first is None else np.asarray(first, dtype=np.float64)
    second = hadamard8() if second is None else np.asarray(second, dtype=np.float64)
    for basis in (first, second):
        if basis.shape != (8, 8) or not np.allclose(basis.T @ basis, np.eye(8), atol=1e-12, rtol=0):
            raise ValueError("A fixed residual basis must be orthogonal.")
    signs = 2 * ((np.arange(256)[:, None] >> np.arange(8)) & 1) - 1
    return np.stack((signs @ first.T, 0.5 * signs @ second.T))


@lru_cache(maxsize=1)
def e8_doubled_points():
    """Finite E8 core v1: zero, then 32,767 ordered antipodal pairs.

    Use doubled coordinates so both cosets and tie order are exact integers.
    The table is an implicit format definition but this CPU reference materializes
    it. Include the table as an extra allocation if a backend retains one.
    """
    pairs = []

    def enumerate_coset(prefix, remaining, grid):
        if len(prefix) == 8:
            if sum(prefix) % 4 == 0 and next((x for x in prefix if x), 0) > 0:
                pairs.append((48 - remaining, prefix))
            return
        for coord in grid:
            after = remaining - coord * coord
            if after >= 0:
                enumerate_coset((*prefix, coord), after, grid)

    enumerate_coset((), 48, (0, -2, 2, -4, 4, -6, 6))
    enumerate_coset((), 48, (-1, 1, -3, 3, -5, 5))
    pairs.sort()
    if len(pairs) < 32767:
        raise RuntimeError("E8 code lattice enumeration is incomplete.")
    table = np.zeros((65535, 8), dtype=np.int8)
    table[1::2] = [coords for _, coords in pairs[:32767]]
    table[2::2] = -table[1::2]
    table.flags.writeable = False
    return table


def _nearest(vectors, points):
    result = []
    norms = np.einsum("ij,ij->i", points, points)
    for start in range(0, len(vectors), 32):
        batch = vectors[start:start + 32]
        result.extend(np.argmin(norms - 2 * batch @ points.T, axis=1))
    return np.asarray(result)


def residual_indices(vectors, books):
    vectors = np.asarray(vectors, dtype=np.float64)
    books = np.asarray(books, dtype=np.float64)
    if (vectors.ndim != 2 or vectors.shape[1] != 8 or books.shape != (2, 256, 8)
            or not np.isfinite(vectors).all() or not np.isfinite(books).all()):
        raise ValueError("Expected finite (n, 8) vectors and two (256, 8) books.")
    first = _nearest(vectors, books[0])
    second = _nearest(vectors - books[0, first], books[1])
    return np.stack((first, second), axis=1).astype(np.uint8)


def _group_scales(values, span):
    padded = np.pad(values, (0, -len(values) % GROUP))
    maxima = np.abs(padded.reshape(-1, GROUP)).max(axis=1)
    smallest = np.nextafter(np.float16(0), np.float16(1), dtype=np.float16)
    scale = np.clip(maxima / span, smallest, np.finfo(np.float16).max).astype("<f2")
    return scale, scale.astype(np.float64).repeat(GROUP)[:len(values)]


def pack(weights, fmt: Format, *, learned_books=None) -> PackedMatrix:
    weights = np.asarray(weights, dtype=np.float32)
    n = _dimensions(weights.shape)
    if not isinstance(fmt, Format) or not np.isfinite(weights).all():
        raise ValueError("Expected finite weights and a known codec.")
    if (learned_books is not None) != (fmt == Format.LEARNED):
        raise ValueError("Supply two learned tables only for the learned format.")
    shape, values = weights.shape, weights.ravel().astype(np.float64)
    if fmt == Format.BF16:
        return PackedMatrix(fmt, shape, _encode_bf16(weights.ravel()))
    if fmt in _VECTOR:
        values = np.pad(values, (0, -n % 8))
    if fmt == Format.ROTATED2:
        values = (values.reshape(-1, 8) @ hadamard8()).ravel()
    spans = {Format.INT8: 127, Format.INT4: 7, Format.SCALAR2: 3, Format.ROTATED2: 3,
             Format.TERNARY: 1, Format.E8: 2, Format.RESIDUAL: 1.5, Format.LEARNED: 1.5}
    scale, repeats = _group_scales(values, spans[fmt])
    normalized = values / repeats
    books = b""
    if fmt == Format.INT8:
        payload = normalized.round().clip(-127, 127).astype("i1").tobytes()
    elif fmt == Format.INT4:
        payload = _pack_bits(normalized.round().clip(-7, 7).astype(np.int8) + 7, 4)
    elif fmt in (Format.SCALAR2, Format.ROTATED2):
        payload = _pack_bits(np.abs(normalized[:, None] - SCALAR_LEVELS).argmin(axis=1), 2)
    elif fmt == Format.TERNARY:
        payload = _pack_trits(normalized.round().clip(-1, 1).astype(np.int8) + 1)
    elif fmt == Format.E8:
        points = e8_doubled_points().astype(np.float64) / 2
        payload = _nearest(normalized.reshape(-1, 8), points).astype("<u2").tobytes()
    else:
        tables = np.asarray(residual_books() if fmt == Format.RESIDUAL else learned_books, dtype="<f2")
        payload = residual_indices(normalized.reshape(-1, 8), tables).tobytes()
        books = tables.tobytes()
    return PackedMatrix(fmt, shape, payload, scale.tobytes(), books)


def unpack(matrix: PackedMatrix) -> np.ndarray:
    n, fmt = _dimensions(matrix.shape), matrix.format
    if fmt == Format.BF16:
        return _decode_bf16(matrix.payload).reshape(matrix.shape)
    count = _pad(n, 8) if fmt in _VECTOR else n
    if fmt == Format.INT8:
        values = np.frombuffer(matrix.payload, dtype="i1")
    elif fmt == Format.INT4:
        values = _unpack_bits(matrix.payload, 4, n) - 7
    elif fmt in (Format.SCALAR2, Format.ROTATED2):
        values = SCALAR_LEVELS[_unpack_bits(matrix.payload, 2, count)]
    elif fmt == Format.TERNARY:
        values = _unpack_trits(matrix.payload, n) - 1
    elif fmt == Format.E8:
        values = e8_doubled_points()[np.frombuffer(matrix.payload, dtype="<u2")].ravel() / 2
    else:
        tables = np.frombuffer(matrix.books, dtype="<f2").astype(np.float64).reshape(2, 256, 8)
        indices = np.frombuffer(matrix.payload, dtype=np.uint8).reshape(-1, 2)
        values = (tables[0, indices[:, 0]] + tables[1, indices[:, 1]]).ravel()
    scale = np.frombuffer(matrix.scales, dtype="<f2").astype(np.float64)
    result = values * scale.repeat(GROUP)[:count]
    if fmt == Format.ROTATED2:
        result = (result.reshape(-1, 8) @ hadamard8().T).ravel()
    return result[:n].astype(np.float32).reshape(matrix.shape)


def linear_reference(inputs, matrix: PackedMatrix):
    """FP64 CPU oracle for a weight matrix; checks representation, no throughput."""
    values = np.asarray(inputs, dtype=np.float64)
    return values @ unpack(matrix).astype(np.float64).T
