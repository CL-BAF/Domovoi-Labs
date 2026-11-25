"""Separate coded matrices, fixed tensors and retained copies in byte receipts."""

from __future__ import annotations

from dataclasses import dataclass

from .codecs import PackedMatrix


@dataclass(frozen=True)
class Storage:
    """One distinct allocation. Represent a tied/aliased tensor once."""

    name: str
    role: str
    size: int

    def __post_init__(self):
        if (not isinstance(self.name, str) or not self.name or
                self.role not in {"embedding", "attention", "norm", "duplicate", "lookup", "other"}
                or type(self.size) is not int or self.size < 0):
            raise ValueError("Invalid retained tensor allocation.")


def representation_bytes(matrices: dict[str, PackedMatrix], tensors: list[Storage]) -> dict:
    counts = {k: 0 for k in ("packed_weights", "scales", "codebooks", "index_metadata", "alignment")}
    for matrix in matrices.values():
        for field, size in matrix.byte_counts().items():
            counts[field] += size
    names = list(matrices) + [x.name for x in tensors]
    if any(not isinstance(name, str) or not name for name in names) or len(set(names)) != len(names):
        raise ValueError("Each allocation needs a unique nonempty name.")
    fixed = {key: sum(t.size for t in tensors if t.role == key)
             for key in ("embedding", "attention", "norm", "duplicate", "lookup", "other")}
    coded = sum(counts.values())
    return {"coded_matrices": counts, "coded_matrix_bytes": coded, "other_allocations": fixed,
            "representation_bytes": coded + sum(fixed.values())}
