"""Exact packed-binary index used for transparent scalability benchmarks."""

from __future__ import annotations

import numpy as np


_POPCOUNT = np.unpackbits(np.arange(256, dtype=np.uint8)[:, None], axis=1).sum(axis=1)


class PackedBinaryIndex:
    """Vectorized exact scan over packed binary descriptors.

    This implementation deliberately does not call radius 16 a small Hamming
    ball. It reports exact-scan cost and shortlist size without enumerating the
    combinatorial 64-bit ball.
    """

    def __init__(self, packed_hashes: np.ndarray):
        hashes = np.asarray(packed_hashes, dtype=np.uint8)
        if hashes.ndim != 2:
            raise ValueError("packed_hashes must have shape [n, bytes]")
        self.hashes = np.ascontiguousarray(hashes)

    @classmethod
    def from_bits(cls, hashes: np.ndarray) -> "PackedBinaryIndex":
        bits = np.asarray(hashes, dtype=np.uint8)
        return cls(np.packbits(bits, axis=1, bitorder="big"))

    def distances(self, query_packed: np.ndarray) -> np.ndarray:
        query = np.asarray(query_packed, dtype=np.uint8).reshape(1, -1)
        if query.shape[1] != self.hashes.shape[1]:
            raise ValueError("query length does not match index descriptors")
        xor = np.bitwise_xor(self.hashes, query)
        return _POPCOUNT[xor].sum(axis=1, dtype=np.uint16)

    def radius_search(self, query_packed: np.ndarray, radius: int) -> tuple[np.ndarray, np.ndarray]:
        distances = self.distances(query_packed)
        indices = np.flatnonzero(distances <= radius)
        return indices, distances[indices]

