#!/usr/bin/env python
"""Benchmark exact packed-Hamming lookup at increasing synthetic index sizes."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / "code" / "src"))

from nth_registry.index import PackedBinaryIndex


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--queries", type=int, default=50)
    parser.add_argument("--radius", type=int, default=16)
    parser.add_argument("--sizes", type=int, nargs="+", default=[1000, 10000, 100000, 1000000])
    parser.add_argument("--output", type=Path, default=PROJECT / "results" / "index_scaling.csv")
    args = parser.parse_args()
    rng = np.random.default_rng(20260830)
    rows = []
    for size in args.sizes:
        hashes = rng.integers(0, 256, size=(size, 8), dtype=np.uint8)
        queries = rng.integers(0, 256, size=(args.queries, 8), dtype=np.uint8)
        index = PackedBinaryIndex(hashes)
        timings = []
        counts = []
        for query in queries:
            start = time.perf_counter_ns()
            found, _ = index.radius_search(query, args.radius)
            timings.append((time.perf_counter_ns() - start) / 1e6)
            counts.append(len(found))
        rows.append({
            "index_size": size,
            "queries": args.queries,
            "radius": args.radius,
            "mean_lookup_ms": float(np.mean(timings)),
            "p95_lookup_ms": float(np.quantile(timings, 0.95)),
            "mean_shortlist": float(np.mean(counts)),
            "p95_shortlist": float(np.quantile(counts, 0.95)),
            "index_bytes": int(hashes.nbytes),
            "method": "vectorized exact scan",
        })
    frame = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False)
    print(frame.to_string(index=False))


if __name__ == "__main__":
    main()

