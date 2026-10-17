#!/usr/bin/env python
"""Compare full-registry shortlist trade-offs for 64-bit classical hashes."""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=PROJECT / "data" / "manifests" / "pair_manifest_main.csv",
    )
    parser.add_argument(
        "--hash-cache",
        type=Path,
        default=PROJECT / "results" / "cache" / "classical_hashes.pkl",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT / "results",
    )
    args = parser.parse_args()
    manifest = pd.read_csv(args.manifest)
    with args.hash_cache.open("rb") as stream:
        caches = pickle.load(stream)["hashes"]
    originals = manifest[["seed_id", "original_path"]].dropna().drop_duplicates("seed_id").sort_values("seed_id")
    queries = manifest[
        manifest["Pair-Type"].eq("positive")
        & manifest["original_split"].eq("test")
        & manifest["transformed_exists"].astype(bool)
    ].drop_duplicates("transformed_path")
    rows = []
    for method in ["aHash", "dHash", "pHash", "wHash-Haar", "NTH-default"]:
        cache = caches[method]
        method_originals = originals[originals["original_path"].isin(cache)].copy()
        seed_to_column = {int(seed): index for index, seed in enumerate(method_originals["seed_id"])}
        method_queries = queries[
            queries["transformed_path"].isin(cache)
            & queries["seed_id"].astype(int).isin(seed_to_column)
        ].copy()
        index_bits = np.stack([cache[path] for path in method_originals["original_path"]])
        query_bits = np.stack([cache[path] for path in method_queries["transformed_path"]])
        chunks = []
        for start in range(0, len(query_bits), 256):
            chunks.append(np.count_nonzero(query_bits[start : start + 256, None, :] != index_bits[None, :, :], axis=2))
        distances = np.concatenate(chunks, axis=0)
        truth = np.asarray([seed_to_column[int(seed)] for seed in method_queries["seed_id"]], dtype=int)
        indices = np.arange(len(method_queries))
        for radius in range(65):
            accepted = distances <= radius
            true_accept = accepted[indices, truth]
            counts = accepted.sum(axis=1)
            false_counts = counts - true_accept.astype(int)
            rows.append({
                "method": method,
                "radius": radius,
                "query_count": len(method_queries),
                "index_size": len(method_originals),
                "true_candidate_recall": float(true_accept.mean()),
                "any_false_candidate_probability": float((false_counts > 0).mean()),
                "mean_shortlist": float(counts.mean()),
                "p95_shortlist": float(np.quantile(counts, 0.95)),
            })
    curves = pd.DataFrame(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    curves.to_csv(args.output_dir / "hash_registry_tradeoff_curves.csv", index=False)
    selected_rows = []
    for target in [0.80, 0.90, 0.95]:
        for method, part in curves.groupby("method"):
            eligible = part[part["true_candidate_recall"] >= target]
            chosen = eligible.sort_values(["mean_shortlist", "radius"]).iloc[0]
            selected_rows.append({"target_recall": target, **chosen.to_dict()})
    selected = pd.DataFrame(selected_rows)
    selected.to_csv(args.output_dir / "hash_registry_target_recall_summary.csv", index=False)
    print(selected.to_string(index=False))


if __name__ == "__main__":
    main()
