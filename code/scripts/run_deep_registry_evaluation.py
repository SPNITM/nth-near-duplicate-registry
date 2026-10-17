#!/usr/bin/env python
"""Evaluate a calibration-selected frozen embedding against the full registry."""

from __future__ import annotations

import argparse
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
        "--deep-results",
        type=Path,
        default=PROJECT / "results" / "deep_frozen_baseline_summary.csv",
    )
    parser.add_argument(
        "--embedding-dir",
        type=Path,
        default=PROJECT / "results" / "raw_embeddings",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT / "results" / "deep_registry_results.csv",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Optional embedding stem. If omitted, use the calibration-selected row.",
    )
    args = parser.parse_args()
    deep = pd.read_csv(args.deep_results)
    if args.model:
        selected = deep[deep["model"].eq(args.model)].iloc[0]
    elif "selected_by_calibration" in deep:
        selected = deep[deep["selected_by_calibration"].astype(bool)].iloc[0]
    else:
        selected = deep.sort_values(
            ["calibration_recall", "calibration_ap"], ascending=False
        ).iloc[0]
    model = str(selected["model"])
    threshold = float(selected["embedding_threshold"])
    embedding_dir = args.embedding_dir
    embeddings = np.load(embedding_dir / f"{model}.npy")
    paths = pd.read_csv(embedding_dir / f"{model}_paths.csv")["path"].astype(str).tolist()
    lookup = {path: index for index, path in enumerate(paths)}
    manifest = pd.read_csv(args.manifest)
    originals = (
        manifest[["seed_id", "original_path"]].dropna().drop_duplicates("seed_id").sort_values("seed_id")
    )
    originals = originals[originals["original_path"].isin(lookup)].copy()
    seed_to_column = {int(seed): index for index, seed in enumerate(originals["seed_id"])}
    queries = manifest[
        manifest["Pair-Type"].eq("positive")
        & manifest["original_split"].eq("test")
        & manifest["transformed_exists"].astype(bool)
    ].drop_duplicates("transformed_path")
    queries = queries[
        queries["transformed_path"].isin(lookup)
        & queries["seed_id"].astype(int).isin(seed_to_column)
    ].copy()
    index_vectors = embeddings[[lookup[path] for path in originals["original_path"]]]
    query_vectors = embeddings[[lookup[path] for path in queries["transformed_path"]]]
    truth = np.asarray([seed_to_column[int(seed)] for seed in queries["seed_id"]], dtype=int)
    accepted_chunks = []
    for start in range(0, len(queries), 128):
        distances = 1.0 - query_vectors[start : start + 128] @ index_vectors.T
        accepted_chunks.append(distances <= threshold)
    accepted = np.concatenate(accepted_chunks, axis=0)
    rows = np.arange(len(queries))
    true_accept = accepted[rows, truth]
    counts = accepted.sum(axis=1)
    false_counts = counts - true_accept.astype(int)
    summary = pd.DataFrame([{
        "model": model,
        "selection_rule": "maximum calibration recall among frozen baselines at target pair FPR",
        "embedding_threshold": threshold,
        "query_count": len(queries),
        "index_size": len(originals),
        "true_query_recall": float(true_accept.mean()),
        "per_query_any_false_match_probability": float((false_counts > 0).mean()),
        "per_nonmatch_false_accept_rate": float(false_counts.sum() / (len(queries) * (len(originals) - 1))),
        "mean_accepted_candidates": float(counts.mean()),
        "p95_accepted_candidates": float(np.quantile(counts, 0.95)),
        "maximum_accepted_candidates": int(counts.max(initial=0)),
        "embedding_comparisons_per_query": len(originals),
    }])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output, index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
