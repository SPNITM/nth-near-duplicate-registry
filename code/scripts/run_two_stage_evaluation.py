#!/usr/bin/env python
"""Evaluate the fixed NTH gate followed by one calibrated embedding threshold.

This script deliberately keeps the NTH radius fixed while sweeping/calibrating
the embedding distance.  It also distinguishes pair-level false-positive rate
from the probability that a query produces any false match in the registry.
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / "code" / "src"))

from nth_registry.evaluation_data import build_evaluation_pairs
from nth_registry.metrics import calibrate_max_threshold, fixed_gate_operating_curve


def hamming(cache: dict[str, np.ndarray], left: pd.Series, right: pd.Series) -> np.ndarray:
    return np.fromiter(
        (np.count_nonzero(cache[a] != cache[b]) for a, b in zip(left, right)),
        dtype=np.int16,
        count=len(left),
    )


def cosine_distance(
    embeddings: np.ndarray,
    lookup: dict[str, int],
    left: pd.Series,
    right: pd.Series,
) -> np.ndarray:
    a = embeddings[[lookup[path] for path in left]]
    b = embeddings[[lookup[path] for path in right]]
    return 1.0 - np.sum(a * b, axis=1)


def summarize(labels: np.ndarray, accepted: np.ndarray) -> dict[str, float | int]:
    labels = labels.astype(bool)
    accepted = accepted.astype(bool)
    tp = int(np.sum(labels & accepted))
    fp = int(np.sum(~labels & accepted))
    fn = int(np.sum(labels & ~accepted))
    tn = int(np.sum(~labels & ~accepted))
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / (tp + fn) if tp + fn else 0.0,
        "fpr": fp / (fp + tn) if fp + tn else 0.0,
    }


def select_projection(embedding_dir: Path, run_table: Path) -> tuple[str, Path, Path]:
    """Select a fixed-margin replicate using calibration AP only."""
    runs = pd.read_csv(run_table)
    eligible = runs[runs["mode"].eq("fixed1.5")].copy()
    if eligible.empty:
        raise ValueError("No fixed1.5 projection-head runs were found")
    selected = eligible.sort_values(
        ["calibration_ap", "calibration_recall", "seed"], ascending=[False, False, True]
    ).iloc[0]
    model = str(selected["model"])
    return model, embedding_dir / f"{model}.npy", embedding_dir / f"{model}_paths.csv"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=PROJECT / "data" / "manifests" / "pair_manifest_main.csv")
    parser.add_argument("--hash-cache", type=Path, default=PROJECT / "results" / "cache" / "classical_hashes.pkl")
    parser.add_argument("--embedding-dir", type=Path, default=PROJECT / "results" / "raw_embeddings")
    parser.add_argument("--run-table", type=Path, default=PROJECT / "results" / "projection_training_runs.csv")
    parser.add_argument("--target-fpr", type=float, default=0.001)
    parser.add_argument("--radii", type=int, nargs="+", default=[2, 4, 8, 12, 16])
    parser.add_argument("--embedding-model", default=None, help="Optional existing embedding stem")
    parser.add_argument("--output-prefix", default="two_stage")
    args = parser.parse_args()

    manifest = pd.read_csv(args.manifest)
    pairs = build_evaluation_pairs(manifest)
    with args.hash_cache.open("rb") as stream:
        nth_cache = pickle.load(stream)["hashes"]["NTH-default"]

    if args.embedding_model:
        model = args.embedding_model
        embedding_file = args.embedding_dir / f"{model}.npy"
        path_file = args.embedding_dir / f"{model}_paths.csv"
        selection_rule = "model selected by calibration performance before test evaluation"
    else:
        model, embedding_file, path_file = select_projection(args.embedding_dir, args.run_table)
        selection_rule = "maximum calibration AP among fixed-margin-1.5 replicates"
    embeddings = np.load(embedding_file)
    paths = pd.read_csv(path_file)["path"].astype(str).tolist()
    lookup = {path: index for index, path in enumerate(paths)}
    usable = pairs[
        pairs["original_path"].isin(nth_cache)
        & pairs["transformed_path"].isin(nth_cache)
        & pairs["original_path"].isin(lookup)
        & pairs["transformed_path"].isin(lookup)
    ].copy()
    usable["nth_distance"] = hamming(nth_cache, usable["original_path"], usable["transformed_path"])
    usable["embedding_distance"] = cosine_distance(
        embeddings, lookup, usable["original_path"], usable["transformed_path"]
    )

    calibration = usable[usable["original_split"].eq("calibration")].copy()
    test = usable[usable["original_split"].eq("test")].copy()
    pair_rows: list[dict] = []
    curve_rows: list[dict] = []
    calibrated_thresholds: dict[int, float] = {}
    for radius in args.radii:
        # NTH failures are represented by +infinity, so no embedding threshold
        # can recover them.  Calibration is performed against all negatives.
        cal_effective_distance = np.where(
            calibration["nth_distance"].to_numpy() <= radius,
            calibration["embedding_distance"].to_numpy(),
            np.inf,
        )
        cal_negative = cal_effective_distance[calibration["Pair-Type"].eq("negative").to_numpy()]
        threshold = calibrate_max_threshold(cal_negative, args.target_fpr)
        calibrated_thresholds[radius] = threshold
        for split_name, part in (("calibration", calibration), ("test", test)):
            labels = part["Pair-Type"].eq("positive").to_numpy()
            stage1 = part["nth_distance"].to_numpy() <= radius
            accepted = stage1 & (part["embedding_distance"].to_numpy() <= threshold)
            row = {
                "selected_embedding_model": model,
                "selection_rule": selection_rule,
                "split": split_name,
                "nth_radius": radius,
                "embedding_threshold": threshold,
                "target_calibration_fpr": args.target_fpr,
                "nth_positive_pass_rate": float(stage1[labels].mean()),
                **summarize(labels, accepted),
            }
            pair_rows.append(row)

        labels = test["Pair-Type"].eq("positive").to_numpy()
        curve = fixed_gate_operating_curve(
            labels,
            test["nth_distance"].to_numpy(),
            test["embedding_distance"].to_numpy(),
            radius,
        )
        curve_rows.extend(
            {
                "selected_embedding_model": model,
                "nth_radius": radius,
                "fpr": float(x),
                "tpr": float(y),
                "fixed_gate_attainable_average_precision": float(curve["ap"]),
                "maximum_gate_recall": float(curve["maximum_recall"]),
                "maximum_gate_fpr": float(curve["maximum_fpr"]),
            }
            for x, y in zip(curve["fpr"], curve["tpr"])
        )

    pair_frame = pd.DataFrame(pair_rows)
    pair_frame.to_csv(PROJECT / "results" / f"{args.output_prefix}_pair_results.csv", index=False)
    pd.DataFrame(curve_rows).to_csv(PROJECT / "results" / f"{args.output_prefix}_fixed_gate_curves.csv", index=False)

    # Registry query evaluation: all supplied originals form the index; held-out
    # test transforms form the queries.  This yields both stage-one shortlist
    # behavior and end-to-end false-match probability per query.
    originals = (
        manifest[["seed_id", "original_path"]]
        .dropna()
        .drop_duplicates("seed_id")
        .sort_values("seed_id")
    )
    originals = originals[originals["original_path"].isin(nth_cache) & originals["original_path"].isin(lookup)].copy()
    queries = manifest[
        manifest["Pair-Type"].eq("positive")
        & manifest["original_split"].eq("test")
        & manifest["transformed_exists"].astype(bool)
    ].drop_duplicates("transformed_path")
    queries = queries[queries["transformed_path"].isin(nth_cache) & queries["transformed_path"].isin(lookup)].copy()
    seed_to_column = {int(seed): index for index, seed in enumerate(originals["seed_id"])}
    queries = queries[queries["seed_id"].astype(int).isin(seed_to_column)].copy()

    index_bits = np.stack([nth_cache[path] for path in originals["original_path"]])
    query_bits = np.stack([nth_cache[path] for path in queries["transformed_path"]])
    index_embeddings = embeddings[[lookup[path] for path in originals["original_path"]]]
    query_embeddings = embeddings[[lookup[path] for path in queries["transformed_path"]]]
    truth = np.asarray([seed_to_column[int(seed)] for seed in queries["seed_id"]], dtype=int)
    rows = np.arange(len(queries))

    registry_rows: list[dict] = []
    # Chunked matrices keep peak memory bounded.
    nth_chunks = []
    emb_chunks = []
    for start in range(0, len(queries), 128):
        stop = min(start + 128, len(queries))
        nth_chunks.append(
            np.count_nonzero(query_bits[start:stop, None, :] != index_bits[None, :, :], axis=2)
        )
        emb_chunks.append(1.0 - query_embeddings[start:stop] @ index_embeddings.T)
    nth_matrix = np.concatenate(nth_chunks, axis=0)
    embedding_matrix = np.concatenate(emb_chunks, axis=0)
    for radius in args.radii:
        stage1 = nth_matrix <= radius
        final = stage1 & (embedding_matrix <= calibrated_thresholds[radius])
        true_stage1 = stage1[rows, truth]
        true_final = final[rows, truth]
        stage1_counts = stage1.sum(axis=1)
        final_counts = final.sum(axis=1)
        false_counts = final_counts - true_final.astype(int)
        nonmatching_comparisons = len(queries) * (len(originals) - 1)
        registry_rows.append({
            "selected_embedding_model": model,
            "query_count": len(queries),
            "index_size": len(originals),
            "nth_radius": radius,
            "embedding_threshold": calibrated_thresholds[radius],
            "true_stage1_recall": float(true_stage1.mean()),
            "true_end_to_end_recall": float(true_final.mean()),
            "per_query_any_false_match_probability": float((false_counts > 0).mean()),
            "per_nonmatch_false_accept_rate": float(false_counts.sum() / nonmatching_comparisons),
            "mean_stage1_shortlist": float(stage1_counts.mean()),
            "p95_stage1_shortlist": float(np.quantile(stage1_counts, 0.95)),
            "mean_final_accepted_candidates": float(final_counts.mean()),
            "p95_final_accepted_candidates": float(np.quantile(final_counts, 0.95)),
            "maximum_final_accepted_candidates": int(final_counts.max(initial=0)),
        })
    registry_frame = pd.DataFrame(registry_rows)
    registry_frame.to_csv(PROJECT / "results" / f"{args.output_prefix}_registry_results.csv", index=False)

    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    for radius, part in pd.DataFrame(curve_rows).groupby("nth_radius"):
        ax.plot(np.maximum(part["fpr"], 1e-6), part["tpr"], label=f"r={radius}")
    ax.set_xscale("log")
    ax.set_xlim(1e-5, 1)
    ax.set_ylim(0, 1.01)
    ax.set_xlabel("False-positive rate (log scale)")
    ax.set_ylabel("True-positive rate")
    ax.grid(alpha=0.25, which="both")
    ax.legend(title="Fixed NTH radius")
    fig.tight_layout()
    figure_path = PROJECT / "figures" / f"{args.output_prefix}_fixed_gate_roc.png"
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_path, dpi=300)
    plt.close(fig)

    print(f"Selected model: {model}")
    print(pair_frame.to_string(index=False))
    print("\nRegistry evaluation\n", registry_frame.to_string(index=False))


if __name__ == "__main__":
    main()
