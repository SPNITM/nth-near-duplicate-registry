#!/usr/bin/env python
"""Calibrate and evaluate frozen or learned embedding files."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / "code" / "src"))

from nth_registry.evaluation_data import build_evaluation_pairs
from nth_registry.metrics import calibrate_max_threshold


def evaluate_one(model: str, embedding_file: Path, path_file: Path, pairs: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    embeddings = np.load(embedding_file)
    paths = pd.read_csv(path_file)["path"].tolist()
    if len(paths) != len(embeddings):
        raise ValueError(f"Path/embedding count mismatch for {model}")
    lookup = {path: i for i, path in enumerate(paths)}
    usable = pairs[pairs["original_path"].isin(lookup) & pairs["transformed_path"].isin(lookup)].copy()
    left = embeddings[[lookup[path] for path in usable["original_path"]]]
    right = embeddings[[lookup[path] for path in usable["transformed_path"]]]
    usable["distance"] = 1.0 - np.sum(left * right, axis=1)
    calibration = usable[usable["original_split"].eq("calibration")]
    test = usable[usable["original_split"].eq("test")]
    threshold = calibrate_max_threshold(
        calibration[calibration["Pair-Type"].eq("negative")]["distance"].to_numpy(), 0.001
    )

    def summarize(part: pd.DataFrame) -> dict:
        labels = part["Pair-Type"].eq("positive").to_numpy()
        distances = part["distance"].to_numpy()
        accepted = distances <= threshold
        tp = int(np.sum(labels & accepted)); fp = int(np.sum(~labels & accepted))
        fn = int(np.sum(labels & ~accepted)); tn = int(np.sum(~labels & ~accepted))
        scores = -distances
        return {
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": tp / (tp + fp) if tp + fp else 0.0,
            "recall": tp / (tp + fn) if tp + fn else 0.0,
            "fpr": fp / (fp + tn) if fp + tn else 0.0,
            "auc": float(roc_auc_score(labels, scores)),
            "ap": float(average_precision_score(labels, scores)),
        }

    cal = summarize(calibration); tst = summarize(test)
    summary = {"model": model, "embedding_threshold": threshold, **{f"calibration_{k}": v for k, v in cal.items()}, **{f"test_{k}": v for k, v in tst.items()}}
    return summary, usable


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=PROJECT / "data" / "manifests" / "pair_manifest_main.csv")
    parser.add_argument("--embedding-dir", type=Path, default=PROJECT / "results" / "raw_embeddings")
    parser.add_argument("--output", type=Path, default=PROJECT / "results" / "deep_baseline_results.csv")
    args = parser.parse_args()
    manifest = pd.read_csv(args.manifest)
    pairs = build_evaluation_pairs(manifest)
    rows = []
    for embedding_file in sorted(args.embedding_dir.glob("*.npy")):
        model = embedding_file.stem
        path_file = args.embedding_dir / f"{model}_paths.csv"
        if not path_file.exists():
            continue
        summary, predictions = evaluate_one(model, embedding_file, path_file, pairs)
        rows.append(summary)
        # Preserve float64 values so reloading the raw predictions reproduces
        # the exact calibrated confusion matrix at threshold boundaries.
        predictions.to_csv(
            args.embedding_dir / f"{model}_pair_predictions.csv",
            index=False,
            float_format="%.17g",
        )
    if not rows:
        raise SystemExit("No matching embedding/path files were found")
    output = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)
    print(output.to_string(index=False))


if __name__ == "__main__":
    main()
