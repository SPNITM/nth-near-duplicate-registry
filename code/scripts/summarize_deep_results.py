#!/usr/bin/env python
"""Create concise deep-baseline tables, cluster-bootstrap CIs, and paired tests."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest

PROJECT = Path(__file__).resolve().parents[2]

DISPLAY_NAMES = {
    "clip_vit_b32": "CLIP ViT-B/32 (LAION-2B)",
    "dinov2_vits14": "DINOv2 ViT-S/14",
    "efficientnet_b0": "EfficientNet-B0 (ImageNet)",
    "mobilenetv3_imagenet": "MobileNetV3-Large (ImageNet)",
    "resnet50_imagenet": "ResNet-50 (ImageNet)",
    "shufflenet_v2_x1": "ShuffleNet V2 1.0x (ImageNet)",
}


def cluster_intervals(predictions: pd.DataFrame, threshold: float, repeats: int, seed: int) -> dict[str, float]:
    test = predictions[predictions["original_split"].eq("test")].copy()
    test["positive"] = test["Pair-Type"].eq("positive")
    test["accepted"] = test["distance"] <= threshold
    test["tp"] = (test["positive"] & test["accepted"]).astype(int)
    test["fn"] = (test["positive"] & ~test["accepted"]).astype(int)
    test["fp"] = (~test["positive"] & test["accepted"]).astype(int)
    test["tn"] = (~test["positive"] & ~test["accepted"]).astype(int)
    clusters = test.groupby("transformed_seed_id")[["tp", "fn", "fp", "tn"]].sum().to_numpy(dtype=np.int64)
    rng = np.random.default_rng(seed)
    recall = np.empty(repeats, dtype=float)
    fpr = np.empty(repeats, dtype=float)
    precision = np.empty(repeats, dtype=float)
    for index in range(repeats):
        sampled = clusters[rng.integers(0, len(clusters), len(clusters))].sum(axis=0)
        tp, fn, fp, tn = sampled
        recall[index] = tp / (tp + fn)
        fpr[index] = fp / (fp + tn)
        precision[index] = tp / (tp + fp)
    return {
        "test_recall_ci_low": float(np.quantile(recall, 0.025)),
        "test_recall_ci_high": float(np.quantile(recall, 0.975)),
        "test_fpr_ci_low": float(np.quantile(fpr, 0.025)),
        "test_fpr_ci_high": float(np.quantile(fpr, 0.975)),
        "test_precision_ci_low": float(np.quantile(precision, 0.025)),
        "test_precision_ci_high": float(np.quantile(precision, 0.975)),
        "bootstrap_unit": "transformed source identity",
        "bootstrap_repeats": repeats,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260830)
    args = parser.parse_args()
    deep = pd.read_csv(PROJECT / "results" / "deep_baseline_results.csv")
    frozen = deep[deep["model"].isin(DISPLAY_NAMES)].copy()
    rows = []
    for item in frozen.itertuples(index=False):
        predictions = pd.read_csv(
            PROJECT / "results" / "raw_embeddings" / f"{item.model}_pair_predictions.csv"
        )
        intervals = cluster_intervals(predictions, float(item.embedding_threshold), args.repeats, args.seed)
        record = item._asdict()
        record["display_name"] = DISPLAY_NAMES[item.model]
        record["training_in_revision"] = "none (frozen public weights)"
        record.update(intervals)
        rows.append(record)
    summary = pd.DataFrame(rows).sort_values(
        ["calibration_recall", "calibration_ap"], ascending=False
    )
    summary["selected_by_calibration"] = False
    summary.loc[summary.index[0], "selected_by_calibration"] = True
    summary.to_csv(PROJECT / "results" / "deep_frozen_baseline_summary.csv", index=False)

    # Projection-head variability is already measured across five independent
    # seeds.  Join the aggregate with the frozen feature extractor result so a
    # reader can see whether training improved rather than assuming it did.
    aggregate = pd.read_csv(PROJECT / "results" / "projection_training_aggregate.csv")
    aggregate.to_csv(PROJECT / "results" / "deep_projection_seed_summary.csv", index=False)

    # Paired exact McNemar test on positive recognition at independently
    # calibrated thresholds: best frozen model versus the calibration-selected
    # fixed-margin projection replicate.
    best_model = str(summary.iloc[0]["model"])
    best_threshold = float(summary.iloc[0]["embedding_threshold"])
    runs = pd.read_csv(PROJECT / "results" / "projection_training_runs.csv")
    selected_run = runs[runs["mode"].eq("fixed1.5")].sort_values(
        ["calibration_ap", "calibration_recall", "seed"], ascending=[False, False, True]
    ).iloc[0]
    projection_model = str(selected_run["model"])
    projection_threshold = float(selected_run["threshold"])
    best = pd.read_csv(PROJECT / "results" / "raw_embeddings" / f"{best_model}_pair_predictions.csv")
    projection = pd.read_csv(PROJECT / "results" / "raw_embeddings" / f"{projection_model}_pair_predictions.csv")
    keys = ["original_path", "transformed_path", "Pair-Type", "original_split"]
    paired = best.merge(projection, on=keys, suffixes=("_best", "_projection"))
    paired = paired[paired["original_split"].eq("test") & paired["Pair-Type"].eq("positive")]
    best_correct = paired["distance_best"].to_numpy() <= best_threshold
    projection_correct = paired["distance_projection"].to_numpy() <= projection_threshold
    best_only = int(np.sum(best_correct & ~projection_correct))
    projection_only = int(np.sum(~best_correct & projection_correct))
    discordant = best_only + projection_only
    p_value = float(binomtest(min(best_only, projection_only), discordant, 0.5).pvalue) if discordant else 1.0
    test_row = pd.DataFrame([{
        "comparison": "paired positive recognition at calibration-selected operating points",
        "best_frozen_model": best_model,
        "projection_model": projection_model,
        "test_positive_pairs": len(paired),
        "best_frozen_only_correct": best_only,
        "projection_only_correct": projection_only,
        "discordant_pairs": discordant,
        "exact_two_sided_mcnemar_p": p_value,
        "interpretation": "best frozen model has higher paired recall" if best_only > projection_only else "no directional advantage",
    }])
    test_row.to_csv(PROJECT / "results" / "deep_paired_significance.csv", index=False)
    print(summary[[
        "display_name", "embedding_threshold", "calibration_recall", "test_precision",
        "test_recall", "test_recall_ci_low", "test_recall_ci_high", "test_fpr",
        "test_fpr_ci_low", "test_fpr_ci_high", "test_auc", "test_ap",
        "selected_by_calibration",
    ]].to_string(index=False))
    print("\n", test_row.to_string(index=False))


if __name__ == "__main__":
    main()
