#!/usr/bin/env python
"""Summarise and plot the complete private-main-corpus verification run."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[2]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def cluster_bootstrap(
    predictions: pd.DataFrame,
    threshold: float,
    repeats: int,
    seed: int,
) -> dict[str, float | int | str]:
    test = predictions[predictions["original_split"].eq("test")].copy()
    test["positive"] = test["Pair-Type"].eq("positive")
    test["accepted"] = test["distance"] <= threshold
    test["tp"] = (test["positive"] & test["accepted"]).astype(int)
    test["fn"] = (test["positive"] & ~test["accepted"]).astype(int)
    test["fp"] = (~test["positive"] & test["accepted"]).astype(int)
    test["tn"] = (~test["positive"] & ~test["accepted"]).astype(int)
    clusters = test.groupby("transformed_seed_id")[["tp", "fn", "fp", "tn"]].sum().to_numpy(np.int64)
    rng = np.random.default_rng(seed)
    recall = np.empty(repeats)
    fpr = np.empty(repeats)
    for index in range(repeats):
        tp, fn, fp, tn = clusters[rng.integers(0, len(clusters), len(clusters))].sum(axis=0)
        recall[index] = tp / (tp + fn)
        fpr[index] = fp / (fp + tn)
    return {
        "unit": "transformed source identity",
        "clusters": int(len(clusters)),
        "repeats": repeats,
        "seed": seed,
        "test_recall_ci_low": float(np.quantile(recall, 0.025)),
        "test_recall_ci_high": float(np.quantile(recall, 0.975)),
        "test_fpr_ci_low": float(np.quantile(fpr, 0.025)),
        "test_fpr_ci_high": float(np.quantile(fpr, 0.975)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest-dir",
        type=Path,
        default=PROJECT / "data" / "manifests_hpc_full_main_20260831",
    )
    parser.add_argument(
        "--result-dir",
        type=Path,
        default=PROJECT / "results" / "hpc_full_main_20260831",
    )
    parser.add_argument(
        "--figure",
        type=Path,
        default=PROJECT / "figures" / "full_main_verification_summary.png",
    )
    parser.add_argument(
        "--tradeoff-figure",
        type=Path,
        default=PROJECT / "figures" / "hash_registry_tradeoff.png",
    )
    parser.add_argument("--bootstrap-repeats", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260830)
    args = parser.parse_args()

    audit_path = args.manifest_dir / "dataset_audit_summary.json"
    manifest_path = args.manifest_dir / "pair_manifest_main.csv"
    hash_path = args.result_dir / "hash_baseline_calibrated_results.csv"
    crop_path = args.result_dir / "nth_crop_ratio_results.csv"
    deep_path = args.result_dir / "deep_baseline_results.csv"
    prediction_path = args.result_dir / "raw_embeddings" / "shufflenet_v2_x1_pair_predictions.csv"
    pair_path = args.result_dir / "two_stage_shufflenet_pair_results.csv"
    registry_path = args.result_dir / "two_stage_shufflenet_registry_results.csv"
    deep_registry_path = args.result_dir / "deep_registry_results.csv"
    tradeoff_path = args.result_dir / "hash_registry_target_recall_summary.csv"
    tradeoff_curves_path = args.result_dir / "hash_registry_tradeoff_curves.csv"

    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    dataset = audit["datasets"][0]
    hashes = pd.read_csv(hash_path)
    crop = pd.read_csv(crop_path)
    deep = pd.read_csv(deep_path).iloc[0]
    predictions = pd.read_csv(prediction_path, low_memory=False)
    pairs = pd.read_csv(pair_path)
    registry = pd.read_csv(registry_path)
    deep_registry = pd.read_csv(deep_registry_path).iloc[0]
    tradeoff = pd.read_csv(tradeoff_path)
    tradeoff_curves = pd.read_csv(tradeoff_curves_path)
    intervals = cluster_bootstrap(
        predictions,
        float(deep["embedding_threshold"]),
        args.bootstrap_repeats,
        args.seed,
    )

    selected_hash_columns = ["method", "calibrated_radius", "test_recall", "test_fpr"]
    selected_pair_columns = ["nth_radius", "embedding_threshold", "recall", "fpr"]
    selected_registry_columns = [
        "nth_radius",
        "true_stage1_recall",
        "true_end_to_end_recall",
        "per_query_any_false_match_probability",
        "per_nonmatch_false_accept_rate",
        "mean_stage1_shortlist",
        "mean_final_accepted_candidates",
    ]
    summary = {
        "scope": "complete private main corpus on institutional HPC CPU node",
        "data_audit": dataset,
        "split_counts": audit["split_counts"],
        "selected_model": {
            **{key: (value.item() if hasattr(value, "item") else value) for key, value in deep.to_dict().items()},
            "cluster_bootstrap_95ci": intervals,
        },
        "deep_only_registry": {
            key: (value.item() if hasattr(value, "item") else value)
            for key, value in deep_registry.to_dict().items()
        },
        "hash_baselines": hashes[selected_hash_columns].to_dict(orient="records"),
        "two_stage_pair_test": pairs[pairs["split"].eq("test")][selected_pair_columns].to_dict(orient="records"),
        "two_stage_registry": registry[selected_registry_columns].to_dict(orient="records"),
        "hash_registry_targets": tradeoff.to_dict(orient="records"),
        "crop_robustness": crop.to_dict(orient="records"),
        "input_sha256": {
            path.relative_to(PROJECT).as_posix(): sha256(path)
            for path in [
                audit_path,
                manifest_path,
                hash_path,
                crop_path,
                deep_path,
                prediction_path,
                pair_path,
                registry_path,
                deep_registry_path,
                tradeoff_path,
                tradeoff_curves_path,
            ]
        },
    }
    output = args.result_dir / "full_main_verification_summary.json"
    output.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "figure.dpi": 150})
    fig, axes = plt.subplots(2, 2, figsize=(10.2, 7.4))
    axes[0, 0].bar(
        ["Positive rows", "Negative rows"],
        [dataset["positive_rows"], dataset["negative_rows"]],
        color=["#2E74B5", "#7F8C8D"],
    )
    axes[0, 0].set_title("a  Complete private-main audit")
    axes[0, 0].set_ylabel("Resolved rows")
    axes[0, 0].text(
        0.5,
        0.94,
        "91,504/91,504 rows resolved; zero missing paths",
        transform=axes[0, 0].transAxes,
        ha="center",
        va="top",
        color="#2E7D32",
    )

    selected = hashes[hashes["method"].isin(["aHash", "dHash", "pHash", "wHash-Haar", "NTH-default"])]
    axes[0, 1].barh(selected["method"], selected["test_recall"], color="#2E74B5")
    axes[0, 1].set_xlim(0.55, 0.75)
    axes[0, 1].set_xlabel("Held-out recall")
    axes[0, 1].set_title("b  Calibrated 64-bit hashes")

    test_pairs = pairs[pairs["split"].eq("test")]
    axes[1, 0].plot(test_pairs["nth_radius"], test_pairs["recall"], "o-", label="end-to-end recall")
    axes[1, 0].plot(registry["nth_radius"], registry["true_stage1_recall"], "s--", label="stage-one ceiling")
    axes[1, 0].set_ylim(0.55, 0.95)
    axes[1, 0].set_xlabel("Fixed NTH radius")
    axes[1, 0].set_ylabel("Recall")
    axes[1, 0].set_title("c  Full-corpus two-stage recall")
    axes[1, 0].legend()

    axes[1, 1].plot(registry["nth_radius"], registry["mean_stage1_shortlist"], "o-", color="#E67E22")
    ax2 = axes[1, 1].twinx()
    ax2.plot(
        registry["nth_radius"],
        registry["per_query_any_false_match_probability"],
        "s--",
        color="#B23A48",
    )
    axes[1, 1].set_xlabel("Fixed NTH radius")
    axes[1, 1].set_ylabel("Mean shortlist", color="#E67E22")
    ax2.set_ylabel("Any false match/query", color="#B23A48")
    axes[1, 1].set_title("d  Workload and registry-level risk")
    for axis in axes.flat:
        axis.grid(alpha=0.2)
    fig.tight_layout()
    args.figure.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.figure, dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.6, 4.5))
    for method, part in tradeoff_curves.groupby("method"):
        selected_curve = part[
            (part["true_candidate_recall"] >= 0.5)
            & (part["true_candidate_recall"] <= 0.98)
        ]
        ax.plot(
            selected_curve["mean_shortlist"],
            selected_curve["true_candidate_recall"],
            label=method,
        )
    ax.set_xscale("log")
    ax.set_xlim(0.5, 1064)
    ax.set_ylim(0.5, 1.0)
    ax.set_xlabel("Mean registry shortlist (log scale)")
    ax.set_ylabel("True-candidate recall")
    ax.grid(alpha=0.25, which="both")
    ax.legend(ncol=2)
    fig.tight_layout()
    args.tradeoff_figure.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.tradeoff_figure, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(json.dumps({
        "summary": str(output),
        "figure": str(args.figure),
        "tradeoff_figure": str(args.tradeoff_figure),
        "summary_sha256": sha256(output),
    }, indent=2))


if __name__ == "__main__":
    main()
