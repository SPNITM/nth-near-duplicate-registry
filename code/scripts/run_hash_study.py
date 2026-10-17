#!/usr/bin/env python
"""Run calibrated classical-hash and NTH ablation studies on available files."""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / "code" / "src"))

from nth_registry.baseline_hashes import BASELINES
from nth_registry.metrics import calibrate_max_threshold, summarize_registry_candidates
from nth_registry.nth import NTHConfig, compute_nth_bits


TARGET_FPR = 0.001


def methods():
    result = dict(BASELINES)
    configs = {
        "NTH-default": NTHConfig(),
        "NTH-include-center": NTHConfig(include_center=True, version="ablation"),
        "NTH-reflect": NTHConfig(padding="reflect", version="ablation"),
        "NTH-edge": NTHConfig(padding="edge", version="ablation"),
        "NTH-zero": NTHConfig(padding="constant", version="ablation"),
        "NTH-median": NTHConfig(threshold="global_median", version="ablation"),
        "NTH-adaptive": NTHConfig(threshold="adaptive_mean", version="ablation"),
        "NTH-nearest": NTHConfig(interpolation="nearest", version="ablation"),
        "NTH-bilinear": NTHConfig(interpolation="bilinear", version="ablation"),
        "NTH-bicubic": NTHConfig(interpolation="bicubic", version="ablation"),
        "NTH-grid12": NTHConfig(grid_size=12, version="ablation"),
        "NTH-grid16": NTHConfig(grid_size=16, version="ablation"),
    }
    result.update({name: (lambda image, c=config: compute_nth_bits(image, c)) for name, config in configs.items()})
    return result


def compute_cache(paths: list[str], method_map: dict, cache_path: Path) -> dict[str, dict[str, np.ndarray]]:
    cache = {}
    if cache_path.exists():
        with cache_path.open("rb") as stream:
            payload = pickle.load(stream)
        if payload.get("paths") == paths:
            cache = {name: values for name, values in payload.get("hashes", {}).items() if name in method_map}
            print(f"Loaded {len(cache)} cached methods from: {cache_path}")
    missing_methods = {name: function for name, function in method_map.items() if name not in cache}
    cache.update({name: {} for name in missing_methods})
    if not missing_methods:
        return cache
    for raw_path in tqdm(paths, desc="Hashing unique images"):
        path = Path(raw_path)
        try:
            with Image.open(path) as opened:
                image = opened.convert("RGB")
                for name, function in missing_methods.items():
                    cache[name][raw_path] = function(image)
        except Exception as exc:
            print(f"Skipping unreadable image {path}: {exc}", file=sys.stderr)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("wb") as stream:
        pickle.dump({"paths": paths, "hashes": cache}, stream, protocol=pickle.HIGHEST_PROTOCOL)
    return cache


def build_evaluation_pairs(frame: pd.DataFrame, negatives_per_positive: int = 5) -> pd.DataFrame:
    """Create adequate deterministic within-split negatives for calibration/test."""
    rng = np.random.default_rng(20260830)
    positives = frame[
        frame["Pair-Type"].eq("positive")
        & frame["usable"].astype(bool)
        & frame["original_split"].isin(["calibration", "test"])
    ].copy()
    rows = [positives]
    for split_name, split_positive in positives.groupby("original_split", sort=False):
        originals = (
            frame[frame["original_split"].eq(split_name)][["seed_id", "original_path"]]
            .dropna()
            .drop_duplicates("seed_id")
        )
        seed_values = originals["seed_id"].astype(int).to_numpy()
        path_by_seed = dict(zip(seed_values, originals["original_path"]))
        negative_rows = []
        for item in split_positive.to_dict("records"):
            candidates = seed_values[seed_values != int(item["seed_id"])]
            chosen = rng.choice(candidates, size=min(negatives_per_positive, len(candidates)), replace=False)
            for negative_seed in chosen:
                negative_rows.append({
                    "Original Image": Path(path_by_seed[int(negative_seed)]).name,
                    "Transformed Image": item["Transformed Image"],
                    "Label": "none",
                    "Pair-Type": "negative",
                    "File Type": item["File Type"],
                    "original_path": path_by_seed[int(negative_seed)],
                    "transformed_path": item["transformed_path"],
                    "original_exists": True,
                    "transformed_exists": True,
                    "seed_id": int(negative_seed),
                    "transformed_seed_id": int(item["seed_id"]),
                    "usable": True,
                    "original_split": split_name,
                    "transformed_split": split_name,
                    "strict_split_usable": True,
                })
        rows.append(pd.DataFrame(negative_rows))
    return pd.concat(rows, ignore_index=True)


def pair_distances(frame: pd.DataFrame, cache: dict[str, np.ndarray]) -> np.ndarray:
    return np.fromiter(
        (
            np.count_nonzero(cache[left] != cache[right])
            for left, right in zip(frame["original_path"], frame["transformed_path"])
        ),
        dtype=np.int16,
        count=len(frame),
    )


def evaluate_method(name: str, frame: pd.DataFrame, cache: dict[str, np.ndarray]) -> tuple[dict, pd.DataFrame]:
    usable = frame[
        frame["strict_split_usable"].astype(bool)
        & frame["original_path"].isin(cache)
        & frame["transformed_path"].isin(cache)
    ].copy()
    usable["distance"] = pair_distances(usable, cache)
    calibration = usable[usable["original_split"].eq("calibration")]
    test = usable[usable["original_split"].eq("test")]
    cal_neg = calibration[calibration["Pair-Type"].eq("negative")]["distance"].to_numpy()
    threshold = calibrate_max_threshold(cal_neg, TARGET_FPR)

    def one(split_frame: pd.DataFrame) -> dict:
        labels = split_frame["Pair-Type"].eq("positive").to_numpy()
        accepted = split_frame["distance"].to_numpy() <= threshold
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

    cal_stats = one(calibration)
    test_stats = one(test)
    summary = {
        "method": name,
        "calibrated_radius": threshold,
        **{f"calibration_{k}": v for k, v in cal_stats.items()},
        **{f"test_{k}": v for k, v in test_stats.items()},
        "calibration_rows": len(calibration),
        "test_rows": len(test),
    }
    positives = test[test["Pair-Type"].eq("positive")].copy()
    per_label = (
        positives.assign(accepted=positives["distance"] <= threshold)
        .groupby("Label", as_index=False)
        .agg(n=("accepted", "size"), recall=("accepted", "mean"), mean_distance=("distance", "mean"))
    )
    per_label.insert(0, "method", name)
    return summary, per_label


def registry_study(
    frame: pd.DataFrame,
    cache: dict[str, np.ndarray],
    radii: list[int],
) -> pd.DataFrame:
    originals = (
        frame[["seed_id", "original_path"]]
        .dropna()
        .drop_duplicates("seed_id")
        .sort_values("seed_id")
    )
    originals = originals[originals["original_path"].isin(cache)].copy()
    seed_to_col = {int(seed): i for i, seed in enumerate(originals["seed_id"])}
    queries = frame[
        frame["Pair-Type"].eq("positive")
        & frame["original_split"].eq("test")
        & frame["transformed_exists"].astype(bool)
        & frame["transformed_path"].isin(cache)
        & frame["seed_id"].astype(int).isin(seed_to_col)
    ].drop_duplicates("transformed_path")
    index_bits = np.stack([cache[path] for path in originals["original_path"]])
    query_bits = np.stack([cache[path] for path in queries["transformed_path"]])
    # XOR broadcasting is chunked to bound memory.
    chunks = []
    for start in range(0, len(query_bits), 256):
        chunks.append(np.count_nonzero(query_bits[start : start + 256, None, :] != index_bits[None, :, :], axis=2))
    matrix = np.concatenate(chunks, axis=0)
    truth = np.array([seed_to_col[int(seed)] for seed in queries["seed_id"]])
    rows = [summarize_registry_candidates(matrix, truth, radius).__dict__ for radius in radii]
    return pd.DataFrame(rows)


def crop_study(frame: pd.DataFrame, cache_method, calibrated_radius: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    originals = (
        frame[frame["original_split"].eq("test")][["seed_id", "original_path"]]
        .drop_duplicates("seed_id")
        .dropna()
    )
    ratios = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    rows = []
    failures = []
    for ratio in ratios:
        distances = []
        for item in originals.itertuples(index=False):
            try:
                with Image.open(item.original_path) as opened:
                    image = opened.convert("RGB")
                    reference = cache_method(image)
                    width, height = image.size
                    dx = int(round(width * ratio / 2))
                    dy = int(round(height * ratio / 2))
                    cropped = image.crop((dx, dy, max(dx + 1, width - dx), max(dy + 1, height - dy))).resize(image.size, Image.Resampling.LANCZOS)
                    distance = int(np.count_nonzero(reference != cache_method(cropped)))
                    distances.append(distance)
                    if distance > calibrated_radius:
                        failures.append({"seed_id": item.seed_id, "crop_fraction_removed": ratio, "distance": distance, "original_path": item.original_path})
            except Exception:
                continue
        values = np.asarray(distances)
        rows.append({
            "crop_fraction_removed": ratio,
            "n": len(values),
            "recall_at_calibrated_radius": float(np.mean(values <= calibrated_radius)),
            "mean_distance": float(values.mean()),
            "p95_distance": float(np.quantile(values, 0.95)),
        })
    failure_frame = pd.DataFrame(failures).sort_values(["crop_fraction_removed", "distance"], ascending=[True, False])
    return pd.DataFrame(rows), failure_frame


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=PROJECT / "data" / "manifests" / "pair_manifest_main.csv")
    parser.add_argument("--output", type=Path, default=PROJECT / "results")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(args.manifest)
    evaluation = build_evaluation_pairs(frame)
    positive_transforms = frame[frame["Pair-Type"].eq("positive") & frame["transformed_exists"].astype(bool)]
    paths = sorted(set(frame["original_path"].dropna()) | set(positive_transforms["transformed_path"].dropna()))
    method_map = methods()
    cache = compute_cache(paths, method_map, args.output / "cache" / "classical_hashes.pkl")

    summaries = []
    label_tables = []
    for name in method_map:
        summary, per_label = evaluate_method(name, evaluation, cache[name])
        summaries.append(summary)
        label_tables.append(per_label)
    summary_frame = pd.DataFrame(summaries)
    summary_frame.to_csv(args.output / "hash_baseline_calibrated_results.csv", index=False)
    pd.concat(label_tables, ignore_index=True).to_csv(args.output / "hash_per_transform_results.csv", index=False)

    registry = registry_study(frame, cache["NTH-default"], [2, 4, 8, 12, 16, 20])
    registry.to_csv(args.output / "nth_registry_query_results.csv", index=False)
    nth_radius = float(summary_frame.loc[summary_frame["method"].eq("NTH-default"), "calibrated_radius"].iloc[0])
    crops, failures = crop_study(frame, method_map["NTH-default"], nth_radius)
    crops.to_csv(args.output / "nth_crop_ratio_results.csv", index=False)
    failures.head(100).to_csv(args.output / "nth_crop_failure_cases.csv", index=False)

    figure_dir = PROJECT / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    plot = summary_frame.sort_values("test_recall", ascending=False)
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    ax.barh(plot["method"], plot["test_recall"], color="#2E74B5")
    ax.set_xlabel("Held-out recall at calibration FPR <= 0.001")
    ax.set_xlim(0, 1.01)
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(figure_dir / "hash_baseline_recall.png", dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.plot(crops["crop_fraction_removed"] * 100, crops["recall_at_calibrated_radius"], marker="o")
    ax.set_xlabel("Central crop: fraction removed from width and height (%)")
    ax.set_ylabel("NTH recall at calibrated radius")
    ax.set_ylim(0, 1.01)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(figure_dir / "nth_crop_robustness.png", dpi=300)
    plt.close(fig)

    print(summary_frame.to_string(index=False))
    print("\nRegistry study\n", registry.to_string(index=False))
    print("\nCrop study\n", crops.to_string(index=False))


if __name__ == "__main__":
    main()
