#!/usr/bin/env python
"""Evaluate main-calibrated policies on supplied COCO/Copydays seed subsets.

These are explicitly seed-source transfer tests under the same synthetic edit
generator, not the official benchmark protocols.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / "code" / "src"))

from nth_registry.nth import compute_nth_bits


def build_pairs(frame: pd.DataFrame, negatives_per_positive: int = 5) -> pd.DataFrame:
    positives = frame[frame["Pair-Type"].eq("positive") & frame["usable"].astype(bool)].copy()
    originals = frame[["seed_id", "original_path"]].dropna().drop_duplicates("seed_id")
    seed_values = originals["seed_id"].astype(int).to_numpy()
    path_by_seed = dict(zip(seed_values, originals["original_path"]))
    rng = np.random.default_rng(20260830)
    negatives = []
    for item in positives.to_dict("records"):
        candidates = seed_values[seed_values != int(item["seed_id"])]
        for negative_seed in rng.choice(candidates, size=min(negatives_per_positive, len(candidates)), replace=False):
            negatives.append({
                "seed_id": int(negative_seed),
                "transformed_seed_id": int(item["seed_id"]),
                "original_path": path_by_seed[int(negative_seed)],
                "transformed_path": item["transformed_path"],
                "Pair-Type": "negative",
                "Label": "none",
            })
    return pd.concat([positives, pd.DataFrame(negatives)], ignore_index=True)


def metrics(labels: np.ndarray, accepted: np.ndarray) -> dict[str, float | int]:
    positive = labels == "positive"
    tp = int(np.sum(positive & accepted)); fp = int(np.sum(~positive & accepted))
    fn = int(np.sum(positive & ~accepted)); tn = int(np.sum(~positive & ~accepted))
    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / (tp + fn) if tp + fn else 0.0,
        "fpr": fp / (fp + tn) if fp + tn else 0.0,
    }


def main() -> None:
    import torch
    from torch.utils.data import DataLoader, Dataset
    from torchvision import models

    torch.set_num_threads(max(1, min(20, os.cpu_count() or 1)))
    weights = models.ShuffleNet_V2_X1_0_Weights.IMAGENET1K_V1
    network = models.shufflenet_v2_x1_0(weights=weights)
    network.fc = torch.nn.Identity()
    network.eval()
    transform = weights.transforms()

    class Paths(Dataset):
        def __init__(self, paths): self.paths = paths
        def __len__(self): return len(self.paths)
        def __getitem__(self, index):
            with Image.open(self.paths[index]) as image:
                return transform(image.convert("RGB")), index

    pair_policy = pd.read_csv(PROJECT / "results" / "two_stage_shufflenet_pair_results.csv")
    thresholds = {
        int(row.nth_radius): float(row.embedding_threshold)
        for row in pair_policy[pair_policy["split"].eq("calibration")].itertuples(index=False)
    }
    deep_threshold = float(
        pd.read_csv(PROJECT / "results" / "deep_frozen_baseline_summary.csv")
        .query("model == 'shufflenet_v2_x1'")["embedding_threshold"].iloc[0]
    )
    rows = []
    per_label_rows = []
    for dataset, manifest_name in [
        ("COCO seed subset", "pair_manifest_coco.csv"),
        ("Copydays seed subset", "pair_manifest_copydays.csv"),
    ]:
        frame = pd.read_csv(PROJECT / "data" / "manifests" / manifest_name)
        pairs = build_pairs(frame)
        paths = sorted(set(pairs["original_path"].dropna()) | set(pairs["transformed_path"].dropna()))
        nth_lookup = {}
        for path in paths:
            with Image.open(path) as opened:
                nth_lookup[path] = compute_nth_bits(opened.convert("RGB"))
        batches = []
        with torch.inference_mode():
            for batch, _ in DataLoader(Paths(paths), batch_size=128, shuffle=False, num_workers=0):
                batches.append(torch.nn.functional.normalize(network(batch).float(), dim=1).numpy())
        embeddings = np.concatenate(batches)
        lookup = {path: index for index, path in enumerate(paths)}
        left = embeddings[[lookup[path] for path in pairs["original_path"]]]
        right = embeddings[[lookup[path] for path in pairs["transformed_path"]]]
        embedding_distance = 1.0 - np.sum(left * right, axis=1)
        nth_distance = np.fromiter(
            (np.count_nonzero(nth_lookup[a] != nth_lookup[b]) for a, b in zip(pairs["original_path"], pairs["transformed_path"])),
            dtype=np.int16,
            count=len(pairs),
        )
        labels = pairs["Pair-Type"].to_numpy()
        policies = [("ShuffleNet only", None, deep_threshold)] + [
            (f"NTH r={radius} + ShuffleNet", radius, threshold)
            for radius, threshold in thresholds.items() if radius in {12, 16}
        ]
        for policy, radius, threshold in policies:
            accepted = embedding_distance <= threshold
            if radius is not None:
                accepted &= nth_distance <= radius
            record = {
                "dataset": dataset,
                "evaluation_scope": "available supplied files; synthetic edit protocol; not official benchmark protocol",
                "policy": policy,
                "nth_radius": radius,
                "embedding_threshold_from_main_calibration": threshold,
                "positive_pairs": int(np.sum(labels == "positive")),
                "negative_pairs": int(np.sum(labels == "negative")),
                **metrics(labels, accepted),
            }
            rows.append(record)
            positives = pairs[labels == "positive"].copy()
            positives["accepted"] = accepted[labels == "positive"]
            per_label = positives.groupby("Label", as_index=False).agg(n=("accepted", "size"), recall=("accepted", "mean"))
            per_label.insert(0, "policy", policy)
            per_label.insert(0, "dataset", dataset)
            per_label_rows.append(per_label)
        cache_dir = PROJECT / "results" / "seed_source_transfer_embeddings"
        cache_dir.mkdir(parents=True, exist_ok=True)
        stem = "coco" if dataset.startswith("COCO") else "copydays"
        np.save(cache_dir / f"{stem}_shufflenet_v2_x1.npy", embeddings)
        pd.DataFrame({"path": paths}).to_csv(cache_dir / f"{stem}_paths.csv", index=False)

    summary = pd.DataFrame(rows)
    summary.to_csv(PROJECT / "results" / "seed_source_transfer_summary.csv", index=False)
    pd.concat(per_label_rows, ignore_index=True).to_csv(
        PROJECT / "results" / "seed_source_transfer_per_transform.csv", index=False
    )
    metadata = {
        "weights": str(weights),
        "policy_source": "thresholds calibrated on the main calibration split only",
        "limitation": "These are not official COCO or Copydays copy-detection protocols.",
    }
    (PROJECT / "results" / "seed_source_transfer_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
