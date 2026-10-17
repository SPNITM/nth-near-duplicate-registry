#!/usr/bin/env python
"""Train a controlled 64-bit supervised deep-hash baseline on frozen features.

The pairwise logistic objective is inspired by DPSH/HashNet-style supervised
hashing, but the MobileNetV3 feature extractor is deliberately frozen so the
experiment remains feasible and isolates the learned binary projection.
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / "code" / "src"))

from nth_registry.evaluation_data import build_evaluation_pairs
from nth_registry.metrics import calibrate_max_threshold


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[11, 23, 37, 53, 71])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, default=PROJECT / "results" / "supervised_deep_hash_runs.csv")
    args = parser.parse_args()

    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    torch.set_num_threads(20)
    embedding_dir = PROJECT / "results" / "raw_embeddings"
    source = np.load(embedding_dir / "mobilenetv3_imagenet.npy").astype(np.float32)
    source_paths = pd.read_csv(embedding_dir / "mobilenetv3_imagenet_paths.csv")["path"].astype(str).tolist()
    lookup = {path: index for index, path in enumerate(source_paths)}
    manifest = pd.read_csv(PROJECT / "data" / "manifests" / "pair_manifest_main.csv")
    train_pairs = build_evaluation_pairs(manifest, splits=("train",), negatives_per_positive=1)
    eval_pairs = build_evaluation_pairs(manifest, splits=("calibration", "test"), negatives_per_positive=5)
    train_pairs = train_pairs[
        train_pairs["original_path"].isin(lookup) & train_pairs["transformed_path"].isin(lookup)
    ].copy()
    eval_pairs = eval_pairs[
        eval_pairs["original_path"].isin(lookup) & eval_pairs["transformed_path"].isin(lookup)
    ].copy()
    train_left = torch.tensor([lookup[path] for path in train_pairs["original_path"]], dtype=torch.long)
    train_right = torch.tensor([lookup[path] for path in train_pairs["transformed_path"]], dtype=torch.long)
    train_labels = torch.tensor(train_pairs["Pair-Type"].eq("positive").to_numpy(), dtype=torch.float32)
    loader = DataLoader(
        TensorDataset(train_left, train_right, train_labels),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
    )
    features = torch.from_numpy(source)
    eval_left = np.asarray([lookup[path] for path in eval_pairs["original_path"]], dtype=int)
    eval_right = np.asarray([lookup[path] for path in eval_pairs["transformed_path"]], dtype=int)

    class HashHead(nn.Module):
        def __init__(self, input_dim: int):
            super().__init__()
            self.network = nn.Sequential(
                nn.Linear(input_dim, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(inplace=True),
                nn.Linear(256, 64),
                nn.Tanh(),
            )
        def forward(self, values): return self.network(values)

    rows = []
    for seed in args.seeds:
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        model = HashHead(source.shape[1]).to(args.device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        for _ in range(args.epochs):
            model.train()
            for left_index, right_index, labels in loader:
                a = model(features[left_index].to(args.device))
                b = model(features[right_index].to(args.device))
                similarity_logits = 6.0 * torch.sum(a * b, dim=1) / 64.0
                pair_loss = nn.functional.binary_cross_entropy_with_logits(
                    similarity_logits, labels.to(args.device)
                )
                quantization = ((a.abs() - 1.0).square().mean() + (b.abs() - 1.0).square().mean())
                balance = a.mean(dim=0).square().mean() + b.mean(dim=0).square().mean()
                loss = pair_loss + 0.05 * quantization + 0.01 * balance
                optimizer.zero_grad(); loss.backward(); optimizer.step()
        model.eval()
        with torch.inference_mode():
            values = np.concatenate([
                model(features[start : start + 4096].to(args.device)).cpu().numpy()
                for start in range(0, len(features), 4096)
            ])
        bits = values >= 0
        distances = np.count_nonzero(bits[eval_left] != bits[eval_right], axis=1)
        evaluated = eval_pairs.copy()
        evaluated["distance"] = distances
        calibration = evaluated[evaluated["original_split"].eq("calibration")]
        test = evaluated[evaluated["original_split"].eq("test")]
        radius = calibrate_max_threshold(
            calibration[calibration["Pair-Type"].eq("negative")]["distance"].to_numpy(), 0.001
        )

        def summarize(part: pd.DataFrame) -> dict:
            positive = part["Pair-Type"].eq("positive").to_numpy()
            accepted = part["distance"].to_numpy() <= radius
            tp = int(np.sum(positive & accepted)); fp = int(np.sum(~positive & accepted))
            fn = int(np.sum(positive & ~accepted)); tn = int(np.sum(~positive & ~accepted))
            return {
                "precision": tp / (tp + fp) if tp + fp else 0.0,
                "recall": tp / (tp + fn) if tp + fn else 0.0,
                "fpr": fp / (fp + tn) if fp + tn else 0.0,
            }
        cal = summarize(calibration); tst = summarize(test)
        name = f"supervised_binary_hash64_seed{seed}"
        rows.append({
            "model": name,
            "seed": seed,
            "epochs": args.epochs,
            "bits": 64,
            "calibrated_radius": radius,
            **{f"calibration_{key}": value for key, value in cal.items()},
            **{f"test_{key}": value for key, value in tst.items()},
        })
        np.save(PROJECT / "results" / "raw_embeddings" / f"{name}_bits.npy", bits)
        torch.save({"state_dict": model.state_dict(), "seed": seed}, PROJECT / "results" / "checkpoints" / f"{name}.pt")
    output = pd.DataFrame(rows)
    output.to_csv(args.output, index=False)
    aggregate = output.agg({
        "test_precision": ["mean", "std"],
        "test_recall": ["mean", "std"],
        "test_fpr": ["mean", "std"],
    })
    aggregate.to_csv(PROJECT / "results" / "supervised_deep_hash_aggregate.csv")
    print(output.to_string(index=False))
    print("\n", aggregate.to_string())


if __name__ == "__main__":
    main()
