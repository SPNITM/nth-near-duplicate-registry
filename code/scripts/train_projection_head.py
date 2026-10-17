#!/usr/bin/env python
"""Train a contrastive projection head on frozen MobileNetV3 features.

This design separates the ImageNet encoder baseline from the contribution of
contrastive metric learning. Identity splits are fixed before training. Model
selection uses calibration AP only; test identities are touched once per run.
"""

from __future__ import annotations

import argparse
import json
import os
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
    parser.add_argument("--manifest", type=Path, default=PROJECT / "data" / "manifests" / "pair_manifest_main.csv")
    parser.add_argument("--embedding-dir", type=Path, default=PROJECT / "results" / "raw_embeddings")
    parser.add_argument("--source-model", default="mobilenetv3_imagenet")
    parser.add_argument("--modes", nargs="+", default=["fixed1.0", "adaptive"])
    parser.add_argument("--seeds", type=int, nargs="+", default=[11, 23, 37, 53, 71])
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    import torch
    from sklearn.metrics import average_precision_score, roc_auc_score
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    torch.set_num_threads(max(1, min(20, os.cpu_count() or 1)))
    source_embeddings = np.load(args.embedding_dir / f"{args.source_model}.npy")
    source_paths = pd.read_csv(args.embedding_dir / f"{args.source_model}_paths.csv")["path"].tolist()
    path_to_index = {path: i for i, path in enumerate(source_paths)}
    manifest = pd.read_csv(args.manifest)
    train_pairs = build_evaluation_pairs(manifest, splits=("train",), negatives_per_positive=1)
    eval_pairs = build_evaluation_pairs(manifest, splits=("calibration", "test"), negatives_per_positive=5)

    def pair_indices(frame: pd.DataFrame):
        usable = frame[frame["original_path"].isin(path_to_index) & frame["transformed_path"].isin(path_to_index)].copy()
        left = np.array([path_to_index[p] for p in usable["original_path"]], dtype=np.int64)
        right = np.array([path_to_index[p] for p in usable["transformed_path"]], dtype=np.int64)
        labels = usable["Pair-Type"].eq("positive").to_numpy(np.float32)
        return usable, left, right, labels

    train_frame, train_left, train_right, train_labels = pair_indices(train_pairs)
    eval_frame, eval_left, eval_right, eval_labels = pair_indices(eval_pairs)
    features = torch.from_numpy(source_embeddings.astype(np.float32))

    class ProjectionHead(nn.Module):
        def __init__(self, input_dim: int):
            super().__init__()
            self.network = nn.Sequential(
                nn.Linear(input_dim, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(inplace=True),
                nn.Linear(512, 256),
            )
        def forward(self, values):
            return nn.functional.normalize(self.network(values), dim=1)

    def metrics(projected: np.ndarray, frame: pd.DataFrame):
        left = projected[eval_left]
        right = projected[eval_right]
        distances = 1.0 - np.sum(left * right, axis=1)
        output = {}
        for split in ("calibration", "test"):
            mask = frame["original_split"].eq(split).to_numpy()
            labels = eval_labels[mask].astype(bool)
            values = distances[mask]
            output[split] = {"labels": labels, "distances": values}
        threshold = calibrate_max_threshold(output["calibration"]["distances"][~output["calibration"]["labels"]], 0.001)
        summary = {"threshold": threshold}
        for split in ("calibration", "test"):
            labels = output[split]["labels"]; values = output[split]["distances"]
            accepted = values <= threshold
            tp = int(np.sum(labels & accepted)); fp = int(np.sum(~labels & accepted))
            fn = int(np.sum(labels & ~accepted)); tn = int(np.sum(~labels & ~accepted))
            summary.update({
                f"{split}_precision": tp / (tp + fp) if tp + fp else 0.0,
                f"{split}_recall": tp / (tp + fn) if tp + fn else 0.0,
                f"{split}_fpr": fp / (fp + tn) if fp + tn else 0.0,
                f"{split}_auc": float(roc_auc_score(labels, -values)),
                f"{split}_ap": float(average_precision_score(labels, -values)),
            })
        return summary, distances

    output_rows = []
    args.embedding_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = PROJECT / "results" / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    for mode in args.modes:
        for seed in args.seeds:
            random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
            model = ProjectionHead(features.shape[1]).to(args.device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
            generator = torch.Generator().manual_seed(seed)
            dataset = TensorDataset(
                torch.from_numpy(train_left), torch.from_numpy(train_right), torch.from_numpy(train_labels)
            )
            loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, generator=generator, num_workers=0)
            best_ap = -np.inf; best_state = None; stale = 0; adaptive_margin = 1.0
            history = []
            for epoch in range(1, args.epochs + 1):
                model.train(); losses = []
                for left_index, right_index, label in loader:
                    left = model(features[left_index].to(args.device))
                    right = model(features[right_index].to(args.device))
                    distance = torch.linalg.vector_norm(left - right, dim=1)
                    if mode == "adaptive":
                        negatives = distance[label.to(args.device) < 0.5].detach()
                        if negatives.numel():
                            observed = float(torch.quantile(negatives, 0.25).cpu())
                            adaptive_margin = 0.9 * adaptive_margin + 0.1 * min(1.5, max(0.5, observed))
                        margin = adaptive_margin
                    else:
                        margin = float(mode.removeprefix("fixed"))
                    target = label.to(args.device)
                    loss = (target * distance.square() + (1 - target) * torch.relu(margin - distance).square()).mean()
                    optimizer.zero_grad(); loss.backward(); optimizer.step(); losses.append(float(loss.detach().cpu()))
                model.eval()
                with torch.inference_mode():
                    chunks = []
                    for start in range(0, len(features), 4096):
                        chunks.append(model(features[start : start + 4096].to(args.device)).cpu().numpy())
                    projected = np.concatenate(chunks)
                current, _ = metrics(projected, eval_frame)
                history.append({"epoch": epoch, "loss": float(np.mean(losses)), "margin": adaptive_margin if mode == "adaptive" else margin, "calibration_ap": current["calibration_ap"]})
                if current["calibration_ap"] > best_ap + 1e-6:
                    best_ap = current["calibration_ap"]; best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}; stale = 0
                else:
                    stale += 1
                    if stale >= args.patience:
                        break
            model.load_state_dict(best_state); model.eval()
            with torch.inference_mode():
                projected = np.concatenate([model(features[start : start + 4096].to(args.device)).cpu().numpy() for start in range(0, len(features), 4096)])
            summary, distances = metrics(projected, eval_frame)
            name = f"mobilenetv3_contrastive_{mode}_seed{seed}"
            np.save(args.embedding_dir / f"{name}.npy", projected)
            pd.DataFrame({"path": source_paths}).to_csv(args.embedding_dir / f"{name}_paths.csv", index=False)
            torch.save({"state_dict": best_state, "seed": seed, "mode": mode, "source_model": args.source_model}, checkpoint_dir / f"{name}.pt")
            pd.DataFrame(history).to_csv(checkpoint_dir / f"{name}_history.csv", index=False)
            output_rows.append({"model": name, "mode": mode, "seed": seed, "epochs_completed": len(history), **summary})
            print(json.dumps(output_rows[-1], indent=2))
    output = pd.DataFrame(output_rows)
    output.to_csv(PROJECT / "results" / "projection_training_runs.csv", index=False)
    aggregate = output.groupby("mode").agg(
        runs=("seed", "size"),
        test_auc_mean=("test_auc", "mean"), test_auc_sd=("test_auc", "std"),
        test_ap_mean=("test_ap", "mean"), test_ap_sd=("test_ap", "std"),
        test_recall_mean=("test_recall", "mean"), test_recall_sd=("test_recall", "std"),
        test_fpr_mean=("test_fpr", "mean"), test_fpr_sd=("test_fpr", "std"),
    ).reset_index()
    aggregate.to_csv(PROJECT / "results" / "projection_training_aggregate.csv", index=False)


if __name__ == "__main__":
    main()
