#!/usr/bin/env python
"""Evaluate an off-chain multiview crop fallback on held-out crop ratios.

Each registry identity stores six off-chain ShuffleNet embeddings/NTH codes:
the full original plus fixed central crops removing 10--50% of width and
height.  Queries use held-out 15/25/35/45% crop ratios, so no query is byte- or
view-identical to a stored reference.  The experiment reports identity-level
retrieval and false acceptance with frozen main-calibration thresholds.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / "code" / "src"))

from nth_registry.nth import compute_nth_bits


def local_seed_path(original_path: str, root: Path) -> Path:
    """Map the generated ``ImageN_orig.ext`` name back to its seed image."""
    name = Path(str(original_path)).name
    stem = Path(name).stem
    if stem.endswith("_orig"):
        stem = stem[:-5]
    return root / f"{stem}{Path(name).suffix}"


def crop_view(image: Image.Image, removed_fraction: float) -> Image.Image:
    image = image.convert("RGB")
    if removed_fraction == 0:
        return image.copy()
    width, height = image.size
    dx = int(round(width * removed_fraction / 2.0))
    dy = int(round(height * removed_fraction / 2.0))
    cropped = image.crop((dx, dy, max(dx + 1, width - dx), max(dy + 1, height - dy)))
    return cropped.resize((width, height), Image.Resampling.LANCZOS)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=PROJECT / "data" / "manifests_hpc_full_main_20260831" / "pair_manifest_main.csv",
    )
    parser.add_argument("--local-original-dir", type=Path, default=PROJECT.parent / "NTH_IMAGES")
    parser.add_argument("--reference-ratios", type=float, nargs="+", default=[0, 0.1, 0.2, 0.3, 0.4, 0.5])
    parser.add_argument("--query-ratios", type=float, nargs="+", default=[0.15, 0.25, 0.35, 0.45])
    parser.add_argument("--nth-radius", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--output-dir", type=Path, default=PROJECT / "results" / "multiview_crop_mitigation_20260901"
    )
    args = parser.parse_args()

    import torch
    from torch.utils.data import DataLoader, Dataset
    from torchvision import models

    torch.set_num_threads(max(1, min(20, os.cpu_count() or 1)))
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    frame = pd.read_csv(args.manifest)
    originals = (
        frame[frame["original_split"].eq("test")][["seed_id", "original_path"]]
        .dropna()
        .drop_duplicates("seed_id")
        .sort_values("seed_id")
        .reset_index(drop=True)
    )
    originals["local_path"] = originals["original_path"].map(
        lambda value: str(local_seed_path(value, args.local_original_dir))
    )
    if not originals["local_path"].map(lambda value: Path(value).is_file()).all():
        raise FileNotFoundError("One or more test originals are absent from the local original-image mirror")

    reference_items = [
        (row.local_path, float(ratio), int(row.seed_id))
        for row in originals.itertuples(index=False)
        for ratio in args.reference_ratios
    ]
    query_items = [
        (row.local_path, float(ratio), int(row.seed_id))
        for ratio in args.query_ratios
        for row in originals.itertuples(index=False)
    ]
    all_items = reference_items + query_items

    weights = models.ShuffleNet_V2_X1_0_Weights.IMAGENET1K_V1
    transform = weights.transforms()

    class Views(Dataset):
        def __init__(self, values):
            self.values = values

        def __len__(self):
            return len(self.values)

        def __getitem__(self, index):
            path, ratio, _ = self.values[index]
            with Image.open(path) as opened:
                view = crop_view(opened, ratio)
            return transform(view), index

    network = models.shufflenet_v2_x1_0(weights=weights)
    network.fc = torch.nn.Identity()
    network = network.eval().to(args.device)
    batches = []
    with torch.inference_mode():
        for batch, _ in DataLoader(Views(all_items), batch_size=args.batch_size, shuffle=False, num_workers=0):
            batches.append(
                torch.nn.functional.normalize(network(batch.to(args.device)).float(), dim=1).cpu().numpy()
            )
    embeddings = np.concatenate(batches)
    hashes = []
    for path, ratio, _ in all_items:
        with Image.open(path) as opened:
            hashes.append(compute_nth_bits(crop_view(opened, ratio)))
    hashes = np.stack(hashes).astype(np.uint8)

    reference_count = len(reference_items)
    reference_embeddings = embeddings[:reference_count]
    query_embeddings = embeddings[reference_count:]
    reference_hashes = hashes[:reference_count]
    query_hashes = hashes[reference_count:]
    reference_seed = np.asarray([item[2] for item in reference_items], dtype=np.int32)
    query_seed = np.asarray([item[2] for item in query_items], dtype=np.int32)
    query_ratio = np.asarray([item[1] for item in query_items], dtype=np.float64)
    identity_values = originals["seed_id"].astype(int).to_numpy()

    embedding_view_distance = 1.0 - query_embeddings @ reference_embeddings.T
    hamming_view_distance = np.count_nonzero(
        query_hashes[:, None, :] != reference_hashes[None, :, :], axis=2
    ).astype(np.int16)
    identity_embedding = np.stack(
        [embedding_view_distance[:, reference_seed == seed].min(axis=1) for seed in identity_values], axis=1
    )
    identity_hamming = np.stack(
        [hamming_view_distance[:, reference_seed == seed].min(axis=1) for seed in identity_values], axis=1
    )

    baseline_indices = np.asarray(
        [index for index, item in enumerate(reference_items) if item[1] == 0], dtype=np.int32
    )
    baseline_embedding = embedding_view_distance[:, baseline_indices]
    baseline_hamming = hamming_view_distance[:, baseline_indices]
    seed_to_column = {int(seed): index for index, seed in enumerate(identity_values)}
    truth = np.asarray([seed_to_column[int(seed)] for seed in query_seed], dtype=np.int32)

    full_summary = json.loads(
        (PROJECT / "results" / "hpc_full_main_20260831" / "full_main_verification_summary.json").read_text(
            encoding="utf-8"
        )
    )
    deep_threshold = float(full_summary["selected_model"]["embedding_threshold"])
    policy_frame = pd.read_csv(PROJECT / "results" / "two_stage_shufflenet_pair_results.csv")
    two_stage_threshold = float(
        policy_frame[
            policy_frame["split"].eq("calibration")
            & policy_frame["nth_radius"].eq(args.nth_radius)
        ]["embedding_threshold"].iloc[0]
    )

    def summarize(name, embedding_matrix, hamming_matrix, use_hash, threshold, mask, reference_views):
        emb = embedding_matrix[mask]
        ham = hamming_matrix[mask]
        target = truth[mask]
        candidate = ham <= args.nth_radius if use_hash else np.ones_like(ham, dtype=bool)
        accepted = candidate & (emb <= threshold)
        true_candidate = candidate[np.arange(len(target)), target]
        true_accepted = accepted[np.arange(len(target)), target]
        false_mask = np.arange(len(identity_values))[None, :] != target[:, None]
        correct = []
        for row in range(len(target)):
            candidates = np.flatnonzero(accepted[row])
            if len(candidates) == 0:
                correct.append(False)
            else:
                correct.append(int(candidates[np.argmin(emb[row, candidates])]) == int(target[row]))
        return {
            "policy": name,
            "query_crop_fraction_removed": float(query_ratio[mask][0]),
            "query_count": int(len(target)),
            "reference_views_per_identity": int(reference_views),
            "true_candidate_recall": float(np.mean(true_candidate)),
            "true_acceptance": float(np.mean(true_accepted)),
            "top1_accepted_retrieval_accuracy": float(np.mean(correct)),
            "any_false_accept_probability": float(np.mean(np.any(accepted & false_mask, axis=1))),
            "mean_candidate_identities": float(np.mean(np.sum(candidate, axis=1))),
            "mean_accepted_identities": float(np.mean(np.sum(accepted, axis=1))),
        }

    rows = []
    for ratio in args.query_ratios:
        mask = np.isclose(query_ratio, ratio)
        rows.extend(
            [
                summarize(
                    "single-view deep fallback", baseline_embedding, baseline_hamming,
                    False, deep_threshold, mask, 1,
                ),
                summarize(
                    "multiview deep fallback", identity_embedding, identity_hamming,
                    False, deep_threshold, mask, len(args.reference_ratios),
                ),
                summarize(
                    f"single-view NTH r={args.nth_radius} + ShuffleNet",
                    baseline_embedding,
                    baseline_hamming,
                    True,
                    two_stage_threshold,
                    mask,
                    1,
                ),
                summarize(
                    f"multiview NTH r={args.nth_radius} + ShuffleNet",
                    identity_embedding,
                    identity_hamming,
                    True,
                    two_stage_threshold,
                    mask,
                    len(args.reference_ratios),
                ),
            ]
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    result = pd.DataFrame(rows)
    result.to_csv(args.output_dir / "multiview_crop_mitigation_results.csv", index=False)
    np.save(args.output_dir / "multiview_crop_embeddings.npy", embeddings)
    pd.DataFrame(
        all_items, columns=["path", "crop_fraction_removed", "seed_id"]
    ).assign(role=["reference"] * len(reference_items) + ["query"] * len(query_items)).to_csv(
        args.output_dir / "multiview_crop_embedding_paths.csv", index=False
    )
    metadata = {
        "strategy": "six off-chain central-crop reference views per identity with a fallback comparison path",
        "reference_ratios": args.reference_ratios,
        "held_out_query_ratios": args.query_ratios,
        "test_identities": len(identity_values),
        "on_chain_storage_change": "none; model/policy identifiers and audit head remain the only anchored decision artefacts",
        "deep_threshold_from_main_calibration": deep_threshold,
        "two_stage_radius": args.nth_radius,
        "two_stage_threshold_from_main_calibration": two_stage_threshold,
        "weights": str(weights),
        "device": args.device,
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "limitation": "Central-crop diagnostic only; not a general local-feature or geometric-alignment solution.",
    }
    (args.output_dir / "multiview_crop_mitigation_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
