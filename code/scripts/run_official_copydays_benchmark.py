#!/usr/bin/env python
"""Evaluate the frozen revision policies on the official INRIA Copydays strong set.

The benchmark consists of 157 original reference images and 229 strongly
attacked queries.  Images are not redistributed by this project.  This script
expects archives obtained from the official INRIA URLs or the VISSL mirror and
records source/checksum metadata beside the aggregate results.

The evaluation deliberately uses thresholds selected on the private main
calibration split.  Copydays labels are used only for final evaluation.  No
Flickr distractors are included, so the resulting retrieval metrics are the
official strong-query/original-reference protocol without the optional large
distractor collection.
"""

from __future__ import annotations

import argparse
import hashlib
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

from nth_registry.baseline_hashes import BASELINES
from nth_registry.nth import compute_nth_bits


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def image_paths(folder: Path) -> list[Path]:
    values = sorted(
        path.resolve()
        for path in folder.rglob("*")
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    if not values:
        raise FileNotFoundError(f"No images found below {folder}")
    return values


def block_id(path: Path) -> int:
    stem = path.stem
    if len(stem) != 6 or not stem.isdigit():
        raise ValueError(f"Unexpected Copydays filename: {path.name}")
    return int(stem[:4])


def extract_embeddings(paths: list[Path], batch_size: int, device: str):
    import torch
    from torch.utils.data import DataLoader, Dataset
    from torchvision import models

    class ImageDataset(Dataset):
        def __init__(self, values, transform):
            self.values = values
            self.transform = transform

        def __len__(self):
            return len(self.values)

        def __getitem__(self, index):
            with Image.open(self.values[index]) as image:
                return self.transform(image.convert("RGB")), index

    torch.set_num_threads(max(1, min(20, os.cpu_count() or 1)))
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")
    weights = models.ShuffleNet_V2_X1_0_Weights.IMAGENET1K_V1
    network = models.shufflenet_v2_x1_0(weights=weights)
    network.fc = torch.nn.Identity()
    network = network.eval().to(device)
    loader = DataLoader(
        ImageDataset(paths, weights.transforms()),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    batches = []
    with torch.inference_mode():
        for batch, _ in loader:
            values = network(batch.to(device))
            batches.append(torch.nn.functional.normalize(values.float(), dim=1).cpu().numpy())
    return np.concatenate(batches), str(weights), torch.__version__, torch.version.cuda


def rank_statistics(distance: np.ndarray, true_indices: np.ndarray) -> dict[str, float]:
    # Stable sorting makes exact ties deterministic and reproducible.
    order = np.argsort(distance, axis=1, kind="stable")
    ranks = np.empty(len(true_indices), dtype=np.int32)
    for row, true_index in enumerate(true_indices):
        ranks[row] = int(np.flatnonzero(order[row] == true_index)[0]) + 1
    return {
        "top1_retrieval_accuracy": float(np.mean(ranks == 1)),
        "recall_at_5": float(np.mean(ranks <= 5)),
        "mean_reciprocal_rank": float(np.mean(1.0 / ranks)),
        "mean_average_precision": float(np.mean(1.0 / ranks)),
        "median_true_rank": float(np.median(ranks)),
        "maximum_true_rank": int(np.max(ranks)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    base = PROJECT / "external_data" / "copydays_official"
    parser.add_argument("--original-dir", type=Path, default=base / "original")
    parser.add_argument("--strong-dir", type=Path, default=base / "strong")
    parser.add_argument("--original-archive", type=Path, default=base / "downloads" / "copydays_original.tar.gz")
    parser.add_argument("--strong-archive", type=Path, default=base / "downloads" / "copydays_strong.tar.gz")
    parser.add_argument("--output-dir", type=Path, default=PROJECT / "results" / "official_copydays_20260901")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    originals = image_paths(args.original_dir)
    queries = image_paths(args.strong_dir)
    if len(originals) != 157 or len(queries) != 229:
        raise RuntimeError(
            f"Expected 157 originals and 229 strong queries; found {len(originals)} and {len(queries)}"
        )
    original_by_block = {block_id(path): index for index, path in enumerate(originals)}
    if len(original_by_block) != len(originals):
        raise RuntimeError("Duplicate original Copydays block identifiers")
    missing_blocks = sorted({block_id(path) for path in queries} - set(original_by_block))
    if missing_blocks:
        raise RuntimeError(f"Queries without originals: {missing_blocks}")
    true_indices = np.asarray([original_by_block[block_id(path)] for path in queries], dtype=np.int32)

    all_paths = originals + queries
    embeddings, weights_id, torch_version, cuda_version = extract_embeddings(
        all_paths, args.batch_size, args.device
    )
    original_embeddings = embeddings[: len(originals)]
    query_embeddings = embeddings[len(originals) :]
    embedding_distance = 1.0 - query_embeddings @ original_embeddings.T

    hashes = np.stack([compute_nth_bits(path) for path in all_paths]).astype(np.uint8)
    original_hashes = hashes[: len(originals)]
    query_hashes = hashes[len(originals) :]
    hamming_distance = np.count_nonzero(
        query_hashes[:, None, :] != original_hashes[None, :, :], axis=2
    ).astype(np.int16)

    deep_summary = rank_statistics(embedding_distance, true_indices)
    hash_summary = rank_statistics(hamming_distance.astype(np.float64), true_indices)
    hash_retrieval_rows = [{"method": "NTH-default", **hash_summary}]
    hash_retrieval_metadata = {"NTH-default": hash_summary}
    for method_name, method in BASELINES.items():
        method_hashes = np.stack([method(path) for path in all_paths]).astype(np.uint8)
        method_originals = method_hashes[: len(originals)]
        method_queries = method_hashes[len(originals) :]
        method_distances = np.count_nonzero(
            method_queries[:, None, :] != method_originals[None, :, :], axis=2
        ).astype(np.int16)
        method_summary = rank_statistics(method_distances.astype(np.float64), true_indices)
        hash_retrieval_rows.append({"method": method_name, **method_summary})
        hash_retrieval_metadata[method_name] = method_summary

    full_summary = json.loads(
        (PROJECT / "results" / "hpc_full_main_20260831" / "full_main_verification_summary.json").read_text(
            encoding="utf-8"
        )
    )
    deep_threshold = float(full_summary["selected_model"]["embedding_threshold"])
    policy_frame = pd.read_csv(PROJECT / "results" / "two_stage_shufflenet_pair_results.csv")
    calibration = policy_frame[policy_frame["split"].eq("calibration")]
    thresholds = {
        int(row.nth_radius): float(row.embedding_threshold)
        for row in calibration.itertuples(index=False)
        if int(row.nth_radius) in {2, 4, 8, 12, 16}
    }

    records = []
    true_deep_distance = embedding_distance[np.arange(len(queries)), true_indices]
    deep_accepted = embedding_distance <= deep_threshold
    records.append(
        {
            "policy": "ShuffleNet only",
            "nth_radius": None,
            "embedding_threshold": deep_threshold,
            "query_count": len(queries),
            "reference_count": len(originals),
            "true_candidate_recall": 1.0,
            "true_end_to_end_acceptance": float(np.mean(true_deep_distance <= deep_threshold)),
            "top1_retrieval_accuracy": deep_summary["top1_retrieval_accuracy"],
            "mean_average_precision": deep_summary["mean_average_precision"],
            "mean_reciprocal_rank": deep_summary["mean_reciprocal_rank"],
            "recall_at_5": deep_summary["recall_at_5"],
            "any_false_accept_probability": float(
                np.mean(
                    np.any(
                        deep_accepted
                        & (np.arange(len(originals))[None, :] != true_indices[:, None]),
                        axis=1,
                    )
                )
            ),
            "mean_shortlist": float(len(originals)),
            "mean_final_accepted_candidates": float(np.mean(np.sum(deep_accepted, axis=1))),
        }
    )

    query_rows = []
    for query_index, query_path in enumerate(queries):
        query_rows.append(
            {
                "query_file": query_path.name,
                "true_original_file": originals[true_indices[query_index]].name,
                "true_nth_distance": int(hamming_distance[query_index, true_indices[query_index]]),
                "true_embedding_distance": float(embedding_distance[query_index, true_indices[query_index]]),
                "deep_true_rank": int(
                    np.flatnonzero(
                        np.argsort(embedding_distance[query_index], kind="stable") == true_indices[query_index]
                    )[0]
                    + 1
                ),
            }
        )

    for radius, threshold in sorted(thresholds.items()):
        stage1 = hamming_distance <= radius
        accepted = stage1 & (embedding_distance <= threshold)
        true_stage1 = stage1[np.arange(len(queries)), true_indices]
        true_accepted = accepted[np.arange(len(queries)), true_indices]
        false_mask = np.arange(len(originals))[None, :] != true_indices[:, None]
        correct_top1 = []
        for row in range(len(queries)):
            candidates = np.flatnonzero(accepted[row])
            if len(candidates) == 0:
                correct_top1.append(False)
                continue
            best = candidates[np.argmin(embedding_distance[row, candidates])]
            correct_top1.append(int(best) == int(true_indices[row]))
        records.append(
            {
                "policy": f"NTH r={radius} + ShuffleNet",
                "nth_radius": radius,
                "embedding_threshold": threshold,
                "query_count": len(queries),
                "reference_count": len(originals),
                "true_candidate_recall": float(np.mean(true_stage1)),
                "true_end_to_end_acceptance": float(np.mean(true_accepted)),
                "top1_retrieval_accuracy": float(np.mean(correct_top1)),
                "mean_average_precision": None,
                "mean_reciprocal_rank": None,
                "recall_at_5": None,
                "any_false_accept_probability": float(np.mean(np.any(accepted & false_mask, axis=1))),
                "mean_shortlist": float(np.mean(np.sum(stage1, axis=1))),
                "mean_final_accepted_candidates": float(np.mean(np.sum(accepted, axis=1))),
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    result_frame = pd.DataFrame(records)
    result_frame.to_csv(args.output_dir / "official_copydays_strong_results.csv", index=False)
    pd.DataFrame(hash_retrieval_rows).to_csv(
        args.output_dir / "official_copydays_hash_retrieval_baselines.csv", index=False
    )
    pd.DataFrame(query_rows).to_csv(args.output_dir / "official_copydays_query_results.csv", index=False)
    np.save(args.output_dir / "official_copydays_shufflenet_embeddings.npy", embeddings)
    pd.DataFrame(
        {
            "path": [str(path) for path in all_paths],
            "split": ["original"] * len(originals) + ["strong"] * len(queries),
        }
    ).to_csv(args.output_dir / "official_copydays_embedding_paths.csv", index=False)

    metadata = {
        "benchmark": "INRIA Copydays",
        "evaluation_subset": "157 original references and 229 strong queries",
        "optional_distractors_included": False,
        "labels_used_for_calibration": False,
        "policy_source": "private-main calibration split, frozen before Copydays evaluation",
        "official_homepage": "https://lear.inrialpes.fr/~jegou/data.php.html#copydays",
        "download_mirror": "https://dl.fbaipublicfiles.com/vissl/datasets/",
        "copyright_notice": "INRIA is the copyright holder; images are not included in the public release.",
        "original_archive_sha256": file_sha256(args.original_archive),
        "strong_archive_sha256": file_sha256(args.strong_archive),
        "original_count": len(originals),
        "strong_query_count": len(queries),
        "model": "shufflenet_v2_x1",
        "weights": weights_id,
        "device": args.device,
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch_version,
        "cuda": cuda_version,
        "deep_retrieval": deep_summary,
        "nth_retrieval": hash_summary,
        "hash_retrieval_baselines": hash_retrieval_metadata,
    }
    (args.output_dir / "official_copydays_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(result_frame.to_string(index=False))
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
