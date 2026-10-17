#!/usr/bin/env python
"""Adaptive score-query evasion stress test for the canonical NTH gate.

The attack is a deterministic, gradient-free coordinate search over the actual
8x8 luminance grid consumed by NTH after its documented Lanczos preprocessing.
At every coordinate it queries the actual NTH Hamming distance for negative,
zero and positive perturbations and retains the best value.  Grid values remain
inside the stated L-infinity budget.  This is a preprocessing-domain diagnostic,
not a claim of optimal source-pixel adversarial risk.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / "code" / "src"))

def local_seed_path(original_path: str, root: Path) -> Path:
    """Map the generated ``ImageN_orig.ext`` name back to its seed image."""
    name = Path(str(original_path)).name
    stem = Path(name).stem
    if stem.endswith("_orig"):
        stem = stem[:-5]
    return root / f"{stem}{Path(name).suffix}"


def luminance_grid(image: Image.Image) -> np.ndarray:
    return np.asarray(
        image.convert("L").resize((8, 8), resample=Image.Resampling.LANCZOS),
        dtype=np.float32,
    )


def nth_bits_from_grid(grid: np.ndarray) -> np.ndarray:
    padded = np.pad(grid.astype(np.float32), 1, mode="wrap")
    sums = np.zeros((8, 8), dtype=np.float32)
    for row_shift in range(3):
        for column_shift in range(3):
            sums += padded[row_shift : row_shift + 8, column_shift : column_shift + 8]
    sums -= grid
    return (sums >= sums.mean()).astype(np.uint8).reshape(-1)


def attack_image(
    grid: np.ndarray,
    epsilon: int,
    reference: np.ndarray,
    radius: int,
    passes: int,
    rng: np.random.Generator,
) -> dict[str, float | int | bool]:
    sign_grid = np.zeros((8, 8), dtype=np.int8)
    best_distance = 0
    queries = 0
    for _ in range(passes):
        for flat_index in rng.permutation(64):
            row, column = divmod(int(flat_index), 8)
            current = int(sign_grid[row, column])
            best_sign = current
            local_best = best_distance
            for candidate in (-1, 0, 1):
                if candidate == current:
                    continue
                sign_grid[row, column] = candidate
                perturbed = np.clip(grid + epsilon * sign_grid, 0, 255)
                distance = int(np.count_nonzero(reference != nth_bits_from_grid(perturbed)))
                queries += 1
                if distance > local_best:
                    local_best = distance
                    best_sign = candidate
            sign_grid[row, column] = best_sign
            best_distance = local_best
            if best_distance > radius:
                # Continue the pass to record attack strength, but no extra pass
                # is required once the operational gate has been evaded.
                pass
    final_values = np.clip(grid + epsilon * sign_grid, 0, 255)
    absolute_change = np.abs(final_values - grid)
    return {
        "maximum_nth_distance": best_distance,
        "evades_radius": best_distance > radius,
        "queries": queries,
        "linf_observed_8bit": int(absolute_change.max()),
        "mean_absolute_change_8bit": float(absolute_change.mean()),
        "changed_grid_cell_fraction": float(np.mean(absolute_change > 0)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=PROJECT / "data" / "manifests_hpc_full_main_20260831" / "pair_manifest_main.csv",
    )
    parser.add_argument("--local-original-dir", type=Path, default=PROJECT.parent / "NTH_IMAGES")
    parser.add_argument("--radius", type=int, default=4)
    parser.add_argument("--passes", type=int, default=2)
    parser.add_argument("--epsilons", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument(
        "--output", type=Path, default=PROJECT / "results" / "nth_adaptive_coordinate_evasion.csv"
    )
    args = parser.parse_args()

    frame = pd.read_csv(args.manifest)
    originals = (
        frame[frame["original_split"].eq("test")][["seed_id", "original_path"]]
        .dropna()
        .drop_duplicates("seed_id")
        .sort_values("seed_id")
    )
    originals["local_path"] = originals["original_path"].map(
        lambda value: str(local_seed_path(value, args.local_original_dir))
    )
    missing = originals[~originals["local_path"].map(lambda value: Path(value).is_file())]
    if not missing.empty:
        raise FileNotFoundError(f"Missing {len(missing)} local test originals")

    rng = np.random.default_rng(20260901)
    rows = []
    for epsilon in args.epsilons:
        for item in originals.itertuples(index=False):
            with Image.open(item.local_path) as opened:
                image = opened.convert("RGB")
            grid = luminance_grid(image)
            reference = nth_bits_from_grid(grid)
            result = attack_image(grid, epsilon, reference, args.radius, args.passes, rng)
            rows.append(
                {
                    "seed_id": int(item.seed_id),
                    "original_file": Path(item.local_path).name,
                    "attack": "adaptive-score-query-on-canonical-8x8-luminance-grid",
                    "linf_epsilon_8bit": epsilon,
                    "passes": args.passes,
                    "radius": args.radius,
                    **result,
                }
            )

    output = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)
    summary = output.groupby(["attack", "linf_epsilon_8bit"], as_index=False).agg(
        images=("seed_id", "size"),
        evasion_rate=("evades_radius", "mean"),
        mean_max_distance=("maximum_nth_distance", "mean"),
        median_queries=("queries", "median"),
        mean_absolute_change_8bit=("mean_absolute_change_8bit", "mean"),
        mean_changed_grid_cell_fraction=("changed_grid_cell_fraction", "mean"),
    )
    summary.to_csv(args.output.with_name("nth_adaptive_coordinate_evasion_summary.csv"), index=False)
    metadata = {
        "attack": "adaptive score-query coordinate search on the canonical 8x8 luminance preprocessing grid",
        "test_identities": len(originals),
        "radius": args.radius,
        "passes": args.passes,
        "seed": 20260901,
        "limitation": "Preprocessing-domain diagnostic non-gradient attack; it is not an optimal, source-pixel, or certified adversarial bound.",
    }
    args.output.with_suffix(".metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
