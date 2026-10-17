#!/usr/bin/env python
"""Black-box bounded-noise evasion stress test for the canonical NTH filter."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / "code" / "src"))

from nth_registry.nth import compute_nth_bits


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=PROJECT / "data" / "manifests" / "pair_manifest_main.csv")
    parser.add_argument("--radius", type=int, default=4)
    parser.add_argument("--trials", type=int, default=32)
    parser.add_argument("--output", type=Path, default=PROJECT / "results" / "nth_bounded_noise_evasion.csv")
    args = parser.parse_args()
    frame = pd.read_csv(args.manifest)
    originals = frame[frame["original_split"].eq("test")][["seed_id", "original_path"]].dropna().drop_duplicates("seed_id")
    rng = np.random.default_rng(20260830)
    rows = []
    for epsilon in [1, 2, 4, 8, 16]:
        for item in originals.itertuples(index=False):
            with Image.open(item.original_path) as opened:
                image = opened.convert("RGB")
                values = np.asarray(image, dtype=np.int16)
                reference = compute_nth_bits(image)
                for attack in ["pixelwise-rademacher", "coherent-8x8-rademacher"]:
                    best_distance = -1
                    for _ in range(args.trials):
                        if attack == "pixelwise-rademacher":
                            signs = rng.choice(np.array([-1, 1], dtype=np.int16), size=values.shape)
                        else:
                            low = rng.choice(np.array([-1, 1], dtype=np.int16), size=(8, 8))
                            mask = Image.fromarray(np.where(low > 0, 255, 0).astype(np.uint8)).resize(
                                image.size, Image.Resampling.NEAREST
                            )
                            signs = np.where(np.asarray(mask, dtype=np.uint8) > 0, 1, -1).astype(np.int16)[..., None]
                        perturbed = np.clip(values + signs * epsilon, 0, 255).astype(np.uint8)
                        distance = int(np.count_nonzero(reference != compute_nth_bits(Image.fromarray(perturbed))))
                        best_distance = max(best_distance, distance)
                    rows.append({
                        "seed_id": int(item.seed_id),
                        "attack": attack,
                        "linf_epsilon_8bit": epsilon,
                        "random_trials": args.trials,
                        "maximum_nth_distance": best_distance,
                        "evades_radius": best_distance > args.radius,
                    })
    output = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)
    summary = output.groupby(["attack", "linf_epsilon_8bit"], as_index=False).agg(
        images=("seed_id", "size"), evasion_rate=("evades_radius", "mean"), mean_max_distance=("maximum_nth_distance", "mean")
    )
    summary.to_csv(args.output.with_name("nth_bounded_noise_evasion_summary.csv"), index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
