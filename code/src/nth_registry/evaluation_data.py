"""Deterministic pair construction shared by hash and embedding evaluations."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def build_evaluation_pairs(
    frame: pd.DataFrame,
    splits: tuple[str, ...] = ("calibration", "test"),
    negatives_per_positive: int = 5,
    random_seed: int = 20260830,
) -> pd.DataFrame:
    rng = np.random.default_rng(random_seed)
    positives = frame[
        frame["Pair-Type"].eq("positive")
        & frame["usable"].astype(bool)
        & frame["original_split"].isin(splits)
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
                    "seed_id": int(negative_seed),
                    "transformed_seed_id": int(item["seed_id"]),
                    "original_split": split_name,
                    "transformed_split": split_name,
                    "usable": True,
                    "strict_split_usable": True,
                })
        rows.append(pd.DataFrame(negative_rows))
    return pd.concat(rows, ignore_index=True)

