"""Deterministic seed-disjoint split utilities."""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
import pandas as pd


SEED_PATTERN = re.compile(r"(?:Image|img_?)(\d+)", re.IGNORECASE)


def seed_id_from_filename(filename: str) -> int:
    match = SEED_PATTERN.search(str(filename))
    if not match:
        raise ValueError(f"Cannot derive seed id from {filename!r}")
    return int(match.group(1))


@dataclass(frozen=True)
class SplitCounts:
    train: int = 745
    calibration: int = 160
    test: int = 159


def make_seed_split(
    seed_ids: list[int] | np.ndarray,
    random_seed: int = 20260830,
    counts: SplitCounts = SplitCounts(),
) -> pd.DataFrame:
    unique_ids = np.array(sorted(set(map(int, seed_ids))), dtype=np.int64)
    expected = counts.train + counts.calibration + counts.test
    if unique_ids.size != expected:
        raise ValueError(f"Expected {expected} unique identities, found {unique_ids.size}")
    shuffled = np.random.default_rng(random_seed).permutation(unique_ids)
    labels = np.empty(expected, dtype=object)
    labels[: counts.train] = "train"
    labels[counts.train : counts.train + counts.calibration] = "calibration"
    labels[counts.train + counts.calibration :] = "test"
    frame = pd.DataFrame({"seed_id": shuffled, "split": labels})
    return frame.sort_values("seed_id", kind="stable").reset_index(drop=True)

