"""Versioned Neighborhood Totality Hash implementation.

The historical project contained implementations that disagreed on whether the
centre pixel belongs to the 3 x 3 sum. This module makes that choice explicit.
The revised default follows the submitted algorithm: eight neighbours, centre
excluded. All alternatives remain available for ablation studies.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
from PIL import Image
from scipy.ndimage import uniform_filter


Padding = Literal["wrap", "reflect", "edge", "constant"]
Threshold = Literal["global_mean", "global_median", "adaptive_mean"]
Interpolation = Literal["nearest", "bilinear", "bicubic", "lanczos"]


_RESAMPLING = {
    "nearest": Image.Resampling.NEAREST,
    "bilinear": Image.Resampling.BILINEAR,
    "bicubic": Image.Resampling.BICUBIC,
    "lanczos": Image.Resampling.LANCZOS,
}


@dataclass(frozen=True)
class NTHConfig:
    grid_size: int = 8
    window_size: int = 3
    padding: Padding = "wrap"
    include_center: bool = False
    threshold: Threshold = "global_mean"
    interpolation: Interpolation = "lanczos"
    version: str = "nth-v2-exclude-center"

    def validate(self) -> None:
        if self.grid_size < 4:
            raise ValueError("grid_size must be at least 4")
        if self.window_size < 3 or self.window_size % 2 == 0:
            raise ValueError("window_size must be odd and at least 3")
        if self.window_size > self.grid_size:
            raise ValueError("window_size cannot exceed grid_size")


def _to_grid(image: Image.Image, config: NTHConfig) -> np.ndarray:
    config.validate()
    gray = image.convert("L").resize(
        (config.grid_size, config.grid_size),
        resample=_RESAMPLING[config.interpolation],
    )
    return np.asarray(gray, dtype=np.float32)


def _neighborhood_sums(grid: np.ndarray, config: NTHConfig) -> np.ndarray:
    radius = config.window_size // 2
    mode = {"edge": "edge", "constant": "constant"}.get(
        config.padding, config.padding
    )
    padded = np.pad(grid, radius, mode=mode)
    out = np.zeros_like(grid, dtype=np.float32)
    for row_shift in range(config.window_size):
        for col_shift in range(config.window_size):
            out += padded[
                row_shift : row_shift + grid.shape[0],
                col_shift : col_shift + grid.shape[1],
            ]
    if not config.include_center:
        out -= grid
    return out


def compute_nth_bits(
    image_or_path: Image.Image | str | Path,
    config: NTHConfig = NTHConfig(),
) -> np.ndarray:
    """Return a row-major binary NTH descriptor."""

    if isinstance(image_or_path, (str, Path)):
        with Image.open(image_or_path) as opened:
            image = opened.copy()
    else:
        image = image_or_path
    grid = _to_grid(image, config)
    sums = _neighborhood_sums(grid, config)
    if config.threshold == "global_mean":
        threshold = np.full_like(sums, sums.mean())
    elif config.threshold == "global_median":
        threshold = np.full_like(sums, np.median(sums))
    elif config.threshold == "adaptive_mean":
        # Local threshold on the neighborhood-sum map. Reflect mode avoids
        # importing the tested padding choice into the threshold estimator.
        threshold = uniform_filter(sums, size=3, mode="reflect")
    else:  # pragma: no cover - protected by the Literal type
        raise ValueError(f"Unknown threshold: {config.threshold}")
    return (sums >= threshold).astype(np.uint8).reshape(-1)


def bits_to_hex(bits: np.ndarray) -> str:
    bits = np.asarray(bits, dtype=np.uint8).reshape(-1)
    if bits.size % 8:
        bits = np.pad(bits, (0, 8 - bits.size % 8))
    return np.packbits(bits, bitorder="big").tobytes().hex()


def hex_to_bits(value: str, bit_count: int = 64) -> np.ndarray:
    raw = bytes.fromhex(value.removeprefix("0x"))
    return np.unpackbits(np.frombuffer(raw, dtype=np.uint8), bitorder="big")[:bit_count]


def hamming_distance(left: np.ndarray, right: np.ndarray) -> int:
    left = np.asarray(left, dtype=np.uint8).reshape(-1)
    right = np.asarray(right, dtype=np.uint8).reshape(-1)
    if left.shape != right.shape:
        raise ValueError(f"Hash lengths differ: {left.size} and {right.size}")
    return int(np.count_nonzero(left != right))

