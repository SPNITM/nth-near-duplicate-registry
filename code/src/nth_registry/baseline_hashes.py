"""Transparent 64-bit classical perceptual-hash baselines."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image
from scipy.fft import dctn


def _open_gray(image_or_path: Image.Image | str | Path) -> Image.Image:
    if isinstance(image_or_path, (str, Path)):
        with Image.open(image_or_path) as opened:
            return opened.convert("L").copy()
    return image_or_path.convert("L")


def average_hash(image_or_path: Image.Image | str | Path) -> np.ndarray:
    image = _open_gray(image_or_path).resize((8, 8), Image.Resampling.LANCZOS)
    values = np.asarray(image, dtype=np.float32)
    return (values >= values.mean()).astype(np.uint8).reshape(-1)


def difference_hash(image_or_path: Image.Image | str | Path) -> np.ndarray:
    image = _open_gray(image_or_path).resize((9, 8), Image.Resampling.LANCZOS)
    values = np.asarray(image, dtype=np.float32)
    return (values[:, 1:] >= values[:, :-1]).astype(np.uint8).reshape(-1)


def perceptual_hash(image_or_path: Image.Image | str | Path) -> np.ndarray:
    image = _open_gray(image_or_path).resize((32, 32), Image.Resampling.LANCZOS)
    values = np.asarray(image, dtype=np.float32)
    low = dctn(values, type=2, norm="ortho")[:8, :8]
    threshold = np.median(low.reshape(-1)[1:])
    return (low >= threshold).astype(np.uint8).reshape(-1)


def wavelet_hash(image_or_path: Image.Image | str | Path) -> np.ndarray:
    """A documented one-level Haar low-frequency 64-bit baseline."""

    image = _open_gray(image_or_path).resize((16, 16), Image.Resampling.LANCZOS)
    values = np.asarray(image, dtype=np.float32)
    low = (
        values[0::2, 0::2]
        + values[1::2, 0::2]
        + values[0::2, 1::2]
        + values[1::2, 1::2]
    ) / 4.0
    return (low >= np.median(low)).astype(np.uint8).reshape(-1)


BASELINES = {
    "aHash": average_hash,
    "dHash": difference_hash,
    "pHash": perceptual_hash,
    "wHash-Haar": wavelet_hash,
}

