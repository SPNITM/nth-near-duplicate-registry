#!/usr/bin/env python
"""Measure embedding latency/throughput with explicit batch and hardware metadata."""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import time
from pathlib import Path

import pandas as pd

PROJECT = Path(__file__).resolve().parents[2]


def model_specs(models):
    """Return reproducible torchvision model/weight/feature-head definitions."""

    return {
        "mobilenetv3_large": (
            models.mobilenet_v3_large,
            models.MobileNet_V3_Large_Weights.IMAGENET1K_V2,
            lambda network, torch: setattr(network, "classifier", torch.nn.Identity()),
            "MobileNetV3-Large",
        ),
        "shufflenetv2_x1_0": (
            models.shufflenet_v2_x1_0,
            models.ShuffleNet_V2_X1_0_Weights.IMAGENET1K_V1,
            lambda network, torch: setattr(network, "fc", torch.nn.Identity()),
            "ShuffleNet V2 1.0x",
        ),
        "efficientnet_b0": (
            models.efficientnet_b0,
            models.EfficientNet_B0_Weights.IMAGENET1K_V1,
            lambda network, torch: setattr(network, "classifier", torch.nn.Identity()),
            "EfficientNet-B0",
        ),
    }


def sample_paths(manifest: Path, image_dir: Path | None, limit: int = 256) -> list[str]:
    if image_dir is not None:
        suffixes = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
        return [
            str(path)
            for path in image_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in suffixes
        ][:limit]
    frame = pd.read_csv(manifest)
    return (
        frame[frame["transformed_exists"].astype(bool)]["transformed_path"]
        .dropna()
        .drop_duplicates()
        .head(limit)
        .astype(str)
        .tolist()
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=PROJECT / "data" / "manifests" / "pair_manifest_main.csv")
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["mobilenetv3_large", "shufflenetv2_x1_0", "efficientnet_b0"],
        choices=["mobilenetv3_large", "shufflenetv2_x1_0", "efficientnet_b0"],
    )
    parser.add_argument("--image-dir", type=Path, default=None)
    parser.add_argument("--platform-label", default="unspecified")
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 8, 32, 64])
    parser.add_argument("--repetitions", type=int, default=10)
    parser.add_argument("--output", type=Path, default=PROJECT / "results" / "embedding_throughput.csv")
    args = parser.parse_args()

    import torch
    from PIL import Image
    from torchvision import models

    torch.set_num_threads(max(1, min(20, os.cpu_count() or 1)))
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but torch.cuda.is_available() is false")
    paths = sample_paths(args.manifest, args.image_dir)
    if len(paths) < max(args.batch_sizes):
        raise SystemExit(
            f"Only {len(paths)} readable candidates were found; "
            f"at least {max(args.batch_sizes)} are required"
        )
    specs = model_specs(models)
    rows = []
    weight_names = {}
    for model_key in args.models:
        constructor, weights, strip_head, display_name = specs[model_key]
        network = constructor(weights=weights)
        strip_head(network, torch)
        network = network.eval().to(args.device)
        transform = weights.transforms()
        tensors = []
        for path in paths:
            with Image.open(path) as image:
                tensors.append(transform(image.convert("RGB")))
        pool = torch.stack(tensors)
        weight_names[model_key] = str(weights)
        for batch_size in args.batch_sizes:
            batch = pool[:batch_size].to(args.device)
            with torch.inference_mode():
                for _ in range(3):
                    network(batch)
                if args.device.startswith("cuda"):
                    torch.cuda.synchronize()
                timings = []
                for _ in range(args.repetitions):
                    start = time.perf_counter_ns()
                    network(batch)
                    if args.device.startswith("cuda"):
                        torch.cuda.synchronize()
                    timings.append((time.perf_counter_ns() - start) / 1e9)
            median_seconds = statistics.median(timings)
            rows.append({
                "model_key": model_key,
                "model": display_name,
                "weights": str(weights),
                "platform_label": args.platform_label,
                "device": args.device,
                "batch_size": batch_size,
                "repetitions": args.repetitions,
                "median_batch_seconds": median_seconds,
                "p95_batch_seconds": sorted(timings)[max(0, int(0.95 * len(timings)) - 1)],
                "median_ms_per_image": median_seconds * 1000 / batch_size,
                "throughput_images_per_second": batch_size / median_seconds,
                "torch_version": torch.__version__,
                "torchvision_version": __import__("torchvision").__version__,
                "python_version": platform.python_version(),
                "platform": platform.platform(),
                "processor": platform.processor(),
                "logical_cpu_count": os.cpu_count(),
                "cuda_device": torch.cuda.get_device_name(0) if args.device.startswith("cuda") else "",
            })
        del network, pool
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output, index=False, float_format="%.17g")
    source = args.image_dir or args.manifest
    try:
        source_label = source.resolve().relative_to(PROJECT.resolve()).as_posix()
    except ValueError:
        source_label = str(source)
    (args.output.with_suffix(".metadata.json")).write_text(
        json.dumps(
            {
                "weights": weight_names,
                "device": args.device,
                "platform_label": args.platform_label,
                "sample_count": len(paths),
                "source": source_label,
                "warmup_batches": 3,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
