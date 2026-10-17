#!/usr/bin/env python
"""Extract frozen deep features under the shared split protocol.

This runner intentionally has no silent fallback. It records a missing
dependency or unavailable GPU as a failed run instead of emitting placeholder
numbers. Supported model names are MobileNetV3, ResNet-50, ShuffleNet V2,
EfficientNet-B0, CLIP ViT-B/32, and DINOv2 ViT-S/14.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[2]


def required_modules(model: str) -> tuple[str, ...]:
    """Return only the imports required by the selected model path."""

    common = ("torch", "torchvision")
    if model == "clip_vit_b32":
        return common + ("open_clip",)
    return common


def dependency_error(model: str) -> None:
    missing = []
    for module in required_modules(model):
        try:
            __import__(module)
        except ImportError:
            missing.append(module)
    if missing:
        raise SystemExit(
            "Deep baseline run not started. Missing packages: "
            + ", ".join(missing)
            + ". Install requirements-gpu.txt in an isolated environment."
        )


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=PROJECT / "data" / "manifests" / "pair_manifest_main.csv")
    parser.add_argument("--model", choices=["mobilenetv3_imagenet", "resnet50_imagenet", "shufflenet_v2_x1", "efficientnet_b0", "clip_vit_b32", "dinov2_vits14"], required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--include-train", action="store_true", help="Extract train identities as well as calibration/test")
    parser.add_argument("--output", type=Path, default=PROJECT / "results" / "raw_embeddings")
    args = parser.parse_args()
    dependency_error(args.model)

    import torch
    from PIL import Image
    from torch.utils.data import DataLoader, Dataset
    from torchvision import models, transforms

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but no CUDA device is available. No results were written.")

    class Paths(Dataset):
        def __init__(self, values, transform):
            self.values = values
            self.transform = transform
        def __len__(self):
            return len(self.values)
        def __getitem__(self, index):
            with Image.open(self.values[index]) as image:
                return self.transform(image.convert("RGB")), index

    weights_id = ""
    if args.model == "mobilenetv3_imagenet":
        weights = models.MobileNet_V3_Large_Weights.IMAGENET1K_V2
        network = models.mobilenet_v3_large(weights=weights)
        network.classifier = torch.nn.Identity()
        preprocess = weights.transforms()
        weights_id = str(weights)
    elif args.model == "resnet50_imagenet":
        weights = models.ResNet50_Weights.IMAGENET1K_V2
        network = models.resnet50(weights=weights)
        network.fc = torch.nn.Identity()
        preprocess = weights.transforms()
        weights_id = str(weights)
    elif args.model == "shufflenet_v2_x1":
        weights = models.ShuffleNet_V2_X1_0_Weights.IMAGENET1K_V1
        network = models.shufflenet_v2_x1_0(weights=weights)
        network.fc = torch.nn.Identity()
        preprocess = weights.transforms()
        weights_id = str(weights)
    elif args.model == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1
        network = models.efficientnet_b0(weights=weights)
        network.classifier = torch.nn.Identity()
        preprocess = weights.transforms()
        weights_id = str(weights)
    elif args.model == "clip_vit_b32":
        import open_clip
        network, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="laion2b_s34b_b79k")
        network.forward = network.encode_image
        weights_id = "open_clip ViT-B-32 laion2b_s34b_b79k"
    else:
        network = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
        preprocess = transforms.Compose([
            transforms.Resize(256, antialias=True),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        weights_id = "facebookresearch/dinov2 dinov2_vits14"

    frame = pd.read_csv(args.manifest)
    selected_splits = ["train", "calibration", "test"] if args.include_train else ["calibration", "test"]
    selected = frame[frame["original_split"].isin(selected_splits)]
    positives = selected[selected["Pair-Type"].eq("positive") & selected["usable"].astype(bool)]
    paths = sorted(set(selected["original_path"].dropna()) | set(positives["transformed_path"].dropna()))
    loader = DataLoader(Paths(paths, preprocess), batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=False)
    network = network.eval().to(args.device)
    output = []
    with torch.inference_mode():
        for batch, _ in loader:
            values = network(batch.to(args.device))
            if isinstance(values, dict):
                values = values.get("x_norm_clstoken", next(iter(values.values())))
            values = torch.nn.functional.normalize(values.float(), dim=1)
            output.append(values.cpu().numpy())
    embeddings = np.concatenate(output)
    args.output.mkdir(parents=True, exist_ok=True)
    np.save(args.output / f"{args.model}.npy", embeddings)
    pd.DataFrame({"path": paths}).to_csv(args.output / f"{args.model}_paths.csv", index=False)
    metadata = {
        "model": args.model,
        "weights": weights_id,
        "device": args.device,
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "image_count": len(paths),
        "manifest_sha256": file_sha256(args.manifest),
    }
    (args.output / f"{args.model}_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
