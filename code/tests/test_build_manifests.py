from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

import pandas as pd


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "build_manifests.py"
SPEC = importlib.util.spec_from_file_location("build_manifests", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)

DEEP_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_deep_baselines.py"
DEEP_SPEC = importlib.util.spec_from_file_location("run_deep_baselines", DEEP_SCRIPT)
assert DEEP_SPEC and DEEP_SPEC.loader
DEEP_MODULE = importlib.util.module_from_spec(DEEP_SPEC)
DEEP_SPEC.loader.exec_module(DEEP_MODULE)


class BuildManifestTests(unittest.TestCase):
    def test_frozen_torchvision_models_do_not_require_optional_model_stacks(self) -> None:
        for model in (
            "mobilenetv3_imagenet",
            "resnet50_imagenet",
            "shufflenet_v2_x1",
            "efficientnet_b0",
            "dinov2_vits14",
        ):
            modules = DEEP_MODULE.required_modules(model)
            self.assertEqual(modules, ("torch", "torchvision"))
        self.assertEqual(
            DEEP_MODULE.required_modules("clip_vit_b32"),
            ("torch", "torchvision", "open_clip"),
        )

    def test_audit_resolves_complete_pair(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            images = root / "images"
            output = root / "output"
            images.mkdir()
            output.mkdir()
            (images / "img_0001_orig.jpg").write_bytes(b"original")
            (images / "img_0001_blur.jpg").write_bytes(b"transformed")
            pd.DataFrame(
                [
                    {
                        "Original Image": "img_0001_orig.jpg",
                        "Transformed Image": "img_0001_blur.jpg",
                        "Pair-Type": "positive",
                        "Label": "blur",
                    }
                ]
            ).to_csv(root / "pairs.csv", index=False)
            summary = MODULE.audit_dataset(
                "test",
                {
                    "csv": Path("pairs.csv"),
                    "original_dirs": [Path("images")],
                    "transformed_dirs": [Path("images")],
                },
                root,
                output,
            )
            self.assertEqual(summary["status"], "audited")
            self.assertEqual(summary["missing_original_references"], 0)
            self.assertEqual(summary["missing_transformed_references"], 0)
            self.assertEqual(summary["usable_positive_rows"], 1)

    def test_missing_pair_csv_is_reported_without_crashing(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            output = root / "output"
            output.mkdir()
            summary = MODULE.audit_dataset(
                "optional",
                {
                    "csv": Path("missing.csv"),
                    "original_dirs": [],
                    "transformed_dirs": [],
                },
                root,
                output,
            )
            self.assertEqual(summary["status"], "not_available")
            self.assertIn("pair CSV", summary["reason"])


if __name__ == "__main__":
    unittest.main()
