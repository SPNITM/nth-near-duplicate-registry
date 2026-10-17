import sys
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nth_registry.nth import NTHConfig, bits_to_hex, compute_nth_bits, hamming_distance


class NTHTests(unittest.TestCase):
    def test_descriptor_has_expected_length(self):
        image = Image.fromarray(np.arange(64, dtype=np.uint8).reshape(8, 8))
        bits = compute_nth_bits(image)
        self.assertEqual(bits.shape, (64,))
        self.assertEqual(len(bits_to_hex(bits)), 16)

    def test_identical_images_have_zero_distance(self):
        image = Image.new("RGB", (32, 32), color=(100, 120, 140))
        left = compute_nth_bits(image)
        right = compute_nth_bits(image.copy())
        self.assertEqual(hamming_distance(left, right), 0)

    def test_center_choice_is_explicit(self):
        pixels = np.zeros((8, 8), dtype=np.uint8)
        pixels[3, 3] = 255
        image = Image.fromarray(pixels)
        excluded = compute_nth_bits(image, NTHConfig(include_center=False))
        included = compute_nth_bits(
            image,
            NTHConfig(include_center=True, version="nth-ablation-include-center"),
        )
        self.assertFalse(np.array_equal(excluded, included))


if __name__ == "__main__":
    unittest.main()

