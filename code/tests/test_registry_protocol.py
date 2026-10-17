import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nth_registry.registry_protocol import ProbeCost


class RegistryProtocolTests(unittest.TestCase):
    def test_probe_embedding_is_charged_for_empty_shortlist(self):
        cost = ProbeCost(
            nth_ms=0.1,
            lookup_ms=0.2,
            probe_embedding_ms=2.0,
            distance_us_per_candidate=5.0,
            shortlist_size=0,
            shortlist_nonempty=False,
        )
        self.assertAlmostEqual(cost.total_ms, 2.3)


if __name__ == "__main__":
    unittest.main()
