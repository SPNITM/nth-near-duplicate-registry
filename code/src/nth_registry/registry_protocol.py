"""Cost model for a registry probe, separated from pair classification."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ProbeCost:
    nth_ms: float
    lookup_ms: float
    probe_embedding_ms: float
    distance_us_per_candidate: float
    shortlist_size: int
    shortlist_nonempty: bool = True  # retained for API compatibility; not used

    @property
    def total_ms(self) -> float:
        comparisons = self.distance_us_per_candidate * self.shortlist_size / 1000.0
        # The selected protocol computes one probe embedding per request even
        # when the stage-one shortlist is empty.
        return self.nth_ms + self.lookup_ms + self.probe_embedding_ms + comparisons
