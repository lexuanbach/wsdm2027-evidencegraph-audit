"""Evidence graph state and dependence-aware claim aggregation."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .effective_evidence import aggregate_stances, covariance_from_dependence
from .schema import SourceEvidence


@dataclass
class EvidenceGraph:
    claim: str
    sources: list[SourceEvidence] = field(default_factory=list)
    _dependence: dict[tuple[str, str], float] = field(default_factory=dict)

    def add_source(self, source: SourceEvidence) -> None:
        if any(existing.source_id == source.source_id for existing in self.sources):
            raise ValueError(f"duplicate source ID: {source.source_id}")
        self.sources.append(source)

    def set_dependence(self, left_id: str, right_id: str, probability: float) -> None:
        if left_id == right_id:
            raise ValueError("self-dependence is fixed at one")
        known = {source.source_id for source in self.sources}
        if left_id not in known or right_id not in known:
            raise KeyError("both sources must be present in the graph")
        if not 0.0 <= probability <= 1.0:
            raise ValueError("dependence probability must be in [0, 1]")
        key = tuple(sorted((left_id, right_id)))
        self._dependence[key] = probability

    def dependence_matrix(self) -> np.ndarray:
        n = len(self.sources)
        if n == 0:
            raise ValueError("graph has no sources")
        matrix = np.eye(n, dtype=np.float64)
        for i, left in enumerate(self.sources):
            for j in range(i + 1, n):
                right = self.sources[j]
                value = self._dependence.get(tuple(sorted((left.source_id, right.source_id))), 0.0)
                matrix[i, j] = matrix[j, i] = value
        return matrix

    def aggregate(self, shrinkage: float = 0.05) -> tuple[float, float]:
        if not self.sources:
            raise ValueError("graph has no sources")
        variances = np.asarray(
            [max(1e-4, 1.0 - source.stance_confidence + 1e-3) for source in self.sources]
        )
        covariance = covariance_from_dependence(
            self.dependence_matrix(), variances=variances, shrinkage=shrinkage
        )
        stances = np.asarray([source.stance for source in self.sources])
        return aggregate_stances(stances, covariance)

    def representative_source_ids(self, threshold: float = 0.5) -> list[str]:
        """Return the strongest source from each thresholded dependence component."""
        if not 0.0 < threshold < 1.0:
            raise ValueError("dependence cluster threshold must be in (0, 1)")

        by_id = {source.source_id: source for source in self.sources}
        remaining = set(by_id)
        representatives: list[SourceEvidence] = []
        while remaining:
            frontier = [min(remaining)]
            component: set[str] = set()
            while frontier:
                current = frontier.pop()
                if current in component:
                    continue
                component.add(current)
                remaining.discard(current)
                frontier.extend(
                    candidate
                    for candidate in list(remaining)
                    if self._dependence.get(tuple(sorted((current, candidate))), 0.0)
                    >= threshold
                )
            representatives.append(
                max(
                    (by_id[source_id] for source_id in component),
                    key=lambda source: (source.stance_confidence, source.source_id),
                )
            )
        return [
            source.source_id
            for source in sorted(
                representatives,
                key=lambda source: (-source.stance_confidence, source.source_id),
            )
        ]
