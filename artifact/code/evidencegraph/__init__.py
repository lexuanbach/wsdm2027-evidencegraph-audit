"""EvidenceGraph research artifact.

The package keeps construction, scoring, aggregation, and evaluation
separate so that every reported value can be traced to raw run records.
"""

from .effective_evidence import effective_evidence, equicorrelation_covariance
from .schema import ClaimEpisode, ProvenanceFamily, SourceEvidence

__all__ = [
    "ClaimEpisode",
    "ProvenanceFamily",
    "SourceEvidence",
    "effective_evidence",
    "equicorrelation_covariance",
]
