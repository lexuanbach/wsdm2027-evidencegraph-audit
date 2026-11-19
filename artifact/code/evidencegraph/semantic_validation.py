"""Faithfulness gate for locally generated provenance paraphrases."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import numpy as np


PARAPHRASE_NLI_MODEL = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"
PARAPHRASE_NLI_REVISION = "6f5cf0a2b59cabb106aca4c287eed12e357e90eb"
PARAPHRASE_NLI_PROTOCOL = "bidirectional-segment-min-entailment-v1"


_NUMBER = re.compile(
    r"(?<!\w)[+-]?(?:(?:\d{1,3}(?:,\d{3})+)|\d+)(?:\.\d+)?%?(?!\w)"
)
_CAPITALIZED = re.compile(r"\b(?:[A-Z][\w'-]*(?:\s+[A-Z][\w'-]*)*)\b")
_INITIAL_STOP = {"A", "An", "The", "This", "These", "It", "In", "On", "At", "By", "From"}


def protected_spans(text: str) -> set[str]:
    spans = set(_NUMBER.findall(text))
    for match in _CAPITALIZED.finditer(text):
        parts = match.group(0).split()
        while parts and parts[0] in _INITIAL_STOP:
            parts.pop(0)
        if parts:
            spans.add(" ".join(parts))
    return spans


def lexical_faithfulness(original: str, candidate: str) -> tuple[bool, list[str]]:
    if not candidate.strip() or candidate.strip() == original.strip():
        return False, ["empty_or_identical"]
    lowered = candidate.casefold()
    missing = sorted(span for span in protected_spans(original) if span.casefold() not in lowered)
    return not missing, [f"missing:{span}" for span in missing]


@dataclass(frozen=True)
class ValidationResult:
    accepted: bool
    forward_entailment: float
    reverse_entailment: float
    reasons: tuple[str, ...]


class MutualEntailmentValidator:
    """Bidirectional sentence-level entailment with a conservative lexical gate."""

    def __init__(
        self,
        model_name: str = PARAPHRASE_NLI_MODEL,
        revision: str = PARAPHRASE_NLI_REVISION,
        threshold: float = 0.80,
        device: str = "cpu",
    ):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.torch = torch
        self.threshold = threshold
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, revision=revision, local_files_only=True
        )
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, revision=revision, local_files_only=True
        ).to(device)
        self.model.eval()
        self.device = device
        label_map = {str(label).casefold(): int(index) for index, label in self.model.config.id2label.items()}
        if "entailment" not in label_map:
            raise ValueError("NLI model has no entailment label")
        self.entailment_index = label_map["entailment"]

    def _entailment(
        self,
        premises: list[str],
        hypotheses: list[str],
        *,
        batch_size: int = 32,
    ) -> np.ndarray:
        if len(premises) != len(hypotheses):
            raise ValueError("premises and hypotheses must have equal length")
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        chunks = []
        for offset in range(0, len(premises), batch_size):
            encoded: dict[str, Any] = self.tokenizer(
                premises[offset : offset + batch_size],
                hypotheses[offset : offset + batch_size],
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )
            encoded = {key: value.to(self.device) for key, value in encoded.items()}
            with self.torch.inference_mode():
                logits = self.model(**encoded).logits
                probabilities = self.torch.softmax(logits, dim=-1)[:, self.entailment_index]
            chunks.append(probabilities.detach().cpu().numpy())
        return np.concatenate(chunks) if chunks else np.empty(0, dtype=float)

    def validate_many(
        self,
        original: str,
        candidates: list[str],
        *,
        batch_size: int = 32,
    ) -> list[ValidationResult]:
        if not candidates:
            return []
        forward = self._entailment(
            [original] * len(candidates), candidates, batch_size=batch_size
        )
        reverse = self._entailment(
            candidates, [original] * len(candidates), batch_size=batch_size
        )
        results = []
        for candidate, forward_score, reverse_score in zip(candidates, forward, reverse):
            lexical_ok, reasons = lexical_faithfulness(original, candidate)
            if float(forward_score) < self.threshold:
                reasons.append("forward_entailment_below_threshold")
            if float(reverse_score) < self.threshold:
                reasons.append("reverse_entailment_below_threshold")
            results.append(
                ValidationResult(
                    accepted=lexical_ok
                    and float(forward_score) >= self.threshold
                    and float(reverse_score) >= self.threshold,
                    forward_entailment=float(forward_score),
                    reverse_entailment=float(reverse_score),
                    reasons=tuple(reasons),
                )
            )
        return results

    def validate_records(
        self,
        records: list[tuple[str, list[str]]],
        *,
        batch_size: int = 32,
    ) -> list[list[ValidationResult]]:
        """Validate aligned evidence segments in shared NLI batches."""
        premises: list[str] = []
        hypotheses: list[str] = []
        record_specs: list[list[tuple[str, list[str], list[str], int | None]]] = []
        for original, record_candidates in records:
            original_segments = [segment.strip() for segment in original.splitlines() if segment.strip()]
            candidate_specs = []
            for candidate in record_candidates:
                candidate_segments = [
                    segment.strip() for segment in candidate.splitlines() if segment.strip()
                ]
                if len(candidate_segments) != len(original_segments):
                    candidate_specs.append(
                        (candidate, original_segments, candidate_segments, None)
                    )
                    continue
                start = len(premises)
                premises.extend(original_segments)
                hypotheses.extend(candidate_segments)
                candidate_specs.append(
                    (candidate, original_segments, candidate_segments, start)
                )
            record_specs.append(candidate_specs)
        forward = self._entailment(premises, hypotheses, batch_size=batch_size)
        reverse = self._entailment(hypotheses, premises, batch_size=batch_size)
        grouped: list[list[ValidationResult]] = []
        for candidate_specs in record_specs:
            record_results = []
            for candidate, original_segments, candidate_segments, start in candidate_specs:
                if start is None:
                    record_results.append(
                        ValidationResult(
                            accepted=False,
                            forward_entailment=0.0,
                            reverse_entailment=0.0,
                            reasons=("segment_count_mismatch",),
                        )
                    )
                    continue
                stop = start + len(original_segments)
                forward_score = float(np.min(forward[start:stop]))
                reverse_score = float(np.min(reverse[start:stop]))
                reasons: list[str] = []
                lexical_ok = True
                for segment_index, (original_segment, candidate_segment) in enumerate(
                    zip(original_segments, candidate_segments)
                ):
                    segment_ok, segment_reasons = lexical_faithfulness(
                        original_segment, candidate_segment
                    )
                    lexical_ok = lexical_ok and segment_ok
                    reasons.extend(
                        f"segment_{segment_index}:{reason}" for reason in segment_reasons
                    )
                if forward_score < self.threshold:
                    reasons.append("forward_entailment_below_threshold")
                if reverse_score < self.threshold:
                    reasons.append("reverse_entailment_below_threshold")
                record_results.append(
                    ValidationResult(
                        accepted=lexical_ok
                        and forward_score >= self.threshold
                        and reverse_score >= self.threshold,
                        forward_entailment=forward_score,
                        reverse_entailment=reverse_score,
                        reasons=tuple(reasons),
                    )
                )
            grouped.append(record_results)
        return grouped
