"""Batched support/refute/neutral edge scorer."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


DEBERTA_LARGE_MODEL = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"
DEBERTA_LARGE_REVISION = "b3546ea6b0346eb6f8d5d68b13c7dc6d0376b3d7"
MINILM_NLI_MODEL = "cross-encoder/nli-MiniLM2-L6-H768"
MINILM_NLI_REVISION = "b95119ce93d3e065de6214e38cd4a97b0f2f2c6d"


@dataclass(frozen=True)
class NLIScores:
    support: float
    neutral: float
    refute: float


class NLIVerifier:
    def __init__(
        self,
        model_name: str = DEBERTA_LARGE_MODEL,
        revision: str | None = None,
        device: str = "cpu",
    ):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, revision=revision, local_files_only=True
        )
        # Score in float32 regardless of the checkpoint's stored dtype: half-precision
        # softmax on MPS drifts from sum-to-one by up to ~4e-4, which the downstream
        # fail-closed probability checks (atol 1e-5) correctly reject.
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            revision=revision,
            local_files_only=True,
            dtype=torch.float32,
        ).to(device)
        self.model.eval()
        self.device = device
        labels = {str(label).casefold(): int(index) for index, label in self.model.config.id2label.items()}
        required = {"entailment", "neutral", "contradiction"}
        if not required.issubset(labels):
            raise ValueError(f"NLI labels do not contain {required}")
        self.labels = labels

    def score(
        self, evidence: list[str], claims: list[str], *, batch_size: int = 32
    ) -> list[NLIScores]:
        if len(evidence) != len(claims) or not evidence:
            raise ValueError("evidence and claims must be aligned non-empty lists")
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        chunks = []
        for offset in range(0, len(evidence), batch_size):
            encoded = self.tokenizer(
                evidence[offset : offset + batch_size],
                claims[offset : offset + batch_size],
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )
            encoded = {key: value.to(self.device) for key, value in encoded.items()}
            with self.torch.inference_mode():
                probabilities = self.torch.softmax(self.model(**encoded).logits, dim=-1)
            chunks.append(np.asarray(probabilities.detach().cpu(), dtype=float))
        values = np.vstack(chunks)
        return [
            NLIScores(
                support=float(row[self.labels["entailment"]]),
                neutral=float(row[self.labels["neutral"]]),
                refute=float(row[self.labels["contradiction"]]),
            )
            for row in values
        ]
