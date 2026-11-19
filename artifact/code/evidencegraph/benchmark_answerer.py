"""Strict frozen-LLM outputs for benchmark verdicts and cited long-form answers."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any

from .local_llm import OllamaClient


BENCHMARK_LABELS = {
    "FEVER": ("SUPPORTS", "REFUTES", "NOT ENOUGH INFO"),
    "HoVer": ("SUPPORTED", "NOT_SUPPORTED"),
    "AVeriTeC": (
        "Supported",
        "Refuted",
        "Not Enough Evidence",
        "Conflicting Evidence/Cherrypicking",
    ),
}
NO_EVIDENCE_LABEL = {
    "FEVER": "NOT ENOUGH INFO",
    "HoVer": "NOT_SUPPORTED",
    "AVeriTeC": "Not Enough Evidence",
}
SUPPORT_LABEL = {
    "FEVER": "SUPPORTS",
    "HoVer": "SUPPORTED",
    "AVeriTeC": "Supported",
}


def answer_context_results(
    method: str, results: list[dict[str, Any]], top_k: int
) -> list[dict[str, Any]]:
    """Preserve graph acquisitions; apply fixed depth only to baseline contexts."""
    if top_k < 1:
        raise ValueError("answer context top-k must be positive")
    return list(results) if method == "evidencegraph" else list(results[:top_k])


@dataclass(frozen=True)
class BenchmarkPrediction:
    label: str
    probabilities: dict[str, float]
    answer: str
    cited_source_ids: tuple[str, ...]
    model: str
    prompt_tokens: int
    completion_tokens: int
    wall_seconds: float
    repair_attempts: int = 0

    def to_dict(self) -> dict[str, Any]:
        record = asdict(self)
        record["cited_source_ids"] = list(self.cited_source_ids)
        return record


@dataclass(frozen=True)
class CitedAnswer:
    answer: str
    claims: tuple[dict[str, Any], ...]
    model: str
    prompt_tokens: int
    completion_tokens: int
    wall_seconds: float

    def to_dict(self) -> dict[str, Any]:
        record = asdict(self)
        record["claims"] = [dict(claim) for claim in self.claims]
        return record


def verdict_schema(labels: tuple[str, ...]) -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["label", "probabilities", "answer", "cited_source_ids"],
        "properties": {
            "label": {"type": "string", "enum": list(labels)},
            "probabilities": {
                "type": "object",
                "additionalProperties": False,
                "required": list(labels),
                "properties": {
                    label: {"type": "number", "minimum": 0.0, "maximum": 1.0}
                    for label in labels
                },
            },
            "answer": {"type": "string", "minLength": 1},
            "cited_source_ids": {
                "type": "array",
                "items": {"type": "string"},
                "uniqueItems": True,
            },
        },
    }


def prediction_from_payload(
    payload: Any,
    response: Any,
    labels: tuple[str, ...],
    available_source_ids: set[str],
) -> BenchmarkPrediction:
    if not isinstance(payload, dict) or set(payload) != {
        "label",
        "probabilities",
        "answer",
        "cited_source_ids",
    }:
        raise ValueError("benchmark answer has the wrong object schema")
    probabilities = {label: float(payload["probabilities"][label]) for label in labels}
    if abs(sum(probabilities.values()) - 1.0) > 1e-5:
        raise ValueError("benchmark label probabilities do not sum to one")
    label = str(payload["label"])
    if label not in labels or label != max(probabilities, key=probabilities.get):
        raise ValueError("benchmark label must equal the probability argmax")
    citations = tuple(map(str, payload["cited_source_ids"]))
    if len(citations) != len(set(citations)) or not set(citations) <= available_source_ids:
        raise ValueError("benchmark answer cites an unavailable or duplicate source")
    answer = str(payload["answer"]).strip()
    if not answer:
        raise ValueError("benchmark answer is empty")
    return BenchmarkPrediction(
        label=label,
        probabilities=probabilities,
        answer=answer,
        cited_source_ids=citations,
        model=response.model,
        prompt_tokens=response.prompt_tokens,
        completion_tokens=response.completion_tokens,
        wall_seconds=response.wall_seconds,
    )


def cited_evidence_records(
    sources: list[dict[str, Any]], prediction: dict[str, Any]
) -> list[dict[str, str]]:
    """Materialize the exact cited passages for scorer and provenance audits."""
    source_index = {}
    for source in sources:
        source_id = str(source["source_id"])
        if source_id in source_index:
            raise ValueError("answer context contains duplicate source IDs")
        source_index[source_id] = str(source["text"])
    if "cited_source_ids" in prediction:
        cited = list(map(str, prediction["cited_source_ids"]))
    else:
        cited = [
            str(source_id)
            for claim in prediction.get("claims", [])
            for source_id in claim.get("cited_source_ids", [])
        ]
    cited = list(dict.fromkeys(cited))
    missing = [source_id for source_id in cited if source_id not in source_index]
    if missing:
        raise ValueError("prediction cites evidence outside its answer context")
    return [
        {"source_id": source_id, "text": source_index[source_id]}
        for source_id in cited
    ]


def deterministic_no_evidence_verdict(dataset: str) -> dict[str, Any]:
    """Return the dataset-defined abstention when retrieval yields no context."""
    if dataset not in BENCHMARK_LABELS:
        raise ValueError("no-evidence verdict requires a supported dataset")
    label = NO_EVIDENCE_LABEL[dataset]
    return {
        "label": label,
        "probabilities": {
            candidate: float(candidate == label) for candidate in BENCHMARK_LABELS[dataset]
        },
        "answer": "No evidence was retrieved; the claim is not established.",
        "cited_source_ids": [],
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "wall_seconds": 0.0,
    }


def apply_graph_calibrated_probability(
    prediction: BenchmarkPrediction,
    dataset: str,
    probability_true: float,
) -> BenchmarkPrediction:
    """Make verdict confidence obey the graph's frozen dev calibrator.

    The graph supplies a binary support probability. The frozen LLM chooses
    the most plausible non-support dataset label; probability mass is then
    assigned only to support versus that alternative so a below-threshold
    graph score cannot still win a multiclass argmax by fragmentation.
    """

    if dataset not in BENCHMARK_LABELS or dataset not in SUPPORT_LABEL:
        raise ValueError("graph calibration requires a verdict benchmark")
    probability = float(probability_true)
    if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
        raise ValueError("graph probability must be finite and in [0, 1]")
    labels = BENCHMARK_LABELS[dataset]
    support = SUPPORT_LABEL[dataset]
    alternatives = [label for label in labels if label != support]
    alternative = max(alternatives, key=lambda label: prediction.probabilities[label])
    probabilities = {label: 0.0 for label in labels}
    probabilities[support] = probability
    probabilities[alternative] = 1.0 - probability
    label = support if probability >= 0.5 else alternative
    answer = prediction.answer
    if label != prediction.label:
        if label == support:
            answer = "The graph-selected independent evidence supports the claim."
        elif label == NO_EVIDENCE_LABEL[dataset]:
            answer = "The graph-selected independent evidence does not establish the claim."
        else:
            answer = "The graph-selected independent evidence refutes the claim."
    return BenchmarkPrediction(
        label=label,
        probabilities=probabilities,
        answer=answer,
        cited_source_ids=prediction.cited_source_ids,
        model=prediction.model,
        prompt_tokens=prediction.prompt_tokens,
        completion_tokens=prediction.completion_tokens,
        wall_seconds=prediction.wall_seconds,
        repair_attempts=prediction.repair_attempts,
    )


class LocalBenchmarkAnswerer:
    def __init__(self, model: str, client: Any | None = None):
        self.model = model
        self.client = client or OllamaClient(timeout=300)

    def predict_verdict(
        self,
        dataset: str,
        query: str,
        sources: list[dict[str, str]],
        *,
        seed: int = 103,
        policy: str = "fixed_rag",
        additional_context: str | None = None,
    ) -> BenchmarkPrediction:
        if dataset not in BENCHMARK_LABELS or not sources:
            raise ValueError("verdict prediction requires a supported dataset and sources")
        labels = BENCHMARK_LABELS[dataset]
        source_ids = {str(source["source_id"]) for source in sources}
        source_text = "\n\n".join(
            f"[{source['source_id']}] {source['text']}" for source in sources
        )
        dependence_instruction = (
            "The list contains one representative per predicted source-origin cluster; do not "
            "increase confidence because several passages repeat the same origin. "
            if policy == "evidencegraph"
            else "Treat the list as the frozen retrieved context. "
        )
        prompt = (
            f"Dataset: {dataset}\nClaim or question: {query}\n\nSources:\n{source_text}\n\n"
            f"{dependence_instruction}Choose exactly one label from {list(labels)}. "
            "Return probabilities for every label that sum to 1, a concise evidence-grounded "
            "answer or justification, and only source IDs from the supplied list."
        )
        if additional_context:
            prompt += (
                "\n\nPrior deliberation to check against the sources:\n"
                f"{additional_context.strip()}"
            )
        messages = [
            {
                "role": "system",
                "content": "You are a calibrated fact verifier. Use only supplied evidence.",
            },
            {"role": "user", "content": prompt},
        ]
        payload, response = self.client.chat_json(
            self.model,
            messages,
            temperature=0.0,
            seed=seed,
            max_tokens=500,
            json_schema=verdict_schema(labels),
        )
        repair_attempts = 0
        try:
            prediction = prediction_from_payload(payload, response, labels, source_ids)
        except ValueError as error:
            if "probability argmax" not in str(error):
                raise
            # One explicit, logged repair: the model chose a label that was not
            # the argmax of its own probabilities. The retry keeps the evidence
            # and instructions identical and only restates the consistency
            # requirement; the attempt count is recorded in the prediction.
            repair_attempts = 1
            repaired = messages[:-1] + [
                {
                    "role": "user",
                    "content": (
                        prompt
                        + "\n\nConsistency requirement: the label must be the label "
                        "with the highest probability in your probabilities object."
                    ),
                }
            ]
            payload, response = self.client.chat_json(
                self.model,
                repaired,
                temperature=0.0,
                seed=seed,
                max_tokens=500,
                json_schema=verdict_schema(labels),
            )
            prediction = prediction_from_payload(payload, response, labels, source_ids)
        return BenchmarkPrediction(
            label=prediction.label,
            probabilities=prediction.probabilities,
            answer=prediction.answer,
            cited_source_ids=prediction.cited_source_ids,
            model=prediction.model,
            prompt_tokens=prediction.prompt_tokens,
            completion_tokens=prediction.completion_tokens,
            wall_seconds=prediction.wall_seconds,
            repair_attempts=repair_attempts,
        )

    def predict_closed_book(
        self, dataset: str, query: str, *, seed: int = 103
    ) -> BenchmarkPrediction:
        if dataset not in BENCHMARK_LABELS:
            raise ValueError("closed-book prediction requires a supported dataset")
        labels = BENCHMARK_LABELS[dataset]
        payload, response = self.client.chat_json(
            self.model,
            [
                {
                    "role": "system",
                    "content": (
                        "You are a calibrated closed-book fact verifier. No external sources "
                        "are available, so do not fabricate citations."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Dataset: {dataset}\nClaim or question: {query}\n"
                        f"Choose one label from {list(labels)}. Return probabilities for every "
                        "label that sum to 1, a concise justification, and an empty citation list."
                    ),
                },
            ],
            temperature=0.0,
            seed=seed,
            max_tokens=400,
            json_schema=verdict_schema(labels),
        )
        return prediction_from_payload(payload, response, labels, set())

    def generate_closed_book_answer(
        self, query: str, *, seed: int = 103
    ) -> CitedAnswer:
        schema = {
            "type": "object",
            "additionalProperties": False,
            "required": ["answer", "claims"],
            "properties": {
                "answer": {"type": "string", "minLength": 1},
                "claims": {
                    "type": "array",
                    "minItems": 1,
                    "items": {"type": "string", "minLength": 1},
                },
            },
        }
        payload, response = self.client.chat_json(
            self.model,
            [
                {
                    "role": "system",
                    "content": (
                        "Answer closed-book. No external sources are available; do not invent "
                        "citations. Separate the answer into atomic factual claims."
                    ),
                },
                {"role": "user", "content": f"Question: {query}"},
            ],
            temperature=0.0,
            seed=seed,
            max_tokens=1000,
            json_schema=schema,
        )
        if not isinstance(payload, dict) or set(payload) != {"answer", "claims"}:
            raise ValueError("closed-book cited answer has the wrong schema")
        answer = str(payload["answer"]).strip()
        claims = [str(claim).strip() for claim in payload["claims"]]
        if not answer or not claims or any(not claim for claim in claims):
            raise ValueError("closed-book answer or claim list is empty")
        return CitedAnswer(
            answer=answer,
            claims=tuple(
                {"claim": claim, "cited_source_ids": []} for claim in claims
            ),
            model=response.model,
            prompt_tokens=response.prompt_tokens,
            completion_tokens=response.completion_tokens,
            wall_seconds=response.wall_seconds,
        )

    def generate_cited_answer(
        self,
        query: str,
        sources: list[dict[str, str]],
        *,
        seed: int = 103,
        policy: str = "fixed_rag",
        additional_context: str | None = None,
    ) -> CitedAnswer:
        if not sources:
            raise ValueError("cited answering requires at least one source")
        source_ids = {str(source["source_id"]) for source in sources}
        source_text = "\n\n".join(
            f"[{source['source_id']}] {source['text']}" for source in sources
        )
        policy_text = (
            "Each supplied source represents a distinct predicted origin cluster. "
            if policy == "evidencegraph"
            else "Use the frozen retrieved source list. "
        )
        schema = {
            "type": "object",
            "additionalProperties": False,
            "required": ["answer", "claims"],
            "properties": {
                "answer": {"type": "string", "minLength": 1},
                "claims": {
                    "type": "array",
                    "minItems": 1,
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["claim", "cited_source_ids"],
                        "properties": {
                            "claim": {"type": "string", "minLength": 1},
                            "cited_source_ids": {
                                "type": "array",
                                "items": {"type": "string"},
                                "uniqueItems": True,
                            },
                        },
                    },
                },
            },
        }
        prompt = (
            f"Question: {query}\n\nSources:\n{source_text}\n\n{policy_text}"
            "Write a useful long-form answer, then list each atomic factual claim and "
            "the exact supplied source IDs supporting it. Omit unsupported claims."
        )
        if additional_context:
            prompt += (
                "\n\nPrior deliberation to check against the sources:\n"
                f"{additional_context.strip()}"
            )
        payload, response = self.client.chat_json(
            self.model,
            [
                {
                    "role": "system",
                    "content": "Answer expert questions only from supplied evidence with claim-level citations.",
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            temperature=0.0,
            seed=seed,
            max_tokens=1200,
            json_schema=schema,
        )
        if not isinstance(payload, dict) or set(payload) != {"answer", "claims"}:
            raise ValueError("cited answer has the wrong object schema")
        answer = str(payload["answer"]).strip()
        claims = []
        for raw_claim in payload["claims"]:
            if not isinstance(raw_claim, dict) or set(raw_claim) != {
                "claim",
                "cited_source_ids",
            }:
                raise ValueError("cited answer claim has the wrong schema")
            claim = str(raw_claim["claim"]).strip()
            citations = tuple(map(str, raw_claim["cited_source_ids"]))
            if not claim or len(citations) != len(set(citations)) or not set(citations) <= source_ids:
                raise ValueError("cited answer claim is empty or cites an unavailable source")
            claims.append({"claim": claim, "cited_source_ids": list(citations)})
        if not answer or not claims:
            raise ValueError("cited answer or claim list is empty")
        return CitedAnswer(
            answer=answer,
            claims=tuple(claims),
            model=response.model,
            prompt_tokens=response.prompt_tokens,
            completion_tokens=response.completion_tokens,
            wall_seconds=response.wall_seconds,
        )

    def draft_atomic_claims(
        self,
        query: str,
        sources: list[dict[str, str]],
        *,
        seed: int = 103,
    ) -> CitedAnswer:
        """Draft an answer and expose its atomic claims for graph verification."""
        if not sources:
            raise ValueError("claim drafting requires at least one source")
        source_text = "\n\n".join(
            f"[{source['source_id']}] {source['text']}" for source in sources
        )
        schema = {
            "type": "object",
            "additionalProperties": False,
            "required": ["answer", "claims"],
            "properties": {
                "answer": {"type": "string", "minLength": 1},
                "claims": {
                    "type": "array",
                    "minItems": 1,
                    "items": {"type": "string", "minLength": 1},
                },
            },
        }
        payload, response = self.client.chat_json(
            self.model,
            [
                {
                    "role": "system",
                    "content": (
                        "Draft from supplied evidence, then decompose the draft into atomic "
                        "factual claims. Do not decide citations; a separate evidence graph "
                        "will verify every claim."
                    ),
                },
                {
                    "role": "user",
                    "content": f"Question: {query}\n\nSources:\n{source_text}",
                },
            ],
            temperature=0.0,
            seed=seed,
            max_tokens=1200,
            json_schema=schema,
        )
        if not isinstance(payload, dict) or set(payload) != {"answer", "claims"}:
            raise ValueError("claim draft has the wrong object schema")
        answer = str(payload["answer"]).strip()
        claims = [str(claim).strip() for claim in payload["claims"]]
        if not answer or not claims or any(not claim for claim in claims):
            raise ValueError("claim draft answer or claim list is empty")
        if len(claims) != len(set(claims)):
            raise ValueError("claim draft contains duplicate atomic claims")
        return CitedAnswer(
            answer=answer,
            claims=tuple(
                {"claim": claim, "cited_source_ids": []} for claim in claims
            ),
            model=response.model,
            prompt_tokens=response.prompt_tokens,
            completion_tokens=response.completion_tokens,
            wall_seconds=response.wall_seconds,
        )

    def compose_supported_answer(
        self,
        query: str,
        supported_claims: list[dict[str, Any]],
        sources: list[dict[str, str]],
        *,
        seed: int = 103,
    ) -> CitedAnswer:
        """Realize only graph-approved claims, preserving their fixed citations."""
        if not supported_claims or not sources:
            raise ValueError("supported-answer composition requires claims and sources")
        source_ids = {str(source["source_id"]) for source in sources}
        normalized_claims = []
        for record in supported_claims:
            if not isinstance(record, dict) or set(record) != {
                "claim",
                "cited_source_ids",
            }:
                raise ValueError("supported claim has the wrong schema")
            claim = str(record["claim"]).strip()
            citations = list(map(str, record["cited_source_ids"]))
            if (
                not claim
                or not citations
                or len(citations) != len(set(citations))
                or not set(citations) <= source_ids
            ):
                raise ValueError("supported claim is empty or has invalid citations")
            normalized_claims.append(
                {"claim": claim, "cited_source_ids": citations}
            )
        source_text = "\n\n".join(
            f"[{source['source_id']}] {source['text']}" for source in sources
        )
        claim_text = "\n".join(
            f"- {record['claim']} CITATIONS={record['cited_source_ids']}"
            for record in normalized_claims
        )
        schema = {
            "type": "object",
            "additionalProperties": False,
            "required": ["answer"],
            "properties": {"answer": {"type": "string", "minLength": 1}},
        }
        payload, response = self.client.chat_json(
            self.model,
            [
                {
                    "role": "system",
                    "content": (
                        "Compose an answer using exactly the supplied verified claims. Do not "
                        "add factual claims, change the fixed citations, or use outside knowledge."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Question: {query}\n\nVerified claims:\n{claim_text}\n\n"
                        f"Cited sources:\n{source_text}"
                    ),
                },
            ],
            temperature=0.0,
            seed=seed,
            max_tokens=1200,
            json_schema=schema,
        )
        if not isinstance(payload, dict) or set(payload) != {"answer"}:
            raise ValueError("supported answer has the wrong object schema")
        answer = str(payload["answer"]).strip()
        if not answer:
            raise ValueError("supported answer is empty")
        return CitedAnswer(
            answer=answer,
            claims=tuple(normalized_claims),
            model=response.model,
            prompt_tokens=response.prompt_tokens,
            completion_tokens=response.completion_tokens,
            wall_seconds=response.wall_seconds,
        )
