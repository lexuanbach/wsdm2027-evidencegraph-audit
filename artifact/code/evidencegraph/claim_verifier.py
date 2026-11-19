"""Frozen local-LLM claim verifier with strict, auditable outputs."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any

from .local_llm import OllamaClient


VERDICTS = {"SUPPORTED", "REFUTED", "NOT_ENOUGH_INFO"}


@dataclass(frozen=True)
class Verification:
    verdict: str
    claim_probability_true: float
    rationale: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    wall_seconds: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class LocalClaimVerifier:
    def __init__(self, model: str, client: OllamaClient | None = None):
        self.model = model
        self.client = client or OllamaClient()

    def verify(
        self,
        claim: str,
        sources: list[dict[str, str]],
        *,
        seed: int = 103,
        dependence_aware: bool = False,
    ) -> Verification:
        if not sources:
            raise ValueError("at least one source is required")
        policy = (
            "The provided list already contains one representative per source-origin cluster. "
            "Do not infer extra confidence from repeated wording. "
            if dependence_aware
            else "Treat each listed document as a retrieved result. "
        )
        prompt = (
            f"Claim: {claim}\n\nSources:\n"
            + "\n".join(
                f"[{source['source_id']}] {source['text']}" for source in sources
            )
            + "\n\n"
            + policy
            + "Decide whether the sources SUPPORT, REFUTE, or provide NOT_ENOUGH_INFO for the "
            "claim. Return one JSON object with exactly verdict, claim_probability_true, and rationale. "
            "verdict must be SUPPORTED, REFUTED, or NOT_ENOUGH_INFO. claim_probability_true is a number "
            "from 0 to 1 estimating the probability that the claim itself is true. rationale "
            "is one short sentence."
        )
        payload, response = self.client.chat_json(
            self.model,
            [
                {
                    "role": "system",
                    "content": "You are a careful evidence-grounded fact verifier. Return valid JSON only.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            seed=seed,
            max_tokens=180,
        )
        if not isinstance(payload, dict) or set(payload) != {
            "verdict",
            "claim_probability_true",
            "rationale",
        }:
            raise ValueError("verifier response has the wrong JSON schema")
        verdict = str(payload["verdict"]).upper()
        probability = float(payload["claim_probability_true"])
        rationale = str(payload["rationale"]).strip()
        if verdict not in VERDICTS or not 0.0 <= probability <= 1.0 or not rationale:
            raise ValueError("verifier response contains an invalid value")
        return Verification(
            verdict=verdict,
            claim_probability_true=probability,
            rationale=rationale,
            model=response.model,
            prompt_tokens=response.prompt_tokens,
            completion_tokens=response.completion_tokens,
            wall_seconds=response.wall_seconds,
        )

    def verify_binary(
        self,
        claim: str,
        sources: list[dict[str, str]],
        *,
        seed: int = 103,
        dependence_aware: bool = False,
    ) -> Verification:
        if not sources:
            raise ValueError("at least one source is required")
        policy = (
            "The list already contains one representative per source-origin cluster. "
            "Do not infer extra confidence from repeated wording. "
            if dependence_aware
            else "Treat each listed document as a retrieved result. "
        )
        prompt = (
            f"Claim: {claim}\n\nSources:\n"
            + "\n".join(
                f"[{source['source_id']}] {source['text']}" for source in sources
            )
            + "\n\n"
            + policy
            + "Is the claim true given the sources? Output exactly one token: TRUE or FALSE."
        )
        probability, response = self.client.chat_binary(
            self.model,
            [
                {
                    "role": "system",
                    "content": "Judge only from the supplied evidence. Answer TRUE or FALSE.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            seed=seed,
        )
        return Verification(
            verdict="SUPPORTED" if probability >= 0.5 else "REFUTED",
            claim_probability_true=probability,
            rationale="Forced binary decision scored from first-token log probabilities.",
            model=response.model,
            prompt_tokens=response.prompt_tokens,
            completion_tokens=response.completion_tokens,
            wall_seconds=response.wall_seconds,
        )
