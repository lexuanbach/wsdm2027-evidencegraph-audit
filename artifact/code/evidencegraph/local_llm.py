"""Minimal deterministic Ollama client used before any paid API call."""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass
from typing import Any

import requests


@dataclass(frozen=True)
class LocalResponse:
    content: str
    model: str
    wall_seconds: float
    prompt_tokens: int
    completion_tokens: int
    logprobs: tuple[dict[str, Any], ...] = ()


def binary_probability(logprobs: tuple[dict[str, Any], ...]) -> float:
    """Normalize first-token TRUE/FALSE log probabilities."""
    if not logprobs:
        raise ValueError("binary response contains no token log probabilities")
    first = logprobs[0]
    candidates = [first, *first.get("top_logprobs", [])]
    scores: dict[str, float] = {}
    for candidate in candidates:
        token = str(candidate.get("token", "")).strip().strip('"\'`.,:;!?').upper()
        if token in {"TRUE", "FALSE"}:
            score = float(candidate["logprob"])
            scores[token] = max(scores.get(token, -math.inf), score)
    if set(scores) != {"TRUE", "FALSE"}:
        raise ValueError("top token probabilities do not contain TRUE and FALSE")
    maximum = max(scores.values())
    true_mass = math.exp(scores["TRUE"] - maximum)
    false_mass = math.exp(scores["FALSE"] - maximum)
    return true_mass / (true_mass + false_mass)


class OllamaClient:
    def __init__(self, base_url: str | None = None, timeout: int = 180):
        resolved = base_url or os.environ.get(
            "OLLAMA_BASE_URL", "http://127.0.0.1:11434"
        )
        self.base_url = resolved.rstrip("/")
        self.timeout = timeout

    def chat(
        self,
        model: str,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.0,
        seed: int = 0,
        max_tokens: int = 512,
        json_mode: bool = False,
        json_schema: dict[str, Any] | None = None,
        keep_alive: str = "10m",
        logprobs: bool = False,
        top_logprobs: int = 0,
    ) -> LocalResponse:
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": False,
            "keep_alive": keep_alive,
            "options": {
                "temperature": temperature,
                "seed": seed,
                "num_predict": max_tokens,
            },
        }
        if json_mode and json_schema is not None:
            raise ValueError("json_mode and json_schema are mutually exclusive")
        if json_schema is not None:
            payload["format"] = json_schema
        elif json_mode:
            payload["format"] = "json"
        if logprobs:
            payload["logprobs"] = True
            payload["top_logprobs"] = top_logprobs
        started = time.monotonic()
        response = requests.post(
            f"{self.base_url}/api/chat", json=payload, timeout=self.timeout
        )
        response.raise_for_status()
        record = response.json()
        return LocalResponse(
            content=str(record["message"]["content"]),
            model=str(record.get("model", model)),
            wall_seconds=time.monotonic() - started,
            prompt_tokens=int(record.get("prompt_eval_count", 0)),
            completion_tokens=int(record.get("eval_count", 0)),
            logprobs=tuple(record.get("logprobs", [])),
        )

    def chat_json(
        self,
        *args: Any,
        json_schema: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> tuple[Any, LocalResponse]:
        response = self.chat(
            *args,
            json_mode=json_schema is None,
            json_schema=json_schema,
            **kwargs,
        )
        return json.loads(response.content), response

    def chat_binary(self, *args: Any, **kwargs: Any) -> tuple[float, LocalResponse]:
        response = self.chat(
            *args,
            max_tokens=1,
            logprobs=True,
            top_logprobs=20,
            **kwargs,
        )
        return binary_probability(response.logprobs), response
