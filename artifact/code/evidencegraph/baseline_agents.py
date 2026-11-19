"""Faithful bounded ReAct and Chain-of-Verification benchmark protocols."""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from typing import Any

from .benchmark_answerer import LocalBenchmarkAnswerer, cited_evidence_records
from .local_llm import LocalResponse, OllamaClient


SearchFunction = Callable[[str, int], list[dict[str, Any]]]


def _usage(responses: list[LocalResponse]) -> dict[str, float | int]:
    return {
        "prompt_tokens": sum(response.prompt_tokens for response in responses),
        "completion_tokens": sum(response.completion_tokens for response in responses),
        "wall_seconds": sum(response.wall_seconds for response in responses),
        "llm_calls": len(responses),
    }


class ReActBenchmarkAgent:
    """Interleave model-chosen searches with environment observations before answering."""

    PROMPT_VERSION = "react-frozen-search-v1"

    def __init__(
        self, model: str, client: Any | None = None, answerer: LocalBenchmarkAnswerer | None = None
    ):
        self.model = model
        self.client = client or OllamaClient(timeout=300)
        self.answerer = answerer or LocalBenchmarkAnswerer(model, self.client)

    def _collect(
        self,
        query: str,
        search: SearchFunction,
        *,
        seed: int,
        maximum_searches: int,
        results_per_search: int,
    ) -> tuple[
        list[dict[str, Any]], list[dict[str, Any]], list[LocalResponse], float
    ]:
        if maximum_searches < 1 or results_per_search < 1:
            raise ValueError("ReAct search budgets must be positive")
        schema = {
            "type": "object",
            "additionalProperties": False,
            "required": ["rationale", "action", "query"],
            "properties": {
                "rationale": {"type": "string", "minLength": 1},
                "action": {"type": "string", "enum": ["search", "finish"]},
                "query": {"type": "string"},
            },
        }
        sources: dict[str, dict[str, Any]] = {}
        trace: list[dict[str, Any]] = []
        responses: list[LocalResponse] = []
        retrieval_wall_seconds = 0.0
        for step in range(maximum_searches + 1):
            observations = [
                {"source_id": source_id, "text": str(source["text"])}
                for source_id, source in sources.items()
            ]
            payload, response = self.client.chat_json(
                self.model,
                [
                    {
                        "role": "system",
                        "content": (
                            "Use the ReAct pattern: briefly reason from observations, then choose "
                            "one search action or finish. Example: question about a person's role; "
                            "search the person and role; inspect the observation; finish only when "
                            "the available evidence can support the answer."
                        ),
                    },
                    {
                        "role": "user",
                        "content": (
                            f"Task: {query}\nSearches remaining: {maximum_searches - step}\n"
                            f"Observations: {json.dumps(observations, ensure_ascii=False)}\n"
                            "Return the next action."
                        ),
                    },
                ],
                temperature=0.0,
                seed=seed + step,
                max_tokens=160,
                json_schema=schema,
            )
            responses.append(response)
            if not isinstance(payload, dict) or set(payload) != {
                "rationale",
                "action",
                "query",
            }:
                raise ValueError("ReAct decision has the wrong schema")
            action = str(payload["action"])
            rationale = str(payload["rationale"]).strip()
            search_query = str(payload["query"]).strip()
            if not rationale or action not in {"search", "finish"}:
                raise ValueError("ReAct decision has invalid content")
            if action == "finish":
                trace.append({"step": step, "action": action, "rationale": rationale})
                break
            if step >= maximum_searches:
                trace.append(
                    {"step": step, "action": "budget_exhausted", "rationale": rationale}
                )
                break
            if not search_query:
                raise ValueError("ReAct search action has an empty query")
            search_started = time.monotonic()
            results = search(search_query, results_per_search)
            search_wall = time.monotonic() - search_started
            retrieval_wall_seconds += search_wall
            new_ids = []
            for result in results:
                source_id = str(result["source_id"])
                if source_id not in sources:
                    sources[source_id] = result
                    new_ids.append(source_id)
            trace.append(
                {
                    "step": step,
                    "action": action,
                    "rationale": rationale,
                    "query": search_query,
                    "observed_source_ids": new_ids,
                    "search_wall_seconds": search_wall,
                }
            )
        if not sources:
            search_started = time.monotonic()
            forced_results = search(query, results_per_search)
            search_wall = time.monotonic() - search_started
            retrieval_wall_seconds += search_wall
            for result in forced_results:
                sources[str(result["source_id"])] = result
            trace.append(
                {
                    "step": len(trace),
                    "action": "forced_initial_search",
                    "query": query,
                    "observed_source_ids": list(sources),
                    "search_wall_seconds": search_wall,
                }
            )
        if not sources:
            raise ValueError("ReAct environment returned no sources")
        return list(sources.values()), trace, responses, retrieval_wall_seconds

    def run_verdict(
        self,
        dataset: str,
        query: str,
        search: SearchFunction,
        *,
        seed: int = 103,
        maximum_searches: int = 3,
        results_per_search: int = 5,
    ) -> dict[str, Any]:
        sources, trace, responses, retrieval_wall_seconds = self._collect(
            query,
            search,
            seed=seed,
            maximum_searches=maximum_searches,
            results_per_search=results_per_search,
        )
        final = self.answerer.predict_verdict(
            dataset, query, sources, seed=seed, policy="react"
        )
        # The answerer's response is represented in its returned accounting.
        final_response = LocalResponse(
            content="",
            model=final.model,
            wall_seconds=final.wall_seconds,
            prompt_tokens=final.prompt_tokens,
            completion_tokens=final.completion_tokens,
        )
        return {
            **final.to_dict(),
            **_usage([*responses, final_response]),
            "protocol": self.PROMPT_VERSION,
            "maximum_searches": maximum_searches,
            "results_per_search": results_per_search,
            "searches": sum(item["action"] == "search" for item in trace),
            "documents_opened": len(sources),
            "retrieval_wall_seconds": retrieval_wall_seconds,
            "cited_evidence": cited_evidence_records(sources, final.to_dict()),
            "trace": trace,
        }

    def run_cited_answer(
        self,
        query: str,
        search: SearchFunction,
        *,
        seed: int = 103,
        maximum_searches: int = 3,
        results_per_search: int = 5,
    ) -> dict[str, Any]:
        sources, trace, responses, retrieval_wall_seconds = self._collect(
            query,
            search,
            seed=seed,
            maximum_searches=maximum_searches,
            results_per_search=results_per_search,
        )
        final = self.answerer.generate_cited_answer(
            query, sources, seed=seed, policy="react"
        )
        final_response = LocalResponse(
            content="",
            model=final.model,
            wall_seconds=final.wall_seconds,
            prompt_tokens=final.prompt_tokens,
            completion_tokens=final.completion_tokens,
        )
        return {
            **final.to_dict(),
            **_usage([*responses, final_response]),
            "protocol": self.PROMPT_VERSION,
            "maximum_searches": maximum_searches,
            "results_per_search": results_per_search,
            "searches": sum(item["action"] == "search" for item in trace),
            "documents_opened": len(sources),
            "retrieval_wall_seconds": retrieval_wall_seconds,
            "cited_evidence": cited_evidence_records(sources, final.to_dict()),
            "trace": trace,
        }


class ChainOfVerificationAgent:
    """Draft, plan checks, answer checks independently, then revise."""

    PROMPT_VERSION = "chain-of-verification-four-phase-v1"

    def __init__(
        self, model: str, client: Any | None = None, answerer: LocalBenchmarkAnswerer | None = None
    ):
        self.model = model
        self.client = client or OllamaClient(timeout=300)
        self.answerer = answerer or LocalBenchmarkAnswerer(model, self.client)

    def _plan(
        self, query: str, draft: str, *, seed: int
    ) -> tuple[list[str], LocalResponse]:
        schema = {
            "type": "object",
            "additionalProperties": False,
            "required": ["questions"],
            "properties": {
                "questions": {
                    "type": "array",
                    "minItems": 1,
                    "maxItems": 3,
                    "uniqueItems": True,
                    "items": {"type": "string", "minLength": 1},
                }
            },
        }
        payload, response = self.client.chat_json(
            self.model,
            [
                {
                    "role": "system",
                    "content": "Plan concise factual questions that can detect errors in a draft.",
                },
                {
                    "role": "user",
                    "content": f"Original task: {query}\nDraft: {draft}\nPlan 1-3 checks.",
                },
            ],
            temperature=0.0,
            seed=seed,
            max_tokens=240,
            json_schema=schema,
        )
        if not isinstance(payload, dict) or set(payload) != {"questions"}:
            raise ValueError("CoVe verification plan has the wrong schema")
        questions = [str(question).strip() for question in payload["questions"]]
        if not 1 <= len(questions) <= 3 or any(not question for question in questions):
            raise ValueError("CoVe verification plan has invalid questions")
        if len(questions) != len(set(questions)):
            raise ValueError("CoVe verification questions are not unique")
        return questions, response

    def _verify(
        self,
        question: str,
        sources: list[dict[str, str]],
        *,
        seed: int,
    ) -> tuple[dict[str, Any], LocalResponse]:
        source_ids = {str(source["source_id"]) for source in sources}
        source_text = "\n\n".join(
            f"[{source['source_id']}] {source['text']}" for source in sources
        )
        schema = {
            "type": "object",
            "additionalProperties": False,
            "required": ["answer", "cited_source_ids"],
            "properties": {
                "answer": {"type": "string", "minLength": 1},
                "cited_source_ids": {
                    "type": "array",
                    "uniqueItems": True,
                    "items": {"type": "string"},
                },
            },
        }
        payload, response = self.client.chat_json(
            self.model,
            [
                {
                    "role": "system",
                    "content": (
                        "Answer this verification question independently from the supplied "
                        "sources. You have not seen the draft or any other verification answer."
                    ),
                },
                {
                    "role": "user",
                    "content": f"Question: {question}\n\nSources:\n{source_text}",
                },
            ],
            temperature=0.0,
            seed=seed,
            max_tokens=320,
            json_schema=schema,
        )
        if not isinstance(payload, dict) or set(payload) != {
            "answer",
            "cited_source_ids",
        }:
            raise ValueError("CoVe verification answer has the wrong schema")
        answer = str(payload["answer"]).strip()
        citations = list(map(str, payload["cited_source_ids"]))
        if not answer or len(citations) != len(set(citations)) or not set(citations) <= source_ids:
            raise ValueError("CoVe verification answer is empty or cites unavailable evidence")
        return {"question": question, "answer": answer, "cited_source_ids": citations}, response

    def _deliberation(
        self,
        query: str,
        draft: str,
        sources: list[dict[str, str]],
        *,
        seed: int,
    ) -> tuple[str, list[dict[str, Any]], list[LocalResponse]]:
        questions, plan_response = self._plan(query, draft, seed=seed + 1)
        checks = []
        responses = [plan_response]
        for index, question in enumerate(questions):
            check, response = self._verify(
                question, sources, seed=seed + 2 + index
            )
            checks.append(check)
            responses.append(response)
        context = json.dumps(
            {"draft": draft, "independent_verification": checks}, ensure_ascii=False
        )
        return context, checks, responses

    def run_verdict(
        self,
        dataset: str,
        query: str,
        sources: list[dict[str, str]],
        *,
        seed: int = 103,
    ) -> dict[str, Any]:
        draft = self.answerer.predict_verdict(
            dataset, query, sources, seed=seed, policy="cove_draft"
        )
        context, checks, middle_responses = self._deliberation(
            query, draft.answer, sources, seed=seed
        )
        final = self.answerer.predict_verdict(
            dataset,
            query,
            sources,
            seed=seed + 10,
            policy="cove_final",
            additional_context=context,
        )
        endpoint_responses = [
            LocalResponse("", draft.model, draft.wall_seconds, draft.prompt_tokens, draft.completion_tokens),
            LocalResponse("", final.model, final.wall_seconds, final.prompt_tokens, final.completion_tokens),
        ]
        return {
            **final.to_dict(),
            **_usage([endpoint_responses[0], *middle_responses, endpoint_responses[1]]),
            "protocol": self.PROMPT_VERSION,
            "draft_answer": draft.answer,
            "verification": checks,
            "documents_opened": len(sources),
            "searches": 0,
        }

    def run_cited_answer(
        self,
        query: str,
        sources: list[dict[str, str]],
        *,
        seed: int = 103,
    ) -> dict[str, Any]:
        draft = self.answerer.generate_cited_answer(
            query, sources, seed=seed, policy="cove_draft"
        )
        context, checks, middle_responses = self._deliberation(
            query, draft.answer, sources, seed=seed
        )
        final = self.answerer.generate_cited_answer(
            query,
            sources,
            seed=seed + 10,
            policy="cove_final",
            additional_context=context,
        )
        endpoint_responses = [
            LocalResponse("", draft.model, draft.wall_seconds, draft.prompt_tokens, draft.completion_tokens),
            LocalResponse("", final.model, final.wall_seconds, final.prompt_tokens, final.completion_tokens),
        ]
        return {
            **final.to_dict(),
            **_usage([endpoint_responses[0], *middle_responses, endpoint_responses[1]]),
            "protocol": self.PROMPT_VERSION,
            "draft_answer": draft.answer,
            "verification": checks,
            "documents_opened": len(sources),
            "searches": 0,
        }
