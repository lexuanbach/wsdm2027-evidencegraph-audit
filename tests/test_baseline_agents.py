from __future__ import annotations

from evidencegraph.baseline_agents import ChainOfVerificationAgent, ReActBenchmarkAgent
from evidencegraph.benchmark_answerer import LocalBenchmarkAnswerer
from evidencegraph.local_llm import LocalResponse


class QueueClient:
    def __init__(self, payloads):
        self.payloads = list(payloads)
        self.messages = []

    def chat_json(self, *args, **kwargs):
        self.messages.append(args[1])
        payload = self.payloads.pop(0)
        return payload, LocalResponse(
            content="{}",
            model="fake",
            wall_seconds=0.1,
            prompt_tokens=10,
            completion_tokens=5,
        )


def _verdict(label="SUPPORTED"):
    return {
        "label": label,
        "probabilities": {
            "SUPPORTED": 0.8 if label == "SUPPORTED" else 0.2,
            "NOT_SUPPORTED": 0.2 if label == "SUPPORTED" else 0.8,
        },
        "answer": "The evidence supports the claim.",
        "cited_source_ids": ["s1"],
    }


def test_react_interleaves_search_observation_and_finish() -> None:
    client = QueueClient(
        [
            {"rationale": "Need evidence.", "action": "search", "query": "claim"},
            {"rationale": "Evidence is sufficient.", "action": "finish", "query": ""},
            _verdict(),
        ]
    )
    searches = []

    def search(query, limit):
        searches.append((query, limit))
        return [{"source_id": "s1", "text": "Supporting evidence."}]

    agent = ReActBenchmarkAgent(
        "fake", client=client, answerer=LocalBenchmarkAnswerer("fake", client)
    )
    result = agent.run_verdict("HoVer", "Claim", search, maximum_searches=2)

    assert searches == [("claim", 5)]
    assert result["searches"] == 1
    assert result["documents_opened"] == 1
    assert result["llm_calls"] == 3
    assert result["prompt_tokens"] == 30


def test_cove_answers_checks_without_exposing_draft_then_revises() -> None:
    client = QueueClient(
        [
            _verdict(),
            {"questions": ["Does the source support the claim?"]},
            {"answer": "Yes.", "cited_source_ids": ["s1"]},
            _verdict(),
        ]
    )
    sources = [{"source_id": "s1", "text": "Supporting evidence."}]
    agent = ChainOfVerificationAgent(
        "fake", client=client, answerer=LocalBenchmarkAnswerer("fake", client)
    )
    result = agent.run_verdict("HoVer", "Claim", sources)

    independent_prompt = client.messages[2]
    assert "Draft:" not in independent_prompt[1]["content"]
    assert result["llm_calls"] == 4
    assert result["prompt_tokens"] == 40
    assert result["verification"][0]["cited_source_ids"] == ["s1"]
