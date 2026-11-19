from evidencegraph.local_llm import binary_probability
from run_duplication import CONTEXT_SLOTS, matched_sources


def test_binary_probability_normalizes_true_false_tokens() -> None:
    probability = binary_probability(
        (
            {
                "token": " TRUE",
                "logprob": -0.2,
                "top_logprobs": [
                    {"token": " TRUE", "logprob": -0.2},
                    {"token": " FALSE", "logprob": -1.2},
                ],
            },
        )
    )
    assert 0.73 < probability < 0.74


def test_matched_duplication_context_has_constant_slots() -> None:
    for copies in (1, 5, 20):
        raw, representatives = matched_sources(
            "A test claim.", copies, "claim-1", seed=103
        )
        assert len(raw) == CONTEXT_SLOTS
        assert sum("repeated" in source["source_id"] for source in raw) == copies
        assert len(representatives) == 1
