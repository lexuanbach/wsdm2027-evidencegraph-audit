from __future__ import annotations

import hashlib
import json

from evidencegraph.benchmark_cases import (
    averitec_cases,
    expertqa_cases,
    fever_wiki_page_id,
    fever_cases,
    parse_expertqa_evidence,
)


def test_hover_titles_map_to_fever_snapshot_page_ids() -> None:
    assert fever_wiki_page_id("The Shining (film)") == "The_Shining_-LRB-film-RRB-"
    assert (
        fever_wiki_page_id("After Hours: Unplugged &amp; Rewired")
        == "After_Hours-COLON-_Unplugged_&_Rewired"
    )


def test_fever_cases_preserve_alternative_evidence_groups(tmp_path) -> None:
    path = tmp_path / "fever.jsonl"
    path.write_text(
        json.dumps(
            {
                "id": 1,
                "verifiable": "VERIFIABLE",
                "label": "SUPPORTS",
                "claim": "Claim.",
                "evidence": [
                    [[1, 1, "Page_A", 0]],
                    [[2, 2, "Page_B", 1], [2, 2, "Page_C", 2]],
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    case = fever_cases(path)[0]

    assert case.gold_evidence == ("Page_A:0", "Page_B:1", "Page_C:2")
    assert case.metadata["evidence_groups"] == [["Page_A:0"], ["Page_B:1", "Page_C:2"]]


def test_fever_case_ids_normalize_unicode_for_index_matching(tmp_path) -> None:
    path = tmp_path / "fever.jsonl"
    path.write_text(
        json.dumps(
            {
                "id": 2,
                "verifiable": "VERIFIABLE",
                "label": "SUPPORTS",
                "claim": "Claim.",
                "evidence": [[[-1, -1, "Jose\u0301", 0]]],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    assert fever_cases(path)[0].gold_evidence == ("José:0",)


def test_averitec_cases_collect_unique_gold_urls(tmp_path) -> None:
    path = tmp_path / "averitec.json"
    path.write_text(
        json.dumps(
            [
                {
                    "claim": "Claim.",
                    "label": "Supported",
                    "justification": "Because.",
                    "questions": [
                        {
                            "question": "Why?",
                            "answers": [
                                {"source_url": "https://example.test/a"},
                                {"source_url": "https://example.test/a"},
                            ],
                        }
                    ],
                }
            ]
        ),
        encoding="utf-8",
    )

    assert averitec_cases(path)[0].gold_evidence == ("https://example.test/a",)


def test_expertqa_cases_join_test_questions_and_evidence(tmp_path) -> None:
    annotated = tmp_path / "annotated.jsonl"
    lfqa = tmp_path / "test.json"
    annotated.write_text(
        json.dumps(
            {
                "question": "Why?",
                "annotator_id": "anon",
                "metadata": {"field": "science"},
                "answers": {
                    "system": {
                        "answer_string": "Because [1].",
                        "claims": [
                            {
                                "claim_string": "Because.",
                                "support": "Complete",
                                "evidence": ["[1] https://example.test\n\nPassage text."],
                            },
                            {
                                "claim_string": "Maybe.",
                                "support": "Partial",
                                "evidence": ["[2] https://example.test/partial\n\nPartial text."],
                            },
                        ],
                    }
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    lfqa.write_text(json.dumps({"example_id": 7, "question": "Why?"}) + "\n", encoding="utf-8")

    case = expertqa_cases(annotated, lfqa)[0]

    assert case.case_id == "7"
    assert case.gold_evidence == ("[1] https://example.test",)
    assert case.metadata["evidence_passages"]["[1] https://example.test"] == "Passage text."
    assert case.metadata["evidence_passages"]["[2] https://example.test/partial"] == "Partial text."
    assert case.metadata["claims"][1]["support"] == "Partial"


def test_parse_expertqa_evidence_content_addresses_passage_only_release_variant() -> None:
    passage = "Passage without a retained per-passage URL."
    expected = hashlib.sha256(passage.encode("utf-8")).hexdigest()

    evidence_id, parsed = parse_expertqa_evidence(f"  {passage}\n")

    assert evidence_id == f"passage:sha256:{expected}"
    assert parsed == passage


def test_parse_expertqa_evidence_rejects_malformed_numbered_source() -> None:
    try:
        parse_expertqa_evidence("[1] not-a-url\n\nPassage")
    except ValueError as error:
        assert "valid HTTP(S) URL" in str(error)
    else:
        raise AssertionError("expected malformed evidence to fail")
