from __future__ import annotations

import json

from evidencegraph.benchmark_loaders import (
    load_averitec,
    load_expertqa,
    load_expertqa_lfqa,
    load_fever,
    load_hover,
)


def test_load_fever_jsonl(tmp_path) -> None:
    path = tmp_path / "fever.jsonl"
    record = {"id": 1, "verifiable": "VERIFIABLE", "label": "SUPPORTS", "claim": "c", "evidence": []}
    path.write_text(json.dumps(record) + "\n", encoding="utf-8")
    assert load_fever(path) == [record]


def test_official_json_loaders(tmp_path) -> None:
    hover = [{"uid": "1", "claim": "c", "supporting_facts": [], "label": "SUPPORTED", "num_hops": 3}]
    averitec = [{"claim": "c", "label": "Supported", "questions": [], "justification": "j"}]
    hover_path, averitec_path = tmp_path / "hover.json", tmp_path / "averitec.json"
    hover_path.write_text(json.dumps(hover), encoding="utf-8")
    averitec_path.write_text(json.dumps(averitec), encoding="utf-8")
    assert load_hover(hover_path) == hover
    assert load_averitec(averitec_path) == averitec


def test_expertqa_json_extension_is_read_as_jsonl(tmp_path) -> None:
    path = tmp_path / "expertqa.json"
    path.write_text('{"question":"one"}\n{"question":"two"}\n', encoding="utf-8")
    assert [row["question"] for row in load_expertqa_lfqa(path)] == ["one", "two"]


def test_expertqa_annotated_loader_requires_model_keyed_answers(tmp_path) -> None:
    path = tmp_path / "expertqa.jsonl"
    path.write_text(
        json.dumps(
            {
                "question": "Why?",
                "annotator_id": "anon",
                "answers": {"system": {"answer_string": "Because."}},
                "metadata": {"field": "science"},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    assert load_expertqa(path)[0]["answers"]["system"]["answer_string"] == "Because."
