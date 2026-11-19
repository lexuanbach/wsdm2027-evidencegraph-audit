import pytest

from evidencegraph.paraphrase_job import (
    PROMPT_PROFILES,
    _parse,
    _prompt,
    candidate_requirements,
    combine_segment_paraphrases,
    exclusive_output_lock,
    generation_batches,
    generation_token_budget,
    response_schema,
    select_pending,
    unique_casefolded,
)


def test_limit_freezes_cohort_before_resume_filter() -> None:
    selection = [{"claim_id": f"c{index}"} for index in range(6)]
    completed = {"c0": {"status": "generated"}, "c2": {"status": "generated"}}

    pending = select_pending(selection, completed, limit=4)

    assert [record["claim_id"] for record in pending] == ["c1", "c3"]


def test_candidate_requirements_match_corruption_design() -> None:
    assert candidate_requirements({"corruption": "duplication"}) == (2, 1)
    assert candidate_requirements({"corruption": "paraphrase_flood"}) == (8, 4)


def test_generation_batches_do_not_mix_requirements() -> None:
    records = [
        {"claim_id": "a", "corruption": "duplication"},
        {"claim_id": "b", "corruption": "duplication"},
        {"claim_id": "c", "corruption": "paraphrase_flood"},
        {"claim_id": "d", "corruption": "duplication"},
    ]
    batches = generation_batches(records, batch_size=4)
    assert [[record["claim_id"] for record in batch] for batch in batches] == [
        ["a", "b", "d"],
        ["c"],
    ]


def test_generation_budget_scales_with_long_faithful_outputs() -> None:
    short = generation_token_budget([("a", "brief evidence")], 2)
    long = generation_token_budget([("a", "x" * 600), ("b", "y" * 600)], 2)
    flood = generation_token_budget([("a", "x" * 600)], 8)

    assert short == 500
    assert long > 1200
    assert flood > long
    assert generation_token_budget([("a", "x" * 10_000)], 8) == 16_000
    assert generation_token_budget([("a", "brief evidence")], 2, scale=2.0) == 1000
    assert generation_token_budget([("a", "brief evidence")], 2, scale=4.0) == 2000


def test_exclusive_output_lock_rejects_concurrent_writer(tmp_path) -> None:
    output = tmp_path / "raw.jsonl"
    with exclusive_output_lock(output):
        with pytest.raises(RuntimeError, match="holds the output lock"):
            with exclusive_output_lock(output):
                pass
    with exclusive_output_lock(output):
        pass


def test_unique_casefolded_does_not_count_duplicate_candidates() -> None:
    assert unique_casefolded(["One.", " one. ", "Two."]) == ["One.", "Two."]


def test_segment_paraphrases_are_recombined_by_candidate_index() -> None:
    parsed = {
        "c::segment-0": ["First A.", "First B."],
        "c::segment-1": ["Second A.", "Second B."],
    }
    assert combine_segment_paraphrases(
        parsed, ["c::segment-0", "c::segment-1"], 2
    ) == ["First A.\nSecond A.", "First B.\nSecond B."]


def test_response_schema_constrains_item_and_candidate_cardinality() -> None:
    schema = response_schema(["a", "b"], candidates_per_item=8)
    items = schema["properties"]["items"]
    paraphrases = items["properties"]["a"]

    assert items["required"] == ["a", "b"]
    assert set(items["properties"]) == {"a", "b"}
    assert paraphrases["minItems"] == paraphrases["maxItems"] == 8
    assert paraphrases["items"]["minLength"] == 1


def test_keyed_response_is_independent_of_json_property_order() -> None:
    payload = {"items": {"b": ["B1", "B2"], "a": ["A1", "A2"]}}

    parsed = _parse(payload, ["a", "b"], candidates_per_item=2)

    assert list(parsed) == ["a", "b"]
    assert parsed == {"a": ["A1", "A2"], "b": ["B1", "B2"]}


def test_keyed_response_rejects_blank_candidate_strings() -> None:
    payload = {"items": {"a": ["A1", "  "]}}

    with pytest.raises(ValueError, match="exactly 2 string paraphrases"):
        _parse(payload, ["a"], candidates_per_item=2)


def test_lexically_anchored_prompt_matches_structural_gate_contract() -> None:
    prompt = _prompt(
        [("a", "The group developed the album.")],
        2,
        prompt_version=PROMPT_PROFILES["lexically_anchored_v12"],
    )

    assert "retain every noun, proper noun, verb, adjective, and adverb" in prompt
    assert "Preserve the original tense, aspect, modality" in prompt
    assert "same agents and patients" in prompt
    assert "Do not return the original string unchanged" in prompt


def test_structure_checklist_prompt_embeds_item_specific_constraints() -> None:
    constraints = {
        "a": {
            "segments": [
                {
                    "content_lemmas": ["ADV:also:count=1"],
                    "semantic_roles": ["uruguay:win:agent:count=1"],
                }
            ]
        }
    }

    prompt = _prompt(
        [("a", "Uruguay also won.")],
        2,
        prompt_version=PROMPT_PROFILES["structure_checklist_v13"],
        item_constraints=constraints,
    )

    assert "required_structure_checklist" in prompt
    assert "ADV:also:count=1" in prompt
    assert "uruguay:win:agent:count=1" in prompt
    assert "do not substitute a synonym" in prompt


def test_token_preserving_preorder_forbids_voice_and_lexical_changes() -> None:
    constraints = {
        "a": {
            "segments": [
                {
                    "content_lemmas": ["NOUN:film:count=1", "VERB:star:count=1"],
                    "semantic_roles": ["film:star:agent:count=1"],
                }
            ]
        }
    }
    prompt = _prompt(
        [("a", "In 1999, the film starred Smith.")],
        2,
        prompt_version=PROMPT_PROFILES["token_preserving_preorder_v16"],
        item_constraints=constraints,
    )

    assert "moving an intact temporal" in prompt
    assert "Never convert active voice to passive voice" in prompt
    assert "never use voice conversion or lexical substitution" in prompt
    assert "film:star:agent:count=1" in prompt
