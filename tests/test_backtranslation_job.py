from evidencegraph.backtranslation_job import (
    MODEL_IDENTITY,
    force_word_ids,
    lexical_constraint_phrases,
)
from evidencegraph.paraphrase_job import (
    PROTECTED_BACKTRANSLATION_PROMPT_VERSION,
    SUPPORTED_PROMPT_VERSIONS,
    validate_generated_record_shape,
)


class FakeGate:
    def prompt_constraints(self, _text: str) -> dict:
        return {
            "segments": [
                {
                    "content_lemmas": [
                        "ADV:also:count=1",
                        "PROPN:uruguay:count=1",
                        "VERB:win:count=1",
                    ]
                }
            ]
        }


class FakeTokenizer:
    def __call__(self, text: str, *, add_special_tokens: bool) -> dict:
        assert not add_special_tokens
        return {"input_ids": [ord(character) for character in text.casefold()]}


def test_constraints_keep_protected_spans_without_forcing_every_lemma() -> None:
    phrases = lexical_constraint_phrases("Uruguay also won in 1930.", FakeGate())

    assert phrases == ["1930", "Uruguay"]


def test_force_word_ids_deduplicates_equal_token_sequences() -> None:
    assert force_word_ids(FakeTokenizer(), ["Uruguay", "uruguay", "win"]) == [
        [ord(character) for character in "uruguay"],
        [ord(character) for character in "win"],
    ]


def test_backtranslation_protocol_is_valid_generation_provenance() -> None:
    assert PROTECTED_BACKTRANSLATION_PROMPT_VERSION in SUPPORTED_PROMPT_VERSIONS
    validate_generated_record_shape(
        {
            "claim_id": "c1",
            "source_record_id": "s1",
            "model": MODEL_IDENTITY,
            "prompt_version": PROTECTED_BACKTRANSLATION_PROMPT_VERSION,
            "candidates_requested": 2,
            "required_validated": 1,
            "original": "Original evidence.",
            "paraphrases": ["Evidence original.", "The evidence is original."],
        }
    )
