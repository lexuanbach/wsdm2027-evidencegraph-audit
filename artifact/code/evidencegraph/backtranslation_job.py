"""Append-only, local constrained backtranslation for hard paraphrase cases."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from .dataset_builder import read_jsonl
from .paraphrase_job import (
    PROTECTED_BACKTRANSLATION_PROMPT_VERSION,
    candidate_requirements,
    combine_segment_paraphrases,
    load_completed,
    select_pending,
    source_segments,
)
from .semantic_validation import protected_spans
from .structural_validation import SpacyLexicallyAnchoredGate
from .wiki_index import WikiIndex


FORWARD_MODEL = "Helsinki-NLP/opus-mt-en-de"
FORWARD_REVISION = "6183067f769a302e3861815543b9f312c71b0ca4"
BACKWARD_MODEL = "Helsinki-NLP/opus-mt-de-en"
BACKWARD_REVISION = "1a922f3b32a8e809e17a47d4b32142d8105924e5"
MODEL_IDENTITY = (
    f"{FORWARD_MODEL}@{FORWARD_REVISION}+"
    f"{BACKWARD_MODEL}@{BACKWARD_REVISION}"
)


def lexical_constraint_phrases(
    original: str, gate: SpacyLexicallyAnchoredGate
) -> list[str]:
    """Constrain high-risk spans without over-constraining fluent decoding."""
    checklist = gate.prompt_constraints(original)
    segments = checklist.get("segments", [])
    if len(segments) != 1:
        raise ValueError("constraint extraction expects exactly one evidence segment")
    phrases = set(protected_spans(original))
    for entry in segments[0].get("phonetic_spans_copy_exactly", []):
        phrase = str(entry).rsplit(":count=", 1)[0].strip()
        if phrase:
            phrases.add(phrase)
    return sorted(phrases, key=lambda value: (value.casefold(), value))


def force_word_ids(tokenizer: Any, phrases: list[str]) -> list[list[int]]:
    """Encode nonempty phrase constraints for Transformers constrained decoding."""
    encoded: list[list[int]] = []
    seen: set[tuple[int, ...]] = set()
    for phrase in phrases:
        token_ids = tokenizer(phrase, add_special_tokens=False)["input_ids"]
        values = tuple(int(value) for value in token_ids)
        if values and values not in seen:
            encoded.append(list(values))
            seen.add(values)
    return encoded


class MarianRoundTrip:
    """Pinned English--German--English local translation pair."""

    def __init__(self, *, device: str = "cpu", local_files_only: bool = False) -> None:
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        self.torch = torch
        self.device = device
        self.forward_tokenizer = AutoTokenizer.from_pretrained(
            FORWARD_MODEL,
            revision=FORWARD_REVISION,
            local_files_only=local_files_only,
        )
        self.forward_model = AutoModelForSeq2SeqLM.from_pretrained(
            FORWARD_MODEL,
            revision=FORWARD_REVISION,
            local_files_only=local_files_only,
        ).to(device)
        self.backward_tokenizer = AutoTokenizer.from_pretrained(
            BACKWARD_MODEL,
            revision=BACKWARD_REVISION,
            local_files_only=local_files_only,
        )
        self.backward_model = AutoModelForSeq2SeqLM.from_pretrained(
            BACKWARD_MODEL,
            revision=BACKWARD_REVISION,
            local_files_only=local_files_only,
        ).to(device)
        self.forward_model.eval()
        self.backward_model.eval()

    def _inputs(self, tokenizer: Any, text: str) -> dict[str, Any]:
        encoded = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        )
        return {key: value.to(self.device) for key, value in encoded.items()}

    def candidates(
        self,
        original: str,
        *,
        count: int,
        constraints: list[str],
        max_new_tokens: int = 256,
    ) -> list[str]:
        if count < 1:
            raise ValueError("candidate count must be positive")
        with self.torch.inference_mode():
            pivot_ids = self.forward_model.generate(
                **self._inputs(self.forward_tokenizer, original),
                num_beams=4,
                do_sample=False,
                max_new_tokens=max_new_tokens,
                renormalize_logits=True,
            )
        pivot = self.forward_tokenizer.decode(
            pivot_ids[0], skip_special_tokens=True
        )
        positive_constraints = force_word_ids(self.backward_tokenizer, constraints)
        beams = max(4, count * 2)
        with self.torch.inference_mode():
            output_ids = self.backward_model.generate(
                **self._inputs(self.backward_tokenizer, pivot),
                num_beams=beams,
                num_return_sequences=count,
                do_sample=False,
                force_words_ids=positive_constraints or None,
                no_repeat_ngram_size=3,
                early_stopping=True,
                max_new_tokens=max_new_tokens,
                renormalize_logits=True,
            )
        return self.backward_tokenizer.batch_decode(
            output_ids, skip_special_tokens=True
        )


def generate_backtranslations(
    selection_path: Path,
    index_path: Path,
    output_path: Path,
    *,
    device: str = "cpu",
    limit: int | None = None,
    local_files_only: bool = False,
) -> dict[str, int]:
    """Generate hard-case alternatives without changing the acceptance gate."""
    selection = read_jsonl(selection_path)
    completed = load_completed(
        output_path,
        expected_prompt_version=PROTECTED_BACKTRANSLATION_PROMPT_VERSION,
    )
    pending = select_pending(selection, completed, limit)
    translator = MarianRoundTrip(
        device=device, local_files_only=local_files_only
    )
    gate = SpacyLexicallyAnchoredGate("en_core_web_sm")
    index = WikiIndex(index_path)
    generated = failures = 0
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with output_path.open("a", encoding="utf-8") as handle:
            for record in pending:
                started = time.perf_counter()
                try:
                    segments = source_segments(record, index)
                    candidates_requested, required_validated = candidate_requirements(
                        record
                    )
                    parsed: dict[str, list[str]] = {}
                    segment_ids: list[str] = []
                    for segment_index, segment in enumerate(segments):
                        segment_id = (
                            f"{record['claim_id']}::segment-{segment_index}"
                        )
                        segment_ids.append(segment_id)
                        parsed[segment_id] = translator.candidates(
                            segment,
                            count=candidates_requested,
                            constraints=lexical_constraint_phrases(segment, gate),
                        )
                    output = {
                        "claim_id": str(record["claim_id"]),
                        "source_record_id": record["source_record_id"],
                        "model": MODEL_IDENTITY,
                        "prompt_version": PROTECTED_BACKTRANSLATION_PROMPT_VERSION,
                        "candidates_requested": candidates_requested,
                        "required_validated": required_validated,
                        "max_tokens_requested": 256,
                        "original": "\n".join(segments),
                        "paraphrases": combine_segment_paraphrases(
                            parsed, segment_ids, candidates_requested
                        ),
                        "status": "generated",
                        "prompt_tokens": 0,
                        "completion_tokens": 0,
                        "batch_wall_seconds": time.perf_counter() - started,
                    }
                    handle.write(
                        json.dumps(output, ensure_ascii=False, sort_keys=True) + "\n"
                    )
                    handle.flush()
                    generated += 1
                except Exception as error:
                    failure = {
                        "claim_ids": [str(record["claim_id"])],
                        "model": MODEL_IDENTITY,
                        "prompt_version": PROTECTED_BACKTRANSLATION_PROMPT_VERSION,
                        "status": "schema_failure",
                        "error_type": type(error).__name__,
                        "error_message": str(error)[:300],
                        "max_tokens_requested": 256,
                    }
                    handle.write(json.dumps(failure, sort_keys=True) + "\n")
                    handle.flush()
                    failures += 1
    finally:
        index.close()
    return {"generated": generated, "schema_failures": failures}
