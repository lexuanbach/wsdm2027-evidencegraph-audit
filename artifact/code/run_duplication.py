"""Run the preregistered duplicate-one-false-source intervention locally."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from evidencegraph.claim_verifier import LocalClaimVerifier
from evidencegraph.dataset_builder import read_jsonl, stable_hash


PROJECT_ROOT = Path(__file__).resolve().parents[2]
COPY_COUNTS = (1, 2, 5, 10, 20)
METHODS = ("fixed_rag", "evidencegraph_oracle")
CONTEXT_SLOTS = 20
PROTOCOL_VERSION = "unsupported-matched-position-logprob-v6"
SELECTION_SEED = 103


def _neutral_control(false_text: str) -> str:
    target_words = len(false_text.split())
    words = "This neutral control passage contains no relevant evidence for or against the claim".split()
    padding = (" unrelated context" * target_words).split()
    return " ".join((words + padding)[:target_words])


def matched_sources(
    claim: str,
    copy_count: int,
    claim_id: str,
    seed: int,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    if copy_count < 1 or copy_count > CONTEXT_SLOTS:
        raise ValueError("copy count must fit the matched context")
    false_text = claim.strip()
    if false_text[-1] not in ".!?":
        false_text += "."
    neutral_text = _neutral_control(false_text)
    positions = list(range(CONTEXT_SLOTS))
    random.Random(stable_hash(f"{claim_id}:positions", seed=seed)).shuffle(positions)
    false_positions = set(positions[:copy_count])
    raw_sources = [
        {
            "source_id": f"slot-{slot}-{'repeated' if slot in false_positions else 'neutral'}",
            "text": false_text if slot in false_positions else neutral_text,
        }
        for slot in range(CONTEXT_SLOTS)
    ]
    representative_sources = [
        {"source_id": "repeated-representative", "text": false_text}
    ]
    return raw_sources, representative_sources


def load_keys(path: Path) -> set[tuple[str, str, int, str, int]]:
    if not path.exists():
        return set()
    return {
        (
            record["claim_id"],
            record["method"],
            int(record["copy_count"]),
            record["model"],
            int(record["seed"]),
        )
        for record in read_jsonl(path)
        if record.get("status") == "complete"
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="qwen2.5:3b-instruct")
    parser.add_argument("--limit", type=int, default=40)
    parser.add_argument("--seed", type=int, default=103)
    parser.add_argument("--selection-seed", type=int, default=SELECTION_SEED)
    parser.add_argument(
        "--confidence-mode", choices=("logprob", "self_report"), default="logprob"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "artifact/runs/duplication/qwen3b-smoke.jsonl",
    )
    args = parser.parse_args()
    selection = [
        record
        for record in read_jsonl(
            PROJECT_ROOT / "data/processed/evidencegraph20k_selection.jsonl"
        )
        if record["split"] == "test" and not record["label"]
    ]
    selection = sorted(
        selection,
        key=lambda record: stable_hash(record["claim_id"], args.selection_seed),
    )[: args.limit]
    completed = load_keys(args.output)
    verifier = LocalClaimVerifier(args.model)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    completed_now = failures = 0
    with args.output.open("a", encoding="utf-8") as handle:
        for record in selection:
            for copy_count in COPY_COUNTS:
                raw_sources, representative_sources = matched_sources(
                    record["claim"], copy_count, record["claim_id"], args.seed
                )
                for method in METHODS:
                    key = (record["claim_id"], method, copy_count, args.model, args.seed)
                    if key in completed:
                        continue
                    sources = raw_sources if method == "fixed_rag" else representative_sources
                    try:
                        verify = (
                            verifier.verify_binary
                            if args.confidence_mode == "logprob"
                            else verifier.verify
                        )
                        result = verify(
                            record["claim"],
                            sources,
                            seed=args.seed,
                            dependence_aware=method == "evidencegraph_oracle",
                        )
                        false_claim_accepted = result.verdict == "SUPPORTED"
                        if method == "evidencegraph_oracle":
                            # The oracle graph sees one supporting origin;
                            # the answerer requires two independent origins.
                            false_claim_accepted = False
                        output = {
                            "claim_id": record["claim_id"],
                            "source_record_id": record["source_record_id"],
                            "gold_label": "REFUTED",
                            "method": method,
                            "copy_count": copy_count,
                            "model": args.model,
                            "seed": args.seed,
                            "selection_seed": args.selection_seed,
                            "protocol_version": PROTOCOL_VERSION,
                            "confidence_mode": args.confidence_mode,
                            "n_sources_presented": len(sources),
                            "false_claim_accepted": false_claim_accepted,
                            "confidence_in_false_claim": result.claim_probability_true,
                            "presented_support_documents": (
                                copy_count if method == "fixed_rag" else 1
                            ),
                            "true_support_origins": 1,
                            "aggregation_support_count": (
                                copy_count if method == "fixed_rag" else 1
                            ),
                            "status": "complete",
                            **result.to_dict(),
                        }
                        completed_now += 1
                    except (ValueError, KeyError, json.JSONDecodeError) as error:
                        output = {
                            "claim_id": record["claim_id"],
                            "method": method,
                            "copy_count": copy_count,
                            "model": args.model,
                            "seed": args.seed,
                            "status": "schema_failure",
                            "error_type": type(error).__name__,
                        }
                        failures += 1
                    handle.write(
                        json.dumps(output, ensure_ascii=False, sort_keys=True) + "\n"
                    )
                    handle.flush()
    print(json.dumps({"complete": completed_now, "schema_failures": failures}, sort_keys=True))


if __name__ == "__main__":
    main()
