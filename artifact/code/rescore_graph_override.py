"""Separate the graph's two end-to-end effects, without any new inference.

The EvidenceGraph answering protocol does two things at once: it changes *which*
passages the answer model sees (one representative per predicted dependence
cluster, after a budgeted controller rollout), and it *overrides* the model's
own verdict distribution with the graph's calibrated support-versus-best-
alternative probability (``apply_graph_calibrated_probability``).

Every EvidenceGraph prediction already stores the model's own verdict and
probabilities alongside the overridden ones (``raw_llm_label`` /
``raw_llm_probabilities``). This script re-scores those frozen rows onto the
model's own verdict, producing a third arm:

* ``fixed_rag``                  -- top-5 context, model's own verdict
* ``evidencegraph_llm_verdict``  -- graph-selected context, model's own verdict
* ``evidencegraph``              -- graph-selected context, graph override

so ``evidencegraph_llm_verdict - fixed_rag`` isolates the *selection* effect and
``evidencegraph - evidencegraph_llm_verdict`` isolates the *override* effect on
identical evidence. No model is called; the output is a deterministic function
of the existing file, and its provenance records the exact source digest.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from evidencegraph.dataset_builder import read_jsonl
from evidencegraph.provenance import sha256
from evidencegraph.result_registry import source_descriptor


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESCORING_PROTOCOL = "graph-selection-with-model-verdict-rescoring-v1"
METHOD = "evidencegraph_llm_verdict"


def rescore(records: list[dict[str, Any]], source: Path) -> list[dict[str, Any]]:
    digest = sha256(source)
    output: list[dict[str, Any]] = []
    for record in records:
        if record.get("status") != "complete":
            # Preserve failed attempts verbatim so the denominators of the
            # re-scored arm match the arm it is derived from exactly.
            output.append(
                {
                    **record,
                    "method": METHOD,
                    "answer_config": {
                        **record["answer_config"],
                        "method": METHOD,
                        "rescoring_protocol": RESCORING_PROTOCOL,
                        "rescored_from_sha256": digest,
                    },
                }
            )
            continue
        if record.get("abstained"):
            # A deterministic no-evidence abstention never consulted the model,
            # so there is no separate model verdict to restore.
            label = record["label"]
            probabilities = record["probabilities"]
        else:
            if "raw_llm_label" not in record or "raw_llm_probabilities" not in record:
                raise ValueError(
                    f"case {record.get('case_id')} lacks the stored model verdict"
                )
            label = str(record["raw_llm_label"])
            probabilities = {
                str(key): float(value)
                for key, value in record["raw_llm_probabilities"].items()
            }
            if abs(sum(probabilities.values()) - 1.0) > 1e-5:
                raise ValueError(f"case {record.get('case_id')} verdict does not sum to one")
            if label != max(probabilities, key=probabilities.get):
                raise ValueError(f"case {record.get('case_id')} verdict is not its argmax")
        rescored = {
            **record,
            "method": METHOD,
            "label": label,
            "probabilities": probabilities,
            "answer_config": {
                **record["answer_config"],
                "method": METHOD,
                "probability_policy": "frozen-llm-structured-probabilities",
                "rescoring_protocol": RESCORING_PROTOCOL,
                "rescored_from_sha256": digest,
            },
        }
        rescored.pop("raw_llm_label", None)
        rescored.pop("raw_llm_probabilities", None)
        output.append(rescored)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, action="append", required=True)
    parser.add_argument(
        "--summary",
        type=Path,
        default=PROJECT_ROOT
        / "artifact/runs/benchmarks/haiku_subset/override_rescoring_manifest.json",
    )
    args = parser.parse_args()
    manifest: dict[str, Any] = {
        "protocol": RESCORING_PROTOCOL,
        "note": "deterministic re-scoring of frozen predictions; no model was called",
        "outputs": [],
    }
    for path in args.predictions:
        records = read_jsonl(path)
        rescored = rescore(records, path)
        output = path.with_name(path.name.replace(
            "_evidencegraph_", "_evidencegraph_llm_verdict_"
        ))
        with output.open("w", encoding="utf-8") as handle:
            for record in rescored:
                handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        overridden = sum(
            1
            for record in records
            if record.get("status") == "complete"
            and not record.get("abstained")
            and str(record.get("raw_llm_label")) != str(record.get("label"))
        )
        complete = sum(1 for record in records if record.get("status") == "complete")
        manifest["outputs"].append(
            {
                "source": source_descriptor(path, PROJECT_ROOT),
                "output": source_descriptor(output, PROJECT_ROOT),
                "complete_rows": complete,
                "verdict_changed_by_override": overridden,
                "override_rate": overridden / complete if complete else None,
            }
        )
        print(
            f"{path.name}: {complete} complete, override flipped the verdict on "
            f"{overridden} ({100 * overridden / complete:.1f}%)"
        )
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    with args.summary.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")


if __name__ == "__main__":
    main()
