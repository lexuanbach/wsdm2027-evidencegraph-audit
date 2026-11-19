"""Fail closed before covariance analysis or model generation; no network calls."""
from pathlib import Path
import argparse, hashlib, json

ROOT = Path(__file__).resolve().parents[1]
REQUIRED = {
    "covariance": ["data/processed/evidencegraph20k_validated_subset.jsonl",
                   "runs/subset_edge_episode/nli/deberta_large_dev.jsonl",
                   "runs/subset_edge_episode/nli/deberta_large_test.jsonl",
                   "runs/subset_edge_episode/dependence/predictions.jsonl"],
    "duplication": ["data/processed/evidencegraph20k_selection.jsonl"],
}

def inspect(root=ROOT):
    result = {}
    for branch, paths in REQUIRED.items():
        inputs = [{"path": rel, "exists": (root / rel).is_file(),
                   "sha256": hashlib.sha256((root / rel).read_bytes()).hexdigest() if (root / rel).is_file() else None} for rel in paths]
        result[branch] = {"inputs": inputs, "status": "blocked_missing_inputs" if any(not i["exists"] for i in inputs) else "requires_provenance_validation"}
    result["duplication"]["measurement_gate"] = "Original frozen protocol uses TRUE/FALSE token log-probabilities. Do not substitute self-reported probability from Anthropic Messages. Requires an explicitly separate, frozen protocol if confidence measurement changes."
    result["covariance"]["weights_required"] = False
    result["covariance"]["qualification_gate"] = "Before fitting: verify original hashes, source/claim joins, retained split counts 1568/359/371, and disjoint claim IDs across splits; fit only dev."
    return result

def validate_splits(rows):
    claims = {}
    for r in rows:
        split = r["split"]
        if split not in {"train", "dev", "test"}: raise ValueError("unknown split")
        claim = str(r["claim_id"])
        if claim in claims and claims[claim] != split: raise ValueError("claim appears across splits")
        claims[claim] = split
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--output", type=Path)
    args = parser.parse_args(); report = inspect()
    if args.output: args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    raise SystemExit(2 if any(r["status"] == "blocked_missing_inputs" for r in report.values()) else 0)
