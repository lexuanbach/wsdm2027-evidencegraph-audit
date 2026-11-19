"""Retrospective frozen audit; numeric-only inputs, no private blinding keys.
Report every fixed threshold; do not select a new threshold on these labels.
"""
from pathlib import Path
import argparse, itertools, json
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, adjusted_rand_score

ROOT = Path(__file__).resolve().parents[1]

def scores(items, field, threshold):
    y = [r["gold"] for r in items]; p = [r[field] >= threshold for r in items]
    pr, rc, f, _ = precision_recall_fscore_support(y, p, average="binary", zero_division=0)
    return {"n": len(y), "precision": float(pr), "recall": float(rc), "f1": float(f),
            "false_merges": sum(a == 0 and b for a, b in zip(y, p)),
            "missed_links": sum(a == 1 and not b for a, b in zip(y, p))}

def cluster_scores(items, threshold):
    aris = []; merges = misses = 0
    for item in items:
        gold = item["gold"]; parent = list(range(len(gold)))
        def find(a):
            while parent[a] != a: parent[a] = parent[parent[a]]; a = parent[a]
            return a
        for pair in item["pairs"]:
            a, b = pair["labels"]
            if pair["prob"] >= threshold: parent[find(a)] = find(b)
        pred = [find(a) for a in range(len(gold))]
        aris.append(adjusted_rand_score(gold, pred))
        for i, j in itertools.combinations(range(len(gold)), 2):
            merges += int(pred[i] == pred[j] and gold[i] != gold[j])
            misses += int(pred[i] != pred[j] and gold[i] == gold[j])
    return {"n_episodes": len(aris), "mean_ari": float(np.mean(aris)),
            "false_merged_pairs": merges, "false_split_pairs": misses}

def analyze(data):
    out = {"status": "retrospective descriptive sensitivity; no threshold selected on audit labels",
           "lineage_thresholds": [.3, .4, .5, .6, .7], "reuse_thresholds": [.7, .8, .9], "scopes": {}}
    for name in ("natural", "constructed"):
        edges = [r for r in data["pairs"] if r["scope"] == name]
        clusters = [r for r in data["clusters"] if r["scope"] == name]
        out["scopes"][name] = {
            "lineage": {str(t): {**scores(edges, "model_dependence_prob", t), **cluster_scores(clusters, t)} for t in out["lineage_thresholds"]},
            "text_reuse": {str(t): scores(edges, "text_reuse_score", t) for t in out["reuse_thresholds"]}}
    for scope, f, ari in (("natural", .568, .401), ("constructed", .967, 1.0)):
        v = out["scopes"][scope]["lineage"]["0.5"]
        assert round(v["f1"], 3) == f and round(v["mean_ari"], 3) == ari, (scope, v)
    return out

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--inputs", type=Path, default=ROOT / "revision/threshold-audit-inputs.json")
    p.add_argument("--output", type=Path, default=ROOT / "revision/threshold-sensitivity.json")
    args = p.parse_args()
    if not args.inputs.is_file(): p.error("numeric audit inputs missing; obtain the checksum-bound audit package")
    result = analyze(json.loads(args.inputs.read_text()))
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print("Reproduced both scopes and every threshold; natural F1=.568 / ARI=.401 at q=.5")
