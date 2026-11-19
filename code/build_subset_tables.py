"""Render the manuscript tables from the checksum-bound JSON summaries.

Tables are written only from complete summaries and every rendered file is
bound to its sources in ``artifact/results/subset_tables_manifest.json``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from evidencegraph.result_registry import source_descriptor


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SUB = PROJECT_ROOT / "artifact/runs/subset_edge_episode"
TABLE_DIR = PROJECT_ROOT / "manuscript/tables"
CORRUPTIONS = (
    ("duplication", "Dup."),
    ("hidden_best_evidence", "Hidden-best"),
    ("support_refute_mix", "Mixture"),
)


def _load(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _pct(value: float) -> str:
    return f"{100 * value:.1f}"


def _ci(value: float, low: float, high: float) -> str:
    return f"{100 * value:.1f} [{100 * low:.1f}, {100 * high:.1f}]"


def edge_table(nli: dict[str, Any], audit: dict[str, Any]) -> str:
    """Render the human-audited lineage transfer table.

    The earlier table emphasized held-out construction pairs and proxy stress.
    The completed two-rater audit now supplies the stronger reference for the
    paper's natural-lineage claim.
    """

    edge_order = (
        ("learned_lineage", r"\textbf{\system{} lineage}"),
        ("text_reuse", "Text-reuse clustering"),
        ("domain_heuristic", "Observed-origin identity"),
    )
    lines = [
        r"\begin{table*}[t]",
        r"\centering\footnotesize\setlength{\tabcolsep}{3pt}",
        r"\caption{Lineage estimation against human labels. Intervals use "
        r"episode-level bootstrap resampling.}",
        r"\label{tab:edges}",
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        r"& \multicolumn{3}{c}{Natural pairs ($n=469$)} & Constructed & Clusters \\",
        r"\cmidrule(lr){2-4}\cmidrule(lr){5-5}\cmidrule(lr){6-6}",
        r"System & Precision & Recall & F1 [95\% CI] & F1 [95\% CI] & ARI [95\% CI] \\",
        r"\midrule",
    ]
    natural_edges = audit["edges"]["natural"]["systems"]
    constructed_edges = audit["edges"]["constructed"]["systems"]
    natural_clusters = audit["clusters"]["natural"]["systems"]
    for key, label in edge_order:
        natural = natural_edges[key]
        constructed = constructed_edges[key]
        cluster = natural_clusters[key]
        f1 = natural["ci"]["f1"]
        constructed_f1 = constructed["ci"]["f1"]
        ari = cluster["ci"]["ari"]
        row = [
            label,
            f"{natural['point']['precision']:.3f}",
            f"{natural['point']['recall']:.3f}",
            f"{f1[0]:.3f} [{f1[1]:.3f},{f1[2]:.3f}]",
            f"{constructed_f1[0]:.3f} [{constructed_f1[1]:.3f},{constructed_f1[2]:.3f}]",
            f"{ari[0]:.3f} [{ari[1]:.3f},{ari[2]:.3f}]",
        ]
        lines.append(" & ".join(row) + r" \\")
    iaa_edges = audit["iaa"]["edges"]["pooled"]
    iaa_clusters = audit["iaa"]["clusters"]
    nli_overall = nli["models"]["DeBERTa-large"]["groups"]["overall"]
    constructed_cluster = audit["clusters"]["constructed"]["systems"]["learned_lineage"]
    note = (
        r"Two-rater agreement: pair $\kappa="
        + f"{iaa_edges['kappa_3way']:.3f}"
        + r"$ ($n=128$), partition ARI $="
        + f"{iaa_clusters['mean_ari']:.3f}"
        + r"$ ($n=12$). \system{} constructed-cluster ARI $="
        + f"{constructed_cluster['point']['ari']:.3f}"
        + r"$. Frozen NLI stance macro-F1 $="
        + f"{100 * nli_overall['macro_f1']:.1f}"
        + f"$ and ECE $={nli_overall['ece']:.3f}"
        + r"$ on 7{,}106 construction edges."
    )
    lines += [
        r"\bottomrule",
        r"\end{tabular}\par\vspace{3pt}",
        r"{\footnotesize\raggedright",
        note + r"\par}",
        r"\end{table*}",
        "",
    ]
    return "\n".join(lines)


def corpus_table(audit: dict[str, Any]) -> str:
    """Render the per-corpus decomposition of the natural-lineage audit.

    Table~\\ref{tab:edges} reports natural pairs POOLED. Pooling hides the fact
    that the three natural corpora differ in what "origin" even means: AVeriTeC
    records a real registered URL per source, while FEVER and HoVer expose only
    a Wikipedia page id. This table separates them, and pairs each corpus with
    the two raters' agreement on it, so the reader can see that the estimator
    tracks how observable lineage is rather than failing uniformly. Constructed
    pairs are omitted here because Table~\\ref{tab:edges} already reports them.
    """

    order = (
        ("AVeriTeC", "AVeriTeC (URL)"),
        ("FEVER", "FEVER (page id)"),
        ("HoVer", "HoVer (page id)"),
    )
    systems = (
        ("learned_lineage", r"\textbf{\system{}}"),
        ("text_reuse", "Reuse"),
        ("domain_heuristic", "Origin ID"),
    )
    by_corpus = audit["by_corpus"]
    agreement = audit["iaa"]["edges"]["pairs"]["A|B"]["by_corpus"]
    lines = [
        r"\begin{table}[t]",
        r"\centering\footnotesize\setlength{\tabcolsep}{3pt}",
        r"\caption{Natural-lineage recovery by corpus, decomposing the pooled "
        r"natural column of Table~\ref{tab:edges}. $\kappa$ is the two raters' "
        r"agreement on that corpus and \emph{Same origin} its audited same-origin rate. "
        r"Edge F1 is reported per corpus.}",
        r"\label{tab:corpus}",
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        r"& & & \multicolumn{3}{c}{Edge F1} \\",
        r"\cmidrule(lr){4-6}",
        r"Corpus & $n$ & Same origin & " + " & ".join(label for _, label in systems) + r" \\",
        r"\midrule",
    ]
    for key, label in order:
        block = by_corpus[key]["systems"]
        row = [
            label,
            f"{by_corpus[key]['n_decided']}",
            f"{by_corpus[key]['positive_rate']:.2f}",
        ] + [f"{block[s]['f1']:.3f}" for s, _ in systems]
        lines.append(" & ".join(row) + r" \\")
    kap = " ".join(
        f"{label.split(' ')[0]} $\\kappa={agreement[key]['kappa_3way']:.3f}$;"
        for key, label in order
    ).rstrip(";")
    lines += [
        r"\bottomrule",
        r"\end{tabular}\par\vspace{3pt}",
        r"{\footnotesize\raggedright",
        "Two-rater agreement by corpus: " + kap + r"\par}",
        r"\end{table}",
        "",
    ]
    return "\n".join(lines)


def stratum_table(audit: dict[str, Any]) -> str:
    """Render what the audit design can and cannot resolve.

    The sample is equal-allocation across corpus x dependence stratum, and the
    stratum variable is the estimator's OWN mean predicted probability. The
    population is concentrated almost entirely in the lowest-score stratum, so
    the audit resolves the region where the estimator fires and cannot resolve
    the region where it stays silent. Showing population share beside achieved
    positives is the honest way to state that scope.
    """

    order = (
        ("q1_lowest", "$[0.0,0.2)$"),
        ("q2_low_mid", "$[0.2,0.5)$"),
        ("q3_high_mid", "$[0.5,0.8)$"),
        ("q4_highest", "$[0.8,1.0]$"),
    )
    natural = ("AVeriTeC", "FEVER", "HoVer")
    cells = audit["by_corpus_stratum"]
    agg: dict[str, dict[str, float]] = {}
    for name, cell in cells.items():
        corpus, stratum = name.split("|")
        if corpus not in natural:
            continue
        acc = agg.setdefault(stratum, {"pop": 0, "n": 0, "pos": 0, "tp": 0.0})
        acc["pop"] += cell["population"] or 0
        acc["n"] += cell["n_decided"]
        acc["pos"] += cell["n_positive"]
        recall = cell["systems"].get("learned_lineage", {}).get("recall")
        if recall is not None and recall == recall:
            acc["tp"] += recall * cell["n_positive"]
    total = sum(agg[key]["pop"] for key, _ in order)
    lines = [
        r"\begin{table}[t]",
        r"\centering\footnotesize\setlength{\tabcolsep}{3pt}",
        r"\caption{What the natural-lineage audit can resolve. Strata are bins of "
        r"the estimator's own mean predicted probability, so the sample is dense "
        r"where it fires and sparse where it stays silent. \emph{Share} is the share "
        r"of all natural pairs in that bin and \emph{Same origin} counts the same-origin pairs the "
        r"audit actually observed there.}",
        r"\label{tab:stratum}",
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Score bin & Share & Audited & Same origin & Recall \\",
        r"\midrule",
    ]
    for key, label in order:
        acc = agg[key]
        recall = acc["tp"] / acc["pos"] if acc["pos"] else float("nan")
        lines.append(
            " & ".join([
                label,
                f"{100 * acc['pop'] / total:.1f}\\%",
                f"{int(acc['n'])}",
                f"{int(acc['pos'])}",
                "--" if recall != recall else f"{recall:.2f}",
            ]) + r" \\"
        )
    lines += [
        r"\bottomrule",
        r"\end{tabular}\par\vspace{3pt}",
        r"{\footnotesize\raggedright",
        f"All {total:,} natural pairs. ".replace(",", "{,}")
        + r"The lowest bin holds most of the population but yielded two audited "
          r"same-origin pairs, so population-level recall is not identified by this "
          r"design, and a population-level rate would require design weights."
        + r"\par}",
        r"\end{table}",
        "",
    ]
    return "\n".join(lines)


def ablation_table(rollout: dict[str, Any], bounded: dict[str, Any]) -> str:
    order = (
        ("fixed_top5", "Fixed top-5 (count)"),
        ("no_dependence", "Active, no lineage"),
        ("evidencegraph_active", "\\textbf{\\system{} active}"),
        ("hard_dependence", "\\quad hard lineage"),
        ("no_effective_evidence", "\\quad no $\\sqrt{\\weff}$ scaling"),
        ("no_contradiction", "\\quad no contradiction"),
        ("no_active_seek", "\\quad no active seek"),
    )
    lines = [
        r"\begin{table*}[t]",
        r"\centering\footnotesize\setlength{\tabcolsep}{3.6pt}",
        r"\caption{Matched controller ablations on 371 test episodes. All policies "
        r"start with five passages and may add three. Accuracy and false acceptance "
        r"(False acc.) are percentages.}",
        r"\label{tab:ablation}",
        r"\begin{tabular}{l rrr rrr rrr r}",
        r"\toprule",
        r"& \multicolumn{3}{c}{Overall (frozen)} & \multicolumn{3}{c}{Overall (bounded)} & "
        r"\multicolumn{3}{c}{Accuracy by condition (bounded)} & \\",
        r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}\cmidrule(lr){8-10}",
        r"Method & Accuracy & ECE & False acc. & Accuracy [95\% CI] & ECE & False acc. & Copies & Hidden & Mixed & Passages \\",
        r"\midrule",
    ]
    froz = rollout["methods"]
    bnd = bounded["rollouts"]["methods"]
    docs = bounded.get("_docs", {})
    for key, label in order:
        f = froz[key]["test"]["groups"]["overall"]
        b = bnd[key]["test"]["groups"]
        o = b["overall"]
        row = [
            label,
            _pct(f["accuracy"]),
            f"{f['ece']:.3f}",
            _pct(f["false_acceptance"]),
            f"{_pct(o['accuracy'])} [{_pct(o['accuracy_low'])}, {_pct(o['accuracy_high'])}]",
            f"{o['ece']:.3f}",
            _pct(o["false_acceptance"]),
        ]
        row += [_pct(b[c]["accuracy"]) for c, _ in CORRUPTIONS]
        row.append(f"{docs.get(key, float('nan')):.1f}")
        lines.append(" & ".join(row) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""]
    return "\n".join(lines)


def baseline_table(base: dict[str, Any]) -> str:
    order = (
        ("count_sum", "Count (no dependence)"),
        ("domain_dedup_vote", "Domain dedup vote"),
        ("near_dup_cluster_vote", "Near-dup cluster vote"),
        ("semantic_dedup_vote", "Semantic dedup vote (SemDeDup)"),
        ("semantic_lexical_union_vote", "Semantic $\\cup$ near-dup vote"),
        ("learned_hard_dependence", "Learned hard lineage"),
        ("learned_soft_dependence", "\\textbf{\\system{} soft GLS}"),
        ("oracle_origin_vote", "Recorded-origin vote"),
        ("oracle_soft_dependence", "Recorded-origin soft GLS"),
        ("conflict_aware_vote", "Conflict-aware transparent vote"),
        ("conflict_aware_xbackbone", "Conflict-aware + cross-backbone"),
    )
    lines = [
        r"\begin{table}[t]",
        r"\centering\scriptsize\setlength{\tabcolsep}{2.4pt}",
        r"\caption{Aggregation on 371 held-out construction episodes. Differences "
        r"compare \system{} soft GLS with each rule. Acc. is accuracy, False acc. is false "
        r"acceptance, and True and Mixed are accuracy on true and mixed claims, all in percentages.}",
        r"\label{tab:baselines}",
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        r"Rule & Acc. & Ours minus rule [CI] & ECE & False acc. & True & Mixed \\",
        r"\midrule",
    ]
    for key, label in order:
        # Rows are emitted only for variants the run actually scored, so a
        # summary produced before an arm existed still renders.
        if key not in base["variants"]:
            continue
        blk = base["variants"][key]
        g = blk["test"]["groups"]["overall"]
        if key == base["reference"]:
            delta = "n/a"
        else:
            p = base["paired_versus_reference"][key]["overall"]["accuracy"]
            delta = f"{100 * p['difference']:+.1f} [{100 * p['low']:+.1f}, {100 * p['high']:+.1f}]"
        row = [
            label,
            _pct(g["accuracy"]),
            delta,
            f"{g['ece']:.3f}",
            _pct(g["false_acceptance"]),
            _pct(blk["test_true_claim_accuracy"]),
            _pct(blk["test"]["groups"]["support_refute_mix"]["accuracy"]),
        ]
        lines.append(" & ".join(row) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    return "\n".join(lines)


def benchmark_table(bench: dict[str, Any], override: dict[str, Any]) -> str:
    names = (("fever_dev", "FEVER"), ("hover_dev", "HoVer"), ("averitec_dev", "AVeriTeC"))
    methods = (
        ("closed_book", "Closed-book"),
        ("fixed_rag", "Fixed top-5 RAG"),
        ("evidencegraph_llm_verdict", "\\textbf{\\system{} context}"),
        ("evidencegraph", "\\textbf{\\system{} with override}"),
    )
    # Selection is compared with fixed RAG; the override is compared directly
    # with selection on every paired generation available for that contrast.
    delta_methods = {"evidencegraph_llm_verdict"}
    lines = [
        r"\begin{table*}[t]",
        r"\centering\footnotesize\setlength{\tabcolsep}{4pt}",
        r"\renewcommand{\arraystretch}{0.93}",
        r"\captionsetup{skip=4pt}",
        r"\caption{End-to-end results on frozen retrieval. Context deltas use "
        r"fixed RAG, while override deltas use context.}",
        r"\label{tab:benchmarks}",
        r"\begin{tabular*}{0.86\textwidth}{@{\extracolsep{\fill}}lrrrr}",
        r"\toprule",
        r"Method & Accuracy [95\% CI] & F1 & ECE & False support \\",
    ]
    for key, name in names:
        if key not in bench["datasets"]:
            continue
        block = bench["datasets"][key]
        lines.append(r"\midrule")
        lines.append(
            rf"\multicolumn{{5}}{{l}}{{\emph{{{name}}} ($n$={block['n_cases_compared']})}} \\"
        )
        for mkey, mlabel in methods:
            if mkey not in block["methods"]:
                # Never reuse "not run at this scale" here: \S5 uses that exact
                # phrase for ReAct and CoVe, which genuinely were never run, so
                # reusing it for a cell that merely missed this build would
                # misdescribe the artifact.
                lines.append(
                    rf"{mlabel} & \multicolumn{{4}}{{l}}{{cell excluded from this build}} \\"
                )
                continue
            m = block["methods"][mkey]
            lines.append(
                " & ".join(
                    [
                        mlabel,
                        f"{_pct(m['accuracy'])} [{_pct(m['accuracy_low'])}, {_pct(m['accuracy_high'])}]",
                        _pct(m["macro_f1"]),
                        f"{m['ece']:.3f}",
                        _pct(m["false_support_rate"]),
                    ]
                )
                + r" \\"
            )
            if mkey in delta_methods and mkey in block["paired_versus_reference"]:
                p = block["paired_versus_reference"][mkey]
                cells = []
                for metric in ("accuracy", "macro_f1", "ece", "false_support_rate"):
                    if metric not in p:
                        cells.append("n/a")
                        continue
                    scale = 1.0 if metric == "ece" else 100.0
                    fmt = "{:+.3f}" if metric == "ece" else "{:+.1f}"
                    if metric == "macro_f1":
                        # Point value only: the interval is the widest cell in
                        # the table and macro-F1 is secondary throughout.
                        cells.append(fmt.format(scale * p[metric]["difference"]))
                        continue
                    cells.append(
                        fmt.format(scale * p[metric]["difference"])
                        + " ["
                        + fmt.format(scale * p[metric]["low"])
                        + ","
                        + fmt.format(scale * p[metric]["high"])
                        + "]"
                    )
                lines.append(
                    " & ".join([r"\quad $\Delta$ context"] + cells) + r" \\"
                )
            if mkey == "evidencegraph" and key in override["datasets"]:
                direct = override["datasets"][key]
                p = direct["paired_versus_reference"]["evidencegraph"]
                cells = []
                for metric in ("accuracy", "macro_f1", "ece", "false_support_rate"):
                    scale = 1.0 if metric == "ece" else 100.0
                    fmt = "{:+.3f}" if metric == "ece" else "{:+.1f}"
                    if metric == "macro_f1":
                        cells.append(fmt.format(scale * p[metric]["difference"]))
                    else:
                        cells.append(
                            fmt.format(scale * p[metric]["difference"])
                            + " ["
                            + fmt.format(scale * p[metric]["low"])
                            + ","
                            + fmt.format(scale * p[metric]["high"])
                            + "]"
                        )
                label = r"\quad $\Delta$ override"
                lines.append(" & ".join([label] + cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular*}", r"\end{table*}", ""]
    return "\n".join(lines)


def natural_lineage_table(nat: dict[str, Any]) -> str:
    order = (
        ("passive_rank", "Passive next-rank"),
        ("mmr_text", "MMR diversity ($\\lambda$=0.75)"),
        ("neardup_diverse", "Near-duplicate diverse (hard)"),
        ("observed_origin_diverse", "Observed-origin diverse (hard)"),
        ("learned_hard_dependence", "Learned hard dependence"),
        ("learned_soft_find_indep", "Learned soft \\act{find\\_indep}"),
        ("oracle_gold_ceiling", "Gold-seeking oracle (ceiling)"),
    )
    datasets = (("averitec", "AVeriTeC"), ("fever", "FEVER"), ("hover", "HoVer"))
    lines = [
        r"\begin{table}[t]",
        r"\centering\scriptsize\setlength{\tabcolsep}{1.5pt}",
        r"\renewcommand{\arraystretch}{0.94}",
        r"\caption{Acquisition on \emph{observed} Web lineage: each policy spends three "
        r"steps over the same frozen top-20 pool. Cells: $\Delta$ gold-evidence recall in "
        r"points vs.\ passive next-rank (95\% paired bootstrap). "
        r"$\bar o$ = mean distinct origins held.}",
        r"\label{tab:natural}",
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        r"& \multicolumn{2}{c}{AVeriTeC ($n$=500)} & \multicolumn{2}{c}{FEVER ($n$=348)} & "
        r"\multicolumn{2}{c}{HoVer ($n$=500)} \\",
        r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}",
        r"Acquisition policy & $\Delta$rec & $\bar o$ & $\Delta$rec & $\bar o$ & $\Delta$rec & $\bar o$ \\",
        r"\midrule",
    ]
    base = []
    for key, _ in datasets:
        block = nat["datasets"][key]["item_recall"]["passive_rank"]
        origins = nat["datasets"][key]["distinct_origins"]["passive_rank"]
        base.extend([f"{100 * block['mean']:.1f}", f"{origins['mean']:.2f}"])
    lines.append("Passive next-rank (absolute) & " + " & ".join(base) + r" \\")
    lines.append(r"\midrule")
    for policy, label in order:
        if policy == "passive_rank":
            continue
        cells = []
        for key, _ in datasets:
            block = nat["datasets"][key]
            d = block["item_recall"][policy]["difference_vs_reference"]
            origins = block["distinct_origins"][policy]["mean"]
            cells.append(
                f"{100 * d['difference']:+.1f} [{100 * d['low']:.1f},{100 * d['high']:.1f}]"
            )
            cells.append(f"{origins:.2f}")
        lines.append(f"{label} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmarks", type=Path, default=PROJECT_ROOT / "artifact/runs/benchmarks/haiku_subset/evaluation_summary.json")
    args = parser.parse_args()
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    sources: dict[str, list[Path]] = {}
    nli = _load(SUB / "nli/summary_test.json")
    human_audit_path = PROJECT_ROOT / "annotation/score/report_2026-08-24.json"
    human_audit = _load(human_audit_path)
    (TABLE_DIR / "table_edges.tex").write_text(edge_table(nli, human_audit), encoding="utf-8")
    sources["table_edges.tex"] = [SUB / "nli/summary_test.json", human_audit_path]
    (TABLE_DIR / "table_corpus.tex").write_text(corpus_table(human_audit), encoding="utf-8")
    sources["table_corpus.tex"] = [human_audit_path]
    (TABLE_DIR / "table_stratum.tex").write_text(stratum_table(human_audit), encoding="utf-8")
    sources["table_stratum.tex"] = [human_audit_path]

    rollout = _load(SUB / "controller_evaluation/summary.json")
    bounded = _load(SUB / "bounded_precision/summary.json")
    # mean documents opened per method from the frozen rollouts
    docs: dict[str, list[int]] = {}
    with (SUB / "controller_evaluation/predictions.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            if record.get("split") == "test":
                docs.setdefault(str(record["method"]), []).append(int(record["documents_opened"]))
    bounded["_docs"] = {key: sum(values) / len(values) for key, values in docs.items()}
    (TABLE_DIR / "table_ablation.tex").write_text(
        ablation_table(rollout, bounded), encoding="utf-8"
    )
    sources["table_ablation.tex"] = [
        SUB / "controller_evaluation/summary.json",
        SUB / "controller_evaluation/predictions.jsonl",
        SUB / "bounded_precision/summary.json",
    ]
    base = _load(SUB / "episode_baselines/summary.json")
    (TABLE_DIR / "table_baselines.tex").write_text(baseline_table(base), encoding="utf-8")
    sources["table_baselines.tex"] = [SUB / "episode_baselines/summary.json"]
    if args.benchmarks.is_file():
        bench = _load(args.benchmarks)
        override_path = args.benchmarks.parent / "override_effect_summary.json"
        override = _load(override_path)
        (TABLE_DIR / "table_benchmarks.tex").write_text(
            benchmark_table(bench, override), encoding="utf-8"
        )
        sources["table_benchmarks.tex"] = [args.benchmarks, override_path]
    natural_path = PROJECT_ROOT / "artifact/runs/natural_lineage_acquisition/summary.json"
    if natural_path.is_file():
        (TABLE_DIR / "table_natural.tex").write_text(
            natural_lineage_table(_load(natural_path)), encoding="utf-8"
        )
        sources["table_natural.tex"] = [natural_path]
    manifest = {
        name: {
            "table": source_descriptor(TABLE_DIR / name, PROJECT_ROOT),
            "sources": [source_descriptor(path, PROJECT_ROOT) for path in paths],
        }
        for name, paths in sources.items()
    }
    out = PROJECT_ROOT / "artifact/results/subset_tables_manifest.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps({name: value["table"]["sha256"][:12] for name, value in manifest.items()}))


if __name__ == "__main__":
    main()
