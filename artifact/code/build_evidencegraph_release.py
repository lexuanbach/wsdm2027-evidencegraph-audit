"""Stage the audited EvidenceGraph-20K corpus, scripts, and license notices."""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
import re
import shutil
from typing import Any

from evidencegraph.dataset_builder import read_jsonl
from evidencegraph.final_artifacts import (
    FINAL_COMBINED_VALIDATION,
    FINAL_WIKIDATA_AUDIT,
    FINAL_WIKIDATA_TARGETS,
    FINAL_WIKIDATA_TERMS,
)
from evidencegraph.result_registry import source_descriptor, verified_source_path
from evidencegraph.wikidata_snapshot import load_provenance_index


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RELEASE_NAME = "evidencegraph20k-v1"
TOP_LEVEL_RELEASE_SCRIPTS = {
    "apply_paraphrase_structural_gate.py",
    "audit_paraphrase_generation.py",
    "audit_paraphrase_validation.py",
    "build_evidencegraph_release.py",
    "build_dataset_registry.py",
    "build_wiki_index.py",
    "check_human_audit_completeness.py",
    "extract_wikidata_snapshot.py",
    "materialize_dataset.py",
    "merge_human_audit_ratings.py",
    "merge_paraphrase_validations.py",
    "preflight_evidence.py",
    "prepare_human_audit.py",
    "run_paraphrase_recovery_ladder.py",
    "run_paraphrase_reserve_recovery.py",
    "run_paraphrases.py",
    "run_deterministic_preordering.py",
    "run_human_audit_rater.py",
    "select_adaptive_reserves.py",
    "select_dataset.py",
    "select_reserves.py",
    "validate_human_audit.py",
    "validate_paraphrases.py",
}
EXCLUDED_PACKAGE_MODULE_TOKENS = (
    "bedrock",
)


ATTRIBUTION = """# EvidenceGraph-20K attribution and redistribution notes

EvidenceGraph-20K is derived from FEVER annotations and FEVER's June-2017
English Wikipedia snapshot. Preserve the original FEVER record, Wikipedia
page, and sentence identifiers embedded in each episode. FEVER states that
incorporated Wikipedia material follows the applicable Wikipedia article
terms, or CC BY-SA 3.0 where those terms are unavailable. See
`LICENSES/fever-data-license.html` and the source manifest.

Date-aligned structured provenance is extracted from the 2017-02-20 Wikidata
dump under Wikidata's CC0 declaration. Synthetic mirror, citation, paraphrase,
counter-source, and intervention records are explicitly marked in the corpus.

AVeriTeC, HoVer, and ExpertQA are evaluation inputs only and are intentionally
not redistributed in this package. Follow their official repositories and
licenses using the pinned URLs/commits in `metadata/source_manifest.json`.

This notice records upstream declarations and is not legal advice.
"""


CODE_NOTICE = """# Project-code notice

The Python files in `scripts/` are included to reproduce the academic
artifact. This data package does not grant a separate software license beyond
the license, if any, published with the project repository. Upstream dataset
and model code retain their own licenses and are not vendored here.
"""


WIKIDATA_NOTICE = """# Wikidata structured-provenance notice

The structured provenance in `provenance/` was extracted only from the
2017-02-20 Wikidata JSON dump. The source audit records the exact archived
URL, byte count, MD5, SHA-1, date, and CC0-1.0 declaration. See also:
https://www.wikidata.org/wiki/Wikidata:Licensing

This package does not substitute current Wikidata statements for missing
historical entities or terms.
"""


RELEASE_RUNBOOK = """# EvidenceGraph-20K v1 reproduction runbook

Run from the release root with Python 3.10. The distributed corpus is already
materialized and audited; `release_manifest.json` binds every included byte.

## Verify the release

Use SHA-256 to verify each `outputs` entry in `release_manifest.json`. The
manifest excludes model weights, API credentials, external benchmark corpora,
and the original multi-gigabyte Wikipedia/Wikidata archives.

## Rebuild from pinned sources

1. Create a virtual environment and install `requirements.txt` plus the pinned
   `en_core_web_sm` model described in the project repository.
2. Download FEVER train, the June-2017 FEVER Wikipedia pages, and the
   2017-02-20 Wikidata dump using `metadata/source_manifest.json`; verify every
   recorded byte count and checksum.
3. Set `PYTHONPATH=scripts`, then build the Wikipedia index and select the
   primary/recovery cohorts:

       python scripts/build_wiki_index.py
       python scripts/select_dataset.py
       python scripts/select_reserves.py --per-stratum 600 \\
         --output data/processed/evidencegraph20k_provenance_reserve.jsonl
       python scripts/preflight_evidence.py
       python scripts/extract_wikidata_snapshot.py --allow-missing

4. Generate the primary paraphrases with local Qwen2.5-1.5B, validate with the
   pinned bidirectional DeBERTa checkpoint, and run the deterministic/local
   recovery ladder. Never lower the NLI or structural thresholds:

       python scripts/run_paraphrases.py --model qwen2.5:1.5b --batch-size 4
       python scripts/validate_paraphrases.py --device mps
       python scripts/apply_paraphrase_structural_gate.py
       python scripts/run_paraphrase_recovery_ladder.py
       python scripts/run_paraphrase_reserve_recovery.py

5. The reserve driver derives a complete final Wikidata subset only when every
   retained page and referenced term is present, materializes exactly 20,000
   episodes, and prepares the blinded audit packet plus the standalone local
   rater interface.
6. Two independent human raters must complete all 3,000 test claims and every
   test source in separate packet copies. Merge their complete exports with
   `merge_human_audit_ratings.py`, then run `validate_human_audit.py` and
   `build_dataset_registry.py`. Model-generated ratings are prohibited.

The full paper evaluation runbook remains in the project repository; it is not
part of this data-only release because AVeriTeC, HoVer, ExpertQA, model caches,
and paid-backbone execution are download-only evaluation dependencies.
"""


def render_release_dataset_card(
    template: str, stats: dict[str, Any], audit: dict[str, Any]
) -> str:
    """Convert the checked-in pre-release card into a measured release card."""

    required_headings = (
        "## Release status",
        "## Intended use",
        "## Data origin and provenance",
        "## Licensing and redistribution boundary",
        "## Human audit",
        "## Known risks and limitations",
        "## Reproduction",
    )
    if any(heading not in template for heading in required_headings):
        raise ValueError("dataset-card template is missing a required section")
    metrics = audit.get("metrics", {})
    release_status = (
        "## Release status\n\n"
        "EvidenceGraph-20K v1 is a checksum-bound audited release containing "
        f"{stats['episodes']:,} episodes with the frozen 14,000/3,000/3,000 "
        "train/dev/test split. The complete 3,000-claim test set and every test "
        "source received two independent blinded ratings. Verdict/family "
        f"Cohen's $\\kappa$ are {metrics['verdict_kappa']:.3f} and "
        f"{metrics['family_kappa']:.3f}, respectively. Construction inputs, "
        "audit files, provenance subsets, scripts, and every distributed file "
        "are bound in `release_manifest.json`.\n\n"
    )
    template = re.sub(
        r"## Release status\n.*?(?=## Intended use)",
        lambda _match: release_status,
        template,
        flags=re.DOTALL,
    )
    template = template.replace(
        "These counts describe the frozen selection manifests, not the unfinished\n"
        "materialized corpus. They must be recomputed from the final release rather\n"
        "than copied into the paper.",
        "These construction-manifest counts are retained for provenance. The "
        "release statistics and paper table are recomputed from the final "
        "materialized corpus and its checksum-bound audit.",
    )
    provenance = (
        "## Data origin and provenance\n\n"
        "The release uses verifiable FEVER training claims and evidence "
        "sentences from FEVER's preprocessed June-2017 English Wikipedia "
        "snapshot. It preserves the FEVER record ID plus Wikipedia page and "
        "sentence identifiers. Mirror and citation records are deterministic "
        "derivatives; paraphrase records are synthetic local-model rewrites "
        "with generator and validation metadata; unrelated padding uses real "
        "unique-page sentences from the same frozen index.\n\n"
        "Each retained evidence page has complete target-and-term provenance "
        "from the pinned 2017-02-20 Wikidata dump. Pages absent from that "
        "historical snapshot were replaced only by validated candidates in the "
        "same split/label/corruption stratum. The final subset audit is "
        "complete and checksum-bound; the preserved globally incomplete parent "
        "extraction was never relabeled complete. No current Wikidata "
        "statements were used to fill historical gaps.\n\n"
    )
    template = re.sub(
        r"## Data origin and provenance\n.*?(?=## Licensing and redistribution boundary)",
        lambda _match: provenance,
        template,
        flags=re.DOTALL,
    )
    human = (
        "## Human audit\n\n"
        f"Two independent blinded raters evaluated all {metrics['n_claims']:,} "
        f"test claims and {metrics['n_sources']:,} test sources under audit "
        "protocol v2. Verdict raw agreement/Cohen's $\\kappa$ are "
        f"{100 * metrics['verdict_raw_agreement']:.1f}\\%/"
        f"{metrics['verdict_kappa']:.3f}; provenance-family agreement/$\\kappa$ "
        f"are {100 * metrics['family_raw_agreement']:.1f}\\%/"
        f"{metrics['family_kappa']:.3f}. Raw disagreements remain in the "
        "released rating files; no model-generated label substitutes for a "
        "human rating.\n\n"
    )
    template = re.sub(
        r"## Human audit\n.*?(?=## Known risks and limitations)",
        lambda _match: human,
        template,
        flags=re.DOTALL,
    )
    reproduction = (
        "## Reproduction\n\n"
        "`metadata/source_manifest.json` pins official source bytes and "
        "repository commits. `RUNBOOK.md` and `scripts/` contain the "
        "construction and audit workflow. `release_manifest.json` records byte "
        "counts and SHA-256 hashes for the complete distributed allowlist. "
        "External AVeriTeC, HoVer, and ExpertQA corpora are download-only and "
        "are not bundled.\n"
    )
    template = re.sub(
        r"## Reproduction\n.*\Z",
        lambda _match: reproduction,
        template,
        flags=re.DOTALL,
    )
    return template.replace(
        "# EvidenceGraph-20K dataset card (pre-release)",
        "# EvidenceGraph-20K dataset card (release v1)",
        1,
    )


def validate_release_inputs(
    *,
    dataset: Path,
    dataset_stats: Path,
    dataset_registry: Path,
    audit_summary: Path,
    final_selection: Path,
    wikidata_targets: Path,
    wikidata_terms: Path,
    wikidata_audit: Path,
) -> dict[str, Any]:
    """Validate the audited release boundary before copying any bytes."""

    with dataset_stats.open(encoding="utf-8") as handle:
        stats = json.load(handle)
    with dataset_registry.open(encoding="utf-8") as handle:
        registry = json.load(handle)
    with audit_summary.open(encoding="utf-8") as handle:
        audit = json.load(handle)
    if (
        stats.get("protocol")
        != "evidencegraph-20k-v5-wikidata-budget-separated-decoy"
        or stats.get("episodes") != 20_000
        or registry.get("status") != "complete"
        or registry.get("display_id") != "dataset"
        or audit.get("status") != "complete"
        or audit.get("target_met") is not True
        or audit.get("metrics", {}).get("n_claims") != 3_000
        or audit.get("metrics", {}).get("rater_count") != 2
    ):
        raise ValueError("release requires the complete dataset and genuine audit gates")
    registry_sources = [
        verified_source_path(descriptor, PROJECT_ROOT)
        for descriptor in registry.get("sources", [])
    ]
    if dataset.resolve() not in {path.resolve() for path in registry_sources}:
        raise ValueError("dataset registry does not bind the release corpus")
    episodes = read_jsonl(dataset)
    if len(episodes) != 20_000:
        raise ValueError("release corpus does not contain exactly 20,000 episodes")
    selection = read_jsonl(final_selection)
    required_pages = {
        str(page_id)
        for record in selection
        for page_id, _line_number in record.get("evidence", [])
    }
    load_provenance_index(
        wikidata_targets,
        wikidata_terms,
        wikidata_audit,
        required_page_ids=required_pages,
    )
    return {
        "episodes": len(episodes),
        "test_claims": sum(record.get("split") == "test" for record in episodes),
        "required_wikidata_pages": len(required_pages),
        "registry_sources": len(registry_sources),
    }


def release_file_specs(args: argparse.Namespace) -> list[tuple[Path, Path]]:
    """Return the explicit release allowlist; external benchmark corpora are absent."""

    fixed = [
        (args.dataset, Path("data/evidencegraph20k.jsonl")),
        (args.dataset_stats, Path("data/evidencegraph20k_stats.json")),
        (args.final_selection, Path("data/evidencegraph20k_final_selection.jsonl")),
        (args.replacements, Path("data/evidencegraph20k_replacements.jsonl")),
        (args.validation, Path("validation/final_combined_validation.jsonl")),
        (args.wikidata_targets, Path("provenance/wikidata_target_entities.jsonl")),
        (args.wikidata_terms, Path("provenance/wikidata_reference_terms.jsonl")),
        (args.wikidata_audit, Path("provenance/wikidata_snapshot_audit.json")),
        (args.dataset_registry, Path("metadata/dataset_registry.json")),
        (args.audit_summary, Path("audit/audit_summary.json")),
        (PROJECT_ROOT / "data/source_manifest.json", Path("metadata/source_manifest.json")),
        (PROJECT_ROOT / "DATASET_CARD.md", Path("DATASET_CARD.md")),
        (PROJECT_ROOT / "RUNBOOK.md", Path("RUNBOOK.md")),
        (PROJECT_ROOT / "requirements.txt", Path("requirements.txt")),
        (
            PROJECT_ROOT / "data/raw/fever_wiki/license.html",
            Path("LICENSES/fever-data-license.html"),
        ),
    ]
    with args.audit_summary.open(encoding="utf-8") as handle:
        audit = json.load(handle)
    for name, descriptor in sorted(audit.get("inputs", {}).items()):
        path = verified_source_path(descriptor, PROJECT_ROOT)
        suffix = path.suffix or ".dat"
        fixed.append((path, Path(f"audit/{name}{suffix}")))
    code_root = PROJECT_ROOT / "artifact/code"
    code_paths = [code_root / name for name in sorted(TOP_LEVEL_RELEASE_SCRIPTS)]
    code_paths.extend(
        path
        for path in sorted((code_root / "evidencegraph").glob("*.py"))
        if not any(token in path.name for token in EXCLUDED_PACKAGE_MODULE_TOKENS)
    )
    validate_script_closure(code_paths, code_root)
    fixed.extend(
        (path, Path("scripts") / path.relative_to(code_root)) for path in code_paths
    )
    destinations = [destination for _source, destination in fixed]
    if len(destinations) != len(set(destinations)):
        raise ValueError("release allowlist contains duplicate destinations")
    forbidden = ("API-key", "bedrock", "averitec_dev_knowledge_store", ".external")
    if any(any(token in str(destination) for token in forbidden) for destination in destinations):
        raise ValueError("release allowlist includes a forbidden external or credential path")
    return fixed


def validate_script_closure(code_paths: list[Path], code_root: Path) -> None:
    """Reject a release whose selected scripts omit a local Python dependency."""

    selected_top = {path.stem for path in code_paths if path.parent == code_root}
    selected_package = {
        path.stem for path in code_paths if path.parent == code_root / "evidencegraph"
    }
    available_top = {path.stem for path in code_root.glob("*.py")}
    missing: set[tuple[str, str]] = set()
    for path in code_paths:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom) or not node.module:
                continue
            if node.module.startswith("evidencegraph."):
                module = node.module.split(".", 1)[1].split(".", 1)[0]
                if module not in selected_package:
                    missing.add((str(path.relative_to(code_root)), module))
            elif path.parent == code_root / "evidencegraph" and node.level == 1:
                module = node.module.split(".", 1)[0]
                if module not in selected_package:
                    missing.add((str(path.relative_to(code_root)), module))
            elif path.parent == code_root and node.module in available_top:
                if node.module not in selected_top:
                    missing.add((str(path.relative_to(code_root)), node.module))
    if missing:
        raise ValueError(f"release scripts omit local dependencies: {sorted(missing)}")


def _write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=Path,
        default=PROJECT_ROOT / "data/processed/evidencegraph20k.jsonl",
    )
    parser.add_argument(
        "--dataset-stats",
        type=Path,
        default=PROJECT_ROOT / "data/processed/evidencegraph20k_stats.json",
    )
    parser.add_argument(
        "--final-selection",
        type=Path,
        default=PROJECT_ROOT / "data/processed/evidencegraph20k_final_selection.jsonl",
    )
    parser.add_argument(
        "--replacements",
        type=Path,
        default=PROJECT_ROOT / "data/processed/evidencegraph20k_replacements.jsonl",
    )
    parser.add_argument("--validation", type=Path, default=FINAL_COMBINED_VALIDATION)
    parser.add_argument("--wikidata-targets", type=Path, default=FINAL_WIKIDATA_TARGETS)
    parser.add_argument("--wikidata-terms", type=Path, default=FINAL_WIKIDATA_TERMS)
    parser.add_argument("--wikidata-audit", type=Path, default=FINAL_WIKIDATA_AUDIT)
    parser.add_argument(
        "--dataset-registry",
        type=Path,
        default=PROJECT_ROOT / "artifact/results/registries/dataset.json",
    )
    parser.add_argument(
        "--audit-summary",
        type=Path,
        default=PROJECT_ROOT / "data/audit/audit_summary.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "release" / RELEASE_NAME,
    )
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to replace an existing release directory")
    validation = validate_release_inputs(
        dataset=args.dataset,
        dataset_stats=args.dataset_stats,
        dataset_registry=args.dataset_registry,
        audit_summary=args.audit_summary,
        final_selection=args.final_selection,
        wikidata_targets=args.wikidata_targets,
        wikidata_terms=args.wikidata_terms,
        wikidata_audit=args.wikidata_audit,
    )
    specs = release_file_specs(args)
    missing = [str(source) for source, _destination in specs if not source.is_file()]
    if missing:
        raise FileNotFoundError("release allowlist inputs are missing: " + ", ".join(missing))
    staging = args.output.with_name(f".{args.output.name}.partial")
    if staging.exists():
        raise FileExistsError("a partial release directory already exists; inspect it")
    staging.mkdir(parents=True)
    for source, destination in specs:
        target = staging / destination
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    with args.dataset_stats.open(encoding="utf-8") as handle:
        stats = json.load(handle)
    with args.audit_summary.open(encoding="utf-8") as handle:
        audit = json.load(handle)
    release_card = render_release_dataset_card(
        (PROJECT_ROOT / "DATASET_CARD.md").read_text(encoding="utf-8"),
        stats,
        audit,
    )
    _write_text(staging / "DATASET_CARD.md", release_card)
    _write_text(staging / "RUNBOOK.md", RELEASE_RUNBOOK)
    _write_text(staging / "ATTRIBUTION.md", ATTRIBUTION)
    _write_text(staging / "LICENSES/PROJECT_CODE_NOTICE.md", CODE_NOTICE)
    _write_text(staging / "LICENSES/WIKIDATA_CC0_NOTICE.md", WIKIDATA_NOTICE)
    outputs = [
        source_descriptor(path, staging)
        for path in sorted(staging.rglob("*"))
        if path.is_file()
    ]
    manifest = {
        "schema_version": 1,
        "protocol": "evidencegraph20k-audited-release-v1",
        "status": "complete",
        "release_name": RELEASE_NAME,
        "validation": validation,
        "inputs": [
            source_descriptor(source, PROJECT_ROOT) for source, _destination in specs
        ],
        "outputs": outputs,
        "exclusions": [
            "API credentials and key loaders",
            "AVeriTeC knowledge store",
            "HoVer and ExpertQA corpora",
            "model weights and caches",
            "coordinator or rater identity information",
        ],
    }
    _write_text(staging / "release_manifest.json", json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    staging.replace(args.output)
    print(json.dumps({"release": str(args.output), "files": len(outputs) + 1}))


if __name__ == "__main__":
    main()
