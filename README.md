# EvidenceGraph: Measuring and Controlling Repeated Evidence in Retrieval-Augmented Verification

Code and stored outputs for the WSDM 2027 paper

> Xuan-Bach Le, Phat T. Tran-Truong, Son Ha Xuan, and Duy Nguyen. 2027.
> *EvidenceGraph: Measuring and Controlling Repeated Evidence in
> Retrieval-Augmented Verification.* In Proceedings of the Twentieth ACM
> International Conference on Web Search and Data Mining (WSDM '27).
> https://doi.org/10.1145/3862322.3863670

Retrieval-augmented fact verification can count the same evidence several times
when passages are copied, mirrored, or paraphrased from one origin.
EvidenceGraph scores each retrieved passage, estimates which passages share an
origin, aggregates evidence so that copies count once, and exposes these
controls as independently switchable stages. This repository contains the
implementation, the stored outputs behind every table and Figure 3 in the
paper, and a one-command script that rebuilds them. Figure 2 (duplication), the
compute totals and several diagnostic numbers in the text come from recorded
summaries whose inputs are not included; see [Limits](#limits-of-this-release). The supplementary material is in
[`supplement.pdf`](supplement.pdf).

## Contents

| Path | What it contains |
|---|---|
| `artifact/code/evidencegraph/` | Library: episode schema and construction, retrieval, NLI and lineage (dependence) features, effective-evidence aggregation, controller, evaluation metrics, bootstrap. |
| `artifact/code/*.py` | Command-line entry points for every pipeline stage (listed below) and for rebuilding results. |
| `artifact/code/reproduce_paper.py` | Rebuilds all tables, Figure 3 and the supplement tables from stored outputs and checks them. |
| `artifact/runs/` | Stored outputs (summaries, frozen predictions, deidentified audit inputs). Each summary lists its input files with SHA-256 digests under `inputs`. |
| `annotation/score/report_2026-08-24.json` | Aggregate human lineage-audit scores (deidentified, no item text). |
| `expected/paper_tables.sha256` | Digests of the camera-ready table fragments, used to check the rebuild byte for byte. |
| `tests/` | Unit and regression tests (run offline in a few seconds). |
| `supplement.pdf` | Supplementary material (threshold sensitivity and historical covariance analyses). |
| `requirements.txt` | Pinned packages for rebuilding results and running tests. |
| `requirements-full.txt` | Additional packages for rerunning the full pipeline (models, retrieval). |
| `Dockerfile` | Container that rebuilds and checks every stored-output result. |
| `SHA256SUMS` | Digests of every file in this release. |

## Quick start: rebuild the paper's results (CPU, about one minute)

Requirements: Python 3.12 or newer, about 50 MB of disk. No GPU, model
download, API key, or network access is needed.

```sh
git clone https://github.com/lexuanbach/wsdm2027-evidencegraph-audit.git
cd wsdm2027-evidencegraph-audit
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
python artifact/code/reproduce_paper.py        # add --no-tex without a LaTeX install
python -m pytest -q                            # optional: unit and regression tests
```

Figure 3 is typeset with LaTeX and the Libertine fonts when `latex` is on the
`PATH`; `--no-tex` draws the same data with matplotlib text instead. Expected
output: every line reads `ok` and the script ends with
`All stored-output results reproduced.` Rebuilt files are written to
`outputs/`. With Docker:

```sh
docker build -t evidencegraph .
docker run --rm evidencegraph                  # runs reproduce_paper.py --no-tex
```

To verify that the release is intact, run `shasum -a 256 -c SHA256SUMS`
(or `sha256sum -c SHA256SUMS`).

## Where each result comes from

`reproduce_paper.py` rebuilds every item marked "rebuilt" from the stored
outputs below. Tables are rebuilt as the exact table fragments used in the paper
and compared byte for byte with the camera-ready versions.

| Paper item | Status | Stored inputs (`artifact/runs/` unless noted) | Script |
|---|---|---|---|
| Table 1, lineage estimation vs. human labels | rebuilt, identical | `subset_edge_episode/nli/summary_test.json`, `annotation/score/report_2026-08-24.json` | `build_subset_tables.py` |
| Table 2, natural-lineage recovery by corpus | rebuilt, identical | `annotation/score/report_2026-08-24.json` | `build_subset_tables.py` |
| Table 3, audit by predicted-probability bin | rebuilt, identical | `annotation/score/report_2026-08-24.json` | `build_subset_tables.py` |
| Table 4, aggregation on 371 held-out episodes | rebuilt, identical | `subset_edge_episode/episode_baselines/summary.json` | `build_subset_tables.py` |
| Table 5, controller ablations | rebuilt, identical | `subset_edge_episode/controller_evaluation/{summary.json,predictions.jsonl}`, `subset_edge_episode/bounded_precision/summary.json` | `build_subset_tables.py` |
| Table 6, end-to-end results on frozen retrieval | rebuilt, identical | `benchmarks/haiku_subset/{evaluation_summary,override_effect_summary}.json` | `build_subset_tables.py` |
| Figure 3, acquisition on natural lineage | rebuilt (plus a CSV of its 18 estimates and a numeric table) | `natural_lineage_acquisition/summary.json` | `make_natural_figure.py` |
| Supplement, threshold tables (3) | rebuilt, matches stored grid | `lineage_threshold/audit_inputs.json` | `threshold_sensitivity.py` |
| Supplement, historical covariance tables (2) | rebuilt, matches stored values | `saved_covariance/*_predictions.jsonl` | `saved_covariance_metrics.py` |
| Figure 1, overview | diagram, no data | | |
| Figure 2, duplication intervention | not rebuilt offline (see limits) | | `run_duplication.py`, `summarize_duplication.py`, `make_duplication_figure.py` |
| Compute totals (Availability paragraph) | recorded totals, not rebuilt offline | | `build_subset_compute_accounting.py` |

Each script can also be run on its own, for example
`python artifact/code/threshold_sensitivity.py` or
`python artifact/code/saved_covariance_metrics.py --verify`.

## Rerunning the full pipeline

The full pipeline rebuilds the stored outputs from the public benchmarks. It
needs the upstream data, model weights and services listed here, and roughly
the compute recorded in the paper: on one Apple-silicon workstation, 3.9
GPU-hours for the duplication intervention, 0.1 hours to score 27,788
entailment edges, and 1.4 hours for 1,500 graph selections; plus 3.6 wall-hours
of hosted inference (4,428 answer calls). Install the extra packages with
`pip install -r requirements-full.txt`. Paths below are relative to the
repository root, which every script treats as its project root.

**Inputs to obtain from their owners** (not redistributed here):

| Input | Expected location |
|---|---|
| FEVER train/dev claims and the 2017 Wikipedia pages | `data/raw/fever_train.jsonl`, `data/raw/fever_dev.jsonl`, `data/raw/fever_wiki/` |
| HoVer v1.1 development set | `.external/hover-official/data/hover/hover_dev_release_v1.1.json` |
| AVeriTeC development claims and knowledge store | `.external/averitec-official/data/dev.json`, `data/raw/averitec_dev_knowledge_store/` |
| Wikidata JSON dump of 2017-02-20 | `data/raw/wikidata-20170220-all.json.bz2` |

**Models** (exact revisions are pinned in the scripts):
NLI `MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli` at revision
`b3546ea6b0346eb6f8d5d68b13c7dc6d0376b3d7` (small scorer:
`cross-encoder/nli-MiniLM2-L6-H768`); `intfloat/e5-large-v2` dense retrieval
and the `cross-encoder/ms-marco-MiniLM-L6-v2` reranker for FEVER/HoVer, and
`mixedbread-ai/mxbai-embed-large-v1` embeddings with MMR for AVeriTeC
(`evidencegraph/retrieval.py`); `qwen2.5:3b-instruct` and `qwen2.5:1.5b` served
by Ollama (`OLLAMA_BASE_URL`, default `http://127.0.0.1:11434`) for paraphrase
generation and the duplication intervention; and
`us.anthropic.claude-haiku-4-5-20251001-v1:0` on Amazon Bedrock at temperature
0 for end-to-end answering (standard AWS credentials; hosted decoding has no
seed guarantee).

**Stages and entry points** (run each with `--help` for all options):

1. Indexes and benchmark cases: `build_wiki_index.py`,
   `build_wiki_title_index.py`, `build_averitec_index.py`,
   `extract_wikidata_snapshot.py`, `prepare_benchmark_cases.py`.
2. Controlled construction (EvidenceGraph-20K): `select_dataset.py` (topic-disjoint
   14,000/3,000/3,000 split), `run_paraphrases.py`, `validate_paraphrases.py`
   (anchor preservation and bidirectional entailment at 0.8),
   `apply_paraphrase_structural_gate.py`, `run_paraphrase_recovery_ladder.py`,
   `run_backtranslation_paraphrases.py`, reserve selection
   (`select_reserves.py`, `select_adaptive_reserves.py`,
   `select_reserve_retries.py`, `make_paraphrase_retry.py`),
   `merge_paraphrase_validations.py`, `run_paraphrase_reserve_recovery.py`,
   then `materialize_validated_subset.py` (the 2,298-episode subset used in the
   paper) or `materialize_dataset.py`; `build_dataset_registry.py` records the
   composition.
3. Edge scoring, lineage estimator and controller:
   `run_edge_episode_pipeline.py` runs `run_nli_scorer.py` and
   `summarize_nli_scorers.py`, `run_dependence_scorer.py` (seeds 103, 104,
   105), `train_controller.py`, `run_episode_evaluation.py`,
   `run_coordinated_fabrication_stress.py` and `run_controller_evaluation.py`
   in order;
   `run_episode_baselines.py` and `reanalyze_bounded_precision.py` produce the
   Table 4 and Table 5 summaries; the covariance controls (`audit_alternative_covariance.py`,
   `audit_joint_error_covariance.py`, `audit_stance_ridge_no_override.py`,
   `freeze_stance_ridge_dev_specs.py`) analyze the same frozen predictions.
4. Duplication intervention (Figure 2): `run_duplication.py --model
   qwen2.5:3b-instruct` and `--model qwen2.5:1.5b` with `--seed` 103, 104 and
   105 on the first 40 stable-hashed test false claims, then
   `summarize_duplication.py` and `make_duplication_figure.py`.
5. Natural lineage (Figure 3): `run_natural_lineage_acquisition.py` on the
   FEVER, HoVer and AVeriTeC retrieval pools with the three dependence models
   (top-20 pool, five initial sources, three actions, MMR lambda 0.75, 1,000
   bootstrap resamples).
6. End-to-end benchmarks (Table 6): `run_lexical_retrieval.py`,
   `run_benchmark_retrieval.py`, `fuse_benchmark_retrieval.py`,
   `run_benchmark_answering.py --backend bedrock`,
   `run_evidencegraph_benchmark_matrix.py`, `run_bedrock_transfer_matrix.py`,
   then `evaluate_haiku_subset.py`, `evaluate_haiku_subset_by_recall_stratum.py`
   and `rescore_graph_override.py`. Subsets use 500 deterministic cases per
   benchmark (stable hash, seed 103).
7. Human lineage audit: `prepare_human_audit.py` creates the blinded rater
   packet; scores are aggregated into `annotation/score/`.
8. Tables and compute totals: `build_subset_tables.py`,
   `build_subset_compute_accounting.py`.

Unless stated otherwise, confidence intervals use 1,000 paired claim (or
claim-cluster) bootstrap resamples, and every learned component is selected on
training and development data before the test split is scored once.

## Limits of this release

- **Not the EvidenceGraph-20K dataset.** The raw episode records, passage-level
  NLI predictions, complete lineage predictions and frozen lineage-estimator
  weights of the 2,298-episode subset are not included, and no dataset licence
  is assigned. The stored summaries record the SHA-256 of each of these inputs.
- **Figure 2 and the compute totals cannot be rebuilt offline.** The
  three-seed duplication summaries (`artifact/runs/duplication/`) and the
  benchmark matrix summary that feed them were not recovered. The values appear
  in the paper; the code to regenerate them from a full rerun is included.
- **Human-audit data are aggregates.** Annotator responses, item text,
  original item IDs and blinding keys are withheld to protect the blinded
  protocol. The threshold analysis uses deidentified numeric consensus labels
  and model scores (`artifact/runs/lineage_threshold/audit_inputs.json`).
- **Covariance predictions are historical.** The three saved prediction arms
  (359 development and 371 test claims) are recomputed, not refit; claim IDs are
  replaced consistently (`artifact/runs/saved_covariance/PROVENANCE.json`).
- **Hosted inference is not seed-controlled.** Rerunning Table 6 can differ
  slightly from the stored predictions.

## Data and licences

The code and documentation written by the authors are released under the MIT
licence (`LICENSE`). Stored outputs derived from FEVER and Wikipedia, HoVer,
AVeriTeC and Wikidata keep the terms of those sources; see
`THIRD_PARTY_NOTICES.md`. No third-party passage text is included.

## Citation

```bibtex
@inproceedings{le2027evidencegraph,
  author    = {Le, Xuan-Bach and Tran-Truong, Phat T. and Ha Xuan, Son and Nguyen, Duy},
  title     = {{EvidenceGraph}: Measuring and Controlling Repeated Evidence in
               Retrieval-Augmented Verification},
  booktitle = {Proceedings of the Twentieth ACM International Conference on
               Web Search and Data Mining (WSDM '27)},
  year      = {2027},
  publisher = {ACM},
  doi       = {10.1145/3862322.3863670}
}
```

## Contact

Xuan-Bach Le (corresponding author); the e-mail address is on the paper.
Please open a GitHub issue for questions about the code.
