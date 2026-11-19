# EvidenceGraph code and audit artifact

Paper 2148, WSDM 2027 Main. Xuan-Bach Le, Phat T. Tran-Truong, Son Ha Xuan,
and Duy Nguyen. Assigned DOI: https://doi.org/10.1145/3862322.3863670

This limited release contains author-approved MIT code and shareable numeric
outputs. It is not the EvidenceGraph-20K dataset. The camera-ready manuscript
is a working revision pending ACM rights and final production approval.

## What this package reproduces

- Seven paper tables byte-for-byte from archived aggregate summaries.
- All retrospective lineage/text-reuse threshold metrics from deidentified
  numeric labels and predictions, including natural F1 .568 / ARI .401.
- Recomputed historical covariance predictions on 359 dev / 371 test claims,
  with paired claim bootstrap (1,000 draws, seed 217) against uniform.
- Controller and revision regression tests, including transitive clustering,
  split-leakage detection and persistent USD25 reservation accounting.

Use Python 3.12 with the pinned dependencies:

```sh
python -m venv .venv
.venv/bin/pip install -r requirements-lock.txt
.venv/bin/python revision/verify_offline.py
.venv/bin/python revision/covariance_saved_audit.py --verify
.venv/bin/python -m pytest -q
```

The Dockerfile runs the same offline verification. Container construction is
separate from the clean-directory Python validation; consult the execution
report for which was actually run.

## What is absent

This is not the full EvidenceGraph-20K construction. The retained construction
has 2,298 episodes, but its raw episode records, passage NLI predictions,
complete lineage predictions, frozen weights and duplication selection file
were not recovered. No new endpoint experiment is included. No full dataset
license is assigned. `python revision/input_preflight.py` deliberately exits 2
and lists the missing inputs. The covariance script also stops before fitting
if they are absent. The three recovered numeric prediction arms passed original-checksum and
split/join validation. Their provenance is in revision/covariance/PROVENANCE.json.
Historical weight uniformity and alternative grid arms are summary-only; they
are not recomputed or refit. The new uncertainty is retrospective and exploratory. Weights are unnecessary if all frozen predictions
and matched episode inputs are present.

The budget guard is tested offline but is not a model runner. No paid request
can be inferred from its presence. The original duplication confidence uses
token log-probabilities; a model's self-reported probability is a different
measurement and cannot silently replace it.

## Manifest history and privacy

`revision/rebuilt-display-manifest.json` verifies 18 current bindings. The
original `results/subset_tables_manifest.json` retains three stale bindings to
an older annotation report. The new manifest does not recreate that old report.
`MANIFEST.sha256.json` binds every staged file. Numeric audit inputs omit text,
original item IDs and private blinding keys. The aggregate report is also
minimized and rebound after sanitization; see PUBLIC-RELEASE-CHANGES.json.
This package excludes reviews,
correspondence, credentials, model caches and third-party passage text.

See `THIRD_PARTY_NOTICES.md` and `LICENSE`. MIT applies only to authors-owned
code/documentation and permitted outputs, not upstream data or dependencies.
