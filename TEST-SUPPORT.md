# Test support and legacy-suite triage

The current code/audit artifact reproduces stored results; it does not certify
the unavailable full construction or historical endpoint runners. The working
archive retains those legacy tests and their failures. No skips or expected-failure
marks were added to hide them, and full-dataset builder gates were not relaxed.

The 2026-11-20 working-copy run improved from **461 passed / 51 failed / 27 errors /
1 skipped** to **472 passed / 40 failed / 27 errors / 1 skipped** (540 collected outcomes).
Eleven failures were repaired: active display paths and the analytical-figure
contract, current dataset-card boundaries, two CLI test paths, and current
checksum/tamper checks. The previous test expecting the empirical duplication
plot now checks that unavailable evidence cannot promote it into the current
submission; replacement with the retired plot is explicitly rejected.

The supported public suite is tested separately. These working-archive counts
are not its test counts, and do not mean all historical claims are reproducible.

## Remaining failures and errors

| Category | Outcomes | Required work |
|---|---:|---|
| external private tooling | 2 | Original private key-loader/credential fixture is absent; never create a real secret file to make this suite pass. |
| full dataset layout and inputs | 2 | Legacy full-dataset builder expects artifact/code layout. Repair requires separate source/layout contract work; full dataset gates remain blocked, not bypassed. |
| legacy claim and layout | 9 | Per-claim pins reference the old artifact/runs layout and prose. Some summaries exist, others are absent; migrate each supported claim individually with provenance before asserting closure. |
| missing historical audit | 2 | Effective-evidence measured-audit file is absent. The full release finalizer remains gated. |
| missing historical inputs or retired claim | 37 | Recover exact historical inputs and claim contract first. Missing evidence does not license reinstating the corresponding manuscript claim. |
| missing orchestration script | 4 | Restore exact shell orchestration source before testing historical runner behavior; not needed for saved-output audit. |
| optional parser dependency | 8 | Install pinned spaCy plus its parser model for construction-only NLP checks; no synthetic fallback. |
| retired build contract | 3 | Legacy full-submission wrapper and withdrawn compute block are not the current audit rebuild contract. Current binding/tamper checks now pass; original gates remain intact. |

The private working record `revision/remain-light-test-triage.json` identifies
each test, evidence and next action. Raw failure logs are not part of the public
artifact. Missing-input checks remain unresolved rather than being relabeled
as successful reproduction. No model call, training or paid API request was run.
