# Availability and reproduction boundary

The authors approved MIT licensing for owned code and shareable outputs on November 19, 2026. The repository https://github.com/lexuanbach/wsdm2027-evidencegraph-audit is public; anonymous API and license access were verified on November 19. The limited code/audit artifact is not EvidenceGraph-20K.

The artifact reproduces seven tables, 18 numerical bindings, the complete retrospective threshold grid, and three historical covariance prediction arms on the same 359 dev / 371 test claims. Paired claim bootstrap uses 1,000 draws and seed 217; it does not refit covariance or select using test labels. Weight uniformity, other grid arms and shared-scorer results remain summary-only. Original hashes, source commit and deidentification are documented. All claim IDs are consistently replaced; the mapping is private and excluded.

No passage text, blinding keys, reviews, correspondence, credentials or recovered pickles are released. Raw retained episodes, passage NLI, full lineage predictions, seed-specific predictions and original duplication contexts remain absent. Full dataset release and raw-input refit remain blocked; the builder gates are unchanged. The 2,298 retained episodes must not be relabeled as a released 20K corpus. Natural-source candidates remain unlabelled.

Source ZIP contains manuscript dependencies and identified revised PDFs. Model endpoint reruns require missing matched inputs and are separate from offline saved-output verification.
