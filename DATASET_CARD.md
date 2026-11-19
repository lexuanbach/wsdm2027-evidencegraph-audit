# EvidenceGraph construction and audit — availability card

This card distinguishes the private historical archive from the public code/audit release of November 19, 2026. It does not announce a dataset release.

- Purpose: diagnostic experiments on repeated evidence, lineage estimation and verification control.
- Initial construction: 20,000 balanced FEVER claims, candidate splits 14,000/3,000/3,000. Retained after gates: 2,298 episodes (1,568/359/371), 44,064 sources, 453,986 within-episode pairs.
- Public: code/schemas, aggregate summaries, deidentified numeric consensus labels and predictions for the 640 pair / 48 partition audit, seven manuscript tables, and three historical covariance prediction arms (359 dev / 371 test). Original response records, blinding keys, passage text and original claim IDs are excluded.
- Absent: complete raw construction and frozen lineage estimator weights. A builder is present but does not establish completed release or passed quality gates.
- Audit design: stratified diagnostic sample, 128 shared pairs and 12 shared partition tasks; unsure/disagreeing pairs excluded under the original primary consensus convention. Natural pair F1/cluster ARI 0.568/0.401 are not population-prevalence estimates.
- Internal release targets: F1 .85 and ARI .80 are unmet on natural evidence. They are artifact gates, not claimed conference requirements.
- Risks: false lineage merges, fabricated counter-sources and confidence overcorrection. Preserve synthetic flags, reserved .invalid domains and origin IDs; do not expose generated false content to indexing.
- Terms: preserve original FEVER/Wikipedia attribution/share-alike obligations and other benchmark licenses. MIT covers authors-owned code and permitted outputs at https://github.com/lexuanbach/wsdm2027-evidencegraph-audit . No full EvidenceGraph-20K dataset license or release is assigned. The AVeriTeC Web knowledge store is not redistributed here.
- Private material: annotation blinding keys remain local and outside the manuscript source ZIP. Reviews and ACM correspondence are not public artifact content.

See revision/verify_offline.py for table reproduction, revision/threshold-sensitivity.json for the retrospective grid, and ARTIFACT-AVAILABILITY.md for limits.
