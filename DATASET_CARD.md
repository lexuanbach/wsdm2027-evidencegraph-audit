# EvidenceGraph construction and audit — availability card

This card documents the downloaded artifact and 2026-11-18 revision; it does not announce a dataset release.

- Purpose: diagnostic experiments on repeated evidence, lineage estimation and verification control.
- Initial construction: 20,000 balanced FEVER claims, candidate splits 14,000/3,000/3,000. Retained after gates: 2,298 episodes (1,568/359/371), 44,064 sources, 453,986 within-episode pairs.
- Present: code and schemas, frozen run summaries/predictions where bundled, 640 pair-audit items and 48 partition tasks with response records, manuscript tables and figures.
- Absent: complete raw construction and frozen lineage estimator weights. A builder is present but does not establish completed release or passed quality gates.
- Audit design: stratified diagnostic sample, 128 shared pairs and 12 shared partition tasks; unsure/disagreeing pairs excluded under the original primary consensus convention. Natural pair F1/cluster ARI 0.568/0.401 are not population-prevalence estimates.
- Internal release targets: F1 .85 and ARI .80 are unmet on natural evidence. They are artifact gates, not claimed conference requirements.
- Risks: false lineage merges, fabricated counter-sources and confidence overcorrection. Preserve synthetic flags, reserved .invalid domains and origin IDs; do not expose generated false content to indexing.
- Terms: preserve original FEVER/Wikipedia attribution/share-alike obligations and other benchmark licenses. No EvidenceGraph-20K license or public URL is assigned. The AVeriTeC Web knowledge store is not redistributed here.
- Private material: annotation blinding keys remain local and outside the manuscript source ZIP. Reviews and ACM correspondence are not public artifact content.

See revision/display-rebuild.log for table reproduction, revision/threshold-sensitivity.json for the retrospective grid, and ARTIFACT-AVAILABILITY.md for limits.
