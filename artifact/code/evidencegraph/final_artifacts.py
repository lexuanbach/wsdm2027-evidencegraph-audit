"""Canonical handoff paths for the finalized EvidenceGraph-20K release."""

from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
FINAL_RECOVERY_DIR = (
    PROJECT_ROOT / "artifact/runs/paraphrases/production_nfc_reserve_recovery"
)
FINAL_COMBINED_VALIDATION = FINAL_RECOVERY_DIR / "final_combined_validation.jsonl"
FINAL_WIKIDATA_TARGETS = (
    PROJECT_ROOT / "data/processed/wikidata20170220_final_target_entities.jsonl"
)
FINAL_WIKIDATA_TERMS = (
    PROJECT_ROOT / "data/processed/wikidata20170220_final_reference_terms.jsonl"
)
FINAL_WIKIDATA_AUDIT = (
    PROJECT_ROOT / "artifact/runs/wikidata20170220_final_snapshot_audit.json"
)
