"""Regression checks for numerical export, transitive merges, leakage and spending."""
from pathlib import Path
import json, sys
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "revision"))
from budget_guard import BudgetLedger
from input_preflight import validate_splits, inspect
from threshold_sensitivity import analyze, cluster_scores

def test_numeric_export_reproduces_every_saved_metric():
    actual = analyze(json.loads((ROOT / "revision/threshold-audit-inputs.json").read_text()))
    expected = json.loads((ROOT / "revision/threshold-sensitivity.json").read_text())
    for scope in expected["scopes"]:
        for kind in expected["scopes"][scope]:
            for t, metrics in expected["scopes"][scope][kind].items():
                assert actual["scopes"][scope][kind][t] == pytest.approx(metrics, abs=1e-14)

def test_numeric_export_has_no_private_identifiers_or_text():
    data = json.loads((ROOT / "revision/threshold-audit-inputs.json").read_text())
    for row in data["pairs"]: assert set(row) == {"scope", "gold", "model_dependence_prob", "text_reuse_score"}
    for row in data["clusters"]:
        assert set(row) == {"scope", "gold", "pairs"}
        assert all(set(p) == {"labels", "prob"} for p in row["pairs"])

def test_transitive_link_changes_partition_not_only_direct_edge():
    row = {"gold": [0, 0, 1], "pairs": [{"labels": [0, 1], "prob": .9}, {"labels": [1, 2], "prob": .6}]}
    assert cluster_scores([row], .5)["false_merged_pairs"] == 2
    assert cluster_scores([row], .7)["false_merged_pairs"] == 0

def test_claim_split_leakage_is_rejected():
    with pytest.raises(ValueError, match="across splits"):
        validate_splits([{"claim_id": "c1", "split": "dev"}, {"claim_id": "c1", "split": "test"}])

def test_missing_inputs_are_not_mistaken_for_completed_run(tmp_path):
    assert all(v["status"] == "blocked_missing_inputs" for v in inspect(tmp_path).values())

def test_budget_resume_preserves_pending_and_charges_retries(tmp_path):
    path = tmp_path / "ledger.json"; BudgetLedger(path).reserve("first", "24")
    resumed = BudgetLedger(path)
    with pytest.raises(RuntimeError): resumed.reserve("retry", "2")
    with pytest.raises(ValueError): resumed.reserve("first", "1")
    resumed.settle("first", "23.5"); resumed.reserve("retry", "1.5"); resumed.settle("retry", None)
    with pytest.raises(RuntimeError): resumed.reserve("another", ".01")

def test_budget_invalid_values_and_overspend_stop(tmp_path):
    ledger = BudgetLedger(tmp_path / "ledger.json")
    for value in ("NaN", "Infinity", "-1", "0"):
        with pytest.raises(ValueError): ledger.reserve("bad", value)
    ledger.reserve("one", "1")
    with pytest.raises(ValueError): ledger.settle("one", "1.01")
    with pytest.raises(ValueError): ledger.settle("one", "-1")
    ledger.settle("one", ".9")
    with pytest.raises(ValueError): ledger.settle("one", ".5")

def test_concurrent_reservations_cannot_overrun_cap(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    path = tmp_path / "ledger.json"
    def reserve(i):
        try: BudgetLedger(path).reserve(str(i), "1"); return True
        except RuntimeError: return False
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(reserve, range(40)))
    assert sum(results) == 25
    assert BudgetLedger.committed(json.loads(path.read_text())) == 25
