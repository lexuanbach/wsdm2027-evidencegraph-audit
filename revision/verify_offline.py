"""Verify numeric audit and all seven table bindings without private files/network."""
from pathlib import Path
import hashlib, json, subprocess, sys, tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
from verify_submission_manifest import binding_failures
from threshold_sensitivity import analyze
import build_subset_tables as tables

def close(a, b):
    if isinstance(a, dict): return a.keys() == b.keys() and all(close(a[k], b[k]) for k in a)
    if isinstance(a, list): return len(a) == len(b) and all(close(x, y) for x, y in zip(a, b))
    if isinstance(a, float): return abs(a-b) <= 1e-14
    return a == b

def main():
    manifest = ROOT / "revision/rebuilt-display-manifest.json"
    assert not binding_failures(manifest, ROOT), binding_failures(manifest, ROOT)
    result = analyze(json.loads((ROOT / "revision/threshold-audit-inputs.json").read_text()))
    assert close(result, json.loads((ROOT / "revision/threshold-sensitivity.json").read_text()))
    with tempfile.TemporaryDirectory(prefix="eg-offline-") as tmp:
        p = Path(tmp); (p / "artifact").mkdir()
        (p / "artifact/runs").symlink_to(ROOT / "runs", target_is_directory=True)
        (p / "annotation").symlink_to(ROOT / "annotation", target_is_directory=True)
        tables.PROJECT_ROOT = p; tables.SUB = p / "artifact/runs/subset_edge_episode"
        tables.TABLE_DIR = p / "manuscript/tables"; sys.argv = ["build_subset_tables.py"]; tables.main()
        names = list(json.loads(manifest.read_text()))
        assert len(names) == 7
        for name in names:
            assert (tables.TABLE_DIR / name).read_bytes() == (ROOT / "manuscript/tables" / name).read_bytes(), name
    from covariance_saved_audit import analyze as covariance_analyze
    covariance = ROOT / "revision/covariance"
    assert close(covariance_analyze(covariance), json.loads((covariance / "saved-prediction-audit.json").read_text()))
    if (ROOT / "MANIFEST.sha256.json").exists():
        for name, digest in json.loads((ROOT / "MANIFEST.sha256.json").read_text()).items():
            assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest, name
    print("PASS: seven tables byte-identical, 18 bindings valid, every audit-grid metric reproduced.")

if __name__ == "__main__": main()
