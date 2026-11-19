"""Regression checks for the analytical figure and recovered aggregation controls."""
from pathlib import Path
import json
import numpy as np
import pytest
from make_information_figure import information
from build_subset_tables import baseline_table

ROOT = Path(__file__).resolve().parents[1]
PUBLIC = (ROOT / 'artifact/runs').is_dir()
RUNS = ROOT / ('artifact/runs' if PUBLIC else 'runs')
SAVED = ROOT / ('artifact/runs/saved_covariance' if PUBLIC else 'revision/covariance')

@pytest.mark.parametrize('rho', [0, .2, .5, .9])
def test_information_matches_matrix_inverse(rho):
    for n in [1, 2, 7, 20]:
        covariance = (1-rho)*np.eye(n)+rho*np.ones((n,n))
        expected = np.ones(n) @ np.linalg.inv(covariance) @ np.ones(n)
        assert information(n, rho) == pytest.approx(expected, abs=1e-11)

def test_duplicate_invariance_for_nontrivial_covariance():
    rng = np.random.default_rng(217)
    for n in [2, 4, 7]:
        m=rng.normal(size=(n,n));cov=m@m.T+np.eye(n)
        before=np.ones(n) @ np.linalg.solve(cov,np.ones(n))
        for j in range(n):
            a=np.vstack([np.eye(n),np.eye(n)[j]])
            after=np.ones(n+1) @ np.linalg.pinv(a@cov@a.T) @ np.ones(n+1)
            assert after == pytest.approx(before, abs=1e-11)
    assert information(20,1)==1

def test_table_preserves_recovered_arm_identities():
    base=json.loads((RUNS/'subset_edge_episode/episode_baselines/summary.json').read_text())
    saved=json.loads((SAVED/'saved-prediction-audit.json').read_text())
    table=baseline_table(base,saved)
    for arm,label in [('ridge_covariance','Covariance ridge'),('ridge_precision','Precision ridge'),('uniform_mean_stance','Uniform')]:
        row=next(line for line in table.splitlines() if line.startswith(label))
        m=saved['metrics'][arm]['test']['overall']
        assert f"{100*m['accuracy']:.1f}" in row
        assert f"{m['ece']:.3f}" in row
    assert 'Diagonal ridge' not in table
