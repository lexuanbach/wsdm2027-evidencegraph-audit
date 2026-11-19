"""Saved-output audit safeguards: joins, leakage and paired uncertainty."""
import copy,sys
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'revision'))
from covariance_saved_audit import paired,validate

def rows():
 return [dict(claim_id=f'c{i}',label=i%2,probability_true=.2 if i%2==0 else .8,split='test',corruption='mixed') for i in range(6)]

def test_pairing_is_by_claim_and_identical_arms_have_zero_interval():
 a=rows();r=paired(a,list(reversed(a)),100)
 for metric in ('accuracy','brier','ece','log_loss'):
  assert r[metric]==dict(difference=0.,low=0.,high=0.)

def test_duplicate_claim_across_splits_is_rejected():
 a=rows();a.append({**a[0],'split':'dev'})
 with pytest.raises(ValueError,match='duplicate'):validate({'a':a})

@pytest.mark.parametrize('field,value',[('label',1),('split','dev'),('corruption','different'),('claim_id','other')])
def test_pair_signature_mismatch_is_rejected(field,value):
 a=rows();b=copy.deepcopy(a);b[0][field]=value
 with pytest.raises(ValueError,match='unaligned'):paired(a,b,100)

def test_nonfinite_probability_is_rejected():
 a=rows();a[0]['probability_true']=float('nan')
 with pytest.raises(ValueError,match='invalid'):validate({'a':a})
