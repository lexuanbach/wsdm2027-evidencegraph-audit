"""Recompute supplement metrics from three saved prediction files.

No model is fitted and no request is made. --verify compares the result
with artifact/runs/saved_covariance/saved-prediction-audit.json.
"""
from pathlib import Path
import hashlib,json,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'artifact/code'))
from evidencegraph.metrics import accuracy,brier_score,expected_calibration_error
ARMS=('ridge_covariance','ridge_precision','uniform_mean_stance')
def log_loss(y,p):
 p=np.clip(p,1e-12,1-1e-12)
 return float(-np.mean(y*np.log(p)+(1-y)*np.log(1-p)))
METRICS={'accuracy':accuracy,'brier':brier_score,'ece':expected_calibration_error,'log_loss':log_loss}
def validate(arms):
 reference=None
 for arm,rows in arms.items():
  keys=[r['claim_id'] for r in rows]
  if len(set(keys))!=len(keys):raise ValueError('duplicate claim or split overlap')
  signature={r['claim_id']:(r['split'],r['label'],r['corruption']) for r in rows}
  if reference is not None and signature!=reference:raise ValueError('unaligned claims, labels, corruption or splits')
  reference=signature
  for r in rows:
   if r['split'] not in ('dev','test') or r['label'] not in (0,1) or not np.isfinite(r['probability_true']) or not 0<=r['probability_true']<=1:raise ValueError('invalid prediction')
def point(rows):
 y=np.array([r['label'] for r in rows]);p=np.array([r['probability_true'] for r in rows])
 return {'n':len(rows),**{k:f(y,p) for k,f in METRICS.items()},'mean_effective_evidence':float(np.mean([r['effective_evidence'] for r in rows]))}
def paired(a,b,n_resamples=1000,seed=217):
 validate({'a':a,'b':b});a=sorted(a,key=lambda r:r['claim_id']);b=sorted(b,key=lambda r:r['claim_id'])
 y=np.array([r['label'] for r in a]);pa=np.array([r['probability_true'] for r in a]);pb=np.array([r['probability_true'] for r in b])
 rng=np.random.default_rng(seed);draws=rng.integers(0,len(y),(n_resamples,len(y)))
 result={'n_claims':len(y)}
 for k,f in METRICS.items():
  delta=float(f(y,pa)-f(y,pb));boot=[f(y[i],pa[i])-f(y[i],pb[i]) for i in draws];lo,hi=np.quantile(boot,[.025,.975])
  result[k]={'difference':delta,'low':float(lo),'high':float(hi)}
 return result
def analyze(directory):
 arms={a:[json.loads(x) for x in (directory/f'{a}_predictions.jsonl').read_text().splitlines()] for a in ARMS};validate(arms)
 for rows in arms.values():
  assert {s:sum(r['split']==s for r in rows) for s in ('dev','test')}=={'dev':359,'test':371}
 result={'scope':'Historical saved-output recomputation, not a new experiment or covariance refit. No test-label selection.','bootstrap':{'resamples':1000,'seed':217,'unit':'claim','interval':'paired percentile 95%, exploratory, unadjusted','difference':'ridge minus uniform'},'metrics':{},'paired_test':{}}
 historical=json.loads((directory/'historical-summary.json').read_text())
 for a,rows in arms.items():
  saved=(historical['ridge_degeneracy']['uniform_mean_stance_limit_arm'] if a=='uniform_mean_stance' else historical['ridge_selected'][a.removeprefix('ridge_')])
  for split in ('dev','test'):
   actual=point([r for r in rows if r['split']==split])
   for k in ('accuracy','brier','ece','log_loss'):
    assert abs(actual[k]-saved[split]['overall'][k])<1e-12,(a,split,k)
 groups=['overall']+sorted({r['corruption'] for r in arms[ARMS[0]]})
 for a,rows in arms.items():
  result['metrics'][a]={s:{g:point([r for r in rows if r['split']==s and (g=='overall' or r['corruption']==g)]) for g in groups} for s in ('dev','test')}
 for a in ARMS[:2]:
  result['paired_test'][a]={g:paired([r for r in arms[a] if r['split']=='test' and (g=='overall' or r['corruption']==g)],[r for r in arms[ARMS[2]] if r['split']=='test' and (g=='overall' or r['corruption']==g)]) for g in ('overall','support_refute_mix')}
 return result
def main():
 d=ROOT/'artifact/runs/saved_covariance';p=d/'saved-prediction-audit.json';result=analyze(d)
 if '--verify' in sys.argv:
  from reproduce_paper import close
  assert close(result,json.loads(p.read_text())),'recomputed metrics differ from the stored values'
 else:
  o=ROOT/'outputs/supplement/saved_covariance_metrics.json';o.parent.mkdir(parents=True,exist_ok=True);o.write_text(json.dumps(result,indent=2)+'\n')
 print(json.dumps({'counts':{'dev':359,'test':371},'paired_test':result['paired_test']}))
if __name__=='__main__':main()
