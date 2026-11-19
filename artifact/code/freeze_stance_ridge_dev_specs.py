"""Freeze the dev-fitted parameters of every arm worth testing (dev only)."""
import json
from pathlib import Path
import numpy as np
from evidencegraph.dataset_builder import read_jsonl
from evidencegraph.episode_evaluation import episode_inputs, index_dependence_records, index_nli_records
from evidencegraph.episode_inference import edge_variances, fit_logistic_calibrator, score_episode
from evidencegraph.stance_conditional import comparative_features, fit_comparative_model
from audit_stance_ridge_no_override import side_spectra, score_record

root = Path(__file__).resolve().parents[2]  # repository root
eps = read_jsonl(root/'data/processed/evidencegraph20k_validated_subset.jsonl')
nli = index_nli_records(read_jsonl(root/'artifact/runs/subset_edge_episode/nli/deberta_large_dev.jsonl'))
dep = index_dependence_records(read_jsonl(root/'artifact/runs/subset_edge_episode/dependence/predictions.jsonl'))
items=[]
for e in eps:
    if str(e['split'])!='dev': continue
    p,d = episode_inputs(e, nli, dep)
    m=np.asarray([[q['support'],q['neutral'],q['refute']] for q in p])
    items.append({'p':p,'d':d,'s':m[:,0]-m[:,2],'y':int(bool(e['label']))})
y=np.asarray([i['y'] for i in items])

out={'note':'dev-fitted parameters; test split never read','arms':{}}

# candidate: stance split, learned dependence, tau=0, ridge=1000 covariance, generous_5p
recs=[score_record(side_spectra(i['p'],i['d'],0.0),1000.0,'covariance') for i in items]
X=np.asarray([comparative_features(r,'generous_5p') for r in recs])
mdl=fit_comparative_model(X,y,form='generous_5p')
out['arms']['stance_ridge_no_override']={'aggregator':'stance-conditional GLS, learned dependence, ridge (Sigma+lambda I)^+',
 'tau':0.0,'ridge':1000.0,'ridge_mode':'covariance','shrinkage':0.05,'comparative_model':mdl.to_dict()}

# signed ridge, learned dependence, lambda=10 covariance
for name,use_dep,lam in (('signed_ridge_learned',True,10.0),('signed_ridge_identity_dependence',False,1.0)):
    lg=np.asarray([score_episode(i['p'],i['d'],use_dependence=use_dep,ridge=lam,ridge_mode='covariance').evidence_logit for i in items])
    c=fit_logistic_calibrator(lg,y)
    out['arms'][name]={'aggregator':'signed GLS collapse with ridge','use_dependence':use_dep,'ridge':lam,'ridge_mode':'covariance','calibrator':c.to_dict()}

# parameter-free limit and the plain sum
for name,fn in (('uniform_mean_sqrt_n_limit',lambda s: float(np.sum(s)/np.sqrt(s.size))),
                ('uniform_sum',lambda s: float(np.sum(s)))):
    lg=np.asarray([fn(i['s']) for i in items])
    c=fit_logistic_calibrator(lg,y)
    out['arms'][name]={'aggregator':'closed form, no dependence, no covariance, no tuned hyperparameter',
      'logit':'sum_i (p_support_i - p_refute_i) / sqrt(n)' if 'limit' in name else 'sum_i (p_support_i - p_refute_i)',
      'calibrator':c.to_dict()}

# count_sum reference calibrator
lg=np.asarray([score_episode(i['p'],i['d'],use_dependence=False).evidence_logit for i in items])
out['arms']['count_sum']={'aggregator':'frozen count baseline','calibrator':fit_logistic_calibrator(lg,y).to_dict()}

path=root/'artifact/runs/revision/v2/stance_ridge/frozen_dev_specs.json'
path.write_text(json.dumps(out,indent=2,sort_keys=True)+'\n')
print(json.dumps(out,indent=2,sort_keys=True))
