"""Reduced CPU ablations: minimal-only horizon 3 and all-data horizons 1 and 2. Fixed original validation-selected configuration; results saved separately.
"""
import argparse
import hashlib
import json
import platform
import warnings
from pathlib import Path
import numpy as np
import sklearn
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits

HASH = '1073e19b5fae25686374df4c6f7f59ddbfe1130a959e8602395ba7905984f862'
C_GRID = [0.01, 0.1, 1.0]
MODELS = {'qwen': ('d5caedf45080a358', (29, 3584)),
          'llama': ('8ef845f862b9e2da', (33, 4096))}
SEEDS = [42, 123, 2026]
LABELS = {'binary': {'reversible': 0, 'irreversible': 1},
          'three_class': {'reversible': 0, 'partially_reversible': 1, 'irreversible': 2}}

def digest(data):
    return hashlib.sha256(data).hexdigest()

def save(path, value):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False))
    tmp.replace(path)

def fit_selected(xt, yt, xv, yv, seed):
    best = None
    for c in C_GRID:
        model = make_pipeline(StandardScaler(), LogisticRegression(
            C=c, solver='lbfgs', max_iter=2000, random_state=seed))
        with warnings.catch_warnings():
            warnings.simplefilter('error', ConvergenceWarning)
            model.fit(xt, yt)
        score = float(balanced_accuracy_score(yv, model.predict(xv)))
        if best is None or score > best[0]:
            best = (score, c, model)
    return best

def choose_layer(rows):
    return min(rows, key=lambda r: (-r['validation_balanced_accuracy'], r['layer']))

def evaluate(model, x, y):
    pred = model.predict(x)
    proba = model.predict_proba(x)
    auc = (roc_auc_score(y, proba[:, 1]) if len(model.classes_) == 2 else
           roc_auc_score(y, proba, labels=model.classes_, multi_class='ovr', average='macro'))
    return dict(n=len(y), accuracy=float(accuracy_score(y, pred)),
                balanced_accuracy=float(balanced_accuracy_score(y, pred)),
                macro_f1=float(f1_score(y, pred, average='macro', zero_division=0)),
                auroc=float(auc), predictions=pred.tolist(), probabilities=proba.tolist(),
                class_order=model.classes_.tolist())

def audit_split(rows, manifest):
    assert manifest['dataset_sha256'] == HASH
    ids = manifest['ids']
    assert set(ids) == {'train', 'validation', 'test'}
    flat = [i for part in ids.values() for i in part]
    by = {r['id']: r for r in rows}
    assert len(flat) == len(set(flat)) == len(rows) and set(flat) == set(by)
    owner = {i: part for part, members in ids.items() for i in members}
    for key in ('scenario_family', 'minimal_pair_group'):
        groups = {}
        for r in rows:
            if r[key] is not None:
                groups.setdefault(r[key], set()).add(owner[r['id']])
        assert all(len(v) == 1 for v in groups.values()), f'{key} leakage'
    return ids

MAIN = {'qwen': 'qwen_sandbox_main_probes_9e373a86af41cb71',
        'llama': 'llama_sandbox_main_probes_59d4e14f0cf12b6c'}

def shuffled(y, seed, repeat):
    # Identical permutations across layers, poolings and models.
    return np.random.default_rng(np.random.SeedSequence([seed, repeat, 2801])).permutation(y)

def lexical(xt, yt, xv, yv, seed):
    from sklearn.feature_extraction.text import CountVectorizer
    best = None
    for c in C_GRID:
        model = make_pipeline(CountVectorizer(), LogisticRegression(C=c, solver='lbfgs', max_iter=2000, random_state=seed))
        with warnings.catch_warnings():
            warnings.simplefilter('error', ConvergenceWarning)
            model.fit(xt, yt)
        score = float(balanced_accuracy_score(yv, model.predict(xv)))
        if best is None or score > best[0]: best = (score, c, model)
    return best

def fit_fixed(x, y, c, seed):
    model=make_pipeline(StandardScaler(),LogisticRegression(C=c,solver='lbfgs',max_iter=2000,random_state=seed))
    with warnings.catch_warnings():
        warnings.simplefilter('error',ConvergenceWarning)
        model.fit(x,y)
    return model


def subset(rows, ids, index, evidence, task, case):
    labelmap=LABELS[task];h=3 if case=='minimal_only_h3' else int(case[-1])
    chosen=[];labels=[]
    for ident in ids:
        r=rows[index[ident]]
        if case=='minimal_only_h3' and not r['minimal_pair_group']:continue
        label=evidence[ident]['by_horizon'][str(h)]['label']
        if label in labelmap:chosen.append(index[ident]);labels.append(labelmap[label])
    return chosen,np.array(labels,dtype=int)

def run(root,name):
    source=root/'data/sandbox_candidate_v2_20261018T095320799792Z'
    raw=(source/'sandbox_candidate.jsonl').read_bytes();assert digest(raw)==HASH
    rows=[json.loads(s) for s in raw.decode().splitlines() if s.strip()];index={r['id']:i for i,r in enumerate(rows)}
    evidence_raw=(source/'private/label_evidence.json').read_bytes()
    entries=json.loads(evidence_raw);evidence={r['id']:r for r in entries}
    assert len(rows)==len(index)==len(entries)==len(evidence)==1200 and set(index)==set(evidence)
    assert all(evidence[r['id']]['by_horizon']['3']['label']==r['label'] for r in rows)
    parent=root/'results'/MAIN[name];pc=json.loads((parent/'config.json').read_text())
    assert digest((parent/'runner_source.py').read_bytes())==pc['runner_sha256']
    original=json.loads((parent/'layer_results.json').read_text());ac=pc['activation_config']
    key,shape=MODELS[name];act=root/'activations'/f'{name}_sandbox1200_{key}'
    assert json.loads((act/'config.json').read_text())==ac and ac['dataset_sha256']==HASH
    splits={seed:json.loads((source/'private'/f'split_{seed}.json').read_text()) for seed in SEEDS}
    for seed,m in splits.items():
        audit_split(rows,m);assert m==pc['split_manifests'][str(seed)]
    cases=['minimal_only_h3','all_h1','all_h2']
    config=dict(model=name,dataset_sha256=HASH,evidence_sha256=digest(evidence_raw),
        parent_config_sha256=digest((parent/'config.json').read_bytes()),source_sha256=digest(Path(__file__).read_bytes()),
        sklearn=sklearn.__version__,numpy=np.__version__,cases=cases,
        method='Deadline-driven reduced ablation after main results observed: fixed original h3 all-data validation-selected layer/C; train-only scaler and probe refitted per condition; no new hyperparameter search',
        horizon_caveat='Same prompts/activations across horizons: label-horizon sensitivity, not horizon-conditioned model behaviour',
        minimal_caveat='Excluded rows are reversible no-ops; comparison changes class/task composition. Matched-subset original-probe scores also reported.')
    out=root/'results'/f'{name}_sandbox_reduced_ablations_{digest(json.dumps(config,sort_keys=True).encode())[:16]}'
    out.mkdir(parents=True,exist_ok=True)
    if (out/'config.json').exists():assert json.loads((out/'config.json').read_text())==config
    else:save(out/'config.json',config)
    (out/'runner_source.py').write_bytes(Path(__file__).read_bytes())
    print('Ablation checkpoints:',out,flush=True);results=[]
    for pooling in ['last','mean']:
        x=np.empty((1200,*shape),dtype=np.float32)
        for i,r in enumerate(rows):
            log=json.loads((act/(r['id']+'.json')).read_text());assert log['id']==r['id'] and log['run_key']==key
            with np.load(act/(r['id']+'.npz'),allow_pickle=False) as f:
                a=f[pooling];assert a.shape==shape and np.isfinite(a).all();x[i]=a
        for seed in SEEDS:
            for task in LABELS:
                block=[r for r in original if (r['seed'],r['task'],r['pooling'])==(seed,task,pooling)]
                assert {r['layer'] for r in block}==set(range(shape[0]))
                w=choose_layer(block);layer=w['layer'];c=w['C']
                for case in cases:
                    parts={p:subset(rows,ids,index,evidence,task,case) for p,ids in splits[seed]['ids'].items()}
                    tr,yt=parts['train'];va,yv=parts['validation'];te,ye=parts['test']
                    assert all(set(y)==set(LABELS[task].values()) for _,y in parts.values()),'Condition lacks a class; stop for review'
                    ident=dict(seed=seed,task=task,pooling=pooling,case=case,layer=layer,C=c)
                    path=out/f'{pooling}_{seed}_{task}_{case}.json'
                    if path.exists():
                        result=json.loads(path.read_text());assert all(result[k]==v for k,v in ident.items())
                        assert result['true_labels']==ye.tolist() and result['test_ids']==[rows[i]['id'] for i in te]
                    else:
                        model=fit_fixed(x[tr,layer],yt,c,seed)
                        result=dict(**ident,true_labels=ye.tolist(),test_ids=[rows[i]['id'] for i in te],test_groups=[rows[i]['scenario_family'] for i in te],
                            partition_sizes={p:len(ii) for p,(ii,y) in parts.items()},
                            class_counts={p:{str(k):int(sum(y==k)) for k in sorted(set(y))} for p,(ii,y) in parts.items()},
                            validation_balanced_accuracy=float(balanced_accuracy_score(yv,model.predict(x[va,layer]))),
                            **evaluate(model,x[te,layer],ye))
                        if case=='minimal_only_h3':
                            lookup={ident:i for i,ident in enumerate(w['test_ids'])}
                            selected=[lookup[rows[i]['id']] for i in te]
                            old=np.array(w['predictions'])[selected]
                            assert np.array_equal(np.array(w['true_labels'])[selected],ye)
                            result['original_all_training_probe_on_same_minimal_test']=dict(balanced_accuracy=float(balanced_accuracy_score(ye,old)),accuracy=float(accuracy_score(ye,old)),predictions=old.tolist())
                        save(path,result)
                    results.append(result)
                    print(name,pooling,seed,task,case,'saved',flush=True)
        del x
    assert len(results)==36
    save(out/'results.json',results)
    groups=[]
    for task in LABELS:
        for pooling in ['last','mean']:
            for case in cases:
                rs=[r for r in results if (r['task'],r['pooling'],r['case'])==(task,pooling,case)]
                g=dict(task=task,pooling=pooling,case=case,test_balanced_accuracy_mean=float(np.mean([r['balanced_accuracy'] for r in rs])),split_sd=float(np.std([r['balanced_accuracy'] for r in rs],ddof=1)),test_n_per_split=[r['n'] for r in rs])
                if case=='minimal_only_h3':g['original_probe_same_minimal_test_mean']=float(np.mean([r['original_all_training_probe_on_same_minimal_test']['balanced_accuracy'] for r in rs]))
                groups.append(g)
    summary=dict(model=name,completed=True,fits_completed=36,scope='Reduced fixed-configuration sandbox ablations only',comparison=groups,saved_to=str(out),final_study_complete=False,
        pending='Ablation statistical audit; ScienceWorld/cross-environment; final annotation; remaining evaluation and submission materials')
    save(out/'summary.json',summary)
    print('REDUCED SANDBOX ABLATIONS COMPLETE');print(json.dumps(summary,indent=2))

def main():
    p=argparse.ArgumentParser();p.add_argument('--model',choices=['qwen','llama','all'],default='all');p.add_argument('--project',type=Path,default=Path('/content/drive/MyDrive/ADNN_Irreversibility'));a=p.parse_args()
    if not a.project.is_dir():
        from google.colab import drive
        drive.mount('/content/drive')
    with threadpool_limits(limits=2):
        for name in (MODELS if a.model=='all' else [a.model]):run(a.project,name)
if __name__=='__main__':main()
