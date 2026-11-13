"""CPU sandbox controls. Quick baselines and resumable full-search negative controls.
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

def run_controls(root, name, stage):
    from sklearn.dummy import DummyClassifier
    source = root / 'data/sandbox_candidate_v2_20261018T095320799792Z'
    raw = (source / 'sandbox_candidate.jsonl').read_bytes()
    assert digest(raw) == HASH
    rows = [json.loads(s) for s in raw.decode().splitlines() if s.strip()]
    index = {r['id']: i for i, r in enumerate(rows)}
    assert len(index) == len(rows) == 1200
    parent = root / 'results' / MAIN[name]
    pc = json.loads((parent / 'config.json').read_text())
    assert pc['dataset_sha256'] == HASH
    assert digest((parent / 'runner_source.py').read_bytes()) == pc['runner_sha256']
    plan_raw = (parent / 'analysis_plan.json').read_bytes()
    assert digest(plan_raw) == pc['analysis_plan_sha256']
    plan = json.loads(plan_raw)
    ac = pc['activation_config']
    key, shape = MODELS[name]
    act = root / 'activations' / f'{name}_sandbox1200_{key}'
    assert json.loads((act / 'config.json').read_text()) == ac
    splits = {s: json.loads((source / 'private' / f'split_{s}.json').read_text()) for s in SEEDS}
    for s, manifest in splits.items():
        audit_split(rows, manifest)
        assert manifest == pc['split_manifests'][str(s)]
    original = json.loads((parent / 'layer_results.json').read_text())
    assert len(original) == shape[0] * 12
    config = dict(parent_config_sha256=digest((parent/'config.json').read_bytes()),
                  source_sha256=digest(Path(__file__).read_bytes()), numpy=np.__version__,
                  sklearn=sklearn.__version__, python=platform.python_version(),
                  C_grid=C_GRID, shuffle_repeats=20,
                  shuffle_method='Only train labels permuted; validation/test labels unchanged; full C/layer validation search per repeat; not a permutation p-value',
                  verb_method='Saved fixed verb map on original task subset; compare control and real accuracy at real validation-selected layer; also report control-selected layer',
                  caveat='Seven verbs are domain-confounded. No test-score tuning.')
    output = root / 'results' / f'{name}_sandbox_controls_{digest(json.dumps(config,sort_keys=True).encode())[:16]}'
    output.mkdir(parents=True, exist_ok=True)
    if (output/'config.json').exists(): assert json.loads((output/'config.json').read_text()) == config
    else: save(output/'config.json',config)
    (output/'runner_source.py').write_bytes(Path(__file__).read_bytes())
    print('Control checkpoints:',output,flush=True)
    results=[]
    for pooling in ('last','mean'):
        x=None
        if stage == 'extended':
            x=np.empty((1200,*shape),dtype=np.float32)
            for i,r in enumerate(rows):
                log=json.loads((act/(r['id']+'.json')).read_text())
                assert log['id']==r['id'] and log['run_key']==key
                with np.load(act/(r['id']+'.npz'),allow_pickle=False) as f:
                    a=f[pooling];assert a.shape==shape and np.isfinite(a).all()
                    x[i]=a
            print('Loaded',name,pooling,flush=True)
        for seed in SEEDS:
            for task,lmap in LABELS.items():
                parts={p:[index[i] for i in ids if rows[index[i]]['label'] in lmap] for p,ids in splits[seed]['ids'].items()}
                y={p:np.array([lmap[rows[i]['label']] for i in ids]) for p,ids in parts.items()}
                tr,va,te=[parts[p] for p in ('train','validation','test')]
                base=dict(seed=seed,task=task,pooling=pooling,test_ids=[rows[i]['id'] for i in te],test_groups=[rows[i]['scenario_family'] for i in te])
                real=[r for r in original if r['seed']==seed and r['task']==task and r['pooling']==pooling]
                assert len(real)==shape[0]
                winner=choose_layer(real)
                assert winner['test_ids']==base['test_ids'] and winner['true_labels']==y['test'].tolist()
                if stage=='quick':
                    # Embedding scores reuse the already audited input pipeline, with no refitting.
                    embedding=next(r for r in real if r['layer']==0)
                    results.append(dict(kind='embedding',**embedding))
                    if pooling=='mean': continue # text/majority do not depend on pooling
                    for kind in ('majority','action_text'):
                        path=output/f'quick_{seed}_{task}_{kind}.json'
                        if path.exists(): record=json.loads(path.read_text())
                        else:
                            if kind=='majority':
                                model=DummyClassifier(strategy='most_frequent').fit(np.zeros((len(tr),1)),y['train'])
                                score=None;c=None;test_x=np.zeros((len(te),1))
                            else:
                                texts={p:[rows[i]['action_text'] for i in ids] for p,ids in parts.items()}
                                score,c,model=lexical(texts['train'],y['train'],texts['validation'],y['validation'],seed)
                                test_x=texts['test']
                            record=dict(**{**base,'pooling':'not_applicable'},kind=kind,C=c,validation_balanced_accuracy=score,
                                        true_labels=y['test'].tolist(),**evaluate(model,test_x,y['test']))
                            save(path,record)
                        results.append(record)
                    continue
                for control in ['verb']+list(range(20)):
                    kind='fixed_verb' if control=='verb' else 'shuffled_train_labels'
                    tag=f'{pooling}_{seed}_{task}_{control}'
                    labels=y
                    if control=='verb':
                        vmap=plan['verb_control_maps'][task]
                        labels={p:np.array([vmap[json.loads(rows[i]['action_text'])['tool']] for i in ids]) for p,ids in parts.items()}
                    else:
                        labels={**y,'train':shuffled(y['train'],seed,control)}
                    assert all(set(v)==set(lmap.values()) for v in labels.values())
                    block=[]
                    for layer in range(shape[0]):
                        path=output/f'{tag}_layer{layer:02d}.json'
                        if path.exists(): record=json.loads(path.read_text())
                        else:
                            score,c,model=fit_selected(x[tr,layer],labels['train'],x[va,layer],labels['validation'],seed)
                            record=dict(**base,kind=kind,repeat=control,layer=layer,C=c,validation_balanced_accuracy=score,
                                        true_labels=labels['test'].tolist(),
                                        class_counts={p:{str(k):int(sum(v==k)) for k in sorted(set(v))} for p,v in labels.items()},
                                        **evaluate(model,x[te,layer],labels['test']))
                            save(path,record)
                        assert record['test_ids']==base['test_ids'] and record['true_labels']==labels['test'].tolist()
                        block.append(record)
                    chosen=dict(choose_layer(block))
                    if control=='verb':
                        matched=next(r for r in block if r['layer']==winner['layer'])
                        chosen['selectivity_at_real_selected_layer']={
                            'layer':winner['layer'],
                            'real_accuracy':winner['accuracy'],'verb_accuracy':matched['accuracy'],
                            'accuracy_difference':winner['accuracy']-matched['accuracy'],
                            'balanced_accuracy_difference':winner['balanced_accuracy']-matched['balanced_accuracy']}
                    results.append(chosen)
                    save(output/'extended_selected_partial.json',results)
                    print(f'{name} {pooling} seed={seed} {task} control={control}: saved',flush=True)
        del x
    save(output/f'{stage}_results.json',results)
    summary=dict(model=name,stage=stage,selected_results=len(results),
                 completed=True,saved_to=str(output),
                 pending='extended controls' if stage=='quick' else 'ablations and statistical audit',
                 final_study_complete=False)
    if stage=='quick':
        summary['scores']=[{k:r[k] for k in ('kind','seed','task','pooling','balanced_accuracy')} for r in results]
    save(output/f'{stage}_summary.json',summary)
    print('SANDBOX CONTROLS STAGE COMPLETE');print(json.dumps(summary,indent=2))

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--model',choices=['qwen','llama','all'],default='all')
    p.add_argument('--stage',choices=['quick','extended'],default='quick')
    p.add_argument('--project',type=Path,default=Path('/content/drive/MyDrive/ADNN_Irreversibility'))
    a=p.parse_args()
    if not a.project.is_dir():
        from google.colab import drive
        drive.mount('/content/drive')
    with threadpool_limits(limits=2):
        for name in (MODELS if a.model=='all' else [a.model]):run_controls(a.project,name,a.stage)
if __name__=='__main__':main()
