"""CPU-only main sandbox probes. Run as a file using %run, not pasted cells.
Controls and statistical inference are separate subsequent tasks.
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

def run(root, name):
    source = root / 'data/sandbox_candidate_v2_20261018T095320799792Z'
    raw = (source / 'sandbox_candidate.jsonl').read_bytes()
    assert digest(raw) == HASH, 'Dataset changed'
    rows = [json.loads(s) for s in raw.decode().splitlines() if s.strip()]
    assert len(rows) == len({r['id'] for r in rows}) == 1200
    assert all(r['horizon'] == 3 for r in rows)
    index = {r['id']: i for i, r in enumerate(rows)}
    splits = {s: json.loads((source / 'private' / f'split_{s}.json').read_text()) for s in SEEDS}
    for m in splits.values():
        audit_split(rows, m)
    run_key, shape = MODELS[name]
    act = root / 'activations' / f'{name}_sandbox1200_{run_key}'
    ac = json.loads((act / 'config.json').read_text())
    assert ac['dataset_sha256'] == HASH and ac['scope'] == 'sandbox_candidate_v2_only'
    assert digest(json.dumps(ac, sort_keys=True).encode())[:16] == run_key
    plan_raw = (act / 'analysis_plan.json').read_bytes()
    assert digest(plan_raw) == ac['analysis_plan_sha256']
    plan = json.loads(plan_raw)
    assert plan['dataset_sha256'] == HASH and plan['seeds'] == SEEDS
    assert plan['primary_horizon'] == 3 and plan['models'][name] == [ac['model'], ac['revision']]
    script = Path(__file__).read_bytes()
    config = dict(dataset_sha256=HASH, activation_config=ac, runner_sha256=digest(script),
                  split_manifests=splits, analysis_plan_sha256=digest(plan_raw),
                  python=platform.python_version(), sklearn=sklearn.__version__, numpy=np.__version__,
                  C_grid=C_GRID, max_iter=2000, solver='lbfgs', seeds=SEEDS,
                  selection='validation balanced accuracy; ties smaller C then shallower layer',
                  preprocessing='StandardScaler fit on train only; no refit on validation',
                  scope='sandbox main probes only; horizon 3; all rows; controls pending')
    key = digest(json.dumps(config, sort_keys=True).encode())[:16]
    output = root / 'results' / f'{name}_sandbox_main_probes_{key}'
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'config.json').exists():
        assert json.loads((output / 'config.json').read_text()) == json.loads(json.dumps(config))
    else:
        save(output / 'config.json', config)
    (output / 'runner_source.py').write_bytes(script)
    (output / 'analysis_plan.json').write_bytes(plan_raw)
    print('Results and checkpoints:', output, flush=True)
    selected, all_results = [], []
    for pooling in ('last', 'mean'):
        # One pooling/model at a time: under 650 MB for the activation array.
        x = np.empty((len(rows), *shape), dtype=np.float32)
        for i, r in enumerate(rows):
            log = json.loads((act / (r['id'] + '.json')).read_text())
            assert log['id'] == r['id'] and log['run_key'] == run_key
            with np.load(act / (r['id'] + '.npz'), allow_pickle=False) as f:
                a = f[pooling]
                assert a.shape == shape and a.dtype == np.float32 and np.isfinite(a).all()
                x[i] = a
        print(f'{name}: verified 1200 {pooling} activation arrays', flush=True)
        for seed in SEEDS:
            for task, label_map in LABELS.items():
                parts = {p: [index[i] for i in members if rows[index[i]]['label'] in label_map]
                         for p, members in splits[seed]['ids'].items()}
                ys = {p: np.array([label_map[rows[i]['label']] for i in members]) for p, members in parts.items()}
                assert all(set(y) == set(label_map.values()) for y in ys.values())
                tr, va, te = (parts[p] for p in ('train', 'validation', 'test'))
                block = []
                for layer in range(shape[0]):
                    checkpoint = output / f'{pooling}_{seed}_{task}_layer{layer:02d}.json'
                    identity = dict(seed=seed, task=task, pooling=pooling, layer=layer, run_key=key)
                    if checkpoint.exists():
                        row = json.loads(checkpoint.read_text())
                        assert all(row[k] == v for k, v in identity.items())
                        assert row['test_ids'] == [rows[i]['id'] for i in te]
                    else:
                        try:
                            score, c, clf = fit_selected(x[tr, layer], ys['train'], x[va, layer], ys['validation'], seed)
                            row = dict(**identity, C=c, validation_balanced_accuracy=score, label_map=label_map,
                                       test_ids=[rows[i]['id'] for i in te],
                                       test_groups=[rows[i]['scenario_family'] for i in te],
                                       true_labels=ys['test'].tolist(), **evaluate(clf, x[te, layer], ys['test']))
                            save(checkpoint, row)
                        except Exception as e:
                            save(output / 'last_failure.json', dict(**identity, error=repr(e)))
                            raise
                    block.append(row)
                    print(f'{name} {pooling} seed={seed} {task}: layer {layer+1}/{shape[0]} saved', flush=True)
                selected.append(choose_layer(block))
                all_results.extend(block)
                save(output / 'selected_results.json', selected)
        del x
    save(output / 'layer_results.json', all_results)
    grouped = []
    for task in LABELS:
        for pooling in ('last', 'mean'):
            rs = [r for r in selected if r['task'] == task and r['pooling'] == pooling]
            grouped.append(dict(task=task, pooling=pooling, selected_layers=[r['layer'] for r in rs],
                                test_balanced_accuracy_mean=float(np.mean([r['balanced_accuracy'] for r in rs])),
                                test_balanced_accuracy_sd=float(np.std([r['balanced_accuracy'] for r in rs], ddof=1)),
                                test_n_per_split=[r['n'] for r in rs]))
    summary = dict(model=name, scope='sandbox only', layer_results=len(all_results), selected_blocks=len(selected),
                   selected_results=grouped, final_study_complete=False,
                   pending='controls, minimal-only/horizon ablations, cluster confidence intervals, ScienceWorld and remaining project deliverables',
                   caveat='Three splits overlap; SD is split variability, not a confidence interval. Do not tune using these test scores.',
                   saved_to=str(output))
    save(output / 'summary.json', summary)
    print('SANDBOX MAIN PROBES COMPLETE', flush=True)
    print(json.dumps(summary, indent=2), flush=True)
    return summary

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['qwen', 'llama', 'all'], default='all')
    parser.add_argument('--project', type=Path, default=Path('/content/drive/MyDrive/ADNN_Irreversibility'))
    args = parser.parse_args()
    if not args.project.is_dir():
        from google.colab import drive
        drive.mount('/content/drive')
    with threadpool_limits(limits=2):
        for name in (MODELS if args.model == 'all' else [args.model]):
            run(args.project, name)

if __name__ == '__main__':
    main()
