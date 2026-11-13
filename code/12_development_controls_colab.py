"""CPU development diagnostics. No model loading or final-study inference."""
import json, hashlib, warnings
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits

RUNS = {
    'qwen': ('qwen_dev96_c9f4151a0c372771', 'qwen_dev_probes_20261018T081513134945Z', (29,3584)),
    'llama': ('llama_dev96_91fcd7afe3fcef85', 'llama_dev_probes_20261018T090343902595Z', (33,4096)),
}
SHUFFLES = 20
BOOTSTRAPS = 1000

def ba(y, pred):
    return float(np.mean([np.mean(pred[y == c] == c) for c in np.unique(y)]))

def bootstrap(y, pred, groups, seed):
    rng = np.random.default_rng(seed)
    unique = sorted(set(groups))
    members = [np.flatnonzero(groups == g) for g in unique]
    values = []
    attempts = 0
    while len(values) < BOOTSTRAPS and attempts < BOOTSTRAPS * 20:
        attempts += 1
        idx = np.concatenate([members[i] for i in rng.integers(len(members), size=len(members))])
        if len(np.unique(y[idx])) != len(np.unique(y)):
            continue
        values.append(ba(y[idx], pred[idx]))
    assert len(values) == BOOTSTRAPS, 'Insufficient bootstrap draws containing every class.'
    return {'scenario_groups':len(unique), 'percentile_95_interval':np.quantile(values,[.025,.975]).tolist(),
            'draws':len(values), 'rejected_missing_class_draws':attempts-len(values)}

def main(project):
    source = project/'data/expanded_development_20261018T075007949936Z'
    raw = (source/'development_dataset.jsonl').read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    assert digest == 'd0ad9dc36284db061aca680deb0985a6bb200931fad28f656da7218d87b25a8a'
    records = [json.loads(s) for s in raw.decode().splitlines() if s.strip()]
    by_id = {r['id']:r for r in records}
    assert len(by_id) == len(records) == 96
    output = project/'results'/('development_controls_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    output.mkdir(parents=True, exist_ok=False)
    verbs = sorted({json.loads(r['action_text'])['tool'] for r in records})
    results = []
    for model_name,(activation_name,result_name,shape) in RUNS.items():
        activation_dir = project/'activations'/activation_name
        config = json.loads((activation_dir/'config.json').read_text())
        assert config['dataset_sha256'] == digest
        selections = json.loads((project/'results'/result_name/'selected_results.json').read_text())
        assert len(selections) == 12
        for row in selections:
            seed,task,pooling = row['seed'],row['task'],row['pooling']
            manifest = json.loads((source/'split_audit_20261018T075128231887Z'/f'split_{seed}.json').read_text())
            assert manifest['dataset_sha256'] == digest
            labels = row['label_map']
            train_ids = [i for i in manifest['ids']['train'] if by_id[i]['label'] in labels]
            eval_ids = [i for i in manifest['ids']['development_eval'] if by_id[i]['label'] in labels]
            assert eval_ids == row['eval_ids'] and not set(train_ids)&set(eval_ids)
            def load(ids):
                vectors = []
                for ident in ids:
                    log = json.loads((activation_dir/(ident+'.json')).read_text())
                    assert log['id'] == ident and log['run_key'] == activation_name.rsplit('_',1)[1]
                    with np.load(activation_dir/(ident+'.npz'),allow_pickle=False) as data:
                        x = data[pooling]
                        assert x.shape == shape and np.isfinite(x).all()
                        vectors.append(x[row['layer']])
                return np.stack(vectors)
            xt,xe = load(train_ids),load(eval_ids)
            yt = np.array([labels[by_id[i]['label']] for i in train_ids])
            ye = np.array([labels[by_id[i]['label']] for i in eval_ids])
            pred = np.array(row['predictions'])
            assert len(pred) == len(ye) and np.isclose(ba(ye,pred),row['balanced_accuracy'])
            groups = np.array([by_id[i]['scenario_family'] for i in eval_ids])
            # Same shuffle schedule for both models and pooling choices.
            rng = np.random.default_rng(seed + (10000 if task == 'three_class' else 0))
            scores = []
            for repeat in range(SHUFFLES):
                shuffled = rng.permutation(yt)
                clf = make_pipeline(StandardScaler(),LogisticRegression(C=row['C'],solver='lbfgs',max_iter=2000,random_state=seed))
                with warnings.catch_warnings():
                    warnings.simplefilter('error',ConvergenceWarning)
                    clf.fit(xt,shuffled)
                scores.append(ba(ye,clf.predict(xe)))
            result = {'model':model_name,'seed':seed,'task':task,'pooling':pooling,
                      'fixed_validation_selected_layer':row['layer'],'fixed_C':row['C'],
                      'observed_balanced_accuracy':ba(ye,pred),
                      'bootstrap':bootstrap(ye,pred,groups,seed),
                      'shuffled_training_label_scores':scores,
                      'shuffle_mean':float(np.mean(scores)), 'shuffle_sd':float(np.std(scores,ddof=1))}
            results.append(result)
            (output/'detailed_results.json').write_text(json.dumps(results,indent=2))
            print(f"Finished {model_name}, seed={seed}, {task}, {pooling}; shuffled mean={np.mean(scores):.3f}",flush=True)
    summary = {'scope':'development_only','completed_blocks':len(results),'shuffles_per_block':SHUFFLES,
               'bootstrap_draws_per_block':BOOTSTRAPS,'verbs':verbs,
               'verb_control_status':'Deferred: only three candidate verbs, confounded with domains; expand verb diversity before interpreting selectivity.',
               'method_notes':[
                   'Training labels shuffled globally; validation-selected layer and C held fixed. Diagnostic only, not a full selection-aware permutation test or p-value.',
                   'Bootstrap resamples evaluation scenario groups within each split, keeping predictions fixed. It excludes training/selection uncertainty.',
                   'Each evaluation split has very few groups. Perfect predictions yield a degenerate bootstrap interval; this does not establish population certainty.',
                   'Overlapping splits are not independent samples; intervals are not pooled across splits.'
               ],'results':[ {k:v for k,v in r.items() if k != 'shuffled_training_label_scores'} for r in results],
               'saved_to':str(output)}
    (output/'summary.json').write_text(json.dumps(summary,indent=2))
    print('DEVELOPMENT CONTROLS COMPLETE')
    print(json.dumps(summary,indent=2))

if __name__ == '__main__':
    from google.colab import drive
    drive.mount('/content/drive')
    with threadpool_limits(limits=2):
        main(Path('/content/drive/MyDrive/ADNN_Irreversibility'))
