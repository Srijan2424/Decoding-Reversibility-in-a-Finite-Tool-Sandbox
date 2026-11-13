"""CPU-only reduced ablation audit; no retraining."""
import argparse, hashlib, json
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score
MAIN={'qwen':'qwen_sandbox_main_probes_9e373a86af41cb71','llama':'llama_sandbox_main_probes_59d4e14f0cf12b6c'}
CONTROL={'qwen':'qwen_sandbox_reduced_controls_f62303af8685fe01','llama':'llama_sandbox_reduced_controls_62368547b9769846'}
HASH='1073e19b5fae25686374df4c6f7f59ddbfe1130a959e8602395ba7905984f862'
MAPS={'binary':{'reversible':0,'irreversible':1},'three_class':{'reversible':0,'partially_reversible':1,'irreversible':2}}
def read(p):return json.loads(p.read_text())
def save(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False))
def winner(rs):return min(rs,key=lambda r:(-r['validation_balanced_accuracy'],r['layer']))
def verify(r, by, manifest, verbmap=None):
    ids=r['test_ids'];mp=MAPS[r['task']]
    expected=[i for i in manifest['ids']['test'] if by[i]['label'] in mp]
    assert ids==expected and len(ids)==len(set(ids))
    truth=[mp[by[i]['label']] if verbmap is None else verbmap[json.loads(by[i]['action_text'])['tool']] for i in ids]
    assert truth==r['true_labels']
    assert r['test_groups']==[by[i]['scenario_family'] for i in ids]
    y=np.array(truth);pred=np.array(r['predictions']);prob=np.array(r['probabilities']);classes=np.array(r['class_order'])
    assert len(pred)==len(y) and prob.shape==(len(y),len(mp)) and np.isfinite(prob).all()
    assert np.all(prob>=0) and np.allclose(prob.sum(axis=1),1)
    assert list(classes)==list(range(len(mp))) and np.array_equal(pred,classes[prob.argmax(axis=1)])
    auc=roc_auc_score(y,prob[:,1]) if len(mp)==2 else roc_auc_score(y,prob,multi_class='ovr',average='macro',labels=classes)
    metrics={'accuracy':accuracy_score(y,pred),'balanced_accuracy':balanced_accuracy_score(y,pred),'macro_f1':f1_score(y,pred,average='macro',zero_division=0),'auroc':auc}
    for k,v in metrics.items():assert np.isclose(v,r[k],atol=1e-10),f'Metric mismatch: {k}'
    return metrics

def bootstrap(r,seed=3001,n=1000):
    y=np.array(r['true_labels']);p=np.array(r['predictions']);g=np.array(r['test_groups']);groups=np.unique(g)
    rng=np.random.default_rng(seed);values=[];missing=0
    for _ in range(n):
        picked=rng.choice(groups,len(groups),replace=True)
        ix=np.concatenate([np.flatnonzero(g==a) for a in picked])
        if set(y[ix])!=set(y):missing+=1;continue
        values.append(float(np.mean([np.mean(p[ix][y[ix]==c]==c) for c in np.unique(y)])))
    assert values,'No valid bootstrap draws'
    lo,hi=np.quantile(values,[.025,.975])
    return dict(ci_low=float(lo),ci_high=float(hi),scenario_groups=len(groups),bootstrap_valid=len(values),bootstrap_missing_class_draws=missing)

ABLATIONS={'qwen':'qwen_sandbox_reduced_ablations_d2e480e5121813fc','llama':'llama_sandbox_reduced_ablations_9e6de01a62f5214b'}
def expected_ids(by,manifest,evidence,task,case,partition):
    h=3 if case=='minimal_only_h3' else int(case[-1]);mp=MAPS[task]
    return [i for i in manifest['ids'][partition]
            if (case!='minimal_only_h3' or by[i]['minimal_pair_group'])
            and evidence[i]['by_horizon'][str(h)]['label'] in mp]

def main(root):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    src=root/'data/sandbox_candidate_v2_20261018T095320799792Z'
    raw=(src/'sandbox_candidate.jsonl').read_bytes();assert hashlib.sha256(raw).hexdigest()==HASH
    records=[json.loads(s) for s in raw.decode().splitlines() if s.strip()];by={r['id']:r for r in records}
    eraw=(src/'private/label_evidence.json').read_bytes();entries=json.loads(eraw);evidence={e['id']:e for e in entries}
    assert len(by)==len(records)==len(entries)==len(evidence)==1200 and set(by)==set(evidence)
    assert all(evidence[i]['by_horizon']['3']['label']==by[i]['label'] for i in by)
    manifests={s:read(src/'private'/f'split_{s}.json') for s in [42,123,2026]}
    for m in manifests.values():
        assert m['dataset_sha256']==HASH
        flat=[i for ids in m['ids'].values() for i in ids];assert len(flat)==len(set(flat))==1200 and set(flat)==set(by)
        owner={i:p for p,ids in m['ids'].items() for i in ids}
        for field in ['scenario_family','minimal_pair_group']:
            groups={}
            for i,r in by.items():
                if r[field] is not None:groups.setdefault(r[field],set()).add(owner[i])
            assert all(len(v)==1 for v in groups.values())
    audited=[];sources={}
    for model,folder in ABLATIONS.items():
        dest=root/'results'/folder;parent=root/'results'/MAIN[model]
        cfg=read(dest/'config.json');pc=read(parent/'config.json')
        assert cfg['dataset_sha256']==HASH and cfg['evidence_sha256']==hashlib.sha256(eraw).hexdigest()
        assert cfg['parent_config_sha256']==hashlib.sha256((parent/'config.json').read_bytes()).hexdigest()
        assert cfg['source_sha256']==hashlib.sha256((dest/'runner_source.py').read_bytes()).hexdigest()
        assert pc['runner_sha256']==hashlib.sha256((parent/'runner_source.py').read_bytes()).hexdigest()
        for seed,m in manifests.items():assert m==pc['split_manifests'][str(seed)]
        mainrows=read(parent/'layer_results.json');results=read(dest/'results.json')
        keys={(s,t,p,c) for s in [42,123,2026] for t in MAPS for p in ['last','mean'] for c in ['minimal_only_h3','all_h1','all_h2']}
        assert len(results)==36 and {(r['seed'],r['task'],r['pooling'],r['case']) for r in results}==keys
        for r in results:
            seed=r['seed'];task=r['task'];case=r['case'];h=3 if case=='minimal_only_h3' else int(case[-1]);mp=MAPS[task]
            m=manifests[seed]
            # Create a label-adjusted view for the shared prediction/metric checks.
            adjusted={i:{**v,'label':evidence[i]['by_horizon'][str(h)]['label']} for i,v in by.items()}
            test=expected_ids(by,m,evidence,task,case,'test')
            assert r['test_ids']==test
            verify(r,adjusted,{'ids':{'test':test}})
            for part in ['train','validation','test']:
                ids=expected_ids(by,m,evidence,task,case,part)
                assert r['partition_sizes'][part]==len(ids)
                labels=[mp[adjusted[i]['label']] for i in ids]
                assert r['class_counts'][part]=={str(c):labels.count(c) for c in sorted(set(labels))}
            assert r['n']==len(test)
            block=[a for a in mainrows if (a['seed'],a['task'],a['pooling'])==(seed,task,r['pooling'])]
            assert {a['layer'] for a in block}==set(range(29 if model=='qwen' else 33))
            w=winner(block);assert r['layer']==w['layer'] and r['C']==w['C']
            row=dict(model=model,**{k:r[k] for k in ['seed','task','pooling','case','layer','C','n','accuracy','balanced_accuracy','macro_f1','auroc']},**bootstrap(r,seed=3201+seed))
            if case=='minimal_only_h3':
                original=r['original_all_training_probe_on_same_minimal_test'];lookup={i:j for j,i in enumerate(w['test_ids'])}
                preds=[w['predictions'][lookup[i]] for i in test]
                assert preds==original['predictions']
                assert [w['true_labels'][lookup[i]] for i in test]==r['true_labels']
                ba=balanced_accuracy_score(r['true_labels'],preds);acc=accuracy_score(r['true_labels'],preds)
                assert np.isclose(ba,original['balanced_accuracy']) and np.isclose(acc,original['accuracy'])
                row['original_probe_matched_test_balanced_accuracy']=float(ba)
                row['matched_test_balanced_accuracy_difference']=r['balanced_accuracy']-float(ba)
            audited.append(row)
        sources[model]=str(dest)
        print(model.upper(),'36 ablations: labels, subsets, metrics and fixed settings verified',flush=True)
    out=root/'results'/('sandbox_ablation_audit_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'));out.mkdir(parents=True)
    df=pd.DataFrame(audited);df.to_csv(out/'ablations_per_split_with_ci.csv',index=False)
    grouped=df.groupby(['model','task','pooling','case']).agg(balanced_accuracy_mean=('balanced_accuracy','mean'),split_sd=('balanced_accuracy','std'),test_n=('n','first')).reset_index()
    grouped.to_csv(out/'ablation_comparison.csv',index=False)
    df[df.case=='minimal_only_h3'].to_csv(out/'minimal_only_matched_comparison.csv',index=False)
    fig,axes=plt.subplots(2,2,figsize=(12,8));cases=['all_h1','all_h2','minimal_only_h3']
    for ax,(task,pool) in zip(axes.flat,[(t,p) for t in MAPS for p in ['last','mean']]):
        for j,model in enumerate(MAIN):
            d=grouped[(grouped.model==model)&(grouped.task==task)&(grouped.pooling==pool)].set_index('case').loc[cases]
            ax.bar(np.arange(3)+(j-.5)*.35,d.balanced_accuracy_mean,width=.35,label=model)
        ax.set_xticks(range(3),['All: horizon 1','All: horizon 2','Minimal only: horizon 3']);ax.set(title=f'{task}, {pool}',ylabel='Mean balanced accuracy',ylim=(0,1.05));ax.legend()
    fig.suptitle('Reduced ablations: different label/subset conditions; fixed original settings');fig.tight_layout();fig.savefig(out/'ablation_comparison.png',dpi=180);fig.savefig(out/'ablation_comparison.pdf');plt.close(fig)
    notes='''# Reduced sandbox ablations
Both models were refitted with training-only scaling at the layer and C originally selected on validation data for horizon 3/all rows. This deadline-driven revision was made after main test results were visible; it is not a full hyperparameter search for each condition.

Minimal-only excludes reversible no-ops. The matched-test table compares the original all-training probe with the minimal-training probe on exactly the same minimal test examples. Raw all-data and minimal-only scores have different test composition.

Horizons 1 and 2 change labels, not prompts or activations. These experiments measure label-horizon sensitivity, not horizon-conditioned reasoning. Binary membership changes when examples become partially reversible or irreversible. Do not infer a monotonic relationship between horizon and accuracy.

Intervals use 1,000 scenario bootstrap draws per split, with only four held-out scenario groups. They may be unstable or degenerate. Overlapping splits are not independent; split SD is not a confidence interval. No claim of statistical significance is made.

ScienceWorld/cross-environment work, final annotation and remaining project deliverables are not completed by this audit.
'''
    (out/'REPORT_NOTES.md').write_text(notes);(out/'audit_source.py').write_bytes(Path(__file__).read_bytes())
    summary=dict(audit_passed=True,ablation_records_verified=len(audited),models=list(ABLATIONS),bootstrap_resamples=1000,source_runs=sources,saved_to=str(out),final_study_complete=False)
    save(out/'summary.json',summary);print('SANDBOX ABLATION AUDIT COMPLETE');print(json.dumps(summary,indent=2))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--project',type=Path,default=Path('/content/drive/MyDrive/ADNN_Irreversibility'));a=p.parse_args()
    if not a.project.is_dir():
        from google.colab import drive
        drive.mount('/content/drive')
    main(a.project)
