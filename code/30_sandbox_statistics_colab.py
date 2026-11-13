"""CPU-only verification and descriptive statistics; no fitting or test-based selection."""
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

def main(root):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    src=root/'data/sandbox_candidate_v2_20261018T095320799792Z'
    raw=(src/'sandbox_candidate.jsonl').read_bytes();assert hashlib.sha256(raw).hexdigest()==HASH
    rows=[json.loads(s) for s in raw.decode().splitlines() if s.strip()];by={r['id']:r for r in rows}
    assert len(by)==1200
    manifests={s:read(src/'private'/f'split_{s}.json') for s in [42,123,2026]}
    for m in manifests.values():
        assert m['dataset_sha256']==HASH
        owner={i:p for p,ids in m['ids'].items() for i in ids}
        flat=[i for ids in m['ids'].values() for i in ids]
        assert len(flat)==len(set(flat))==1200 and set(flat)==set(by)
        for key in ['scenario_family','minimal_pair_group']:
            owners={}
            for r in rows:
                if r[key] is not None:owners.setdefault(r[key],set()).add(owner[r['id']])
            assert all(len(v)==1 for v in owners.values())
    selected=[];controls=[];layers=[];count=0
    provenance={}
    for model in MAIN:
        parent=root/'results'/MAIN[model];ctrl=root/'results'/CONTROL[model]
        pc=read(parent/'config.json');cc=read(ctrl/'config.json');plan=read(parent/'analysis_plan.json')
        assert cc['parent_config_sha256']==hashlib.sha256((parent/'config.json').read_bytes()).hexdigest()
        assert pc['dataset_sha256']==HASH and cc['shuffle_repeats']==5
        assert hashlib.sha256((parent/'runner_source.py').read_bytes()).hexdigest()==pc['runner_sha256']
        assert hashlib.sha256((ctrl/'runner_source.py').read_bytes()).hexdigest()==cc['source_sha256']
        assert hashlib.sha256((parent/'analysis_plan.json').read_bytes()).hexdigest()==pc['analysis_plan_sha256']
        for s in manifests:assert manifests[s]==pc['split_manifests'][str(s)]
        allrows=read(parent/'layer_results.json');chosen=read(parent/'selected_results.json');cr=read(ctrl/'extended_results.json')
        nl=29 if model=='qwen' else 33
        assert len(allrows)==nl*12 and len(chosen)==12 and len(cr)==72
        assert len({(r['seed'],r['task'],r['pooling'],r['layer']) for r in allrows})==nl*12
        assert len({(r['seed'],r['task'],r['pooling']) for r in chosen})==12
        for r in allrows:
            verify(r,by,manifests[r['seed']]);count+=1
            layers.append(dict(model=model,**{k:r[k] for k in ['seed','task','pooling','layer','balanced_accuracy','validation_balanced_accuracy']}))
        for seed in [42,123,2026]:
            for task in MAPS:
                for pool in ['last','mean']:
                    block=[r for r in allrows if (r['seed'],r['task'],r['pooling'])==(seed,task,pool)]
                    assert {r['layer'] for r in block}==set(range(nl))
                    w=winner(block)
                    assert next(r for r in chosen if (r['seed'],r['task'],r['pooling'])==(seed,task,pool))==w
                    selected.append(dict(model=model,**{k:w[k] for k in ['seed','task','pooling','layer','C','n','accuracy','balanced_accuracy','macro_f1','auroc']},**bootstrap(w,seed=3001+seed)))
                    cb=[r for r in cr if (r['seed'],r['task'],r['pooling'])==(seed,task,pool)]
                    assert len(cb)==6 and {r['repeat'] for r in cb}=={'verb',0,1,2,3,4}
                    for r in cb:
                        assert r['layer']==w['layer'] and r['C']==w['C']
                        vm=plan['verb_control_maps'][task] if r['repeat']=='verb' else None
                        verify(r,by,manifests[seed],vm);count+=1
                        if vm is not None:
                            delta=w['accuracy']-r['accuracy']
                            assert np.isclose(delta,r['selectivity_at_real_selected_layer']['accuracy_difference'])
                        else:delta=None
                        controls.append(dict(model=model,seed=seed,task=task,pooling=pool,repeat=r['repeat'],kind=r['kind'],balanced_accuracy=r['balanced_accuracy'],accuracy_selectivity=delta))
        provenance[model]=dict(main=str(parent),controls=str(ctrl))
        print(model.upper(),'saved predictions, metrics and validation selection verified',flush=True)
    out=root/'results'/('sandbox_statistics_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'));out.mkdir(parents=True)
    df=pd.DataFrame(selected);cf=pd.DataFrame(controls);lf=pd.DataFrame(layers)
    df.to_csv(out/'selected_per_split_with_ci.csv',index=False);cf.to_csv(out/'reduced_controls.csv',index=False);lf.to_csv(out/'all_layers.csv',index=False)
    summary=df.groupby(['model','task','pooling']).agg(balanced_accuracy_mean=('balanced_accuracy','mean'),split_sd=('balanced_accuracy','std'),accuracy_mean=('accuracy','mean'),macro_f1_mean=('macro_f1','mean'),auroc_mean=('auroc','mean')).reset_index()
    summary.to_csv(out/'main_comparison.csv',index=False)
    fig,axes=plt.subplots(2,2,figsize=(12,8))
    for ax,(task,pool) in zip(axes.flat,[(t,p) for t in MAPS for p in ['last','mean']]):
        for model in MAIN:
            d=lf[(lf.task==task)&(lf.pooling==pool)&(lf.model==model)].groupby('layer').balanced_accuracy.mean()
            ax.plot(d.index,d.values,label=model)
        ax.axhline(.5 if task=='binary' else 1/3,color='gray',linestyle='--',label='chance')
        ax.set(title=f'{task}, {pool}',xlabel='Hidden-state index (0 = embedding)',ylabel='Mean test balanced accuracy',ylim=(0,1.02));ax.legend()
    fig.suptitle('Descriptive layer curves; selection used validation only');fig.tight_layout();fig.savefig(out/'layer_curves.png',dpi=180);fig.savefig(out/'layer_curves.pdf');plt.close(fig)
    fig,ax=plt.subplots(figsize=(10,9));positions=np.arange(len(df))
    ax.hlines(positions,df.ci_low,df.ci_high,color='tab:blue');ax.plot(df.balanced_accuracy,positions,'o',color='tab:blue')
    ax.set_yticks(positions,[f'{r.model} | {r.task} | {r.pooling} | seed {r.seed}' for r in df.itertuples()]);ax.set(xlabel='Test balanced accuracy; scenario-bootstrap 95% interval',xlim=(0,1.02));fig.tight_layout();fig.savefig(out/'per_split_intervals.png',dpi=180);plt.close(fig)
    notes='''# Sandbox results: scope and limitations
Main probes and reduced controls were checked against saved test labels, predictions and probabilities. Layers were selected using validation scores only.

Intervals use 1,000 scenario-cluster bootstrap draws separately per split. There are only four held-out scenario groups per split: intervals can be unstable or degenerate and do not demonstrate broad generalisation. Splits overlap; their SD is descriptive variability, not a confidence interval. No pooled independent-sample significance test is reported.

The five-shuffle control is a deadline-driven revision made after main results were visible. Layer and C are fixed from original validation choices; it is not the original full-search control or a formal permutation test. All five repetitions are retained. Verb labels are easy to decode from seven familiar, domain-confounded verbs; negative selectivity must be reported.

Layer curves are descriptive and must not be used to choose new settings. Minimal-only and horizon ablations, ScienceWorld transfer, final annotation and remaining submission work are not completed by this script.
'''
    (out/'REPORT_NOTES.md').write_text(notes);(out/'audit_source.py').write_bytes(Path(__file__).read_bytes())
    result=dict(audit_passed=True,prediction_records_verified=count,selected_main_blocks=len(df),reduced_control_results=len(cf),bootstrap_resamples=1000,comparison=summary.to_dict(orient='records'),source_runs=provenance,saved_to=str(out),final_study_complete=False)
    save(out/'summary.json',result)
    print('SANDBOX STATISTICAL AUDIT COMPLETE');print(json.dumps(result,indent=2))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--project',type=Path,default=Path('/content/drive/MyDrive/ADNN_Irreversibility'));a=p.parse_args()
    if not a.project.is_dir():
        from google.colab import drive
        drive.mount('/content/drive')
    main(a.project)
