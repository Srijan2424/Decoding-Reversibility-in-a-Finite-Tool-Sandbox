import json, hashlib, warnings, time
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
import sklearn
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.dummy import DummyClassifier
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits

C_GRID=[0.01,0.1,1.0]

def fit_selected(Xtrain,ytrain,Xval,yval,seed,kind="probe"):
    best=None
    for C in C_GRID:
        clf=LogisticRegression(C=C,solver="lbfgs",max_iter=2000,random_state=seed)
        model=make_pipeline(StandardScaler(),clf) if kind=="probe" else make_pipeline(CountVectorizer(),clf)
        with warnings.catch_warnings():
            warnings.simplefilter("error",ConvergenceWarning)
            model.fit(Xtrain,ytrain)
        score=float(balanced_accuracy_score(yval,model.predict(Xval)))
        # Grid is ascending: ties favour stronger regularization.
        if best is None or score>best[0]:best=(score,C,model)
    return best

def evaluate(model,X,y):
    pred=model.predict(X);proba=model.predict_proba(X);classes=model.classes_
    if len(classes)==2:
        auc=roc_auc_score(y,proba[:,1],labels=classes)
    else:
        auc=roc_auc_score(y,proba,labels=classes,multi_class="ovr",average="macro")
    return {"n":len(y),"balanced_accuracy":float(balanced_accuracy_score(y,pred)),
            "macro_f1":float(f1_score(y,pred,average="macro",zero_division=0)),"auroc":float(auc),
            "predictions":pred.tolist(),"probabilities":proba.tolist(),"class_order":classes.tolist()}

def run_probes(project):
    project=Path(project)
    source=project/"data/expanded_development_20261018T075007949936Z"
    split_dir=source/"split_audit_20261018T075128231887Z"
    activation_dir=project/"activations/llama_dev96_91fcd7afe3fcef85"
    raw=(source/"development_dataset.jsonl").read_bytes()
    digest=hashlib.sha256(raw).hexdigest()
    assert digest=="d0ad9dc36284db061aca680deb0985a6bb200931fad28f656da7218d87b25a8a"
    records=[json.loads(line) for line in raw.decode().splitlines() if line.strip()]
    ids=[r["id"] for r in records];index={ident:i for i,ident in enumerate(ids)}
    assert len(ids)==len(index)==96
    config=json.loads((activation_dir/"config.json").read_text())
    assert config["dataset_sha256"]==digest and config["scope"]=="development_only"
    arrays={"last":[],"mean":[]}
    for ident in ids:
        log=json.loads((activation_dir/(ident+".json")).read_text())
        assert log["id"]==ident and log["run_key"]=="91fcd7afe3fcef85"
        with np.load(activation_dir/(ident+".npz"),allow_pickle=False) as saved:
            for name in arrays:
                x=saved[name]
                assert x.shape==(33,4096) and np.isfinite(x).all()
                arrays[name].append(x)
    arrays={name:np.stack(values) for name,values in arrays.items()}
    output=project/"results"/("llama_dev_probes_"+datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
    output.mkdir(parents=True,exist_ok=False)
    (output/"config.json").write_text(json.dumps({"activation_config":config,"C_grid":C_GRID,"scikit_learn":sklearn.__version__,"numpy":np.__version__,"seed_values":[42,123,2026],"selection":"C per layer and layer per pooling selected on validation balanced accuracy; ties choose first","scope":"development only; full controls and statistical analysis pending"},indent=2))
    results=[];selected=[];baselines=[]
    for seed in (42,123,2026):
        manifest=json.loads((split_dir/f"split_{seed}.json").read_text())
        assert manifest["dataset_sha256"]==digest
        for task in ("binary","three_class"):
            label_map={"reversible":0,"irreversible":1} if task=="binary" else {"reversible":0,"partially_reversible":1,"irreversible":2}
            partitions={name:[index[i] for i in members if records[index[i]]["label"] in label_map] for name,members in manifest["ids"].items()}
            tr,va,ev=[partitions[n] for n in ("train","validation","development_eval")]
            assert not set(tr)&set(va) and not set(tr)&set(ev) and not set(va)&set(ev)
            y={name:np.array([label_map[records[i]["label"]] for i in members]) for name,members in partitions.items()}
            yt,yv,ye=[y[n] for n in ("train","validation","development_eval")]
            texts={name:[records[i]["action_text"] for i in members] for name,members in partitions.items()}
            for kind in ("majority","action_text"):
                if kind=="majority":
                    clf=DummyClassifier(strategy="most_frequent").fit(np.zeros((len(tr),1)),yt)
                    metrics=evaluate(clf,np.zeros((len(ev),1)),ye)
                    val_score=None;C=None
                else:
                    val_score,C,clf=fit_selected(texts["train"],yt,texts["validation"],yv,seed,kind="lexical")
                    metrics=evaluate(clf,texts["development_eval"],ye)
                baselines.append({"seed":seed,"task":task,"kind":kind,"C":C,"validation_balanced_accuracy":val_score,"eval_ids":[ids[i] for i in ev],"label_map":label_map,**metrics})
            for pooling,X in arrays.items():
                layer_rows=[]
                for layer in range(X.shape[1]):
                    score,C,clf=fit_selected(X[tr,layer],yt,X[va,layer],yv,seed)
                    row={"seed":seed,"task":task,"pooling":pooling,"layer":layer,"C":C,"validation_balanced_accuracy":score,"label_map":label_map,"eval_ids":[ids[i] for i in ev],**evaluate(clf,X[ev,layer],ye)}
                    layer_rows.append(row);results.append(row)
                winner=max(layer_rows,key=lambda row:row["validation_balanced_accuracy"])
                selected.append(winner)
                print(f"Finished seed={seed}, task={task}, pooling={pooling}; validation-selected layer={winner['layer']}",flush=True)
                # Preserve completed blocks if execution is interrupted.
                (output/"layer_results.json").write_text(json.dumps(results,indent=2))
                (output/"selected_results.json").write_text(json.dumps(selected,indent=2))
                (output/"baseline_results.json").write_text(json.dumps(baselines,indent=2))
    grouped=[]
    for task in ("binary","three_class"):
        for pooling in ("last","mean"):
            subset=[r for r in selected if r["task"]==task and r["pooling"]==pooling]
            grouped.append({"task":task,"pooling":pooling,"validation_selected_layers":[r["layer"] for r in subset],"eval_balanced_accuracy_mean":float(np.mean([r["balanced_accuracy"] for r in subset])),"eval_balanced_accuracy_sd":float(np.std([r["balanced_accuracy"] for r in subset],ddof=1)),"eval_n_per_split":[r["n"] for r in subset]})
    summary={"scope":"development_only","layer_probe_results":len(results),"selected_results":grouped,"baseline_balanced_accuracy":[{k:r[k] for k in ("seed","task","kind","balanced_accuracy")} for r in baselines],"final_study_complete":False,"pending":"shuffled-label/verb controls, scenario bootstrap CIs, broader datasets, cross-environment testing, RQ2 and RQ3","saved_to":str(output)}
    (output/"summary.json").write_text(json.dumps(summary,indent=2))
    return summary

from google.colab import drive
drive.mount("/content/drive")
with threadpool_limits(limits=2):
    summary=run_probes("/content/drive/MyDrive/ADNN_Irreversibility")
print("DEVELOPMENT PROBE RUN COMPLETE")
print(json.dumps(summary,indent=2))
