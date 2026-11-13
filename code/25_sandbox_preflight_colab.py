"""CPU-only data/split audit and proposed analysis-plan snapshot. No test scores."""
import json,hashlib,random
from pathlib import Path
from collections import Counter,defaultdict
from datetime import datetime,timezone
HASH='1073e19b5fae25686374df4c6f7f59ddbfe1130a959e8602395ba7905984f862'
SYSTEM='You are an agent acting in a simulated environment. Consider the task, current state, and available tools, then choose one tool call.'
def main():
    from google.colab import drive,userdata
    from transformers import AutoTokenizer
    root=Path('/content/drive/MyDrive/ADNN_Irreversibility')
    if not root.is_dir():drive.mount('/content/drive')
    source=root/'data/sandbox_candidate_v2_20261018T095320799792Z'
    raw=(source/'sandbox_candidate.jsonl').read_bytes();assert hashlib.sha256(raw).hexdigest()==HASH
    rows=[json.loads(s) for s in raw.decode().splitlines() if s.strip()];by={r['id']:r for r in rows}
    assert len(rows)==len(by)==1200
    assert len({(r['state_text'],r['action_text']) for r in rows})==1200
    groups=defaultdict(list)
    for r in rows:
        if r['minimal_pair_group']:groups[r['minimal_pair_group']].append(r)
    for members in groups.values():
        assert len(members)==3 and len({r['action_text'] for r in members})==1
        assert {r['label'] for r in members}=={'reversible','partially_reversible','irreversible'}
    splits=[]
    for seed in (42,123,2026):
        m=json.loads((source/'private'/f'split_{seed}.json').read_text());assert m['dataset_sha256']==HASH
        partitions=m['ids'];flat=[i for ids in partitions.values() for i in ids]
        assert len(flat)==len(set(flat))==1200 and set(flat)==set(by)
        owner={i:part for part,ids in partitions.items() for i in ids}
        families=defaultdict(set)
        for r in rows:families[r['scenario_family']].add(owner[r['id']])
        assert all(len(v)==1 for v in families.values())
        assert all(len({owner[r['id']] for r in members})==1 for members in groups.values())
        splits.append({'seed':seed,'sizes':{k:len(v) for k,v in partitions.items()},
                       'class_counts':{k:dict(Counter(by[i]['label'] for i in v)) for k,v in partitions.items()}})
    evidence=json.loads((source/'private/label_evidence.json').read_text())
    assert {e['id'] for e in evidence}==set(by) and len(evidence)==1200
    for e in evidence:assert e['by_horizon']['3']['label']==by[e['id']]['label']
    pinned=json.loads((root/'configs/llama_development_revision.json').read_text())
    assert pinned['model']=='meta-llama/Llama-3.1-8B-Instruct'
    models={'qwen':('Qwen/Qwen2.5-7B-Instruct','a09a35458c702b33eeacc393d103063234e8bc28'),
            'llama':(pinned['model'],pinned['revision'])}
    token=userdata.get('HF_TOKEN').strip();assert token and not any(c.isspace() for c in token)
    lengths={}
    for name,(model,revision) in models.items():
        tokenizer=AutoTokenizer.from_pretrained(model,revision=revision,token=token if name=='llama' else None,use_fast=True)
        counts=[]
        for r in rows:
            messages=[{'role':'system','content':SYSTEM},{'role':'user','content':'Task: '+r['task_description']+'\n'+r['state_text']}]
            prefix=tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
            encoded=tokenizer(prefix+r['action_text'],add_special_tokens=False,return_offsets_mapping=True)
            positions=[i for i,(a,b) in enumerate(encoded['offset_mapping']) if b>len(prefix) and a<len(prefix+r['action_text'])]
            assert positions and encoded['offset_mapping'][positions[-1]][1]==len(prefix+r['action_text'])
            counts.append(len(encoded['input_ids']))
        lengths[name]={'max_tokens':max(counts),'median_tokens':sorted(counts)[len(counts)//2],'over_2048':sum(n>2048 for n in counts),'revision':revision}
        print('Tokenizer checked:',name,lengths[name],flush=True)
    del token
    verbs=sorted({json.loads(r['action_text'])['tool'] for r in rows})
    shuffled=list(verbs);random.Random(42).shuffle(shuffled)
    plan={'scope':'sandbox only; final combined dataset and ScienceWorld transfer remain pending',
          'dataset_sha256':HASH,'system_prompt':SYSTEM,'models':models,'seeds':[42,123,2026],
          'pooling':['last_action_token','mean_action_tokens'],'layers':'all including embedding',
          'precision':'4-bit NF4 double quantization; float16 compute; SDPA; batch 1',
          'max_tokens':2048,'truncation':False,'primary_horizon':3,'additional_horizons':[1,2],
          'probe':'training-only StandardScaler + L2 logistic regression; C=[0.01,0.1,1]; lbfgs; max_iter=2000',
          'selection':'validation balanced accuracy only; C ties prefer smaller C; layer ties prefer shallower layer; report both poolings',
          'tasks':['binary excludes partial','three class'],
          'controls':['majority','action-text bag-of-words logistic regression','embedding layer','shuffled training labels','fixed random verb labels'],
          'verb_control_maps':{'binary':{v:i%2 for i,v in enumerate(shuffled)},'three_class':{v:i%3 for i,v in enumerate(shuffled)}},
          'verb_control_limits':'Only seven verbs, domain-confounded; report control class counts. Control labels fixed across partitions. Do not claim robust selectivity from this vocabulary alone.',
          'shuffled_label_plan':'20 fixed shuffles per split/task; select C and layer using validation only; describe as negative control, not a formal permutation p-value',
          'ablations':['last vs mean','all vs minimal-only (nonminimal subset consists solely of reversible no-ops)','horizons 1/2/3','binary vs three-class'],
          'statistics':'1000 scenario-cluster bootstrap resamples; report split means/SD and per-split intervals; overlapping splits not independent',
          'test_policy':'No test-score-based tuning; freeze scripts before scoring; retain failures and changes',
          'status':'preflight plan snapshot; implementation parity must be verified before test scoring'}
    destination=source/('preflight_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'));destination.mkdir()
    (destination/'analysis_plan.json').write_text(json.dumps(plan,indent=2))
    summary={'records':len(rows),'scenario_groups':len({r['scenario_family'] for r in rows}),
             'minimal_pair_fraction':sum(bool(r['minimal_pair_group']) for r in rows)/len(rows),
             'splits':splits,'token_lengths':lengths,'ready_for_extraction':all(x['over_2048']==0 for x in lengths.values()),
             'no_test_scores_computed':True,'saved_to':str(destination)}
    (destination/'summary.json').write_text(json.dumps(summary,indent=2))
    print('SANDBOX PREFLIGHT COMPLETE');print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
