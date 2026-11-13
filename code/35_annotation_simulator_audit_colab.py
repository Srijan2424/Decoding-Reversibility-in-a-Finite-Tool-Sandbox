"""CPU audit of frozen human/AI labels against sandbox automatic labels."""
import csv,io,json,hashlib
from pathlib import Path
from collections import Counter
from datetime import datetime,timezone
LABELS=['reversible','partially_reversible','irreversible']
HUMAN_HASH='c8c0ad54b7cd9d7ef80a2831e17149efdb99bee2ae64fa6f49c2cbd3ac782226'
def read(p):return json.loads(p.read_text())
def pairs(a,b):
    n=len(a);po=sum(x==y for x,y in zip(a,b))/n;ca=Counter(a);cb=Counter(b);pe=sum(ca[x]*cb[x] for x in LABELS)/n**2
    return dict(n=n,agreement=po,kappa=None if pe==1 else (po-pe)/(1-pe),confusion_rows_first_columns_second=[[sum(x==l and y==k for x,y in zip(a,b)) for k in LABELS] for l in LABELS])
def scenario_matches(actual, expected, field):
    if actual == expected:
        return True
    if field != 'state_text':
        return False
    # Permit only line-ending normalization and newlines replaced by spaces.
    normalized = expected.replace('\r\n', '\n').replace('\r', '\n')
    return actual == normalized or actual == normalized.replace('\n', ' ')

def main():
    from google.colab import drive,files
    root=Path('/content/drive/MyDrive/ADNN_Irreversibility')
    if not root.is_dir():drive.mount('/content/drive')
    print('Upload your original completed human TSV (not the AI CSV).')
    uploaded=files.upload();assert len(uploaded)==1
    raw=next(iter(uploaded.values()));assert hashlib.sha256(raw).hexdigest()==HUMAN_HASH,'Human file differs from preserved original; do not silently replace annotations.'
    human=list(csv.DictReader(io.StringIO(raw.decode('utf-8-sig')),delimiter='\t'))
    src=root/'data/sandbox_candidate_v2_20261018T095320799792Z'
    data=(src/'sandbox_candidate.jsonl').read_bytes();assert hashlib.sha256(data).hexdigest()=='1073e19b5fae25686374df4c6f7f59ddbfe1130a959e8602395ba7905984f862'
    by={r['id']:r for r in map(json.loads,data.decode().splitlines())}
    mapping=read(src/'private/annotation_mapping.json')
    blank=list(csv.DictReader(io.StringIO((src/'annotation_audit/ai_blinded.csv').read_text())))
    ai_dir=root/'annotations/qwen_blinded_cd40c9f41866c9b5';cfg=read(ai_dir/'config.json')
    assert hashlib.sha256((src/'annotation_audit/ai_blinded.csv').read_bytes()).hexdigest()==cfg['packet_sha256']
    assert hashlib.sha256((ai_dir/'runner_source.py').read_bytes()).hexdigest()==cfg['source_sha256']
    expected={f'S{i:03d}' for i in range(1,101)}
    assert len(human)==len(blank)==len(mapping)==100 and {r['annotation_id'] for r in human}=={r['annotation_id'] for r in blank}==set(mapping)==expected
    hd={r['annotation_id']:r for r in human};bd={r['annotation_id']:r for r in blank};rows=[];formatting_changes=[]
    ai_csv={r['annotation_id']:r for r in csv.DictReader(io.StringIO((ai_dir/'ai_annotations.csv').read_text()))};assert set(ai_csv)==expected
    for i in sorted(expected):
        h=hd[i];b=bd[i];r=by[mapping[i]]
        for field in ['task','state_text','action_text','horizon']:
            assert scenario_matches(h[field],b[field],field), f'Substantive scenario mismatch: {i} {field}'
            if h[field]!=b[field]:formatting_changes.append(dict(annotation_id=i,field=field,change='line endings/newlines normalized to spaces; original human file retained'))
        assert b['task']==r['task_description'] and b['state_text']==r['state_text'] and b['action_text']==r['action_text'] and int(b['horizon'])==r['horizon']
        log=read(ai_dir/(i+'.json'));assert log['annotation_id']==i and log['run_key']==ai_dir.name.removeprefix('qwen_blinded_')
        scenario={k:b[k] for k in ['task','state_text','action_text','horizon']}
        expected_messages=[{'role':'system','content':cfg['system']},{'role':'user','content':cfg['rules']+'\nSCENARIO DATA:\n'+json.dumps(scenario,ensure_ascii=False)}]
        assert log['messages']==expected_messages and log['parse_error'] is None
        parsed=json.loads(log['raw_response']);assert parsed==log['annotation'] and parsed['label']==ai_csv[i]['label']
        assert h['label'] in LABELS and parsed['label'] in LABELS and r['label'] in LABELS
        rows.append(dict(annotation_id=i,dataset_id=r['id'],human=h['label'],ai=parsed['label'],automatic=r['label'],human_notes=h['notes'],ai_notes=parsed['notes']))
    scores={f'{a}_vs_{b}':pairs([r[a] for r in rows],[r[b] for r in rows]) for a,b in [('human','ai'),('human','automatic'),('ai','automatic')]}
    out=root/'annotations'/('agreement_audit_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'));out.mkdir()
    (out/'human_original.tsv').write_bytes(raw)
    (out/'formatting_changes.json').write_text(json.dumps(formatting_changes,indent=2))
    with (out/'comparison.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    summary=dict(audit_passed=True,rows=100,scope='sandbox-only human-AI and automatic-label agreement',class_order=LABELS,counts={a:dict(Counter(r[a] for r in rows)) for a in ['human','ai','automatic']},comparisons=scores,original_labels_unchanged=True,formatting_changes_accepted=len(formatting_changes),limitations=['Constant human labels and repeated rationale limit annotation quality.','Qwen is also evaluated in the study.','Simulator labels are an operational reference, not independently established ground truth.','Does not cover ScienceWorld.'],saved_to=str(out))
    (out/'summary.json').write_text(json.dumps(summary,indent=2));(out/'audit_source.py').write_bytes(Path(__file__).read_bytes())
    print('ANNOTATION SIMULATOR AUDIT COMPLETE');print(json.dumps(summary,indent=2));files.download(str(out/'summary.json'))
if __name__=='__main__':main()
