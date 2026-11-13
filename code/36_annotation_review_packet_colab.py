"""Export deterministic post-annotation review evidence. Never changes labels."""
import csv,json,hashlib,random,zipfile
from pathlib import Path
LABELS=['reversible','partially_reversible','irreversible']
def select(rows):
    rng=random.Random(36100);selected=[]
    for label in LABELS:
        candidates=sorted([r for r in rows if r['automatic']==label and (r['human']!=label or r['ai']!=label)],key=lambda r:r['annotation_id'])
        assert len(candidates)>=4
        selected.extend(rng.sample(candidates,4))
    return selected

def main():
    from google.colab import drive,files
    root=Path('/content/drive/MyDrive/ADNN_Irreversibility')
    if not root.is_dir():drive.mount('/content/drive')
    audit=root/'annotations/agreement_audit_20261107T075359253495Z'
    src=root/'data/sandbox_candidate_v2_20261018T095320799792Z'
    comparison_raw=(audit/'comparison.csv').read_bytes()
    with (audit/'comparison.csv').open(newline='') as f:comparisons=list(csv.DictReader(f))
    assert len(comparisons)==100 and len({r['annotation_id'] for r in comparisons})==100
    raw=(src/'sandbox_candidate.jsonl').read_bytes();assert hashlib.sha256(raw).hexdigest()=='1073e19b5fae25686374df4c6f7f59ddbfe1130a959e8602395ba7905984f862'
    data={r['id']:r for r in map(json.loads,raw.decode().splitlines())}
    evidence={r['id']:r for r in json.loads((src/'private/label_evidence.json').read_text())}
    mapping=json.loads((src/'private/annotation_mapping.json').read_text())
    selected=select(comparisons);packet=[]
    text=['# Post-annotation disagreement review','Original labels and agreement scores are frozen. This deliberately selected diagnostic sample is not a new blinded annotation or an estimate of error prevalence. Four cases per automatic class, sampled with seed 36100. Automatic evidence must also be scrutinised.','']
    for r in selected:
        ident=r['dataset_id'];assert mapping[r['annotation_id']]==ident
        item=data[ident];ev=evidence[ident]
        assert r['automatic']==item['label']==ev['by_horizon']['3']['label']
        packet.append(dict(comparison=r,scenario=item,simulator_evidence=ev))
        text.extend([f"## {r['annotation_id']}",f"Task: {item['task_description']}",f"Candidate: {item['action_text']}",f"Horizon: {item['horizon']}",f"Human: {r['human']}. Reason: {r['human_notes']}",f"AI: {r['ai']}. Reason: {r['ai_notes']}",f"Automatic: {r['automatic']}",'### Scenario',item['state_text'],'### Simulator evidence','```json',json.dumps(ev,indent=2),'```','### Review questions','1. Is the proposed recovery tool actually available?','2. Are its prerequisites satisfied, and how many calls are needed?','3. Does recovery restore every relevant field, including money and content?','4. Does the saved witness or search evidence support the automatic label?','Review conclusion: pending.',''])
    out=audit/'post_annotation_review';out.mkdir(exist_ok=True)
    manifest=dict(purpose='Post-annotation diagnostic review, not relabelling',seed=36100,selected_ids=[r['annotation_id'] for r in selected],comparison_sha256=hashlib.sha256(comparison_raw).hexdigest(),original_labels_unchanged=True)
    target=out/'annotation_review_12.zip'
    with zipfile.ZipFile(target,'w',zipfile.ZIP_DEFLATED) as z:
        z.writestr('review_packet.json',json.dumps(packet,indent=2));z.writestr('review_packet.md','\n\n'.join(text));z.writestr('selection_manifest.json',json.dumps(manifest,indent=2))
    print('ANNOTATION REVIEW PACKET READY');print(json.dumps(manifest,indent=2));files.download(str(target))
if __name__=='__main__':main()
