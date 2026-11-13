"""Audit saved restricted BFS coverage; no simulator, no new dataset labels."""
import json,hashlib
from pathlib import Path
from collections import Counter
from datetime import datetime,timezone
SOURCE=Path('/content/drive/MyDrive/ADNN_Irreversibility/results/scienceworld_fresh_validation_20261107T134439885002Z')
def audit_nodes(nodes,horizon):
    lookup={tuple(n['undo']):n for n in nodes}
    assert len(lookup)==len(nodes),'Duplicate evaluated search sequence'
    expected={()};checked=set();missing=[]
    for depth in range(horizon+1):
        current=sorted(s for s in expected if len(s)==depth)
        for seq in current:
            if seq not in lookup:missing.append(list(seq));continue
            checked.add(seq);node=lookup[seq]
            if depth<horizon and not node['done']:
                assert isinstance(node['children'],list)
                assert len(node['children'])==len(set(node['children']))
                expected.update(seq+(a,) for a in node['children'])
    # Missing ancestors can hide additional descendants; missing count is not total remaining workload.
    return {'enumerated_tree_complete':not missing,'expected_sequences_known':len(expected),
            'evaluated_expected_sequences':len(checked),'missing_frontier_count':len(missing),
            'missing_frontier_sample':missing[:5],
            'limitation':'Coverage of saved enumerated actions only; does not independently validate action enumeration or physical state scope.'}
def main():
    assert SOURCE.is_dir(),'Mount Drive and check source folder.'
    records=json.loads((SOURCE/'results.json').read_text());out=SOURCE.parent/('scienceworld_coverage_audit_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'));out.mkdir()
    results=[];errors=[];hashes={}
    for record in records:
        cid=record['id']
        try:
            files=list(SOURCE.glob('*/'+cid+'/nodes.json'));assert len(files)==1,'Missing or duplicate node file'
            path=files[0];nodes=json.loads(path.read_text());hashes[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
            assert len(nodes)==record['nodes_tested']
            assert dict(Counter(str(len(n['undo'])) for n in nodes))==record['nodes_by_depth']
            node_map={tuple(n['undo']):n for n in nodes}
            for kind,w in record['witnesses'].items():
                assert w['depth']==len(w['undo']) and w['independent_replay_matches']
                assert node_map[tuple(w['undo'])]['kind']==kind,'Witness not in evaluated nodes'
            horizons={}
            for h in (1,2,3):
                c=audit_nodes(nodes,h)
                valid=record['status'] in ('budget_exhausted','restricted_search_exhausted','exact_witness_found')
                exact=record['witnesses'].get('exact');thermal=record['witnesses'].get('thermal')
                if not valid:result='unresolved_validation_error'
                elif exact and exact['depth']<=h:result='exact_recovery_witness_under_projection'
                elif not c['enumerated_tree_complete']:result='unresolved_incomplete_search'
                elif thermal and thermal['depth']<=h:result='thermal_only_recovery_no_exact_within_restricted_search'
                else:result='no_exact_or_thermal_recovery_within_restricted_search'
                horizons[str(h)]={**c,'operational_result':result}
            r={'id':cid,'source_status':record['status'],'source_error':record.get('error'),'horizons':horizons,'final_dataset_label':None}
            results.append(r);print(cid,{h:v['operational_result'] for h,v in horizons.items()},flush=True)
        except Exception as exc:errors.append({'id':cid,'error':str(exc)})
    summary={'cases_audited':len(results),'counts_by_horizon':{str(h):dict(Counter(r['horizons'][str(h)]['operational_result'] for r in results)) for h in (1,2,3)},
       'case_validation_errors':[{'id':r['id'],'error':r['source_error']} for r in results if r['source_status']=='validation_error'],
       'audit_errors':errors,'final_labels_created':0,'scope_coverage_validated':False,
       'note':'Repairs conservative horizon reporting using archived branch coverage; no further simulator calls or full-environment irreversibility claims.',
       'saved_to':str(out)}
    hashes[str(SOURCE/'results.json')]=hashlib.sha256((SOURCE/'results.json').read_bytes()).hexdigest()
    for name,data in [('summary.json',summary),('results.json',results),('input_hashes.json',hashes)]: (out/name).write_text(json.dumps(data,indent=2))
    (out/'runner.py').write_bytes(Path(__file__).read_bytes())
    print('SAVED SEARCH COVERAGE AUDIT COMPLETE');print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
