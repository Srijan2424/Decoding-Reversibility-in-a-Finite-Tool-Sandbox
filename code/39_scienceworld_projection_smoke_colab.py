"""Offline projection smoke test on old diagnostics. No labels and no simulator needed."""
import gzip, json, hashlib, math
from pathlib import Path
from datetime import datetime, timezone

ROOT=Path('/content/drive/MyDrive/ADNN_Irreversibility/results')
PILOT=ROOT/'scienceworld_physical_pilot_20261107T091929052223Z'
RECOVERY=ROOT/'scienceworld_recovery_diagnostic_20261107T094243161617Z'

def index(tree):
    nodes={}; parents={}
    def visit(node,parent=None):
        if not isinstance(node,dict) or 'uuid' not in node:raise ValueError('Unsupported object schema')
        uid=str(node['uuid'])
        if uid in nodes:raise ValueError('Duplicate UUID')
        nodes[uid]=node;parents[uid]=parent
        children=node.get('contents',{})
        if not isinstance(children,dict):raise ValueError('Unsupported contents schema')
        for c in children.values():visit(c,uid)
    visit(tree);return nodes,parents

def descendants(nodes,roots):
    result=set()
    def visit(uid):
        if uid in result or uid not in nodes:return
        result.add(uid)
        for c in nodes[uid].get('contents',{}).values():visit(str(c['uuid']))
    for uid in roots:visit(uid)
    return result

def make_scope(before,ids):
    nodes,parents=index(before)
    if not ids or any(i not in nodes for i in ids):raise ValueError('Binding objects absent')
    core=descendants(nodes,ids)
    context={parents[i] for i in core if parents[i] is not None}-core
    return {'roots':sorted(ids),'core':sorted(core),'context':sorted(context)}

def project(tree,scope):
    nodes,parents=index(tree)
    ids=set(scope['core'])|set(scope['context'])|descendants(nodes,scope['roots'])
    result={}
    for uid in sorted(ids):
        if uid not in nodes:result[uid]={'missing':True};continue
        n=nodes[uid]
        result[uid]={'parent_uuid':parents[uid],
          'properties':{k:v for k,v in n.items() if k!='contents'},
          'children':sorted(str(c['uuid']) for c in n.get('contents',{}).values())}
    return result

def differences(a,b,path=()):
    if type(a)!=type(b):return [(path,a,b)]
    if isinstance(a,dict):
        result=[]
        for k in sorted(a.keys()|b.keys()):
            if k not in a or k not in b:result.append((path+(k,),a.get(k),b.get(k)))
            else:result.extend(differences(a[k],b[k],path+(k,)))
        return result
    return [] if a==b else [(path,a,b)]

def compare(a,b):
    ds=differences(a,b)
    def thermal(d):
        path,x,y=d
        return path[-2:]==('propMaterial','temperatureC') and all(type(v) in (int,float) and math.isfinite(v) for v in (x,y))
    return {'exact_projection_match':not ds,'thermal_only_residual':bool(ds) and all(thermal(d) for d in ds),
      'difference_count':len(ds),'differences':[{'path':'/'.join(p),'before':x,'after':y} for p,x,y in ds]}

def main():
    assert PILOT.is_dir() and RECOVERY.is_dir(),'Mount Drive; original pilot and recovery directories are required.'
    out=ROOT/('scienceworld_projection_smoke_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'));out.mkdir()
    rows=[];errors=[];manifest={}
    for case in sorted(RECOVERY.glob('P[0-9][0-9][0-9]')):
        cid=case.name
        try:
            binding_path=case/'action_bindings.json'
            bindings=json.loads(binding_path.read_text())['bindings']
            candidates={tuple(str(i) for i in b['obj_ids']) for b in bindings}
            if len(candidates)!=1:raise ValueError('Missing or ambiguous object-ID binding')
            ids=next(iter(candidates))
            paths=list(PILOT.glob('task_*/'+cid+'/snapshots.json.gz'));assert len(paths)==1
            with gzip.open(paths[0],'rt') as f:snap=json.load(f)
            record=json.loads((case/'summary.json').read_text())
            if record['candidate'].startswith('pour ') and len(ids)>=2 and ids[0]==ids[1]:
                rows.append({'id':cid,'status':'excluded_self_pour','final_label':None});continue
            scope=make_scope(snap['before']['tree'],set(ids));before=project(snap['before']['tree'],scope)
            row={'id':cid,'candidate':record['candidate'],'scope':scope,
                 'candidate_comparison':compare(before,project(snap['after']['tree'],scope)),
                 'recoveries':[],'final_label':None,'status':'diagnostic_only'}
            inputs=[binding_path,paths[0],case/'summary.json']
            for file in sorted(case.glob('targeted_*.json.gz')):
                with gzip.open(file,'rt') as f:trial=json.load(f)
                row['recoveries'].append({'undo':trial['test']['undo'],
                  'saved_replay_stable':trial['test']['replay_stable'],
                  'comparison':compare(before,project(trial['end']['tree'],scope))})
                inputs.append(file)
            for file in inputs:manifest[str(file)]=hashlib.sha256(file.read_bytes()).hexdigest()
            rows.append(row)
            print(cid,'recovery comparisons:',[(t['comparison']['exact_projection_match'],t['comparison']['thermal_only_residual']) for t in row['recoveries']])
        except Exception as exc:errors.append({'id':cid,'error':str(exc)})
    tests=[t for r in rows for t in r.get('recoveries',[])]
    summary={'old_cases_reviewed':len(rows),'recovery_comparisons':len(tests),
       'exact_projection_matches':sum(t['comparison']['exact_projection_match'] for t in tests),
       'thermal_only_residuals':sum(t['comparison']['thermal_only_residual'] for t in tests),
       'excluded_self_pours':[r['id'] for r in rows if r['status']=='excluded_self_pour'],
       'errors':errors,'final_labels_created':0,'scope_coverage_validated':False,
       'partial_labels_require_exhaustive_search_for_exact_recovery':True,
       'fresh_validation_complete':False,'saved_to':str(out)}
    for name,data in [('summary.json',summary),('results.json',rows),('input_hashes.json',manifest)]:
        (out/name).write_text(json.dumps(data,indent=2,allow_nan=False))
    (out/'runner.py').write_bytes(Path(__file__).read_bytes())
    print('PROJECTED STATE SMOKE TEST COMPLETE');print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
