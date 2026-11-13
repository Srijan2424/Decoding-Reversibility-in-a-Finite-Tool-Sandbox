"""Fresh restricted-action development validation; not full-world labels."""
import argparse,gzip,hashlib,importlib.metadata,json,os,runpy,time,traceback
from collections import deque,Counter
from pathlib import Path
from datetime import datetime,timezone
COMMIT='e8216d6044e8e39be9fcb185e3b2dfb602584b52'
JAR='e77b0fee7d68abe3ca5b12e57d86e2bfe7603f20c5200da0b0f085939faeb465'
MANIFEST={'version':'fresh_restricted_v1','variations':[1,2],
 'task_keywords':['boil','freeze','mix','melt'],
 'recovery_prefixes':['move ','activate ','deactivate ','open ','close ','pour ','mix '],
 'waiting':['wait1'], 'horizons':[1,2,3],
 'role_rule':'Initial candidate-bound objects and their descendants, plus immediate original parents; exclude sewer/universe/terminal objects. Add later descendants of original bound objects except terminals.',
 'selection':'Evenly spaced physical actions along supplied gold trajectory, before any recovery outcome is seen.',
 'scope':'v2 drainage-aware projection; initial parent subtrees additionally tracked for recovery roles',
 'interpretation':'Exhaustive only within declared action-role set. Action-count cap aborts, never truncates.',
 'partial':'Thermal-only residual with no exact witness after complete bounded search.',
 'read_only_actions':'Not members of this restricted action set.',
 'full_state_coverage_verified':False}
def save(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False))
def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True,allow_nan=False).encode()).hexdigest()
def physical(a):return a.startswith(('move ','activate ','deactivate ','pour ','mix '))
def bindings(env,action):
    vals={tuple(str(i) for i in r['obj_ids']) for r in env.get_valid_action_object_combinations_with_templates() if r['action']==action}
    if len(vals)!=1:raise ValueError('Ambiguous or missing candidate binding')
    return set(next(iter(vals)))
def allowed(env,roles,roots,tree,helper,max_actions):
    nodes,_=helper['index'](tree)
    dynamic=set(roles)|helper['descendants'](nodes,roots)
    dynamic={i for i in dynamic if i in nodes and not nodes[i].get('name','').lower().startswith('terminal ') and nodes[i].get('name') not in ('universe','sewer')}
    grouped={}
    for row in env.get_valid_action_object_combinations_with_templates():
        a=row['action'];ids=tuple(str(i) for i in row['obj_ids'])
        grouped.setdefault(a,set()).add(ids)
    actions=[]
    for a,bs in grouped.items():
        if not a.startswith(tuple(MANIFEST['recovery_prefixes'])):continue
        eligible=[ids for ids in bs if ids and set(ids)<=dynamic and not (a.startswith('pour ') and len(ids)>1 and ids[0]==ids[1])]
        if not eligible:continue
        if len(bs)!=1:raise ValueError('Ambiguous recovery action: '+a)
        actions.append(a)
    valid=set(env.get_valid_action_object_combinations())
    actions.extend(a for a in MANIFEST['waiting'] if a in valid)
    actions=sorted(set(actions))
    if len(actions)>max_actions:raise ValueError(f'Action budget exceeded ({len(actions)}>{max_actions}); no actions silently discarded')
    return actions

def main(args):
    option='-Xlog:os+container=off'
    if option not in os.environ.get('JAVA_TOOL_OPTIONS','').split():os.environ['JAVA_TOOL_OPTIONS']=(os.environ.get('JAVA_TOOL_OPTIONS','')+' '+option).strip()
    from scienceworld import ScienceWorldEnv
    from scienceworld.constants import JAR_PATH
    dist=importlib.metadata.distribution('scienceworld')
    assert json.loads(dist.read_text('direct_url.json') or '{}').get('vcs_info',{}).get('commit_id')==COMMIT
    assert hashlib.sha256(Path(JAR_PATH).read_bytes()).hexdigest()==JAR
    helper_path=Path('/content/40_scienceworld_drainage_recheck_colab.py')
    helper=runpy.run_path(str(helper_path),run_name='projection_helper')
    root=Path(args.project);assert root.is_dir(),'Mount Drive first.'
    out=root/'results'/('scienceworld_fresh_validation_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'));out.mkdir()
    save(out/'protocol.json',{'manifest':MANIFEST,'arguments':vars(args),'commit':COMMIT,'jar_sha256':JAR,'projection_sha256':hashlib.sha256(helper_path.read_bytes()).hexdigest()})
    (out/'runner.py').write_bytes(Path(__file__).read_bytes());(out/'projection.py').write_bytes(helper_path.read_bytes())
    env=None;results=[];errors=[];started=time.monotonic()
    try:
        env=ScienceWorldEnv(envStepLimit=250);names=env.get_task_names();chosen=[]
        for k in MANIFEST['task_keywords']:
            t=next((t for t in names if k in t.lower() and t not in chosen),None)
            if t:chosen.append(t)
        for task in chosen:
            for variation in MANIFEST['variations']:
                folder=out/f'{len(results):03d}_{task}_{variation}';folder.mkdir()
                try:
                    print('Discovering:',task,'variation',variation,flush=True)
                    assert variation<env.get_max_variations(task),'Variation unavailable'
                    env.load(task,variation,'',generateGoldPath=True);gold=env.get_gold_action_sequence()
                    assert gold and not any(a.startswith('ERROR') for a in gold),'Gold path unavailable'
                    save(folder/'gold.json',gold)
                    def snap(done):
                        return {'tree':env.getObjectTree(),'look':env.look(),'inventory':env.inventory(),
                            'valid_actions':sorted(set(env.get_valid_action_object_combinations())),
                            'moves':int(env.get_num_moves()),'score':float(env.lastStepScore),
                            'goal_progress':env.get_goal_progress(),'done':bool(done)}
                    def replay(seq):
                        env.load(task,variation,'',generateGoldPath=False);env.reset();done=False;trace=[]
                        for a in seq:
                            if done:raise ValueError('Terminal replay boundary')
                            obs,reward,done,_=env.step(a);trace.append([a,obs,reward,bool(done),int(env.get_num_moves())])
                        return snap(done),trace
                    env.load(task,variation,'',generateGoldPath=False);env.reset();prefix=[];candidates=[]
                    for a in gold[:100]:
                        if physical(a) and a in env.get_valid_action_object_combinations():
                            try:
                                ids=bindings(env,a)
                                if not (a.startswith('pour ') and len(ids)<2):candidates.append((list(prefix),a,ids))
                            except ValueError:pass
                        _,_,done,_=env.step(a);prefix.append(a)
                        if done:break
                    if not candidates:raise ValueError('No eligible physical candidates')
                    if args.per_trajectory==1:indices=[len(candidates)//2]
                    else:indices=sorted({round(i*(len(candidates)-1)/(args.per_trajectory-1)) for i in range(args.per_trajectory)})
                    save(folder/'selected_candidates.json',[{'prefix':candidates[i][0],'action':candidates[i][1],'ids':sorted(candidates[i][2])} for i in indices])
                    for index in indices:
                        prefix,action,roots=candidates[index];cid=f'V{len(results)+1:03d}';case=folder/cid;case.mkdir()
                        print(cid,action,flush=True)
                        row={'id':cid,'task':task,'variation':variation,'prefix':prefix,'candidate':action,'final_dataset_label':None}
                        nodes=[];witnesses={};queue=deque([[]]);reason=None;case_start=time.monotonic()
                        try:
                            before,trace=replay(prefix);again,trace2=replay(prefix)
                            assert before==again and trace==trace2,'Starting replay unstable'
                            after,at=replay(prefix+[action]);again,at2=replay(prefix+[action])
                            assert after==again and at==at2,'Candidate replay unstable'
                            all_nodes,parents=helper['index'](before['tree'])
                            roles=helper['descendants'](all_nodes,roots)
                            roles|={parents[i] for i in roles if parents[i] is not None}
                            roles={i for i in roles if all_nodes[i].get('name') not in ('universe','sewer') and not all_nodes[i].get('name','').startswith('terminal ')}
                            scope=helper['make_scope'](before['tree'],roots|roles)
                            row['scope']=scope;row['recovery_role_ids']=sorted(roles)
                            reference=helper['project'](before['tree'],scope)
                            with gzip.open(case/'initial.json.gz','wt') as f:json.dump({'before':before,'after':after},f)
                            while queue:
                                if len(nodes)>=args.max_nodes or time.monotonic()-case_start>=args.seconds_per_case:
                                    reason='budget_exhausted';break
                                undo=queue.popleft();end,etrace=replay(prefix+[action]+undo)
                                comp=helper['compare'](reference,helper['project'](end['tree'],scope))
                                kind='exact' if comp['exact_projection_match'] else 'thermal' if comp['thermal_only_residual'] else None
                                if kind and kind not in witnesses:
                                    repeat,rtrace=replay(prefix+[action]+undo)
                                    assert end==repeat and etrace==rtrace,'Witness replay unstable'
                                    witnesses[kind]={'undo':undo,'depth':len(undo),'independent_replay_matches':True}
                                    with gzip.open(case/(kind+'_witness.json.gz'),'wt') as f:json.dump({'end':end,'trace':etrace,'comparison':comp},f)
                                next_actions=[]
                                if len(undo)<3 and not end['done']:
                                    next_actions=allowed(env,roles,roots,end['tree'],helper,args.max_actions)
                                    queue.extend(undo+[a] for a in next_actions)
                                nodes.append({'undo':undo,'snapshot_sha256':digest(end),'kind':kind,'done':end['done'],'children':next_actions})
                                if kind=='exact':reason='exact_witness_found';break
                            if reason is None:reason='restricted_search_exhausted'
                        except Exception as exc:reason='validation_error';row['error']=str(exc);(case/'error.txt').write_text(traceback.format_exc())
                        counts=Counter(len(n['undo']) for n in nodes)
                        per_h={}
                        for h in [1,2,3]:
                            exact=witnesses.get('exact');thermal=witnesses.get('thermal')
                            if exact and exact['depth']<=h:label='reversible_under_projection'
                            elif reason=='restricted_search_exhausted':label='partially_reversible_under_projection' if thermal and thermal['depth']<=h else 'irreversible_under_action_set'
                            else:label='unresolved'
                            per_h[str(h)]=label
                        row.update(status=reason,provisional_results_by_horizon=per_h,witnesses=witnesses,
                            nodes_tested=len(nodes),nodes_by_depth=dict(counts),queued_remaining=len(queue),
                            elapsed_seconds=round(time.monotonic()-case_start,2),scope_coverage_validated=False)
                        save(case/'nodes.json',nodes);save(case/'result.json',row);results.append(row)
                        save(out/'progress.json',results);print(' ',reason,per_h,flush=True)
                except Exception as exc:
                    errors.append({'task':task,'variation':variation,'error':str(exc)});(folder/'error.txt').write_text(traceback.format_exc())
                    print('Trajectory error saved:',str(exc),flush=True)
    finally:
        summary={'fresh_cases':len(results),'status_counts':dict(Counter(r['status'] for r in results)),
          'provisional_horizon3_counts':dict(Counter(r['provisional_results_by_horizon']['3'] for r in results)),
          'errors':errors,'elapsed_minutes':round((time.monotonic()-started)/60,2),
          'final_labels_created':0,'scope_coverage_validated':False,'scienceworld_requirement_met':False,'saved_to':str(out)}
        save(out/'results.json',results);save(out/'summary.json',summary)
        if env:
            try:env.close()
            except Exception:pass
    print('FRESH RESTRICTED VALIDATION COMPLETE');print(json.dumps(summary,indent=2))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--project',default='/content/drive/MyDrive/ADNN_Irreversibility')
    p.add_argument('--per-trajectory',type=int,default=1);p.add_argument('--max-nodes',type=int,default=512)
    p.add_argument('--seconds-per-case',type=float,default=90);p.add_argument('--max-actions',type=int,default=24)
    args=p.parse_args();assert min(args.per_trajectory,args.max_nodes,args.seconds_per_case,args.max_actions)>0;main(args)
