"""CPU diagnostics only: no final reversibility labels or approximate equality."""
import os, json, hashlib, inspect, importlib.metadata, traceback
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

EXPECTED_COMMIT='e8216d6044e8e39be9fcb185e3b2dfb602584b52'
EXPECTED_JAR='e77b0fee7d68abe3ca5b12e57d86e2bfe7603f20c5200da0b0f085939faeb465'
MAX_SEARCH_NODES=24
HORIZON=3

def digest(obj):
    return hashlib.sha256(json.dumps(obj,sort_keys=True,allow_nan=False).encode()).hexdigest()

def diff(a,b,path='$',out=None):
    if out is None: out=[]
    if len(out)>=8:return out
    if type(a)!=type(b):out.append({'path':path,'before':str(a)[:100],'after':str(b)[:100]})
    elif isinstance(a,dict):
        for k in sorted(set(a)|set(b)):
            if k not in a or k not in b:out.append({'path':path+'/'+k,'kind':'missing_key'})
            else:diff(a[k],b[k],path+'/'+k,out)
            if len(out)>=8:break
    elif isinstance(a,list):
        if len(a)!=len(b):out.append({'path':path,'kind':'list_length','before':len(a),'after':len(b)})
        for i,(x,y) in enumerate(zip(a,b)):
            diff(x,y,path+'/'+str(i),out)
            if len(out)>=8:break
    elif a!=b:out.append({'path':path,'before':a,'after':b})
    return out[:8]

def save(path,obj):path.write_text(json.dumps(obj,indent=2,allow_nan=False))

def run(project):
    option='-Xlog:os+container=off'
    if option not in os.environ.get('JAVA_TOOL_OPTIONS','').split():
        os.environ['JAVA_TOOL_OPTIONS']=(os.environ.get('JAVA_TOOL_OPTIONS','')+' '+option).strip()
    from scienceworld import ScienceWorldEnv
    from scienceworld.constants import JAR_PATH
    dist=importlib.metadata.distribution('scienceworld')
    direct=json.loads(dist.read_text('direct_url.json') or '{}')
    jar=hashlib.sha256(Path(JAR_PATH).read_bytes()).hexdigest()
    assert direct.get('vcs_info',{}).get('commit_id')==EXPECTED_COMMIT, 'Use the previously pinned ScienceWorld installation; do not upgrade silently.'
    assert jar==EXPECTED_JAR,'Simulator JAR differs from the verified pilot.'
    output=project/'results'/('scienceworld_semantics_audit_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    output.mkdir(parents=True,exist_ok=False)
    save(output/'installation.json',{'version':dist.version,'source':direct,'jar_sha256':jar,
         'horizon':HORIZON,'max_search_nodes':MAX_SEARCH_NODES})
    (output/'installed_wrapper_source.py').write_text(inspect.getsource(ScienceWorldEnv))
    env=None;reports=[];errors=[]
    try:
        env=ScienceWorldEnv(envStepLimit=100)
        tasks=env.get_task_names();save(output/'tasks.json',tasks)
        # Initial states from three different physical-task definitions.
        selected=[]
        for keyword in ('boil','melt','freeze'):
            matches=[t for t in tasks if keyword in t.lower()]
            if matches and matches[0] not in selected:selected.append(matches[0])
        if not selected:raise RuntimeError('No matching physical tasks found; see saved task list.')
        def snapshot():
            tree=env.getObjectTree()
            return {'tree':tree,'tree_sha256':digest(tree),'moves':int(env.get_num_moves()),
                    'valid_actions':sorted(set(env.get_valid_action_object_combinations())),
                    'last_step_score':float(env.lastStepScore)}
        def replay(task,seq):
            env.load(task,0,'');observation,info=env.reset()
            initial=snapshot();trace=[]
            for action in seq:
                if action not in env.get_valid_action_object_combinations():
                    raise ValueError('Unavailable replay action: '+action)
                observation,reward,done,info=env.step(action)
                trace.append({'action':action,'observation':observation,'reward':reward,'done':bool(done),'info':info,'moves':int(env.get_num_moves())})
                if done and len(trace)<len(seq):raise RuntimeError('Episode terminated before replay finished')
            return initial,snapshot(),trace
        for number,task in enumerate(selected):
            folder=output/f'task_{number}';folder.mkdir()
            print('Auditing task:',task,flush=True)
            try:
                initial,_,_=replay(task,[])
                twice=snapshot()
                save(folder/'initial.json',initial)
                stable_reads=initial==twice
                actions=initial['valid_actions']
                door=next((a for a in actions if a.startswith('open door to ')),None)
                candidates=[a for a in ('look around','inventory','wait','wait1') if a in actions]
                if door:candidates.append(door)
                action_reports=[]
                for action in candidates:
                    b,after,log=replay(task,[action]);b2,after2,log2=replay(task,[action])
                    record={'action':action,'initial_replay_matches':b==b2,'full_audit_snapshot_replay_matches':after==after2,
                            'observation_trace_replay_matches':log==log2,
                            'elapsed_moves':after['moves']-b['moves'],
                            'exported_tree_unchanged':after['tree']==b['tree'],
                            'valid_actions_unchanged':after['valid_actions']==b['valid_actions'],
                            'difference_sample':diff(b['tree'],after['tree'])}
                    action_reports.append(record)
                    save(folder/f'action_{len(action_reports)}.json',{'summary':record,'before':b,'after':after,'repeat_after':after2,'log':log})
                    print('  Checked:',action,flush=True)
                door_report=None;search_report=None
                if door:
                    close='close '+door[len('open '):]
                    _,opened,_=replay(task,[door])
                    if close in opened['valid_actions']:
                        b,closed,log=replay(task,[door,close])
                        door_report={'actions':[door,close],'elapsed_moves':closed['moves']-b['moves'],
                                     'exact_exported_tree_restored':closed['tree']==b['tree'],
                                     'differences':diff(b['tree'],closed['tree']),'matched_duration_controls':[]}
                        for waits in (['wait'],['wait1'],['wait','wait'],['wait1','wait1']):
                            if waits[0] not in actions:continue
                            wb,wa,wl=replay(task,waits)
                            if wa['moves']-wb['moves']==door_report['elapsed_moves']:
                                door_report['matched_duration_controls'].append({'actions':waits,'initial_match':wb==b,
                                    'final_tree_matches_door':wa['tree']==closed['tree'],
                                    'interpretation':'time-relative comparison only; not original-state restoration'})
                        save(folder/'door_comparison.json',{'summary':door_report,'final':closed,'trace':log})
                    # Bounded search without unsafe tree-only deduplication.
                    # It measures feasibility; hitting the budget NEVER implies irreversible.
                    q=deque([[]]);tested=0;witness=None;unstable=False;logs=[]
                    while q and tested<MAX_SEARCH_NODES:
                        seq=q.popleft();b,end,trace=replay(task,[door]+seq)
                        b2,end2,trace2=replay(task,[door]+seq)
                        tested+=1
                        if b!=b2 or end!=end2 or trace!=trace2:
                            unstable=True;break
                        logs.append({'undo_sequence':seq,'tree_sha256':end['tree_sha256'],'moves':end['moves'],'trace':trace})
                        if end['tree']==b['tree']:
                            witness=seq;break
                        if len(seq)<HORIZON and not (trace and trace[-1]['done']):
                            # Heuristic order only; all enumerated valid actions remain in queue.
                            next_actions=sorted(end['valid_actions'],key=lambda a:(a!=close,a))
                            q.extend(seq+[a] for a in next_actions)
                    search_report={'candidate':door,'tested_nodes':tested,'queued_nodes_remaining':len(q),
                                   'exported_tree_witness':witness,'replay_instability':unstable,
                                   'enumerated_search_exhausted':not q and witness is None and not unstable,
                                   'status':'replay_unstable' if unstable else 'exported_tree_witness_only' if witness is not None else 'unresolved',
                                   'full_state_coverage_verified':False,'final_label':None}
                    save(folder/'bounded_search.json',{'summary':search_report,'nodes':logs})
                report={'task':task,'variation':0,'simplifications':'','consecutive_reads_match':stable_reads,
                        'initial_valid_action_count':len(actions),'action_checks':action_reports,
                        'door_comparison':door_report,'bounded_search':search_report}
                reports.append(report);save(folder/'summary.json',report)
            except Exception as exc:
                errors.append({'task':task,'type':type(exc).__name__,'message':str(exc)})
                (folder/'error.txt').write_text(traceback.format_exc())
                print('  Task audit failed; error saved:',type(exc).__name__,flush=True)
            save(output/'progress.json',{'reports':reports,'errors':errors})
    finally:
        if env is not None:env.close()
    summary={'purpose':'state semantics and search feasibility; not dataset generation',
             'task_reports':reports,'errors':errors,'full_state_coverage_verified':False,
             'final_labels_generated':0,'saved_to':str(output)}
    save(output/'summary.json',summary)
    print('SCIENCEWORLD SEMANTICS AUDIT FINISHED')
    print(json.dumps(summary,indent=2))
    return summary

if __name__=='__main__':
    from google.colab import drive
    drive.mount('/content/drive')
    run(Path('/content/drive/MyDrive/ADNN_Irreversibility'))
