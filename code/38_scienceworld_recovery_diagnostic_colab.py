"""Read pilot37 and test targeted recoveries. Never emits final training labels."""
import argparse, gzip, hashlib, importlib.metadata, json, os, time, traceback
from pathlib import Path
from datetime import datetime, timezone
from collections import Counter
COMMIT='e8216d6044e8e39be9fcb185e3b2dfb602584b52'
JAR='e77b0fee7d68abe3ca5b12e57d86e2bfe7603f20c5200da0b0f085939faeb465'
SOURCE='scienceworld_physical_pilot_20261107T091929052223Z'

def save(p,v):
    tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(v,indent=2,allow_nan=False));tmp.replace(p)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def world(s):return {k:s[k] for k in ('tree','look','inventory','valid_actions')}
def diff(a,b,path='$'):
    if type(a)!=type(b):return [{'path':path,'before':a,'after':b}]
    if isinstance(a,dict):
        result=[]
        for k in sorted(a.keys()|b.keys()):
            if k not in a or k not in b:result.append({'path':path+'/'+str(k),'before':a.get(k),'after':b.get(k),'missing_key':True})
            else:result.extend(diff(a[k],b[k],path+'/'+str(k)))
        return result
    if isinstance(a,list):
        if a==b:return []
        return [{'path':path,'before':a,'after':b}]
    return [] if a==b else [{'path':path,'before':a,'after':b}]
def objects(tree):
    out={}
    def walk(node,parent=None):
        if not isinstance(node,dict):return
        uid=str(node.get('uuid',''))
        if uid:out[uid]={'name':node.get('name',''),'parent':parent}
        contents=node.get('contents',{})
        for child in (contents.values() if isinstance(contents,dict) else contents if isinstance(contents,list) else []):walk(child,uid or parent)
    walk(tree);return out

def suggestions(action,before,after):
    valid=set(after['valid_actions']); options=[]
    for a,b in [('activate ','deactivate '),('deactivate ','activate ')]:
        if action.startswith(a):options.append(b+action[len(a):])
    if action.startswith('move '):
        old,new=objects(before['tree']),objects(after['tree'])
        for uid,item in old.items():
            if uid in new and item['parent']!=new[uid]['parent'] and item['parent'] in old:
                options.append('move '+item['name']+' to '+old[item['parent']]['name'])
    return [[x] for x in dict.fromkeys(options) if x in valid][:3]

def comparison(a,b):
    changes=diff(world(a),world(b))
    temps=[x for x in changes if x['path'].endswith('/temperatureC')]
    return {'exact_exported_world_match':not changes,'difference_count':len(changes),
            'temperature_field_differences':len(temps),
            'other_difference_count':len(changes)-len(temps),'differences':changes,
            'note':'Temperature-only differences are not ignored or classified as reversible.'}

def main(args):
    source=Path(args.source);assert (source/'results.json').exists(),'Pilot folder missing. Mount Drive and check source path.'
    config=json.loads((source/'config.json').read_text())
    assert config['commit']==COMMIT and config['jar_sha256']==JAR
    option='-Xlog:os+container=off'
    if option not in os.environ.get('JAVA_TOOL_OPTIONS','').split():os.environ['JAVA_TOOL_OPTIONS']=(os.environ.get('JAVA_TOOL_OPTIONS','')+' '+option).strip()
    from scienceworld import ScienceWorldEnv
    from scienceworld.constants import JAR_PATH
    dist=importlib.metadata.distribution('scienceworld')
    assert json.loads(dist.read_text('direct_url.json') or '{}').get('vcs_info',{}).get('commit_id')==COMMIT,'Install pinned version.'
    assert sha(Path(JAR_PATH))==JAR,'Unexpected simulator JAR; stop.'
    out=source.parent/('scienceworld_recovery_diagnostic_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'));out.mkdir()
    save(out/'config.json',{'source':str(source),'source_config_sha256':sha(source/'config.json'),
        'source_results_sha256':sha(source/'results.json'),'protocol':'targeted one-call inverse tests; optional wait1 duration control',
        'full_state_coverage_verified':False,'final_labels_created':0,'clock_and_goal_metadata_not_part_of_exported_world_comparison':True})
    (out/'runner.py').write_bytes(Path(__file__).read_bytes())
    records=json.loads((source/'results.json').read_text());rows=[];errors=[];env=None;started=time.monotonic()
    try:
        env=ScienceWorldEnv(envStepLimit=250)
        for record in records:
            cid=record['id'];folder=out/cid;folder.mkdir()
            try:
                print(cid,record['task'],record['candidate'],flush=True)
                archived=list(source.glob('task_*/'+cid+'/snapshots.json.gz'));assert len(archived)==1
                with gzip.open(archived[0],'rt') as f: saved=json.load(f)
                def snap(done):
                    return {'tree':env.getObjectTree(),'look':env.look(),'inventory':env.inventory(),
                        'valid_actions':sorted(set(env.get_valid_action_object_combinations())),
                        'moves':int(env.get_num_moves()),'score':float(env.lastStepScore),
                        'goal_progress':env.get_goal_progress(),'done':bool(done)}
                def replay(seq):
                    env.load(record['task'],record['variation'],'',generateGoldPath=False);env.reset();done=False;trace=[]
                    for a in seq:
                        if done:raise RuntimeError('Replay would continue past terminal state.')
                        obs,reward,done,info=env.step(a)
                        trace.append({'action':a,'observation':obs,'reward':reward,'done':bool(done),'moves':int(env.get_num_moves())})
                    return snap(done),trace
                prefix=record['prefix'];action=record['candidate']
                before,_=replay(prefix);after,trace=replay(prefix+[action])
                assert before==saved['before'] and after==saved['after'],'Current replay differs from archived snapshots; stop this case.'
                # Bindings are inspected at the pre-action state, where the action is available.
                replay(prefix)
                bindings=[x for x in env.get_valid_action_object_combinations_with_templates() if x.get('action')==action]
                ambiguity={'bindings':bindings,'distinct_object_id_lists':list({json.dumps(x.get('obj_ids')) for x in bindings}),
                           'same_displayed_pour_names':action.startswith('pour ') and len(action[5:].split(' into '))==2 and action[5:].split(' into ')[0]==action[5:].split(' into ')[1]}
                save(folder/'action_bindings.json',ambiguity)
                save(folder/'candidate_changes.json',comparison(before,after))
                old_search=json.loads((archived[0].parent/'search.json').read_text()) if (archived[0].parent/'search.json').exists() else {}
                depths=Counter(len(n['undo']) for n in old_search.get('nodes',[]))
                branching=len(after['valid_actions'])
                coverage={'prior_nodes_by_depth':dict(depths),'prior_queued_remaining':old_search.get('queued_remaining'),
                    'initial_recovery_branching':branching,'constant_branching_size_illustration_0_to_3':sum(branching**h for h in range(4)),
                    'illustration_is_not_a_bound_or_measured_total':True,
                    'prior_tested_one_step_fraction':depths[1]/branching if branching else None}
                save(folder/'search_coverage.json',coverage)
                tests=[]
                for undo in suggestions(action,before,after):
                    end,endtrace=replay(prefix+[action]+undo);repeat,repeattrace=replay(prefix+[action]+undo)
                    stable=end==repeat and endtrace==repeattrace
                    comp=comparison(before,end)
                    test={'undo':undo,'replay_stable':stable,'comparison':comp,'elapsed_moves':end['moves']-before['moves'],'trace':endtrace}
                    # A measured duration control, never a replacement for literal restoration.
                    if 'wait1' in before['valid_actions'] and not before['done']:
                        control,controltrace=replay(prefix+['wait1'])
                        duration=control['moves']-before['moves']
                        test['wait_control']={'actions':['wait1'],'elapsed_moves':duration,'duration_matches':duration==test['elapsed_moves'],
                            'time_relative_exported_world_match':world(control)==world(end) if duration==test['elapsed_moves'] else None,
                            'not_original_state_restoration':True,'trace':controltrace}
                    with gzip.open(folder/('targeted_'+str(len(tests))+'.json.gz'),'wt') as f:json.dump({'test':test,'end':end,'repeat_end':repeat},f)
                    tests.append({'undo':undo,'replay_stable':stable,'exact_exported_world_match':comp['exact_exported_world_match'],
                        'temperature_field_differences':comp['temperature_field_differences'],'other_difference_count':comp['other_difference_count'],
                        'wait_control':test.get('wait_control')})
                row={'id':cid,'candidate':action,'archive_replay_matches':True,'tests':tests,
                     'no_inverse_tested':not tests,'same_displayed_pour_names':ambiguity['same_displayed_pour_names'],
                     'search_coverage':coverage,'final_label':None}
                rows.append(row);save(folder/'summary.json',row)
                print('  targeted recoveries:',len(tests),'| exact exported matches:',sum(t['exact_exported_world_match'] and t['replay_stable'] for t in tests),flush=True)
            except Exception as exc:
                errors.append({'id':cid,'error':str(exc)});(folder/'error.txt').write_text(traceback.format_exc())
                print('  Case error saved:',str(exc),flush=True)
            save(out/'progress.json',{'rows':rows,'errors':errors})
    finally:
        tests=[t for r in rows for t in r['tests']]
        summary={'cases_checked':len(rows),'targeted_recoveries_tested':len(tests),
            'stable_exact_exported_world_witnesses':sum(t['replay_stable'] and t['exact_exported_world_match'] for t in tests),
            'recoveries_with_only_temperature_differences':sum(t['temperature_field_differences']>0 and t['other_difference_count']==0 for t in tests),
            'ambiguous_display_pour_cases':[r['id'] for r in rows if r['same_displayed_pour_names']],
            'cases_without_tested_inverse':[r['id'] for r in rows if r['no_inverse_tested']],
            'errors':errors,'elapsed_minutes':round((time.monotonic()-started)/60,2),
            'final_labels_created':0,'full_state_coverage_verified':False,'scienceworld_requirement_met':False,'saved_to':str(out)}
        save(out/'summary.json',summary);save(out/'results.json',rows)
        if env is not None:
            try:env.close()
            except Exception:pass
    print('TARGETED RECOVERY DIAGNOSTIC COMPLETE');print(json.dumps(summary,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',default='/content/drive/MyDrive/ADNN_Irreversibility/results/'+SOURCE)
    main(p.parse_args())
