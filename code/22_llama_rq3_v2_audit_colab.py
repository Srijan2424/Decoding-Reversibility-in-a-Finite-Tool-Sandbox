"""Strict CPU replay audit for corrected Llama run. Original files remain unchanged."""
import json, hashlib, math, random, runpy
from pathlib import Path
from datetime import datetime, timezone
from collections import Counter

RUN='llama_rq3_v2_b0c2ec1d9ef80433'
METRICS=('task_success','caution','safe_route_executed','irreversible_action_taken','checked_before_execution','asked_before_execution')

def fingerprint(x):return hashlib.sha256(json.dumps(x,sort_keys=True).encode()).hexdigest()
def interval(values):
    v=sorted(x for x in values if x is not None)
    if not v:return None
    def q(p):
        x=(len(v)-1)*p;i=int(x);f=x-i
        return v[i]*(1-f)+v[min(i+1,len(v)-1)]*f
    return [q(.025),q(.975)]
def statistics(pairs):
    result={}
    for version in ('reversible','irreversible'):
        rows=[p[version] for p in pairs]
        for metric in METRICS:result[version+'/'+metric]=sum(r[metric] for r in rows)/len(rows)
        events=[r for r in rows if r['irreversible_action_taken']]
        result[version+'/caution_given_irreversible_execution']=sum(r['caution_before_irreversible'] for r in events)/len(events) if events else None
    result['caution_difference']=sum(p['irreversible']['caution']-p['reversible']['caution'] for p in pairs)/len(pairs)
    result['A_more_cautious_irreversible']=sum(p['irreversible']['caution']>p['reversible']['caution'] for p in pairs)/len(pairs)
    return result

def audit(root):
    source=root/'results'/RUN
    config=json.loads((source/'config.json').read_text())
    assert fingerprint(config)[:16]=='b0c2ec1d9ef80433','Config identity mismatch'
    engine_path=source/'engine_source.py'
    assert hashlib.sha256(engine_path.read_bytes()).hexdigest()==config['engine_sha256']
    runner_path=source/'runner_source.py'
    assert hashlib.sha256(runner_path.read_bytes()).hexdigest()==config['runner_sha256']
    engine=runpy.run_path(str(engine_path),run_name='audit_engine')
    runner=runpy.run_path(str(runner_path),run_name='audit_parser')
    tasks=json.loads((source/'effective_tasks_private.json').read_text())
    assert fingerprint(tasks)==config['effective_task_sha256']
    assert len(tasks)==len({t['id'] for t in tasks})==120
    details=[];errors=[]
    for task in tasks:
        ident=task['id'];record=json.loads((source/(ident+'.json')).read_text())
        assert record['id']==ident and record['run_key']=='b0c2ec1d9ef80433' and record['complete']
        for field in ('pair_id','context_group','version','domain'):assert record[field]==task[field]
        env=engine['BehaviorSandbox'](task)
        messages=[{'role':'system','content':runner['SYSTEM']},{'role':'user','content':task['rules']+'\nTask: '+task['task']+'\nInitial state: '+json.dumps(task['initial_state'],sort_keys=True)}]
        events=record['events'];assert 1<=len(events)<=15
        ever=False;undid=False
        for step,event in enumerate(events,1):
            assert event['step']==step and event['before_state']==env.state,(ident,step,'before state')
            assert event['prompt_messages']==messages,(ident,step,'prompt history')
            error_kind=None
            try:parsed=runner['parse'](event['raw_response'])
            except json.JSONDecodeError:parsed=None;error_kind='json_syntax'
            except (ValueError,TypeError):parsed=None;error_kind='schema'
            assert parsed==event['parsed_call'] and error_kind==event['parse_error_kind']
            was_success=env.success();response=env.step(parsed)
            assert response==event['tool_response'] and env.state==event['after_state'],(ident,step,'transition')
            assert env.logs[-1]['irreversible_execution']==event['irreversible_execution']
            if response['ok'] and parsed['tool']=='undo_pending' and was_success:undid=True
            ever|=env.success()
            messages.extend([{'role':'assistant','content':event['raw_response']},{'role':'user','content':'Tool result: '+json.dumps(response)}])
            if not response['ok']:errors.append({'id':ident,'step':step,'kind':error_kind or 'tool_precondition','raw_response':event['raw_response'],'tool_response':response,'status_before':event['before_state']['status'],'prepared_before':event['before_state']['prepared']})
        assert messages==record['messages']
        assert env.finished or len(events)==15
        assert record['termination']==('finish' if env.finished else 'step_limit')
        metrics={**env.score(),'parse_failures':sum(e['parse_error'] is not None for e in events),
                 'json_syntax_failures':sum(e['parse_error_kind']=='json_syntax' for e in events),
                 'schema_failures':sum(e['parse_error_kind']=='schema' for e in events)}
        assert metrics==record['metrics'],(ident,'metrics')
        details.append({**{k:task[k] for k in ('id','pair_id','context_group','version','domain')},**metrics,
                        'lost_success':ever and not env.success(),'undo_after_success':undid,
                        'tool_sequence':[e['parsed_call']['tool'] if e['parsed_call'] else '<parse failure>' for e in events]})
    original=json.loads((source/'summary.json').read_text())
    assert original['task_successes']==sum(r['task_success'] for r in details)
    for version in ('reversible','irreversible'):
        rows=[r for r in details if r['version']==version]
        for k,v in original['by_version'][version].items():
            assert v==(len(rows) if k=='n' else sum(r[k] for r in rows)),('Summary mismatch',version,k)
    pairs=[]
    for pid in sorted({r['pair_id'] for r in details}):
        rows=[r for r in details if r['pair_id']==pid];assert len(rows)==2
        pair={r['version']:r for r in rows};assert set(pair)=={'reversible','irreversible'}
        assert rows[0]['context_group']==rows[1]['context_group']
        pairs.append(pair)
    assert len(pairs)==60
    def analyze(selected):
        groups=sorted({p['reversible']['context_group'] for p in selected})
        rng=random.Random(42);point=statistics(selected)
        draws=[statistics([p for g in rng.choices(groups,k=len(groups)) for p in selected if p['reversible']['context_group']==g]) for _ in range(1000)]
        more=sum(p['irreversible']['caution']>p['reversible']['caution'] for p in selected)
        less=sum(p['irreversible']['caution']<p['reversible']['caution'] for p in selected);n=more+less
        pvalue=min(1.,2*sum(math.comb(n,k) for k in range(min(more,less)+1))/2**n) if n else 1.
        return {'n_pairs':len(selected),'context_groups':len(groups),'point_estimates':point,
                'cluster_bootstrap_95_intervals':{k:interval([d[k] for d in draws]) for k in point},
                'bootstrap_draws_with_defined_metric':{k:sum(d[k] is not None for d in draws) for k in point},
                'caution_pair_counts':{'more_irreversible':more,'more_reversible':less,'tied':len(selected)-n},
                'mcnemar_exact_two_sided_p':pvalue,
                'irreversible_execution_denominators':{v:sum(p[v]['irreversible_action_taken'] for p in selected) for v in ('reversible','irreversible')}}
    overall=analyze(pairs)
    domains={d:analyze([p for p in pairs if p['reversible']['domain']==d]) for d in ('files','email','wallet')}
    out=source/('strict_audit_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'));out.mkdir()
    summary={'episodes_replayed':120,'all_before_after_states_responses_prompts_and_scores_verified':True,
             'invalid_call_counts':dict(Counter(e['kind'] for e in errors)),
             'lost_success_ids':[r['id'] for r in details if r['lost_success']],
             'failure_breakdown':{v:{'n':sum(r['version']==v for r in details),
                 'final_failures':sum(r['version']==v and not r['task_success'] for r in details),
                 'lost_success':sum(r['version']==v and r['lost_success'] for r in details),
                 'undo_after_success':sum(r['version']==v and r['undo_after_success'] for r in details)} for v in ('reversible','irreversible')},
             'invalid_tool_frequencies':dict(Counter(e['raw_response'] for e in errors)),
             'failure_examples':[{'id':r['id'],'version':r['version'],'domain':r['domain'],'tool_sequence':r['tool_sequence'],'lost_success':r['lost_success']} for r in details if not r['task_success']][:8],
             'overall':overall,'domains':domains,'bootstrap_resamples':1000,
             'caveats':['Intervals resample 20 context groups, keeping paired versions and domains together.',
                        'Conventional McNemar assumes independent pairs; shared contexts weaken that assumption.',
                        'Constant outcomes give degenerate bootstrap intervals, not certainty about generalisation.',
                        'Conditional caution is undefined where irreversible execution count is zero.',
                        'A is a paired proportion, not the difference between marginal caution rates.'],
             'blindness_gap':None,'pending':'Stated knowledge K, Llama behaviour, ScienceWorld and remaining final study.',
             'saved_to':str(out)}
    for filename,obj in [('summary.json',summary),('episodes.json',details),('pairs.json',pairs),('invalid_calls.json',errors)]:
        (out/filename).write_text(json.dumps(obj,indent=2))
    report='# Corrected Llama behaviour audit\n\nAll 120 episodes passed strict replay, including before/after snapshots, prompt histories, parsing, tool responses and metrics.\n\n'
    report+='Paired results (proportions):\n\n'+json.dumps(overall,indent=2)+'\n\nLimitations: '+ ' '.join(summary['caveats'])+'\n\nThe blindness gap remains uncomputed because K has not been measured. Original runs remain preserved. Llama uses the corrected interface shared with Qwen.\n'
    (out/'PROGRESS_AUDITED.md').write_text(report)
    print('LLAMA V2 STRICT AUDIT PASSED');print(json.dumps(summary,indent=2))
    return summary

if __name__=='__main__':
    from google.colab import drive
    root=Path('/content/drive/MyDrive/ADNN_Irreversibility')
    if not root.is_dir():drive.mount('/content/drive')
    audit(root)
