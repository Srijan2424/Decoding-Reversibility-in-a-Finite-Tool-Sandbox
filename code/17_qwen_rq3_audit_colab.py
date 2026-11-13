"""Read-only episode audit; writes new analysis files, never replaces model logs."""
import json, hashlib, math, random, runpy
from pathlib import Path
from datetime import datetime, timezone
from collections import Counter
from copy import deepcopy

def main():
    from google.colab import drive
    root=Path('/content/drive/MyDrive/ADNN_Irreversibility')
    if not root.is_dir():drive.mount('/content/drive')
    source=root/'results/qwen_rq3_33a9369f3e575b3d'
    config=json.loads((source/'config.json').read_text())
    engine_path=source/'engine_source.py'
    assert hashlib.sha256(engine_path.read_bytes()).hexdigest()==config['engine_sha256']
    engine=runpy.run_path(str(engine_path),run_name='audit_engine')
    pack=root/'behavior/rq3_candidate_20261018T163731128446Z/tasks_private.jsonl'
    raw=pack.read_bytes();assert hashlib.sha256(raw).hexdigest()==config['task_sha256']
    tasks={r['id']:r for r in map(json.loads,raw.decode().splitlines())}
    out=source/('audit_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'));out.mkdir()
    details=[];errors=[];reasons=Counter();domains={};before_state_anomalies=[]
    for ident,task in tasks.items():
        record=json.loads((source/(ident+'.json')).read_text())
        assert record['id']==ident and record['run_key']=='33a9369f3e575b3d' and record['complete']
        env=engine['BehaviorSandbox'](task);ever_success=False;undo_after_success=False
        expected_before=[]
        for event in record['events']:
            expected_before.append(deepcopy(env.state))
            was_success=env.success()
            response=env.step(event['parsed_call'])
            assert response==event['tool_response'] and env.state==event['after_state'],ident
            assert env.logs[-1]['irreversible_execution']==event['irreversible_execution']
            if response['ok'] and event['parsed_call']['tool']=='undo_pending' and was_success:undo_after_success=True
            ever_success|=env.success()
            if not response['ok']:
                reason='parse_failure' if event['parse_error'] is not None else response['error']
                reasons[reason]+=1
                errors.append({'id':ident,'domain':task['domain'],'step':event['step'],'kind':reason,
                               'raw_response':event['raw_response'],'parsed_call':event['parsed_call'],'response':response})
        for index,event in enumerate(record['events']):
            if event['before_state'] != expected_before[index]:
                matches_legacy = event['before_state'] == env.logs[index]['before']
                anomaly={'id':ident,'step':event['step'],'matches_reproduced_legacy_alias_bug':matches_legacy,
                         'stored_before':event['before_state'],'reconstructed_before':expected_before[index]}
                before_state_anomalies.append(anomaly)
                assert matches_legacy, ('Unexplained before-state mismatch; do not accept run',ident,event['step'])
        metrics={**env.score(),'parse_failures':sum(e['parse_error'] is not None for e in record['events'])}
        assert metrics==record['metrics'],('Score mismatch',ident)
        row={'id':ident,'pair_id':task['pair_id'],'context_group':task['context_group'],'domain':task['domain'],
             'version':task['version'],'termination':record['termination'],'ever_successful':ever_success,
             'undid_after_success':undo_after_success,'lost_success':ever_success and not env.success(),
             'tool_sequence':[e['parsed_call']['tool'] if isinstance(e['parsed_call'],dict) else '<parse failure>' for e in record['events']],
             **metrics}
        details.append(row)
    assert len(details)==120
    pairs=[]
    for pid in sorted({r['pair_id'] for r in details}):
        a=next(r for r in details if r['pair_id']==pid and r['version']=='reversible')
        b=next(r for r in details if r['pair_id']==pid and r['version']=='irreversible')
        pairs.append({'pair_id':pid,'context_group':a['context_group'],'rev_caution':int(a['caution']),
                      'irr_caution':int(b['caution']),'rev_success':int(a['task_success']),'irr_success':int(b['task_success'])})
    def paired_statistics(sample):
        n=len(sample)
        return {'caution_difference':sum(p['irr_caution']-p['rev_caution'] for p in sample)/n,
                'A_more_cautious_irreversible':sum(p['irr_caution']>p['rev_caution'] for p in sample)/n,
                'success_difference':sum(p['irr_success']-p['rev_success'] for p in sample)/n}
    point=paired_statistics(pairs);groups=sorted({p['context_group'] for p in pairs});rng=random.Random(42)
    draws=[paired_statistics([p for g in rng.choices(groups,k=len(groups)) for p in pairs if p['context_group']==g]) for _ in range(1000)]
    def percentile(values,q):
        v=sorted(values);x=(len(v)-1)*q;i=int(x);f=x-i
        return v[i]*(1-f)+v[min(i+1,len(v)-1)]*f
    cis={k:[percentile([d[k] for d in draws],.025),percentile([d[k] for d in draws],.975)] for k in point}
    more=sum(p['irr_caution']>p['rev_caution'] for p in pairs)
    less=sum(p['irr_caution']<p['rev_caution'] for p in pairs)
    discordant=more+less
    pvalue=min(1.,2*sum(math.comb(discordant,k) for k in range(min(more,less)+1))/2**discordant) if discordant else 1.
    for domain in ('files','email','wallet'):
        subset=[r for r in details if r['domain']==domain]
        domains[domain]={'n':len(subset),'successes':sum(r['task_success'] for r in subset),
                         'episodes_with_parse_failures':sum(r['parse_failures']>0 for r in subset),
                         'no_execution':sum(r['no_execution'] for r in subset),
                         'lost_success':sum(r['lost_success'] for r in subset)}
    summary={'episodes_replayed':120,
             'before_state_logging_anomalies':len(before_state_anomalies),
             'logging_note':'After an invalid call, the original engine aliased its before-state log to mutable live state. Every reported anomaly was reproduced using the archived engine. Responses, after-states and final scores are checked independently. Originals preserved; reconstructed before-states saved separately.','all_saved_scores_match_replay':True,'domains':domains,
             'invalid_call_counts':dict(reasons),'episodes_that_lost_success':[r['id'] for r in details if r['lost_success']],
             'episodes_with_undo_after_success':[r['id'] for r in details if r['undid_after_success']],
             'paired_caution_counts':{'more_cautious_irreversible':more,'more_cautious_reversible':less,'tied':60-discordant},
             'paired_metrics':point,'context_cluster_bootstrap_95_intervals':cis,
             'bootstrap_resamples':1000,'context_groups':len(groups),
             'mcnemar_exact_two_sided_p':pvalue,
             'mcnemar_caveat':'Conventional test treats task pairs as independent; shared templates/contexts weaken that assumption. Cluster bootstrap separately accounts for context grouping.',
             'conditional_caution_before_irreversible':None,
             'conditional_caution_note':'Undefined when no irreversible executions occurred; not zero.',
             'blindness_gap':None,'blindness_gap_note':'K still requires separate stated-knowledge measurement on the same pairs.',
             'saved_to':str(out)}
    for name,obj in [('summary.json',summary),('episode_audit.json',details),('invalid_calls.json',errors),('paired_results.json',pairs),('before_state_anomalies.json',before_state_anomalies)]:
        (out/name).write_text(json.dumps(obj,indent=2))
    print('QWEN BEHAVIOUR AUDIT PASSED');print(json.dumps(summary,indent=2))
    print('FIRST 6 INVALID CALLS (full list saved in invalid_calls.json):')
    print(json.dumps(errors[:6],indent=2))

if __name__=='__main__':main()
