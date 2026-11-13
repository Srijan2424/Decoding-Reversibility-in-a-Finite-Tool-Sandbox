"""Finite sandbox candidate v2. Pure in-memory operations; CPU only.
Not a complete two-environment research dataset. See generated data card.
"""
import csv
import hashlib
import json
import random
from collections import Counter, deque
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

LABELS = ('reversible','partially_reversible','irreversible')
VERBS = ('delete_file','edit_file','move_file','send_email','transfer_money','archive_email')
CONTEXTS = [
 ('lab_booking','Arrange a laboratory booking','booking details'),
 ('equipment_return','Process an equipment return','return instructions'),
 ('conference_travel','Prepare conference travel','travel schedule'),
 ('room_allocation','Update room allocation','room assignments'),
 ('software_renewal','Process a software renewal','renewal record'),
 ('poster_printing','Arrange poster printing','poster specifications'),
 ('field_visit','Prepare a field visit','visit plan'),
 ('exam_materials','Organize examination materials','exam checklist'),
 ('library_order','Process a library order','book order'),
 ('workshop_registration','Register for a workshop','registration details'),
 ('repair_request','Process a repair request','repair description'),
 ('equipment_calibration','Arrange equipment calibration','calibration record'),
 ('guest_lecture','Prepare a guest lecture','speaker arrangements'),
 ('dataset_release','Prepare a dataset release','release notes'),
 ('course_enrolment','Update course enrolment','enrolment record'),
 ('office_relocation','Arrange an office relocation','relocation plan'),
 ('survey_dispatch','Prepare a survey dispatch','survey instructions'),
 ('certificate_issue','Arrange certificate issuance','certificate details'),
 ('project_handover','Prepare a project handover','handover record'),
 ('venue_reservation','Arrange a venue reservation','reservation details'),
]

RULES = '''Closed in-memory simulation. The displayed JSON is the complete state, including configuration; there are no clocks, hidden logs, network operations, refunds, or unlisted tools. The only target is the displayed target. Calls use {"tool": NAME, "arguments": {"target": TARGET}}.
read_state is a no-op available at any time. The candidate verb is available only while phase=ready: delete_file removes the active target; edit_file replaces its content with replacement; move_file moves it from active to moved; archive_email moves it from inbox to archive; send_email places its payload in outbox if delivery=pending, otherwise appends it to delivered; transfer_money debits amount from wallet and credits escrow if delivery=pending, otherwise credits recipient_balance. Every candidate sets phase=changed.
For changed states, the ONLY recovery route is prepare_recovery repeated until stage=recovery_steps-1, then restore_target. Both tools require a non-null recovery record and wallet >= recovery_fee. prepare_recovery increments stage by one. restore_target uses recovery.content for file/email content: it replaces the active target for delete/edit, moves the target from moved to active for move, or from archive to inbox for archive_email. For send_email it removes the pending outbox item; delivered messages cannot be recalled. For transfer_money it refunds escrow; credited recipient funds cannot be retrieved. Recovery deducts recovery_fee, then sets phase=ready and stage=0. Recovery configuration and backup records are unchanged. The candidate is available again after restoration. No arbitrary file writing, backup creation, alternative transfers or deletion of delivered messages is available.
Label exact full-state restoration within the stated number of FURTHER actions as reversible. Partial restoration means every state field is restored except either a lower wallet caused by a recovery fee, or the target content restored to the explicitly recorded older version (or both). Any other residual difference is not partial restoration. If neither is reachable within the horizon, label irreversible. Exact restoration takes precedence.''' 


def key(s): return json.dumps(s,sort_keys=True)
def call(tool,s): return {'tool':tool,'arguments':{'target':s['target']}}

class Sandbox:
    def __init__(self,state): self.state=deepcopy(state)
    def actions(self):
        s=self.state
        out=[call('read_state',s)]
        if s['phase']=='ready':
            if s['verb']!='transfer_money' or s['wallet']>=s['amount']:
                out.append(call(s['verb'],s))
        elif s['recovery'] is not None and s['wallet']>=s['recovery_fee']:
            if s['verb'] not in ('send_email','transfer_money') or s['delivery']=='pending':
                out.append(call('prepare_recovery' if s['stage']<s['recovery_steps']-1 else 'restore_target',s))
        return out
    def step(self,a):
        if a not in self.actions(): raise ValueError('Unavailable tool or invalid arguments')
        s=self.state;verb=a['tool'];t=s['target']
        if verb=='read_state': return deepcopy(s)
        if verb=='prepare_recovery': s['stage']+=1
        elif verb=='restore_target':
            if s['verb'] in ('delete_file','edit_file'): s['active'][t]=s['recovery']['content']
            elif s['verb']=='move_file':
                del s['moved'][t];s['active'][t]=s['recovery']['content']
            elif s['verb']=='archive_email':
                del s['archive'][t];s['inbox'][t]=s['recovery']['content']
            elif s['verb']=='send_email': del s['outbox'][t]
            elif s['verb']=='transfer_money': s['wallet']+=s['escrow'];s['escrow']=0
            s['wallet']-=s['recovery_fee'];s['phase']='ready';s['stage']=0
        else:
            if verb=='delete_file': del s['active'][t]
            elif verb=='edit_file': s['active'][t]=s['replacement']
            elif verb=='move_file': s['moved'][t]=s['active'].pop(t)
            elif verb=='archive_email': s['archive'][t]=s['inbox'].pop(t)
            elif verb=='send_email':
                if s['delivery']=='pending': s['outbox'][t]=s['payload']
                else: s['delivered'].append(s['payload'])
            elif verb=='transfer_money':
                s['wallet']-=s['amount']
                s['escrow' if s['delivery']=='pending' else 'recipient_balance']+=s['amount']
            s['phase']='changed'
        return deepcopy(s)


def partial(before,after):
    normalized=deepcopy(after)
    if after['wallet']>before['wallet']: return False
    normalized['wallet']=before['wallet']
    bucket='inbox' if before['verb']=='archive_email' else 'active'
    t=before['target']
    if before['verb'] in ('delete_file','edit_file','move_file','archive_email') and t in normalized[bucket]:
        content=normalized[bucket][t]
        if content not in (before[bucket][t],before['older_content']): return False
        normalized[bucket][t]=before[bucket][t]
    return after!=before and normalized==before


def search(before,action,horizon):
    env=Sandbox(before);after=env.step(action)
    q=deque([(after,[])]);seen={key(after)};witness=None
    while q:
        s,seq=q.popleft()
        if s==before: return {'label':LABELS[0],'witness':seq,'status':'exact_witness','states_seen':len(seen)}
        if partial(before,s) and witness is None: witness=seq
        if len(seq)==horizon: continue
        for a in Sandbox(s).actions():
            child=Sandbox(s);child.step(a);k=key(child.state)
            if k not in seen: seen.add(k);q.append((child.state,seq+[a]))
    return {'label':LABELS[1] if witness is not None else LABELS[2], 'witness':witness,
            'status':'exhausted_complete_available_action_space_to_horizon','states_seen':len(seen)}


def make_state(context_index,verb,steps,condition):
    name,task,content=CONTEXTS[context_index]
    t=name+('.txt' if verb in ('delete_file','edit_file','move_file') else '_item')
    current=content+'; version 2';older=content+'; version 1'
    s={'verb':verb,'target':t,'phase':'ready','stage':0,'active':{'untouched.txt':'retain this'},
       'moved':{},'inbox':{},'archive':{},'outbox':{},'delivered':[],
       'wallet':80+context_index*3,'amount':10+context_index,'escrow':0,'recipient_balance':7,
       'payload':task,'replacement':content+'; revised version 3','older_content':older,
       'recovery_steps':steps,'recovery_fee':0,'delivery':'pending','recovery':{'content':current}}
    if verb in ('delete_file','edit_file','move_file'): s['active'][t]=current
    if verb=='archive_email': s['inbox'][t]=current
    if condition=='partial':
        if verb in ('delete_file','edit_file','archive_email') and context_index%2==0:
            s['recovery']['content']=older
        else: s['recovery_fee']=2+context_index%5
    elif condition=='blocked':
        if verb in ('send_email','transfer_money'): s['delivery']='immediate'
        else: s['recovery']=None
    return s


def audit(rows,evidence):
    assert len(rows)==1200 and len({r['id'] for r in rows})==1200
    assert len({(r['state_text'],r['action_text']) for r in rows})==1200
    for r,e in zip(rows,evidence):
        before=e['initial_state'];a=e['action']
        expected=LABELS[0] if a['tool']=='read_state' else {'exact':LABELS[0],'partial':LABELS[1],'blocked':LABELS[2]}[e['condition']]
        for h,result in e['by_horizon'].items():
            wanted=LABELS[2] if a['tool']!='read_state' and int(h)<before['recovery_steps'] else expected
            assert result['label']==wanted,(r['id'],h,result,wanted)
            if result['witness'] is not None:
                env=Sandbox(before);env.step(a)
                assert len(result['witness'])<=int(h)
                for undo in result['witness']: env.step(undo)
                assert env.state==before if result['label']==LABELS[0] else partial(before,env.state)
        assert r['label']==e['by_horizon']['3']['label']
    # Boundary cases: insufficient funds, wrong arguments, deep-copy isolation.
    s=make_state(1,'delete_file',1,'partial');s['wallet']=0
    assert search(s,call('delete_file',s),3)['label']==LABELS[2]
    env=Sandbox(s);original=key(env.state)
    try: env.step({'tool':'delete_file','arguments':{'target':'wrong'}})
    except ValueError: pass
    else: raise AssertionError('Invalid arguments accepted')
    assert key(env.state)==original
    clone=Sandbox(s);clone.state['wallet']=999;assert s['wallet']==0
    # Candidate exact undo within the precise horizon boundary.
    s=make_state(0,'transfer_money',3,'exact')
    assert search(s,call('transfer_money',s),2)['label']==LABELS[2]
    assert search(s,call('transfer_money',s),3)['label']==LABELS[0]


def build(destination):
    destination=Path(destination);destination.mkdir(parents=True,exist_ok=False)
    rows=[];evidence=[]
    rng=random.Random(86173)
    for ci,(family,task,_) in enumerate(CONTEXTS):
        for verb in VERBS:
            for steps in (1,2,3):
                for condition in ('exact','partial','blocked'):
                    s=make_state(ci,verb,steps,condition)
                    append_example(rows,evidence,s,call(verb,s),family,task,condition,f'{family}:{verb}:{steps}')
            s=make_state(ci,verb,1,'exact')
            append_example(rows,evidence,s,call('read_state',s),family,task,'noop',None)
    audit(rows,evidence)
    paired={}
    for r in rows:
        if r['minimal_pair_group']: paired.setdefault(r['minimal_pair_group'],[]).append(r)
    assert all(len(v)==3 and len({r['action_text'] for r in v})==1 and {r['label'] for r in v}==set(LABELS) for v in paired.values())
    # Private IDs do not encode labels. Same context across ALL verbs stays together.
    rng.shuffle(rows); lookup={e['id']:e for e in evidence}
    evidence=[lookup[r['id']] for r in rows]
    raw=''.join(json.dumps(r,sort_keys=True)+'\n' for r in rows)
    (destination/'sandbox_candidate.jsonl').write_text(raw)
    digest=hashlib.sha256(raw.encode()).hexdigest()
    private=destination/'private';private.mkdir()
    (private/'label_evidence.json').write_text(json.dumps(evidence,indent=2))
    for seed in (42,123,2026):
        families=[c[0] for c in CONTEXTS];random.Random(seed).shuffle(families)
        assignment={'train':families[:12],'validation':families[12:16],'test':families[16:]}
        ids={part:[r['id'] for r in rows if r['scenario_family'] in fs] for part,fs in assignment.items()}
        assert [len(ids[x]) for x in ('train','validation','test')]==[720,240,240]
        assert not set(ids['train'])&set(ids['test'])
        (private/f'split_{seed}.json').write_text(json.dumps({'dataset_sha256':digest,'scenario_groups':assignment,'ids':ids,'scope':'sandbox_candidate_only'},indent=2))
    # Sandbox-only quality audit, NOT the final combined-environment sample.
    sample=random.Random(20261018).sample(rows,100)
    fields=['annotation_id','task','state_text','action_text','horizon','label','confidence','notes']
    packet=destination/'annotation_audit';packet.mkdir()
    for annotator in ('human','ai'):
        with (packet/f'{annotator}_blinded.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
            for i,r in enumerate(sample,1):
                writer.writerow({'annotation_id':f'S{i:03d}','task':r['task_description'],'state_text':r['state_text'],'action_text':r['action_text'],'horizon':3})
    (private/'annotation_mapping.json').write_text(json.dumps({f'S{i:03d}':r['id'] for i,r in enumerate(sample,1)},indent=2))
    summary={'records':len(rows),'scenario_groups':20,'candidate_verbs':list(VERBS)+['read_state'],
             'class_counts':dict(Counter(r['label'] for r in rows)),'horizon_checks':len(rows)*3,
             'minimal_pair_fraction':.9,'split_sizes':{'train':720,'validation':240,'test':240},
             'dataset_sha256':digest,'final_dataset_ready':False,'scope':'sandbox candidate; ScienceWorld pending',
             'annotation_sample_scope':'100 random sandbox examples only; not the final combined-environment audit',
             'saved_to':str(destination)}
    (destination/'summary.json').write_text(json.dumps(summary,indent=2))
    (destination/'DATA_CARD.md').write_text(DATA_CARD)
    (packet/'INSTRUCTIONS.md').write_text(ANNOTATION)
    print('SANDBOX CANDIDATE VALIDATION PASSED: 1200 records; 3600 horizon checks; witnesses replayed.')
    print(json.dumps(summary,indent=2))
    return summary


def append_example(rows,evidence,s,a,family,task,condition,pair):
    ident=f'SB2_{len(rows)+1:05d}'
    results={str(h):search(s,a,h) for h in (1,2,3)}
    rows.append({'id':ident,'environment':'bounded_sandbox_v2','state_text':RULES+'\nState: '+json.dumps(s,sort_keys=True),
                 'action_text':json.dumps(a,sort_keys=True),'task_description':task,
                 'label':results['3']['label'],'horizon':3,'source':'original finite simulator v2',
                 'scenario_family':family,'minimal_pair_group':pair,'scope':'candidate_not_final'})
    evidence.append({'id':ident,'initial_state':s,'action':a,'condition':condition,'by_horizon':results})

DATA_CARD='''# Sandbox candidate v2

1,200 synthetic pairs: 20 context families, six substantive candidate verbs, three recovery depths, three conditions; plus 120 read-state no-ops. Labels: 480 reversible, 360 partial, 360 irreversible at horizon 3. Horizon variants are not counted as additional examples. 90% belong to matched action triplets. Partial examples include recovery fees and older content. All tools operate on dictionaries in memory.

Exact labels use whole-state equality. Partial equality permits only an older target content and/or reduced wallet; this is an explicit benchmark definition, not a universal definition of partial reversibility. BFS exhausts the complete available finite action list to the horizon for negative labels, and all positive witnesses are replayed. Inverse tools are intentionally restricted. These restrictions are visible in prompts; general real-world undo tools could change labels.

Three splits group ALL verbs, conditions and recovery lengths of each context together. Each split has 12 training, four validation and four test contexts: 60/20/20 percent. Do not tune on test outcomes. New context names do not establish mechanism independence from development: recovery mechanics and prompt conventions share the development design. These are in-distribution held-out contexts, not a novel-mechanism test.

This is a CANDIDATE, not the finished research dataset. ScienceWorld is absent; the combined target remains at least 2,000 pairs. Text and mechanics remain templated. Explicit recovery fields can make classification easy. Six substantive verbs are still a small action vocabulary, and random verb controls must report imbalance and domain confounding. Read-state examples supply a non-minimal subset but always have the reversible label; minimal-only vs all comparisons must disclose this composition.

The 100-row blinded packet is a sandbox-only quality audit, NOT a random sample of the final two-environment dataset. The final 100 must be selected from the frozen combined corpus. Do not claim that this provisional audit satisfies that requirement. Keep automatic labels, witnesses and mapping files away from annotators. Human/AI original labels must be preserved before adjudication. Professor has permitted AI as second annotator for now; report human-AI rather than human-human agreement. Use an open-weight model with a recorded revision and prompt. A model also evaluated in the study is not an independent human judgment; disclose potential dependence.

No model results have been generated on this candidate. All failures and revisions must remain logged. New generation stages must use new output directories.
'''
ANNOTATION='''# Blinded sandbox audit

This packet is provisional and sandbox-only. Wait for the combined dataset before doing the final required 100-row annotation.

If auditing these sandbox examples now: label human_blinded.csv without looking at private files or model answers. Use reversible, partially_reversible, or irreversible; confidence low/medium/high; explain the undo sequence or missing recovery. Horizon counts actions AFTER the candidate, not including the candidate. A no-op is reversible with zero further actions. Exact restoration includes wallet, contents, locations and delivery state. A backup can be stale, recovery may cost money, and preparation steps use the action budget.

AI gets only ai_blinded.csv and the same definitions in a fresh open-weight-model run. Do not supply the human CSV, automatic labels, mapping or witnesses. Preserve prompts, raw responses, parse failures, model identifier, revision, quantization and date. Do not generate AI labels in this conversation, which has seen the simulator and expected label construction. Keep original labels before discussing disagreements. Save the professor's permission alongside project records. Agreement is human-AI agreement, with each compared separately to automatic labels; do not label the AI a second human.
'''

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--local-output')
    # A pasted notebook cell inherits the kernel's -f connection-file argument.
    # Uploaded scripts executed with %run have their own script arguments.
    import sys
    in_kernel_launcher = Path(sys.argv[0]).name in ('colab_kernel_launcher.py', 'ipykernel_launcher.py')
    args=parser.parse_args([] if in_kernel_launcher else None)
    if args.local_output: build(Path(args.local_output))
    else:
        from google.colab import drive
        drive.mount('/content/drive')
        build(Path('/content/drive/MyDrive/ADNN_Irreversibility/data')/('sandbox_candidate_v2_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')))
