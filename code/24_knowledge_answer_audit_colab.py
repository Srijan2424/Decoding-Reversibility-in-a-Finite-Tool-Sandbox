"""CPU answer audit; preserves original knowledge outputs."""
import json, hashlib
from pathlib import Path
from collections import Counter
from datetime import datetime, timezone
RUNS={'qwen':('qwen_paired_knowledge_ce4e9484ddc6bb02','qwen_rq3_v2_fb175ab231fc8381'),
      'llama':('llama_paired_knowledge_6cb1efb5c73041af','llama_rq3_v2_b0c2ec1d9ef80433')}
def fp(x):return hashlib.sha256(json.dumps(x,sort_keys=True).encode()).hexdigest()
def main():
    from google.colab import drive
    root=Path('/content/drive/MyDrive/ADNN_Irreversibility')
    if not root.is_dir():drive.mount('/content/drive')
    reports={}
    for model,(knowledge,behavior) in RUNS.items():
        source=root/'results'/knowledge
        config=json.loads((source/'config.json').read_text())
        assert fp(config)[:16]==knowledge.rsplit('_',1)[1]
        tasks=json.loads((root/'results'/behavior/'effective_tasks_private.json').read_text())
        assert fp(tasks)==config['effective_task_sha256']
        counts=Counter();matrix=Counter();examples=[];answers={};correct=0
        for task in tasks:
            a=json.loads((source/(task['id']+'.json')).read_text())
            assert a['id']==task['id'] and a['run_key']==knowledge.rsplit('_',1)[1]
            expected='reversible' if task['initial_state']['cancellation_enabled'] else 'irreversible'
            assert expected==task['version']
            message=task['rules']+'\nTask context: '+task['task']+'\nInitial state: '+json.dumps(task['initial_state'],sort_keys=True)+'\nCandidate action: '+json.dumps({'tool':task['primary_tool'],'arguments':{}},sort_keys=True)+'\n'+config['question']
            assert a['messages']==[{'role':'system','content':config['system']},{'role':'user','content':message}]
            try:
                obj=json.loads(a['raw_response'])
                label=obj['label'] if isinstance(obj,dict) and set(obj)=={'label'} and obj['label'] in ('reversible','partially_reversible','irreversible') else None
            except (ValueError,TypeError,KeyError):label=None
            assert label==a['predicted_label']
            answers[task['id']]=label;correct+=label==expected
            counts[label or '<invalid>']+=1;matrix[(task['domain'],expected,label or '<invalid>')]+=1
            # One complete response for every observed domain / truth / prediction combination.
            if matrix[(task['domain'],expected,label or '<invalid>')]==1:
                examples.append({'id':task['id'],'domain':task['domain'],'expected':expected,'prediction':label,'raw_response':a['raw_response']})
        pair_correct=0
        for pid in {t['pair_id'] for t in tasks}:
            group=[t for t in tasks if t['pair_id']==pid];assert len(group)==2
            pair_correct+=all(answers[t['id']]==t['version'] for t in group)
        summary=json.loads((source/'summary.json').read_text())
        assert correct/120==summary['individual_accuracy'] and pair_correct/60==summary['overall']['point']['K']
        reports[model]={'queries_verified':120,'correct_individual_answers':correct,'correct_pairs':pair_correct,
                        'predicted_label_counts':dict(counts),'confusion_rows':[{'domain':d,'expected':e,'predicted':p,'count':n} for (d,e,p),n in sorted(matrix.items())],
                        'examples':examples,'prompt_parser_and_accuracy_checks_passed':True}
    destination=root/'results'/('knowledge_answer_audit_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    destination.mkdir();(destination/'summary.json').write_text(json.dumps(reports,indent=2))
    print('KNOWLEDGE ANSWER AUDIT PASSED');print(json.dumps(reports,indent=2));print('Saved to:',destination)
if __name__=='__main__':main()
