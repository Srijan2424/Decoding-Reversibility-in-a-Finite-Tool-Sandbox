"""CPU-only source inspection of the pinned export, not a simulator experiment."""
import argparse,hashlib,json,re,subprocess
from pathlib import Path
COMMIT='e8216d6044e8e39be9fcb185e3b2dfb602584b52'
PATHS=['scienceworld/struct/EnvObject.scala','scienceworld/runtime/pythonapi/PythonInterface.scala','scienceworld/objects/devices/StopWatch.scala','scienceworld/objects/agent/Agent.scala']
def audit(folder):
 texts={p:(folder/p.replace('/','_')).read_text() for p in PATHS}
 env=texts[PATHS[0]];export=env.split('def toJSON():String = {',1)[1].split('\nobject EnvObject',1)[0]
 active='\n'.join(line for line in export.splitlines() if not line.lstrip().startswith('//'))
 findings=[]
 for field in ['propChromosomePairs','propPollination']:
  assert 'var '+field in env and field not in active
  findings.append({'field':field,'declared':True,'absent_from_active_base_export':True,'basis':'Export statements are commented out.'})
 watch=texts[PATHS[2]]
 for field in ['stopwatchCurTime','stopwatchUserTime']:
  assert 'var '+field in watch and field not in active and 'def toJSON' not in watch
  findings.append({'field':field,'declared':True,'absent_from_active_base_export':True,'basis':'Subclass counters influence tick/description, but no subclass toJSON override exists.'})
 agent=texts[PATHS[3]]
 assert 'var waitingCounter' in agent and 'waitingCounter' not in active and 'def toJSON' not in agent
 findings.append({'field':'waitingCounter','declared':True,'absent_from_active_base_export':True,'basis':'Agent waiting state is absent from inherited serialization.'})
 assert 'agentInterface.get.universe.toJSON()' in texts[PATHS[1]]
 return {'commit':COMMIT,'audit_type':'static source inspection, not a dynamic simulator validation','getObjectTree_delegates_to_universe_toJSON':True,'omitted_fields':findings,
 'conclusion':'Object-tree equality alone is insufficient to establish equality of full simulator state.',
 'limits':['Does not establish that these omitted fields changed in the earlier 18 or 8 cases.','Does not determine a correct task-specific physical-state equivalence.','Does not classify new state-action pairs or satisfy the ScienceWorld dataset requirement.'],
 'sources':[{'path':p,'url':'https://github.com/allenai/ScienceWorld/blob/'+COMMIT+'/simulator/src/main/scala/'+p,'sha256':hashlib.sha256((folder/p.replace('/','_')).read_bytes()).hexdigest()} for p in PATHS]}
def main():
 p=argparse.ArgumentParser();p.add_argument('--source-dir',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--download',action='store_true');a=p.parse_args();a.source_dir.mkdir(parents=True,exist_ok=True)
 if a.download:
  for path in PATHS:subprocess.run(['curl','-sSL','--fail','--max-time','60','https://raw.githubusercontent.com/allenai/ScienceWorld/'+COMMIT+'/simulator/src/main/scala/'+path,'-o',str(a.source_dir/path.replace('/','_'))],check=True)
 result=audit(a.source_dir);a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2));print('SOURCE AUDIT COMPLETE: five omitted fields confirmed; zero new labels created.')
if __name__=='__main__':main()
