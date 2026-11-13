"""CPU diagnostic only. No final labels; no approximate or time-matched equality."""
import argparse
import gzip
import hashlib
import importlib.metadata
import inspect
import json
import os
import platform
import time
import traceback
from collections import deque, Counter
from datetime import datetime, timezone
from pathlib import Path

COMMIT = 'e8216d6044e8e39be9fcb185e3b2dfb602584b52'
JAR_SHA = 'e77b0fee7d68abe3ca5b12e57d86e2bfe7603f20c5200da0b0f085939faeb465'

def save(path, value):
    data = json.dumps(value, indent=2, allow_nan=False)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(data)
    tmp.replace(path)

def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()

def physical(action):
    return action.lower().startswith(('activate ', 'deactivate ', 'mix ', 'pour ', 'move ', 'use '))

def projected(snapshot):
    # Diagnostic exported world only. Clock/history/goal progress remain separate.
    return {k: snapshot[k] for k in ('tree', 'look', 'inventory', 'valid_actions')}

def search(replay, prefix, candidate, reference, max_nodes, seconds):
    """BFS over the enumerated valid interface; budgets never imply irreversibility."""
    queue = deque([[]]); nodes = []; start = time.monotonic(); witness = None
    reason = 'enumerated_space_exhausted'
    while queue:
        if len(nodes) >= max_nodes or time.monotonic() - start >= seconds:
            reason = 'budget_exhausted'; break
        undo = queue.popleft()
        end, trace = replay(prefix + [candidate] + undo)
        matches = projected(end) == projected(reference)
        nodes.append({'undo': undo, 'snapshot_sha256': digest(end),
                      'exported_world_matches': matches, 'done': end['done'], 'trace': trace})
        if matches:
            repeat, repeat_trace = replay(prefix + [candidate] + undo)
            if repeat != end or repeat_trace != trace:
                reason = 'witness_replay_unstable'; break
            witness = undo; reason = 'exported_world_witness_only'; break
        if len(undo) < 3 and not end['done']:
            queue.extend(undo + [a] for a in end['valid_actions'])
    return {'status': reason, 'nodes_tested': len(nodes), 'queued_remaining': len(queue),
            'witness': witness, 'elapsed_seconds': round(time.monotonic()-start, 3),
            'final_label': None, 'nodes': nodes,
            'coverage_caveat': 'Enumerated valid interface only; full simulator state coverage unverified.'}

def run(args):
    option = '-Xlog:os+container=off'
    if option not in os.environ.get('JAVA_TOOL_OPTIONS', '').split():
        os.environ['JAVA_TOOL_OPTIONS'] = (os.environ.get('JAVA_TOOL_OPTIONS', '')+' '+option).strip()
    from scienceworld import ScienceWorldEnv
    from scienceworld.constants import JAR_PATH
    dist = importlib.metadata.distribution('scienceworld')
    source = json.loads(dist.read_text('direct_url.json') or '{}')
    assert source.get('vcs_info', {}).get('commit_id') == COMMIT, 'Install the pinned commit using the notebook setup cell.'
    assert hashlib.sha256(Path(JAR_PATH).read_bytes()).hexdigest() == JAR_SHA, 'JAR differs from prior pilot. Stop and share this error.'
    root = Path(args.project)
    assert root.is_dir(), 'Mount Drive first; project folder is missing.'
    out = root/'results'/('scienceworld_physical_pilot_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    out.mkdir(parents=True)
    save(out/'config.json', {'commit': COMMIT, 'jar_sha256': JAR_SHA, 'version': dist.version,
        'python': platform.python_version(), 'arguments': vars(args), 'simplifications': '',
        'scope': 'Physical-action diagnostics, not dataset labels', 'comparison': 'Exact exported tree plus look/inventory/valid actions',
        'excluded_from_restoration_comparison': ['moves', 'goal_progress', 'score', 'history', 'hidden simulator internals'],
        'full_state_coverage_verified': False, 'partial_recovery_definition': None})
    (out/'runner.py').write_bytes(Path(__file__).read_bytes())
    (out/'installed_wrapper.py').write_text(inspect.getsource(ScienceWorldEnv))
    env = None; results = []; errors = []; discoveries = []
    started = time.monotonic()
    def summary():
        return {'diagnostic_run_complete': True, 'candidate_checks': len(results),
            'stable_candidates': sum(r.get('replay_stable', False) for r in results),
            'search_status_counts': dict(Counter(r.get('search', {}).get('status', 'not_run') for r in results)),
            'tasks_attempted': [d['task'] for d in discoveries], 'errors': errors,
            'elapsed_minutes': round((time.monotonic()-started)/60, 2), 'final_labels_created': 0,
            'full_state_coverage_verified': False, 'scienceworld_requirement_met': False, 'saved_to': str(out)}
    try:
        env = ScienceWorldEnv(envStepLimit=250)
        tasks = env.get_task_names(); save(out/'available_tasks.json', tasks)
        chosen = []
        for keyword in ('boil', 'freeze', 'mix', 'melt'):
            match = next((t for t in tasks if keyword in t.lower() and t not in chosen), None)
            if match: chosen.append(match)
        if not chosen: raise RuntimeError('No physical task names matched; inspect available_tasks.json.')
        for task in chosen:
            discovery = {'task': task, 'variation': 0, 'candidates': []}; discoveries.append(discovery)
            folder = out/('task_'+str(len(discoveries))); folder.mkdir()
            try:
                print('Discovering physical actions:', task, flush=True)
                env.load(task, 0, '', generateGoldPath=True)
                gold = env.get_gold_action_sequence()
                save(folder/'gold_path.json', gold)
                if not gold or any(a.startswith('ERROR') for a in gold):
                    raise RuntimeError('Gold trajectory unavailable; no candidate actions invented.')
                def snap(done=False):
                    return {'tree': env.getObjectTree(), 'look': env.look(), 'inventory': env.inventory(),
                        'valid_actions': sorted(set(env.get_valid_action_object_combinations())),
                        'moves': int(env.get_num_moves()), 'score': float(env.lastStepScore),
                        'goal_progress': env.get_goal_progress(), 'done': bool(done)}
                def replay(actions):
                    env.load(task, 0, '', generateGoldPath=False)
                    env.reset(); trace = []; done = False
                    for action in actions:
                        if done: raise RuntimeError('Replay would step beyond terminal state.')
                        obs, reward, done, info = env.step(action)
                        trace.append({'action': action, 'observation': obs, 'reward': reward,
                                      'done': bool(done), 'moves': int(env.get_num_moves())})
                    return snap(done), trace
                # Follow supplied trajectory to reach physical situations; never infer success from text alone.
                env.load(task, 0, '', generateGoldPath=False); env.reset()
                prefix = []; candidates = []
                for action in gold[:args.max_prefix]:
                    valid = sorted(set(env.get_valid_action_object_combinations()))
                    if physical(action) and action in valid:
                        candidates.append((list(prefix), action))
                    obs, reward, done, info = env.step(action)
                    prefix.append(action)
                    if done: break
                # Evenly sample trajectory positions rather than taking only its first actions.
                if len(candidates) > args.per_task:
                    indices = sorted({round(i*(len(candidates)-1)/(args.per_task-1)) for i in range(args.per_task)}) if args.per_task > 1 else [0]
                    candidates = [candidates[i] for i in indices]
                discovery['candidates'] = [{'prefix': p, 'action': a} for p,a in candidates]
                save(folder/'discovery.json', discovery)
                for prefix, action in candidates:
                    cid = f'P{len(results)+1:03d}'; case = folder/cid; case.mkdir()
                    print(cid, task, ':', action, flush=True)
                    before, t0 = replay(prefix); before2, t02 = replay(prefix)
                    after, t1 = replay(prefix+[action]); after2, t12 = replay(prefix+[action])
                    stable = before == before2 and after == after2 and t0 == t02 and t1 == t12
                    record = {'id': cid, 'task': task, 'variation': 0, 'prefix': prefix, 'candidate': action,
                        'replay_stable': stable, 'candidate_enumerated': action in before['valid_actions'],
                        'exported_world_changed': projected(before) != projected(after),
                        'elapsed_moves': after['moves']-before['moves'],
                        'valid_recovery_actions': len(after['valid_actions']), 'final_label': None}
                    with gzip.open(case/'snapshots.json.gz', 'wt') as f:
                        json.dump({'before': before, 'repeat_before': before2, 'after': after, 'repeat_after': after2,
                                   'prefix_trace': t0, 'candidate_trace': t1}, f)
                    if stable and not before['done']:
                        checked = search(replay, prefix, action, before, args.max_nodes, args.seconds_per_case)
                        save(case/'search.json', checked)
                        record['search'] = {k:v for k,v in checked.items() if k != 'nodes'}
                    results.append(record); save(case/'result.json', record)
                    save(out/'progress.json', {'results': results, 'errors': errors})
                    print('  replay stable:', stable, '|', record.get('search', {}).get('status', 'search skipped'), flush=True)
            except Exception as exc:
                errors.append({'task': task, 'error': str(exc), 'type': type(exc).__name__})
                (folder/'error.txt').write_text(traceback.format_exc())
                print('Task diagnostic error saved:', task, str(exc), flush=True)
            save(out/'discoveries.json', discoveries)
    finally:
        save(out/'results.json', results); save(out/'summary.json', summary())
        if env is not None:
            try: env.close()
            except Exception: pass
    print('SCIENCEWORLD PHYSICAL PILOT FINISHED — DIAGNOSTICS ONLY')
    print(json.dumps(summary(), indent=2))
    return out

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--project', default='/content/drive/MyDrive/ADNN_Irreversibility')
    p.add_argument('--per-task', type=int, default=5)
    p.add_argument('--max-prefix', type=int, default=100)
    p.add_argument('--max-nodes', type=int, default=12)
    p.add_argument('--seconds-per-case', type=float, default=45)
    args = p.parse_args()
    assert min(args.per_task,args.max_prefix,args.max_nodes,args.seconds_per_case) > 0
    run(args)
