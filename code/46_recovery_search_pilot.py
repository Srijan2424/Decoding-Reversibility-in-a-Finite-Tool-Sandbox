"""Validate comparator controls and benchmark a legal-action search on fresh cases."""
import argparse
from collections import Counter, deque
import hashlib
import json
from pathlib import Path
import time

from importlib.util import module_from_spec, spec_from_file_location

spec = spec_from_file_location("diagnostic", Path(__file__).with_name("45_recovery_state_pilot.py"))
diagnostic = module_from_spec(spec)
spec.loader.exec_module(diagnostic)


def replay(env, task, variation, actions):
    env.load(task, variation, "", generateGoldPath=False)
    env.reset()
    trace = []
    done = False
    for action in actions:
        if done:
            raise ValueError("Action after terminal state")
        observation, reward, done, _ = env.step(action)
        trace.append([action, observation, reward, done])
    return trace, done


def full_snapshot(env):
    return env.getObjectTree(), diagnostic.jvm_state(env)


def flattened(tree):
    pending = [tree]
    while pending:
        obj = pending.pop()
        yield obj
        pending.extend(obj.get("contents", {}).values())


def controls(env):
    task, variation = "boil", 2
    def snapshot(commands):
        replay(env, task, variation, commands)
        return full_snapshot(env)
    before = snapshot([])
    repeat = snapshot([])
    waited = snapshot(["wait1"])
    watch = snapshot(["activate stopwatch", "deactivate stopwatch"])
    freeze = diagnostic.CASES[1]
    prefix, actions = freeze[2:]
    replay(env, freeze[0], freeze[1], prefix + ["wait1"])
    sink_control = env.getObjectTree()
    replay(env, freeze[0], freeze[1], prefix + actions)
    sink_result = env.getObjectTree()
    before_drain = next(o for o in flattened(sink_control) if o["name"] == "sewer")
    after_drain = next(o for o in flattened(sink_result) if o["name"] == "sewer")
    # Take a legal pickup from the same kitchen and check that the object ID persists.
    replay(env, task, variation, [])
    actions = set(env.get_valid_action_object_combinations())
    pickup = next(a for a in sorted(actions) if a == "pick up thermometer")
    moved = snapshot([pickup])
    inventory = next(o for o in flattened(moved[0]) if o["name"] == "inventory")
    results = {
        "independent_resets_equal": before == repeat,
        "time_changes_temperatures": diagnostic.difference_counts(before[0], waited[0]).get("temperature", 0) > 0,
        "stopwatch_export_collision_detected": watch[0] == waited[0] and
        watch[1]["stopwatch_counters"] != waited[1]["stopwatch_counters"],
        "sink_drainage_detected": before_drain != after_drain and
        set(after_drain["contents"]) > set(before_drain["contents"]),
        "pickup_location_detected": any(o["name"] == "thermometer" for o in flattened(inventory)) and before != moved,
    }
    return results


def search(env, task, variation, limit, seconds, horizon):
    trace, _ = replay(env, task, variation, [])
    reference = full_snapshot(env)
    replay(env, task, variation, [])
    assert reference == full_snapshot(env), "Reset snapshots differ"
    initial = sorted(set(env.get_valid_action_object_combinations()))
    candidate = next(a for a in initial if a.startswith("open ") and "door" in a)
    candidate_trace, candidate_done = replay(env, task, variation, [candidate])
    candidate_snapshot = full_snapshot(env)
    assert candidate_snapshot != reference, "Candidate did not change compared state"
    repeat_trace, _ = replay(env, task, variation, [candidate])
    assert candidate_snapshot == full_snapshot(env) and candidate_trace == repeat_trace
    actions = sorted(set(env.get_valid_action_object_combinations())) if not candidate_done else []
    pending = deque([(a,) for a in actions])
    visited = Counter()
    witness = None
    enumerated = Counter({1: len(actions)})
    started = time.monotonic()
    while pending and sum(visited.values()) < limit and time.monotonic() - started < seconds:
        path = pending.popleft()
        trace, done = replay(env, task, variation, [candidate, *path])
        depth = len(path)
        visited[depth] += 1
        if sum(visited.values()) % 100 == 0:
            print(f"{task} {variation}: {sum(visited.values())} branches evaluated", flush=True)
        if env.getObjectTree() == reference[0] and diagnostic.jvm_state(env) == reference[1]:
            second_trace, _ = replay(env, task, variation, [candidate, *path])
            if trace != second_trace or full_snapshot(env) != reference:
                raise ValueError("Recovery witness failed independent replay")
            witness = list(path)
            break
        if depth < horizon and not done:
            children = sorted(set(env.get_valid_action_object_combinations()))
            enumerated[depth + 1] += len(children)
            pending.extend(path + (a,) for a in children)
    frontier = Counter(map(len, pending))
    return {
        "task": task, "variation": variation, "candidate": candidate,
        "candidate_actions_available_after": len(actions),
        "expanded_by_depth": dict(visited), "enumerated_by_depth": dict(enumerated),
        "queued_by_depth": dict(frontier),
        "complete_horizons": [h for h in range(1, horizon + 1) if not any(d <= h for d in frontier)],
        "witness_on_compared_fields": witness, "seconds": round(time.monotonic() - started, 2),
        "status": "witness" if witness else "exhausted" if not pending else "budget_exhausted",
        "accepted_label": None,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--nodes", type=int, default=32)
    p.add_argument("--seconds", type=float, default=40)
    p.add_argument("--horizon", type=int, choices=(1, 2, 3), default=3)
    p.add_argument("--case", choices=("all", "boil", "freeze", "chemistry-mix"), default="all")
    p.add_argument("--skip-controls", action="store_true")
    args = p.parse_args()
    if args.nodes < 1 or args.seconds <= 0:
        p.error("Search budget must be positive")
    from scienceworld import ScienceWorldEnv
    from scienceworld.constants import JAR_PATH
    jar_digest = hashlib.sha256(Path(JAR_PATH).read_bytes()).hexdigest()
    assert jar_digest == diagnostic.JAR_SHA
    env = ScienceWorldEnv(envStepLimit=250)
    report = {"jar_sha256": jar_digest, "accepted_labels": 0, "full_state_coverage_verified": False,
              "budget_per_case": {"nodes": args.nodes, "seconds": args.seconds, "horizon": args.horizon}, "cases": []}
    def save():
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    try:
        if not args.skip_controls:
            report["controls"] = controls(env)
            print("Controls:", report["controls"], flush=True)
            save()
        for task, variation in (("boil", 3), ("freeze", 4), ("chemistry-mix", 3)):
            if args.case != "all" and args.case != task:
                continue
            result = search(env, task, variation, args.nodes, args.seconds, args.horizon)
            report["cases"].append(result)
            save()
            print(result, flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    main()
