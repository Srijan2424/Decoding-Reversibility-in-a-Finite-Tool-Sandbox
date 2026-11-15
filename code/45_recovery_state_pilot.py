"""CPU diagnostic for state coverage and replay at the pinned ScienceWorld revision."""
import argparse
import hashlib
import json
from pathlib import Path

COMMIT = "e8216d6044e8e39be9fcb185e3b2dfb602584b52"
JAR_SHA = "e77b0fee7d68abe3ca5b12e57d86e2bfe7603f20c5200da0b0f085939faeb465"
CASES = [
    ("boil", 1, ["open door to hallway", "go to hallway"], ["open door to kitchen", "close door to kitchen"]),
    ("freeze", 1, ["open door to hallway", "go to hallway", "open door to kitchen", "go to kitchen"], ["activate sink", "deactivate sink"]),
    ("chemistry-mix", 2, [], ["open cupboard", "close cupboard"]),
    ("boil", 2, [], ["activate stopwatch", "deactivate stopwatch"]),
]


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def difference_counts(before, after):
    if before == after:
        return {}
    counts = {"temperature": 0, "other": 0}
    def visit(a, b, path):
        if a == b:
            return
        if isinstance(a, dict) and isinstance(b, dict):
            for key in a.keys() | b.keys():
                if key in a and key in b:
                    visit(a[key], b[key], path + (key,))
                else:
                    counts["other"] += 1
        elif path[-2:] == ("propMaterial", "temperatureC"):
            counts["temperature"] += 1
        else:
            counts["other"] += 1
    visit(before, after, ())
    return counts


def jvm_state(env):
    server = env.server
    interface = server.agentInterface().get()
    agent = server.agent().get()
    universe = interface.universe()
    stack = [universe]
    seen = set()
    watches = {}
    omitted_genetics = []
    portals = {}
    while stack:
        obj = stack.pop()
        ident = str(obj.uuid())
        if ident in seen:
            continue
        seen.add(ident)
        cls = obj.getClass().getSimpleName()
        if cls == "StopWatch":
            watches[ident] = [obj.stopwatchCurTime(), obj.stopwatchUserTime()]
        if cls == "Portal":
            portals[ident] = json.loads(obj.toJSON())
        if obj.propChromosomePairs().isDefined() or obj.propPollination().isDefined():
            omitted_genetics.append(ident)
        for children in (obj.getContainedObjects(True), obj.getPortals(True)):
            iterator = children.iterator()
            while iterator.hasNext():
                stack.append(iterator.next())
    return {
        "stopwatch_counters": watches,
        "portal_objects": portals,
        "waiting_counter": agent.waitingCounter(),
        "objects_with_genetic_or_pollination_state": sorted(omitted_genetics),
    }


def execute(env, task, variation, commands):
    env.load(task, variation, "", generateGoldPath=False)
    env.reset()
    trace = []
    for action in commands:
        valid = set(env.get_valid_action_object_combinations())
        if action not in valid:
            raise ValueError(f"Unavailable action {action!r}. Related: {[x for x in sorted(valid) if x.startswith(action.split()[0]+' ')][:12]}")
        before = env.get_num_moves()
        obs, reward, done, _ = env.step(action)
        trace.append([action, obs, reward, done, env.get_num_moves() - before])
        if done:
            raise ValueError("Episode ended while replaying")
    return {
        "tree": env.getObjectTree(),
        "extra": jvm_state(env),
        "moves": env.get_num_moves(),
        "trace": trace,
    }


def verify_case(env, task, variation, prefix, actions):
    before = execute(env, task, variation, prefix)
    candidate = execute(env, task, variation, prefix + actions[:1])
    after = execute(env, task, variation, prefix + actions)
    replicates = [execute(env, task, variation, seq) for seq in (prefix, prefix + actions[:1], prefix + actions)]
    stable = all(a == b for a, b in zip((before, candidate, after), replicates))
    control = execute(env, task, variation, prefix + ["wait1"])
    duration = after["moves"] - before["moves"]
    control_duration = control["moves"] - before["moves"]
    matched = duration == control_duration
    return {
        "task": task, "variation": variation, "prefix": prefix,
        "candidate": actions[0], "attempted_inverse": actions[1:],
        "reset_and_replay_stable": stable,
        "candidate_changed_tree": candidate["tree"] != before["tree"],
        "candidate_changed_extra_state": candidate["extra"] != before["extra"],
        "inverse_restored_tree": after["tree"] == before["tree"],
        "inverse_restored_extra_state": after["extra"] == before["extra"],
        "residual_in_tree": difference_counts(before["tree"], after["tree"]),
        "changed_extra_fields": sorted(k for k in before["extra"] if before["extra"][k] != after["extra"][k]),
        "snapshots_sha256": [digest(x) for x in (before, candidate, after)],
        "elapsed_moves": duration,
        "control_elapsed_moves": control_duration,
        "equal_duration_control": matched,
        "inverse_tree_matches_control": (after["tree"] == control["tree"]) if matched else None,
        "inverse_extra_state_matches_control": (after["extra"] == control["extra"]) if matched else None,
        "inverse_tree_vs_control": difference_counts(control["tree"], after["tree"]) if matched else None,
        "stopwatch_counters": {k: v["extra"]["stopwatch_counters"] for k, v in
                              zip(("before", "candidate", "after_inverse", "after_wait"), (before, candidate, after, control))},
        "count_objects_with_unserialized_genetic_state": len(before["extra"]["objects_with_genetic_or_pollination_state"]),
        "trace": after["trace"],
        "label": None,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    from scienceworld import ScienceWorldEnv
    from scienceworld.constants import JAR_PATH
    jar_hash = hashlib.sha256(Path(JAR_PATH).read_bytes()).hexdigest()
    if jar_hash != JAR_SHA:
        raise ValueError(f"Unexpected simulator JAR: {jar_hash}")
    env = ScienceWorldEnv(envStepLimit=250)
    report = {
        "source_revision": COMMIT, "jar_sha256": jar_hash, "scope": "diagnostic inverse sequences",
        "complete_physical_state_verified": False, "accepted_labels": 0, "cases": [],
    }
    try:
        for task, variation, prefix, actions in CASES:
            try:
                case = verify_case(env, task, variation, prefix, actions)
            except (AssertionError, ValueError) as exc:
                case = {"task": task, "variation": variation, "error": str(exc), "label": None}
            report["cases"].append(case)
            print(task, variation, case.get("residual_in_tree", case.get("error")), flush=True)
    finally:
        env.close()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
