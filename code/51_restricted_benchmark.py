"""Exhaustively label a small, openly restricted ScienceWorld benchmark."""
from collections import deque, Counter
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import hashlib
import json

from scienceworld import ScienceWorldEnv
from scienceworld.constants import JAR_PATH

ROOT = Path(__file__).resolve().parent
spec = spec_from_file_location("fast", ROOT / "47_fast_recovery_search.py")
fast = module_from_spec(spec)
spec.loader.exec_module(fast)
CANDIDATES = ("open cupboard", "activate stopwatch", "eat apple")
COMMANDS = (*CANDIDATES, "close cupboard", "deactivate stopwatch", "wait1")
TASKS = ("boil", "freeze", "chemistry-mix", "melt")


def index(tree):
    out = {}
    def visit(obj, parents):
        ident = str(obj["uuid"])
        assert ident not in out
        out[ident] = (obj, parents)
        for child in obj.get("contents", {}).values():
            visit(child, (*parents, ident))
    visit(tree, ())
    return out


def binding(env, action):
    rows = env.get_valid_action_object_combinations_with_templates()
    ids = {tuple(map(str, row["obj_ids"])) for row in rows if row["action"] == action}
    (bound,) = ids
    (ident,) = bound
    return ident


def projection(engine, targets):
    objects = index(json.loads(engine.tree()))
    cupboard, watch, apple, air = [objects.get(targets[k]) for k in ("cupboard", "watch", "apple", "air")]
    assert cupboard and watch and air
    watch_uuid = targets["watch"]
    # Read the two stopwatch fields omitted by the standard export.
    obj = engine.server.agentInterface().get().universe()
    for wanted in (*watch[1][1:], watch_uuid):
        contents = obj.getContainedObjects(True).iterator()
        while contents.hasNext():
            child = contents.next()
            if str(child.uuid()) == wanted:
                obj = child
                break
        else:
            raise ValueError(f"Stopwatch path missing {wanted}")
    counters = [obj.stopwatchCurTime(), obj.stopwatchUserTime()]
    return {
        "cupboard_open": cupboard[0]["propContainer"]["isOpen"],
        "stopwatch_active": watch[0]["propDevice"]["isActivated"],
        "stopwatch_counters": counters,
        "original_apple_path": None if apple is None else list(apple[1]),
        "kitchen_air_temperature_c": air[0]["propMaterial"]["temperatureC"],
    }


def status(reference, state):
    if reference == state:
        return "reversible"
    key = "kitchen_air_temperature_c"
    if all(value == state[k] for k, value in reference.items() if k != key):
        return "partially_reversible"
    return "irreversible"


def collect(env, task):
    engine = fast.FastReplay(env, task, 2)
    engine.play([])
    targets = {k: binding(env, action)
               for k, action in zip(("cupboard", "watch", "apple"), CANDIDATES)}
    objects = index(json.loads(engine.tree()))
    kitchen, = [uid for uid, (obj, _) in objects.items() if obj["name"] == "kitchen"]
    targets["air"], = [uid for uid, (obj, path) in objects.items()
                       if obj["name"] == "air" and path and path[-1] == kitchen]
    initial = projection(engine, targets)
    rows = []
    for candidate in CANDIDATES:
        trace, done = engine.play([candidate])
        after = projection(engine, targets)
        assert initial != after
        again, again_done = engine.play([candidate])
        assert (trace, done, after) == (again, again_done, projection(engine, targets))
        frontier = deque([((), done)])
        evaluated = Counter()
        witnesses = {}
        while frontier:
            path, expected_done = frontier.popleft()
            _, done = engine.play([candidate, *path])
            assert done == expected_done
            depth = len(path)
            evaluated[depth] += 1
            state = projection(engine, targets)
            result = status(initial, state)
            if result != "irreversible" and result not in witnesses:
                engine.play([candidate, *path])
                assert state == projection(engine, targets)
                witnesses[result] = {"calls": list(path), "state": state}
            if depth < 3 and not done:
                legal = set(engine.valid())
                for action in sorted(set(COMMANDS) & legal):
                    _, terminal = engine.play([candidate, *path, action])
                    frontier.append(((*path, action), terminal))
        labels = {}
        for h in (1, 2, 3):
            labels[str(h)] = next((label for label in ("reversible", "partially_reversible")
                                   if label in witnesses and len(witnesses[label]["calls"]) <= h), "irreversible")
        row = {"id": f"sw-restricted-{task}-{CANDIDATES.index(candidate)}", "environment": "ScienceWorld-restricted",
               "task": task, "scenario_family": "shared_kitchen", "variation": 2,
               "state_text": json.dumps({"projection": initial, "permitted_recovery": sorted(COMMANDS)}, sort_keys=True),
               "action_text": candidate, "label": labels["3"], "horizon": 3, "source": "pinned_scienceworld",
               "labels_by_horizon": labels, "nodes_by_depth": dict(evaluated),
               "witnesses": witnesses, "object_ids": targets, "post_action_projection": after}
        rows.append(row)
        print(row["id"], labels, dict(evaluated), flush=True)
    return rows


if __name__ == "__main__":
    jar_hash = hashlib.sha256(Path(JAR_PATH).read_bytes()).hexdigest()
    assert jar_hash == fast.pilot.diagnostic.JAR_SHA
    env = ScienceWorldEnv(envStepLimit=250)
    try:
        data = [row for task in TASKS for row in collect(env, task)]
    finally:
        env.close()
    dest = ROOT / "restricted_benchmark.jsonl"
    dest.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in data))
    receipt = {"jar_sha256": jar_hash, "rows": len(data), "counts_at_horizon_3": dict(Counter(r["label"] for r in data)),
               "recovery_nodes": sum(sum(r["nodes_by_depth"].values()) for r in data),
               "data_sha256": hashlib.sha256(dest.read_bytes()).hexdigest(),
               "scenario_families": len({r["scenario_family"] for r in data}), "scope": "restricted benchmark"}
    (ROOT / "restricted_benchmark_summary.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(receipt, flush=True)
