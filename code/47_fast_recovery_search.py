"""Benchmark faster replay against the public Python wrapper on the pinned JVM."""
import argparse
from collections import Counter, deque
import hashlib
from importlib.util import spec_from_file_location, module_from_spec
import json
from pathlib import Path
import time

spec = spec_from_file_location("pilot", Path(__file__).with_name("46_recovery_search_pilot.py"))
pilot = module_from_spec(spec)
spec.loader.exec_module(pilot)


class FastReplay:
    def __init__(self, env, task, variation):
        self.env = env
        self.server = env.server
        env.load(task, variation, "", generateGoldPath=False)

    def play(self, commands):
        self.server.reset()
        score = 0
        trace = []
        done = False
        # ScienceWorldEnv.reset() performs this initial observation.
        for index, action in enumerate(("look around", *commands)):
            if done:
                raise ValueError("Action after terminal state")
            observation = self.server.step(action)
            next_score = int(round(100 * self.server.getScore()))
            done = bool(self.server.getCompleted() or
                        self.server.getNumMoves() > self.env.envStepLimit or next_score < 0)
            if index:
                trace.append([action, observation, next_score - score, done])
            score = next_score
        return trace, done

    def tree(self):
        return self.server.getObjectTree("")

    def valid(self):
        return sorted(set(self.server.getValidActionObjectCombinations()))


def benchmark(env, task, variation):
    base = FastReplay(env, task, variation)
    base.play([])
    initial = base.tree()
    actions = base.valid()
    assert actions == sorted(set(env.get_valid_action_object_combinations()))
    candidate = next(a for a in actions if a.startswith("open ") and "door" in a)
    base.play([candidate])
    after = base.valid()
    expected = json.loads(initial)
    equality_count = 0
    digest = hashlib.sha256()
    public_seconds = fast_seconds = 0
    for command in after:
        start = time.monotonic()
        old_trace, old_done = pilot.replay(env, task, variation, [candidate, command])
        old_tree = env.getObjectTree()
        public_seconds += time.monotonic() - start
        start = time.monotonic()
        new_trace, new_done = base.play([candidate, command])
        new_tree = base.tree()
        fast_seconds += time.monotonic() - start
        assert old_trace == new_trace and old_done == new_done, command
        assert old_tree == json.loads(new_tree), command
        if old_tree == expected:
            equality_count += 1
        digest.update(json.dumps([command, old_trace, old_done, old_tree], sort_keys=True).encode())
    return {
        "task": task, "variation": variation, "candidate": candidate, "matched_branches": len(after),
        "equal_export_count": equality_count, "trace_and_tree_digest": digest.hexdigest(),
        "public_seconds": round(public_seconds, 3), "fast_seconds": round(fast_seconds, 3),
        "speedup": round(public_seconds / fast_seconds, 2),
    }


def search(env, task, variation, limit, seconds):
    base = FastReplay(env, task, variation)
    base.play([])
    reference = json.loads(base.tree())
    extra = pilot.diagnostic.jvm_state(env)
    candidate = next(a for a in base.valid() if a.startswith("open ") and "door" in a)
    base.play([candidate])
    choices = base.valid()
    frontier = deque((a,) for a in choices)
    visited = Counter()
    discovered = Counter({1: len(choices)})
    witness = None
    # Support early witnesses without changing the breadth-first coverage count.
    started = time.monotonic()
    while frontier and sum(visited.values()) < limit and time.monotonic() - started < seconds:
        path = frontier.popleft()
        trace, done = base.play([candidate, *path])
        depth = len(path)
        visited[depth] += 1
        if sum(visited.values()) % 1000 == 0:
            print(task, dict(visited), flush=True)
        if json.loads(base.tree()) == reference and pilot.diagnostic.jvm_state(env) == extra:
            public_trace, _ = pilot.replay(env, task, variation, [candidate, *path])
            if public_trace != trace or pilot.full_snapshot(env) != (reference, extra):
                raise ValueError("Witness failed public-wrapper replay")
            witness = list(path)
            break
        if not done and depth < 3:
            children = base.valid()
            discovered[depth + 1] += len(children)
            frontier.extend(path + (child,) for child in children)
    pending = Counter(map(len, frontier))
    return {
        "task": task, "variation": variation, "candidate": candidate,
        "evaluated": dict(visited), "enumerated": dict(discovered), "remaining": dict(pending),
        "complete_horizons": [h for h in (1, 2, 3) if not any(d <= h for d in pending)],
        "witness_on_compared_fields": witness,
        "seconds": round(time.monotonic() - started, 2), "accepted_labels": 0,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("benchmark", "search"), required=True)
    parser.add_argument("--nodes", type=int, default=2400)
    parser.add_argument("--seconds", type=float, default=165)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    from scienceworld import ScienceWorldEnv
    from scienceworld.constants import JAR_PATH
    digest = hashlib.sha256(Path(JAR_PATH).read_bytes()).hexdigest()
    assert digest == pilot.diagnostic.JAR_SHA
    env = ScienceWorldEnv(envStepLimit=250)
    try:
        if args.mode == "benchmark":
            outcome = benchmark(env, "freeze", 4)
        else:
            outcome = search(env, "freeze", 4, args.nodes, args.seconds)
        args.output.write_text(json.dumps({"jar_sha256": digest, **outcome}, indent=2) + "\n")
        print(outcome, flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    main()
