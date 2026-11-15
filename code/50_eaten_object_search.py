"""Enumerate API-listed one-call recovery after a uniquely bound eat action."""
import hashlib
from importlib.util import module_from_spec, spec_from_file_location
import json
from pathlib import Path
import time

from scienceworld import ScienceWorldEnv
from scienceworld.constants import JAR_PATH

ROOT = Path(__file__).resolve().parent
spec = spec_from_file_location("fast", ROOT / "47_fast_recovery_search.py")
fast = module_from_spec(spec)
spec.loader.exec_module(fast)


def contains(tree, uuid):
    return any(str(obj["uuid"]) == uuid for obj in fast.pilot.flattened(json.loads(tree)))


def run(env):
    task, variation = "freeze", 2
    engine = fast.FastReplay(env, task, variation)
    engine.play([])
    (bound,) = [row for row in env.get_valid_action_object_combinations_with_templates()
                if row["action"] == "eat apple"]
    (uuid,) = map(str, bound["obj_ids"])
    assert contains(engine.tree(), uuid)
    _, done = engine.play(["eat apple"])
    assert not done and not contains(engine.tree(), uuid)
    branches = engine.valid()
    started = time.monotonic()
    restores = []
    for action in branches:
        engine.play(["eat apple", action])
        if contains(engine.tree(), uuid):
            restores.append(action)
    return {"task": task, "variation": variation, "candidate": "eat apple",
            "original_apple_id": uuid, "one_call_checked": len(branches),
            "one_call_identity_restorations": restores, "elapsed_seconds": round(time.monotonic() - started, 2),
            "scope": "One-call API enumeration, original-object identity only",
            "accepted_labels": 0}


if __name__ == "__main__":
    assert hashlib.sha256(Path(JAR_PATH).read_bytes()).hexdigest() == fast.pilot.diagnostic.JAR_SHA
    env = ScienceWorldEnv(envStepLimit=250)
    try:
        result = run(env)
        (ROOT / "eaten_object_search.json").write_text(json.dumps(result, indent=2) + "\n")
        print(result, flush=True)
    finally:
        env.close()
