"""Check temperature-only residuals and two candidate invariants on pinned ScienceWorld."""
from importlib.util import module_from_spec, spec_from_file_location
import hashlib
import json
from pathlib import Path

from scienceworld import ScienceWorldEnv
from scienceworld.constants import JAR_PATH

ROOT = Path(__file__).resolve().parent
spec = spec_from_file_location("pilot", ROOT / "46_recovery_search_pilot.py")
pilot = module_from_spec(spec)
spec.loader.exec_module(pilot)


def stopwatch(extra):
    (timer,) = extra["stopwatch_counters"].items()
    return timer


def rule(before, after):
    residual = pilot.diagnostic.difference_counts(before[0], after[0])
    if before[1] != after[1] or residual.get("other"):
        return "nonthermal_residual"
    return "thermal_only_residual" if residual.get("temperature") else "equal_on_compared_fields"


def run(env):
    checks = []
    for task, variation in (("freeze", 2), ("chemistry-mix", 2), ("melt", 2)):
        pilot.replay(env, task, variation, [])
        rooms = set(env.get_valid_action_object_combinations())
        apples = sorted(a for a in rooms if a.startswith("eat ") and "apple" in a and "pineapple" not in a)
        if not "activate stopwatch" in rooms or not apples:
            checks.append({"task": task, "variation": variation, "status": "kitchen_commands_unavailable", "relevant_commands": sorted(a for a in rooms if "stopwatch" in a or a.startswith("eat "))[:25]})
            continue
        (apple_action,) = [a for a in env.get_valid_action_object_combinations_with_templates()
                           if a["action"] == apples[0]]
        (apple_id,) = apple_action["obj_ids"]
        before = pilot.full_snapshot(env)
        wait_trace, _ = pilot.replay(env, task, variation, ["wait1"])
        wait = pilot.full_snapshot(env)
        wait_moves = env.get_num_moves()
        pilot.replay(env, task, variation, ["activate stopwatch", "deactivate stopwatch"])
        cycle = pilot.full_snapshot(env)
        assert env.get_num_moves() == wait_moves
        watch_id, start = stopwatch(before[1])
        after_id, after = stopwatch(cycle[1])
        assert watch_id == after_id and start == [0, 0] and after == [0, 1]
        assert rule(wait, cycle) == "nonthermal_residual"
        pilot.replay(env, task, variation, [apples[0]])
        eaten = env.getObjectTree()
        old_ids = {str(o["uuid"]) for o in pilot.flattened(before[0])}
        new_ids = {str(o["uuid"]) for o in pilot.flattened(eaten)}
        assert str(apple_id) in old_ids - new_ids
        assert not any(str(apple_id) in [str(i) for i in a["obj_ids"]]
                       for a in env.get_valid_action_object_combinations_with_templates())
        result = {"task": task, "variation": variation, "status": "checked",
                  "stopwatch_id": watch_id, "display_before": start[1], "display_after_cycle": after[1],
                  "timer_cycle_vs_wait": rule(wait, cycle), "timer_cycle_vs_pre_action": rule(before, cycle),
                  "wait_vs_pre_action": rule(before, wait),
                  "eat_command": apples[0], "eat_apple_object_id": apple_id, "apple_missing_from_world_and_action_bindings": True}
        checks.append(result)
        print(result, flush=True)
    return checks


if __name__ == "__main__":
    assert hashlib.sha256(Path(JAR_PATH).read_bytes()).hexdigest() == pilot.diagnostic.JAR_SHA
    env = ScienceWorldEnv(envStepLimit=250)
    try:
        cases = run(env)
        receipt = {"checks": cases, "accepted_labels": 0,
                   "scope": "Checks observed residuals, not validation of the partial-recovery definition."}
        (ROOT / "recovery_rule_checks.json").write_text(json.dumps(receipt, indent=2) + "\n")
    finally:
        env.close()
