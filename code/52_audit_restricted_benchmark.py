"""Audit the restricted dataset receipt and replay its witnesses via the public wrapper."""
from collections import Counter
from importlib.util import module_from_spec, spec_from_file_location
import hashlib
import json
from pathlib import Path

from scienceworld import ScienceWorldEnv

ROOT = Path(__file__).resolve().parent
spec = spec_from_file_location("benchmark", ROOT / "51_restricted_benchmark.py")
b = module_from_spec(spec)
spec.loader.exec_module(b)
raw = (ROOT / "restricted_benchmark.jsonl").read_bytes()
receipt = json.loads((ROOT / "restricted_benchmark_summary.json").read_text())
rows = [json.loads(line) for line in raw.splitlines()]
assert hashlib.sha256(raw).hexdigest() == receipt["data_sha256"]
assert len(rows) == len({r["id"] for r in rows}) == receipt["rows"]
assert Counter(r["label"] for r in rows) == receipt["counts_at_horizon_3"]
assert sum(sum(r["nodes_by_depth"].values()) for r in rows) == receipt["recovery_nodes"]
env = ScienceWorldEnv(envStepLimit=250)
replayed = 0
try:
    for row in rows:
        fixed = json.loads(row["state_text"])
        engine = b.fast.FastReplay(env, row["task"], row["variation"])
        b.fast.pilot.replay(env, row["task"], row["variation"], [])
        assert b.projection(engine, row["object_ids"]) == fixed["projection"]
        for label, witness in row["witnesses"].items():
            assert set(witness["calls"]) <= set(fixed["permitted_recovery"])
            b.fast.pilot.replay(env, row["task"], row["variation"], [row["action_text"], *witness["calls"]])
            final = b.projection(engine, row["object_ids"])
            assert final == witness["state"]
            assert b.status(fixed["projection"], final) == label
            replayed += 1
        for h in (1, 2, 3):
            label = "irreversible"
            for possible in ("partially_reversible", "reversible"):
                if possible in row["witnesses"] and len(row["witnesses"][possible]["calls"]) <= h:
                    label = possible
            assert row["labels_by_horizon"][str(h)] == label
        assert row["label"] == row["labels_by_horizon"]["3"]
finally:
    env.close()
counts = Counter((r["state_text"], r["action_text"]) for r in rows)
print(json.dumps({"rows": len(rows), "unique_projected_state_action_pairs": len(counts),
                  "witnesses_replayed": replayed, "horizon_labels_checked": len(rows) * 3}, indent=2))
