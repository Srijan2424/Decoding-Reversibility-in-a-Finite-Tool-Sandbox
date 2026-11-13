# Run in the SAME CPU notebook after the previous pilot.
import os, json, hashlib
from datetime import datetime, timezone
from pathlib import Path
from scienceworld import ScienceWorldEnv

option = "-Xlog:os+container=off"
if option not in os.environ.get("JAVA_TOOL_OPTIONS", "").split():
    os.environ["JAVA_TOOL_OPTIONS"] = (os.environ.get("JAVA_TOOL_OPTIONS", "") + " " + option).strip()

assert "RUN" in globals(), "Run the storage cell of notebook 03 first."
audit_dir = RUN / ("state_audit_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
audit_dir.mkdir(parents=True, exist_ok=False)

def fingerprint(tree):
    return hashlib.sha256(json.dumps(tree, sort_keys=True, allow_nan=False).encode()).hexdigest()

def differences(a, b, path="$", found=None):
    if found is None: found = []
    if len(found) >= 6: return found
    if type(a) is not type(b):
        found.append({"path":path,"a":str(a)[:180],"b":str(b)[:180]})
    elif isinstance(a, dict):
        for key in sorted(set(a) | set(b)):
            if key not in a or key not in b:
                found.append({"path":path+"/"+key,"difference":"key missing from one tree"})
            else: differences(a[key], b[key], path+"/"+key, found)
            if len(found) >= 6: break
    elif isinstance(a, list):
        if len(a) != len(b): found.append({"path":path,"length_a":len(a),"length_b":len(b)})
        for i, (x,y) in enumerate(zip(a,b)):
            differences(x,y,path+"/"+str(i),found)
            if len(found) >= 6: break
    elif a != b: found.append({"path":path,"a":str(a)[:180],"b":str(b)[:180]})
    return found[:6]

# A new Java environment is required because the pilot closed the previous one.
audit_env = ScienceWorldEnv(envStepLimit=30)
traces, read_checks = [], []
sequence = ["open art studio door", "go to art studio", "go to door"]
try:
    for repetition in range(3):
        audit_env.load("boil", 0, "")
        audit_env.reset()
        first = audit_env.getObjectTree()
        repeated_read = audit_env.getObjectTree()
        read_checks.append(fingerprint(first) == fingerprint(repeated_read))
        snapshots = [first]
        for action in sequence:
            observation, reward, done, info = audit_env.step(action)
            snapshots.append(audit_env.getObjectTree())
            if done: break
        traces.append(snapshots)
        (audit_dir / f"object_trees_run_{repetition}.json").write_text(json.dumps(snapshots, indent=2, allow_nan=False))
finally:
    audit_env.close()

matches = [[fingerprint(a) == fingerprint(b) for a,b in zip(traces[0], trace)] for trace in traces[1:]]
first_mismatch = None
for run_index, trace in enumerate(traces[1:], 1):
    for step_index, (a,b) in enumerate(zip(traces[0], trace)):
        if fingerprint(a) != fingerprint(b):
            first_mismatch = {"repeat":run_index,"step":step_index,"differences":differences(a,b)}
            break
    if first_mismatch: break
summary = {
    "consecutive_reads_match":read_checks,
    "snapshot_counts":[len(t) for t in traces],
    "replay_tree_matches_by_step":matches,
    "action_changes_exported_tree":[fingerprint(a) != fingerprint(b) for a,b in zip(traces[0],traces[0][1:])],
    "tree_root_type":type(traces[0][0]).__name__,
    "root_keys_sample":list(traces[0][0])[:12] if isinstance(traces[0][0],dict) else None,
    "first_replay_mismatch":first_mismatch,
    "export_completeness_validated":False,
    "saved_to":str(audit_dir),
}
(audit_dir / "summary.json").write_text(json.dumps(summary,indent=2))
print(json.dumps(summary,indent=2))
print("TREE PREVIEW (truncated):")
print(json.dumps(traces[0][0],indent=2)[:2500])
