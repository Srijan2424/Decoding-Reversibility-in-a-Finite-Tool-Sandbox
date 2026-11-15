"""Compare the JVM diagnostic against all 419 Python reset-and-replay branches."""
import base64
import hashlib
from importlib.util import module_from_spec, spec_from_file_location
import json
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parent
spec = spec_from_file_location("fast", ROOT / "47_fast_recovery_search.py")
fast = module_from_spec(spec)
spec.loader.exec_module(fast)

from scienceworld import ScienceWorldEnv
from scienceworld.constants import JAR_PATH

if hashlib.sha256(Path(JAR_PATH).read_bytes()).hexdigest() != fast.pilot.diagnostic.JAR_SHA:
    sys.exit("Unexpected simulator JAR")

with tempfile.TemporaryDirectory(prefix="adnn-jvm-") as class_dir:
    subprocess.run(["javac", "-classpath", str(JAR_PATH), "-d", class_dir,
                    str(ROOT / "RecoveryReplayBench.java")], check=True)
    run = ["java", "-Xmx1536m", "-classpath", f"{class_dir}:{JAR_PATH}", "RecoveryReplayBench"]
    output = subprocess.check_output(run + ["benchmark", "0", "120"], text=True)
    checks = {}
    for line in output.splitlines():
        fields = line.split("\t")
        if fields[0] == "CHECK":
            _, command, digest, terminal, equals = fields
            checks[base64.b64decode(command).decode()] = (digest, terminal == "true", equals == "true")
        if fields[0] == "SUMMARY":
            _, total, checked, listed, _, seconds, interrupted = fields
    assert len(checks) == int(total) == int(checked) == 419 and interrupted == "false"
    env = ScienceWorldEnv(envStepLimit=250)
    try:
        engine = fast.FastReplay(env, "freeze", 4)
        engine.play([])
        reference = engine.tree()
        candidate = "open door to art studio"
        engine.play([candidate])
        assert set(engine.valid()) == checks.keys()
        for action, check in checks.items():
            _, terminal = engine.play([candidate, action])
            tree = engine.tree()
            assert check == (hashlib.sha256(tree.encode()).hexdigest(), terminal, tree == reference), action
    finally:
        env.close()
    assert not any(equals for _, _, equals in checks.values())
    print(f"419 command states and terminal flags verified. JVM benchmark including command enumeration: {seconds}s.", flush=True)
    result = subprocess.check_output(run + ["search", "3000", "90"], text=True)
    lines = [ln.split("\t") for ln in result.splitlines() if ln.startswith(("SUMMARY\t", "MATCH\t", "MATCH2\t"))]
    summary = next(row for row in lines if row[0] == "SUMMARY")
    _, total2, checked2, listed2, second_checked, elapsed, incomplete = summary
    assert total2 == "419" and (incomplete == "true" or listed == listed2)
    receipt = {
        "verified_one_call_branches": len(checks), "jvm_validation_seconds_including_enumeration": float(seconds),
        "search": {"one_call_listed": int(total2), "one_call_checked": int(checked2),
                   "two_call_listed": int(listed2), "two_call_checked": int(second_checked),
                   "seconds": float(elapsed), "export_matches": [r[1:] for r in lines if r[0].startswith("MATCH")]},
        "scope": "export comparison diagnostic on freeze variation 4",
        "accepted_labels": 0,
    }
    (ROOT / "jvm_replay_progress.json").write_text(json.dumps(receipt, indent=2)+"\n")
    print(json.dumps(receipt, indent=2))
