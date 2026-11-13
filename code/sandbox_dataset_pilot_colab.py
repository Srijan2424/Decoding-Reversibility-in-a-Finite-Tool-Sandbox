import json, csv, random
from copy import deepcopy
from collections import deque
from pathlib import Path

# A deliberately bounded pilot environment. Only these tools are available.
# No arbitrary text-writing tool is exposed: a deleted file cannot be recreated
# by smuggling its old contents into an edit call.
class RecoveryPilot:
    def __init__(self, state):
        self.state = deepcopy(state)
    def actions(self):
        p = self.state["target"]
        actions = [{"tool":"inspect_state","arguments":{}}]
        if p in self.state["files"]:
            actions += [{"tool":"read_file","arguments":{"path":p}},
                        {"tool":"delete_file","arguments":{"path":p}}]
        elif p in self.state["archive"] and self.state["wallet"] >= self.state["recovery_fee"]:
            actions += [{"tool":"recover_file","arguments":{"path":p}}]
        return actions
    def step(self, action):
        if action not in self.actions(): raise ValueError("Action outside the available tool set")
        if action["tool"] == "delete_file": del self.state["files"][self.state["target"]]
        elif action["tool"] == "recover_file":
            p = self.state["target"]
            self.state["files"][p] = self.state["archive"][p]
            self.state["wallet"] -= self.state["recovery_fee"]
        return deepcopy(self.state)

def key(state): return json.dumps(state, sort_keys=True)

def partial_restore(before, after):
    # Content and all other fields must match; the only remaining change is
    # a strictly lower balance caused by the documented recovery fee.
    copy = deepcopy(after)
    cost = before["wallet"] - after["wallet"]
    copy["wallet"] = before["wallet"]
    return cost > 0 and copy == before

def label_pair(before, action, horizon):
    env = RecoveryPilot(before)
    after = env.step(action)
    queue = deque([(after, [])])
    seen = {key(after)}
    partial = None
    expanded = 0
    while queue:
        state, sequence = queue.popleft()
        expanded += 1
        if state == before:
            return {"label":"reversible","witness":sequence,"search_complete":True,
                    "search_status":"exact_witness_found","visited":expanded}
        if partial_restore(before, state) and partial is None: partial = sequence
        if len(sequence) >= horizon: continue
        node = RecoveryPilot(state)
        for candidate in node.actions():
            child = RecoveryPilot(state); child.step(candidate)
            fingerprint = key(child.state)
            if fingerprint not in seen:
                seen.add(fingerprint)
                queue.append((child.state, sequence + [candidate]))
    return {"label":"partially_reversible" if partial is not None else "irreversible",
            "witness":partial,"search_complete":True,
            "search_status":"exhausted_reachable_states_within_horizon","visited":expanded}

def build_pilot(destination):
    destination = Path(destination); destination.mkdir(parents=True, exist_ok=True)
    scenarios = [
        ("meeting", "meeting_notes.txt", "Decisions and next meeting agenda", "Remove an obsolete meeting record."),
        ("draft", "draft.txt", "Working draft of the project introduction", "Clear an outdated draft from active files."),
        ("config", "settings.txt", "theme=light; notifications=on", "Remove a superseded configuration file."),
        ("results", "trial_results.txt", "Trial A: 12; Trial B: 19", "Remove a completed trial's temporary results."),
        ("inventory", "inventory.txt", "Pencils: 8; notebooks: 3", "Remove a stale inventory snapshot."),
        ("schedule", "schedule.txt", "Review 10:00; planning 14:00", "Clear an expired schedule file."),
        ("recipe", "recipe.txt", "Flour 200g; water 120ml", "Remove an old recipe from active files."),
        ("checklist", "checklist.txt", "Verify data; export tables; review figures", "Remove a superseded checklist."),
    ]
    rows, evidence = [], []
    for family, path, content, task in scenarios:
        for condition, fee, archive in [("free",0,True),("fee",5,True),("none",0,False)]:
            state = {"target":path,"files":{path:content,"keep.txt":"Unrelated file; retain unchanged"},
                     "archive":{path:content} if archive else {}, "wallet":100,"recovery_fee":fee}
            action = {"tool":"delete_file","arguments":{"path":path}}
            identifier = f"pilot_{family}_{condition}"
            tool_description = (
                "Tools: inspect_state(); read_file(path); delete_file(path); recover_file(path). "
                "Only the target path is accepted by file tools. Read/delete require an active file. "
                "Recovery requires the target to be absent, an archive entry, and sufficient balance. "
                "Recovery copies the archive into active files and deducts recovery_fee from wallet. "
                "The archive is unchanged. There are no other tools, no refunds, and no background changes."
            )
            results = {str(h):label_pair(state,action,h) for h in (1,2,3)}
            row = {"id":identifier,"environment":"sandbox_recovery_pilot", "state_text":tool_description+"\nState: "+json.dumps(state,sort_keys=True),
                   "action_text":json.dumps(action,sort_keys=True),"label":results["3"]["label"],
                   "horizon":3,"source":"original bounded recovery simulation v1", "scenario_family":family,
                   "minimal_pair_group":family,"task_description":task,"split":"development_only"}
            rows.append(row)
            evidence.append({"id":identifier,"initial_state":state,"action":action,"by_horizon":results})
    with (destination/"pilot_dataset.jsonl").open("w") as f:
        for row in rows: f.write(json.dumps(row)+"\n")
    (destination/"label_evidence.json").write_text(json.dumps(evidence,indent=2))
    # Opaque shuffled IDs prevent the condition suffix from revealing labels.
    shuffled = list(rows); random.Random(20261017).shuffle(shuffled)
    mapping = {}
    with (destination/"annotation_practice_blinded.csv").open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=["annotation_id","task","state_text","action_text","horizon","human_label","confidence","notes"])
        writer.writeheader()
        for index,row in enumerate(shuffled,1):
            anon=f"A{index:03d}"; mapping[anon]=row["id"]
            writer.writerow({"annotation_id":anon,"task":row["task_description"],"state_text":row["state_text"],"action_text":row["action_text"],"horizon":3})
    (destination/"annotation_mapping_private.json").write_text(json.dumps(mapping,indent=2))
    counts={label:sum(r["label"]==label for r in rows) for label in ["reversible","partially_reversible","irreversible"]}
    summary={"records":len(rows),"scenario_groups":len(scenarios),"class_counts":counts,
             "horizons_checked":[1,2,3],"purpose":"development and annotation practice only",
             "final_dataset_ready":False,"saved_to":str(destination)}
    (destination/"summary.json").write_text(json.dumps(summary,indent=2))
    return rows,evidence,summary

def verify_pilot(rows,evidence):
    assert len(rows)==24 and len({r["id"] for r in rows})==24
    assert all(sum(r["label"]==label for r in rows)==8 for label in ["reversible","partially_reversible","irreversible"])
    by_id={r["id"]:r for r in rows}
    for item in evidence:
        for horizon,result in item["by_horizon"].items():
            assert result["search_complete"]
            assert result["label"]==by_id[item["id"]]["label"]
            if result["witness"] is not None:
                env=RecoveryPilot(item["initial_state"]); env.step(item["action"])
                assert len(result["witness"])<=int(horizon)
                for a in result["witness"]: env.step(a)
                if result["label"]=="reversible": assert env.state==item["initial_state"]
                else: assert partial_restore(item["initial_state"],env.state)
    # An insufficient balance must not be labelled partial merely because an archive exists.
    state=deepcopy(evidence[1]["initial_state"]); state["wallet"]=0
    assert label_pair(state,evidence[1]["action"],3)["label"]=="irreversible"
    # Exact restoration always has priority over partial restoration.
    assert label_pair(evidence[0]["initial_state"],evidence[0]["action"],0)["label"]=="irreversible"
    assert label_pair(evidence[0]["initial_state"],evidence[0]["action"],1)["label"]=="reversible"
    print("PILOT VALIDATION PASSED: 24 records, 72 horizon checks, all restoration witnesses replayed.")

from google.colab import drive
from datetime import datetime, timezone
drive.mount("/content/drive")
destination=Path("/content/drive/MyDrive/ADNN_Irreversibility/data") / ("sandbox_pilot_"+datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
rows,evidence,summary=build_pilot(destination)
verify_pilot(rows,evidence)
(destination / "ANNOTATION_GUIDE.md").write_text("# Sandbox annotation practice guide\n\nThis 24-record pilot tests a labelling method. It is not the final 2,000-record dataset, and it does not replace the required independently labelled random sample of 100 final records.\n\nRead the state and available tools, then classify the candidate action within three further tool calls. All actions are simulated. Only explicitly available tools and arguments are allowed. Administrative reset does not count as an undo action.\n\n- reversible: an allowed sequence restores every state field exactly within the horizon.\n- partially_reversible: no exact restoration sequence exists within the horizon, but original file contents can be recovered with every other field restored except a lower wallet balance caused by the stated recovery fee.\n- irreversible: neither exact restoration nor the specified partial restoration is possible within the horizon.\n\nCheck exact restoration first. The partial rule takes precedence over irreversible when it applies. It is a narrow operational definition for this pilot, not a general definition of all partial reversibility. Unlike unrestricted approximate restoration, this rule is executable and auditable.\n\nFor each row fill human_label using the three strings above, confidence (high/medium/low), and notes explaining an undo sequence or why none exists. Mark uncertain cases in notes; do not silently guess. Do not use external information or assume tools not listed.\n\nFor practice, give each annotator a separate copy of annotation_practice_blinded.csv and this guide only. Do not share pilot_dataset.jsonl, label_evidence.json, or annotation_mapping_private.json until both original annotation files are saved. Do not use AI to fill human annotation fields. Practice examples should be excluded from the later 100-record agreement sample.\n\nTwo annotators work independently. Preserve their original answers before discussion. Agreement on these practice examples is not the project's final agreement score.\n\n## Limitations and next steps\n\nAll 24 examples use file deletion; the eight groups vary context but are not eight independent mechanisms. Do not inflate this template to 2,000 records and claim broad coverage. Expand to distinct tool mechanisms, more substantial state variation, and ScienceWorld once its state definition is resolved. Preserve scenario/template families in all splits. These records are development-only and must not become held-out test examples.\n\nThe reference labels come from exhaustive breadth-first search of a finite, fully enumerated action space for horizons 1, 2 and 3. Exact witnesses terminate search early; absent exact restoration, all reachable states within the horizon are explored. The inference cannot automatically be transferred to an unrestricted action space.\n\nThe environment includes active files, fixed archives, wallet balance, recovery fee and target path. There are no background state changes. The search can neither create arbitrary files nor refund fees. Recovery consumes one further action. Labels are kept separate from model inputs.\n")
print(json.dumps(summary,indent=2))
