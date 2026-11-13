import json, hashlib, random
from pathlib import Path
from collections import Counter, defaultdict

def audit_and_split(source_dir, output_dir):
    source_dir, output_dir = Path(source_dir), Path(output_dir)
    dataset_path = source_dir / "development_dataset.jsonl"
    raw = dataset_path.read_bytes()
    records = [json.loads(line) for line in raw.decode().splitlines() if line.strip()]
    assert len(records) == 96, "This development audit expects the 96-record pilot."
    assert len({r["id"] for r in records}) == len(records), "Duplicate IDs"
    labels={"reversible","partially_reversible","irreversible"}
    assert all(r["label"] in labels and r["horizon"]==3 for r in records)
    assert all(r["split"]=="development_only" for r in records)
    groups=defaultdict(list)
    for row in records: groups[row["scenario_family"]].append(row)
    file_groups=sorted(g for g,rs in groups.items() if {r["environment"] for r in rs}=={"sandbox_recovery_pilot"})
    transaction_groups=sorted(g for g,rs in groups.items() if {r["environment"] for r in rs}=={"sandbox_transaction_pilot"})
    assert len(file_groups)==len(transaction_groups)==8
    # All three labels should exist under each exact candidate action string.
    by_action=defaultdict(set)
    for row in records: by_action[row["action_text"]].add(row["label"])
    assert all(classes==labels for classes in by_action.values())
    output_dir.mkdir(parents=True,exist_ok=False)
    summaries=[]
    for seed in (42,123,2026):
        rng=random.Random(seed)
        files=file_groups.copy(); transactions=transaction_groups.copy()
        rng.shuffle(files);rng.shuffle(transactions)
        # Whole groups make exact 20% impossible here. 21/96 is the nearest
        # attainable size to 20% with these group sizes. Final dataset splits
        # will be designed separately; none of these examples is a final test set.
        assignments={"development_eval":files[:1]+transactions[:2],
                     "validation":files[1:2]+transactions[2:4],
                     "train":files[2:]+transactions[4:]}
        split_rows={name:[r for group in names for r in groups[group]] for name,names in assignments.items()}
        assigned=[r["id"] for rows in split_rows.values() for r in rows]
        assert len(assigned)==len(set(assigned))==96
        name_sets={name:set(names) for name,names in assignments.items()}
        for a in name_sets:
            for b in name_sets:
                if a!=b: assert not name_sets[a]&name_sets[b]
        # Exact state/action duplicates must not cross boundaries either.
        owner={}
        pair_owner={}
        for split,rows in split_rows.items():
            for row in rows:
                signature=(row["state_text"],row["action_text"])
                assert signature not in owner or owner[signature]==split
                owner[signature]=split
                pair=row["minimal_pair_group"]
                assert pair not in pair_owner or pair_owner[pair]==split
                pair_owner[pair]=split
        manifest={"seed":seed,"dataset_sha256":hashlib.sha256(raw).hexdigest(),
                  "scope":"development only; not final research splits",
                  "groups":assignments,"ids":{s:[r["id"] for r in rows] for s,rows in split_rows.items()}}
        (output_dir/f"split_{seed}.json").write_text(json.dumps(manifest,indent=2))
        summaries.append({"seed":seed,"sizes":{s:len(rs) for s,rs in split_rows.items()},
                          "class_counts":{s:dict(Counter(r["label"] for r in rs)) for s,rs in split_rows.items()}})
    summary={"records":len(records),"scenario_groups":len(groups),"seeds":[42,123,2026],
             "group_leakage_detected":False,"minimal_pair_leakage_detected":False,
             "exact_duplicate_leakage_detected":False,"candidate_actions_have_all_three_labels":True,
             "development_eval_fraction":21/96,"final_research_test_set":False,
             "splits":summaries,"saved_to":str(output_dir)}
    (output_dir/"summary.json").write_text(json.dumps(summary,indent=2))
    return summary

from google.colab import drive
from datetime import datetime, timezone
drive.mount("/content/drive")
# Explicitly choose the dataset you just generated; never silently pick a newer one.
source_dir=Path("/content/drive/MyDrive/ADNN_Irreversibility/data/expanded_development_20261018T075007949936Z")
output_dir=source_dir/("split_audit_"+datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
summary=audit_and_split(source_dir,output_dir)
print("DEVELOPMENT SPLIT AUDIT PASSED")
print(json.dumps(summary,indent=2))
