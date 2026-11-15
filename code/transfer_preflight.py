"""Report class support, group support and overlap for proposed transfer datasets."""
import argparse
from collections import Counter
import json
from pathlib import Path

LABELS = ("reversible", "partially_reversible", "irreversible")


def read(path):
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if any(row["horizon"] != 3 or row["label"] not in LABELS for row in rows):
        raise ValueError(f"Unrecognized label or horizon: {path}")
    signatures = {(r["state_text"], r["action_text"]) for r in rows}
    return rows, {
        "rows": len(rows), "unique_state_action_inputs": len(signatures),
        "scenario_families": len({r["scenario_family"] for r in rows}),
        "counts": {label: sum(r["label"] == label for r in rows) for label in LABELS},
        "families_per_label": {
            label: len({r["scenario_family"] for r in rows if r["label"] == label}) for label in LABELS
        },
    }


def audit_split(path, rows):
    by_id = {r["id"]: r for r in rows}
    groups = json.loads(path.read_text())["ids"]
    assert set(groups) == {"train", "validation", "test"}
    ids = [i for part in groups.values() for i in part]
    assert len(ids) == len(set(ids)) == len(by_id)
    ownership = {}
    for part, indices in groups.items():
        for i in indices:
            family = by_id[i]["scenario_family"]
            assert ownership.setdefault(family, part) == part
    return {part: dict(Counter(by_id[i]["label"] for i in indices)) for part, indices in groups.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sandbox", type=Path, required=True)
    parser.add_argument("--scienceworld", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sandbox_rows, sandbox = read(args.sandbox)
    _, scienceworld = read(args.scienceworld)
    outcome = {
        "sandbox": sandbox, "scienceworld": scienceworld,
        "sandbox_existing_grouped_splits": {
            p.stem: audit_split(p, sandbox_rows)
            for p in sorted((args.sandbox.parent / "private").glob("split_*.json"))
        },
        "mandatory_direction_requirements": {
            "sandbox_to_scienceworld": "All evaluated classes must be represented in the target for comparable macro scores.",
            "scienceworld_to_sandbox": "At least two classes are needed for fitting, and disjoint labelled source groups are needed to validate settings.",
        },
        "current_binary_exact_vs_irreversible_possible": all(scienceworld["counts"][label]
                                                           for label in ("reversible", "irreversible")),
        "current_three_class_possible": all(scienceworld["counts"].values()),
        "current_grouped_scienceworld_train_validation_test_possible": scienceworld["scenario_families"] >= 3,
    }
    args.output.write_text(json.dumps(outcome, indent=2) + "\n")
    print(json.dumps(outcome, indent=2))


if __name__ == "__main__":
    main()
