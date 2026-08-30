"""
Compares judge output (data/batch_results.jsonl) against human labels
(data/validation_labels.jsonl) to check whether the judge's verdicts
match what a human reading the same issue + patch would conclude.

Not statistical at this scale (n=5-10) -- raw per-field agreement counts
and full mismatch detail, not confidence intervals or kappa scores.
Deliberately excludes unexplained_concern from comparison: it's an
open-ended catch-all, not a field a human can independently verify the
same way as a yes/no correctness or grounding question.

Appends a summary row to data/validation_runs.jsonl each run, so
agreement rates can be tracked over time as more cases get labeled and
the judge/prompt evolves.

Run via: python -m scripts.validate_judge
"""

import json
from datetime import datetime, timezone

BATCH_RESULTS_PATH = "data/batch_results.jsonl"
LABELS_PATH = "data/validation_labels.jsonl"
RUNS_LOG_PATH = "data/validation_runs.jsonl"

COMPARED_FIELDS = [
    "resolves_issue",
    "introduces_regression_risk",
    "context_faithfulness",
    "reasoning_relevancy",
]


def _load_batch_results() -> dict[str, dict]:
    results = {}
    with open(BATCH_RESULTS_PATH) as f:
        for line in f:
            row = json.loads(line)
            if "error" in row:
                continue  # skip failed runs, nothing to validate
            results[row["source_id"]] = row
    return results


def _load_labels() -> dict[str, dict]:
    labels = {}
    with open(LABELS_PATH) as f:
        for line in f:
            row = json.loads(line)
            labels[row["source_id"]] = row
    return labels


def _judge_value(result: dict, field: str):
    if field == "introduces_regression_risk":
        return result["risk"]["introduces_regression_risk"]
    return result["judge_output"].get(field)


def main():
    results = _load_batch_results()
    labels = _load_labels()

    result_ids = set(results)
    label_ids = set(labels)

    unlabeled = result_ids - label_ids
    orphan_labels = label_ids - result_ids
    if unlabeled:
        print(f"WARNING: {len(unlabeled)} result(s) have no label, skipped: {sorted(unlabeled)}")
    if orphan_labels:
        print(f"WARNING: {len(orphan_labels)} label(s) have no matching result, skipped: {sorted(orphan_labels)}")

    compared_ids = sorted(result_ids & label_ids)
    if not compared_ids:
        print("No overlapping cases between results and labels. Nothing to compare.")
        return

    print(f"\nComparing {len(compared_ids)} case(s): {compared_ids}\n")

    field_matches = {field: 0 for field in COMPARED_FIELDS}
    mismatches = []

    for source_id in compared_ids:
        result = results[source_id]
        label = labels[source_id]

        for field in COMPARED_FIELDS:
            judge_val = _judge_value(result, field)
            human_val = label.get(field)
            match = judge_val == human_val
            if match:
                field_matches[field] += 1
            else:
                mismatches.append({
                    "source_id": source_id,
                    "field": field,
                    "judge": judge_val,
                    "human": human_val,
                    "label_note": label.get("note", ""),
                })

    print("=== Per-field agreement ===")
    for field in COMPARED_FIELDS:
        n = field_matches[field]
        total = len(compared_ids)
        print(f"{field}: {n}/{total}")

    if mismatches:
        print("\n=== Mismatches ===")
        for m in mismatches:
            print(f"[{m['source_id']}] {m['field']}: judge={m['judge']!r} human={m['human']!r}"
                  + (f" (note: {m['label_note']})" if m['label_note'] else ""))
    else:
        print("\nNo mismatches.")

    summary_row = {
        "date": datetime.now(timezone.utc).isoformat(),
        "n_cases": len(compared_ids),
        "agreement": {field: f"{field_matches[field]}/{len(compared_ids)}" for field in COMPARED_FIELDS},
        "n_mismatches": len(mismatches),
    }
    with open(RUNS_LOG_PATH, "a") as f:
        f.write(json.dumps(summary_row) + "\n")

    print(f"\nSummary appended to {RUNS_LOG_PATH}")


if __name__ == "__main__":
    main()