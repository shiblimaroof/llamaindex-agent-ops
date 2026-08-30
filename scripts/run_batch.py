import argparse
import json
from issue_worker.orchestrator import run_pipeline

GOLDEN_SET_PATH = "data/golden_set.jsonl"
OUTPUT_PATH = "data/run_batch_results.jsonl"

def run_batch(limit: int | None = None) -> list[dict]:
    results = []
    with open(GOLDEN_SET_PATH) as f:
        entries = [json.loads(line) for line in f if json.loads(line).get("passes_golden_filter")]

    if limit is not None:
        entries = entries[:limit]

    with open(OUTPUT_PATH, "w") as out:
        for i, entry in enumerate(entries, start=1):
            source_id = entry["source_id"]
            print(f"[{i}/{len(entries)}] {source_id}...", end=" ")
            try:
                result = run_pipeline(source_id)
                row = {"source_id": source_id, "result": result}
                print("OK")
            except Exception as e:
                row = {"source_id": source_id, "error": str(e)}
                print(f"FAILED ({type(e).__name__}: {e})")

            results.append(row)
            out.write(json.dumps(row, default=str) + "\n")
            out.flush()

    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="Only run the first N golden-set issues (for a dry run)")
    args = parser.parse_args()

    results = run_batch(limit=args.limit)
    print(f"\nDone. {len(results)} results written to {OUTPUT_PATH}")