import json
from issue_worker.orchestrator import run_pipeline

GOLDEN_SET_PATH = "data/golden_set.jsonl"

def run_batch() -> list[dict]:
    results = []
    with open(GOLDEN_SET_PATH) as f:
        for line in f:
            entry = json.loads(line)
            if not entry.get("passes_golden_filter"):
                continue
            source_id = entry["source_id"]
            try:
                result = run_pipeline(source_id)
                results.append({"source_id": source_id, "result": result})
            except Exception as e:
                results.append({"source_id": source_id, "error": str(e)})
    return results

if __name__ == "__main__":
    results = run_batch()
    print(results)