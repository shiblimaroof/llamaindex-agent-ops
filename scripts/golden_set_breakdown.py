"""
golden_set_breakdown.py

For each issue in the golden set, fetch its linked fix PR diff and
classify which files it touches: llama-index-core vs
llama-index-integrations/*. Reports the split.
"""
import json
import re
import time
import urllib.request

GOLDEN_SET_PATH = "data/golden_set.jsonl"
GRADING_KEY_PATH = "data/grading_key.jsonl"


def load_golden_ids(path: str) -> list[str]:
    ids = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("passes_golden_filter") is True:
                ids.append(row["source_id"])
    return ids


def load_grading_key(path: str) -> dict[str, dict]:
    key = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            key[row["source_id"]] = row
    return key


def fetch_diff(url: str) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": "issue-worker-eval"})
    with urllib.request.urlopen(req, timeout=15) as resp:
        return resp.read().decode("utf-8", errors="replace")


def classify_diff(diff_text: str) -> str:
    """Return 'core', 'integration', 'mixed', or 'other' based on
    which top-level package the changed files belong to."""
    paths = re.findall(r"^diff --git a/(\S+) b/\S+", diff_text, re.MULTILINE)
    if not paths:
        return "unknown"

    touches_core = any(
        p.startswith("llama-index-core/") for p in paths
    )
    touches_integration = any(
        p.startswith("llama-index-integrations/") for p in paths
    )

    if touches_core and touches_integration:
        return "mixed"
    elif touches_core:
        return "core"
    elif touches_integration:
        return "integration"
    else:
        return "other"  # e.g. docs, root-level config, CI


def main():
    golden_ids = load_golden_ids(GOLDEN_SET_PATH)
    grading_key = load_grading_key(GRADING_KEY_PATH)

    print(f"Golden set size: {len(golden_ids)}")

    counts = {"core": 0, "integration": 0, "mixed": 0, "other": 0, "unknown": 0, "no_diff_url": 0}
    details = []

    for source_id in golden_ids:
        row = grading_key.get(source_id)
        if not row or not row.get("linked_fix_diff_url"):
            counts["no_diff_url"] += 1
            details.append((source_id, "no_diff_url"))
            continue

        url = row["linked_fix_diff_url"]
        try:
            diff_text = fetch_diff(url)
        except Exception as e:
            print(f"  [{source_id}] fetch failed: {e}")
            counts["unknown"] += 1
            details.append((source_id, "fetch_failed"))
            continue

        category = classify_diff(diff_text)
        counts[category] += 1
        details.append((source_id, category))

        time.sleep(0.3)  # be polite to GitHub, avoid rate limiting

    print("\n--- Breakdown ---")
    for k, v in counts.items():
        pct = (v / len(golden_ids) * 100) if golden_ids else 0
        print(f"{k:12s}: {v:3d}  ({pct:.1f}%)")

    print("\n--- Per-issue detail ---")
    for source_id, category in details:
        print(f"{source_id:>8s}  {category}")


if __name__ == "__main__":
    main()