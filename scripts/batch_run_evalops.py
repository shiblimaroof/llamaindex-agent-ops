"""
Batch runner: runs Step A + Step B + combiner across the golden set
(data/golden_set.jsonl, passes_golden_filter == true), writing one result
row per issue to data/batch_results.jsonl.

top_chunks is recomputed fresh per issue via chunk_repo + build_query +
retrieve (not stored anywhere from the original resolve run), the same
sequence the real pipeline uses.

On a per-issue failure, logs an error row and continues rather than
aborting the whole run -- one bad issue shouldn't cost the rest of the
results. A fixed 2s sleep between issues guards against OpenRouter's
free-tier rate limits.

"""

import argparse
import json
import time
import traceback
import uuid
import subprocess

from dotenv import load_dotenv
load_dotenv()

from evalops.regression.orchestrator import run_regression_checks
from evalops.judge.runner import run_judge
from evalops.regression.combiner import compute_regression_risk
from issue_worker.retrieval.chunker import chunk_repo
from issue_worker.retrieval.query_builder import build_query
from issue_worker.retrieval.retriever import retrieve

GOLDEN_SET_PATH = "data/golden_set.jsonl"
RAW_ISSUES_PATH = "data/raw_issues.jsonl"
OUTPUT_PATH = "data/batch_results.jsonl"
WORKTREE_ROOT = "data/repo_cache/worktree"
SLEEP_SECONDS = 2

def _worktree_has_changes(worktree_path: str) -> bool:
    """True if the worktree has any uncommitted changes (patch was applied)."""
    result = subprocess.run(
        ["git", "-C", worktree_path, "status", "--porcelain"],
        capture_output=True,
        text=True,
    )
    return bool(result.stdout.strip())


def _load_golden_set_ids() -> list[str]:
    ids = []
    with open(GOLDEN_SET_PATH) as f:
        for line in f:
            row = json.loads(line)
            if row.get("passes_golden_filter") is True:
                ids.append(row["source_id"])
    return ids


def _load_issues() -> dict[str, dict]:
    issues = {}
    with open(RAW_ISSUES_PATH) as f:
        for line in f:
            row = json.loads(line)
            issues[row["source_id"]] = row
    return issues


def _run_one_issue(source_id: str, issue: dict) -> dict:
    worktree_path = f"{WORKTREE_ROOT}/{source_id}"
    issue_body = issue["body"]
    run_id = str(uuid.uuid4())

    if not _worktree_has_changes(worktree_path):
        return {
            "source_id": source_id,
            "run_id": run_id,
            "error": "worktree has no uncommitted changes — no patch to judge",
            "stage": "no_patch_to_judge",
        }

    regression_results = run_regression_checks(worktree_path, issue_body)

    full_chunks = chunk_repo(worktree_path, source_id)
    query = build_query(issue, full_chunks)
    top_chunks = retrieve(query, full_chunks, top_k=5)

    judge_output = run_judge(
        worktree_path, issue_body, regression_results, top_chunks, source_id, run_id
    )
    risk = compute_regression_risk(regression_results, judge_output)

    return {
        "source_id": source_id,
        "run_id": run_id,
        "regression_results": regression_results,
        "judge_output": judge_output,
        "risk": risk,
    }


def main(limit: int | None = None):
    golden_ids = _load_golden_set_ids()
    if limit is not None:
        golden_ids = golden_ids[:limit]
    issues = _load_issues()

    print(f"Running batch over {len(golden_ids)} golden-set issues")

    with open(OUTPUT_PATH, "w") as out:
        for i, source_id in enumerate(golden_ids, start=1):
            print(f"[{i}/{len(golden_ids)}] {source_id}...", end=" ")

            issue = issues.get(source_id)
            if issue is None:
                error_row = {
                    "source_id": source_id,
                    "error": "no matching entry in raw_issues.jsonl",
                    "stage": "load_issue",
                }
                out.write(json.dumps(error_row) + "\n")
                print("FAILED (no issue found)")
                continue

            try:
                result = _run_one_issue(source_id, issue)
                out.write(json.dumps(result, default=str) + "\n")
            except Exception as e:
                error_row = {
                    "source_id": source_id,
                    "error": str(e),
                    "stage": type(e).__name__,
                    "traceback": traceback.format_exc(),
                }
                out.write(json.dumps(error_row) + "\n")
                print(f"FAILED ({type(e).__name__}: {e})")
                out.flush()
                time.sleep(SLEEP_SECONDS)
                continue

            if "risk" in result:
                print(f"OK (risk={result['risk']['introduces_regression_risk']})")
            else:
                print(f"SKIPPED ({result.get('stage', 'unknown')}: {result.get('error', '')})")

            out.flush()
            time.sleep(SLEEP_SECONDS)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="Only run the first N golden-set issues")
    args = parser.parse_args()
    main(limit=args.limit)