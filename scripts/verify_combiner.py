"""
Real end-to-end verification of regression/combiner.py against issue 22068.

Chains run_regression_checks() -> run_judge() -> compute_regression_risk(),
same pattern as scripts/verify_judge_runner.py: real worktree, real issue
body from data/raw_issues.jsonl, real judge call. top_chunks is recomputed
fresh via chunk_repo + build_query + retrieve, since it isn't stored
anywhere from the original resolve run. Run via:

    python -m scripts.verify_combiner
"""

import json
import uuid

from dotenv import load_dotenv
load_dotenv()

from evalops.regression.orchestrator import run_regression_checks
from evalops.judge.runner import run_judge
from evalops.regression.combiner import compute_regression_risk
from issue_worker.retrieval.chunker import chunk_repo
from issue_worker.retrieval.query_builder import build_query
from issue_worker.retrieval.retriever import retrieve

SOURCE_ID = "22068"
WORKTREE_PATH = f"data/repo_cache/worktree/{SOURCE_ID}"


def _load_issue(source_id: str) -> dict:
    with open("data/raw_issues.jsonl") as f:
        for line in f:
            row = json.loads(line)
            if str(row.get("id")) == source_id or str(row.get("source_id")) == source_id:
                return row
    raise ValueError(f"issue {source_id} not found in data/raw_issues.jsonl")


def main():
    run_id = str(uuid.uuid4())
    issue = _load_issue(SOURCE_ID)
    issue_body = issue["body"]

    regression_results = run_regression_checks(WORKTREE_PATH, issue_body)
    print("=== regression_results ===")
    print(json.dumps(regression_results, indent=2, default=str))

    full_chunks = chunk_repo(WORKTREE_PATH, SOURCE_ID)
    query = build_query(issue, full_chunks)
    top_chunks = retrieve(query, full_chunks, top_k=5)

    judge_output = run_judge(
        WORKTREE_PATH, issue_body, regression_results, top_chunks, SOURCE_ID, run_id
    )
    print("\n=== judge_output ===")
    print(json.dumps(judge_output, indent=2, default=str))

    risk = compute_regression_risk(regression_results, judge_output)
    print("\n=== introduces_regression_risk ===")
    print(json.dumps(risk, indent=2))


if __name__ == "__main__":
    main()