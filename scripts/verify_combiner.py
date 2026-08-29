"""
Real end-to-end verification of regression/combiner.py against issue 22068.

Chains run_regression_checks() -> run_judge() -> compute_regression_risk(),
same pattern as scripts/verify_judge_runner.py: real worktree, real issue
body from data/raw_issues.jsonl, real judge call. Run via:

    python -m scripts.verify_combiner

(not `python scripts/verify_combiner.py`, which fails with
ModuleNotFoundError since it doesn't add repo root to sys.path)
"""

import json
import uuid

from dotenv import load_dotenv
load_dotenv()

from evalops.regression.orchestrator import run_regression_checks
from evalops.judge.runner import run_judge
from evalops.regression.combiner import compute_regression_risk

SOURCE_ID = "22068"
WORKTREE_PATH = f"data/repo_cache/worktree/{SOURCE_ID}"


def _load_issue_body(source_id: str) -> str:
    with open("data/raw_issues.jsonl") as f:
        for line in f:
            row = json.loads(line)
            if str(row.get("id")) == source_id or str(row.get("source_id")) == source_id:
                return row["body"]
    raise ValueError(f"issue {source_id} not found in data/raw_issues.jsonl")


def main():
    run_id = str(uuid.uuid4())
    issue_body = _load_issue_body(SOURCE_ID)

    regression_results = run_regression_checks(WORKTREE_PATH, issue_body)
    print("=== regression_results ===")
    print(json.dumps(regression_results, indent=2, default=str))

    judge_output = run_judge(
        WORKTREE_PATH, issue_body, regression_results, SOURCE_ID, run_id
    )
    print("\n=== judge_output ===")
    print(json.dumps(judge_output, indent=2, default=str))

    risk = compute_regression_risk(regression_results, judge_output)
    print("\n=== introduces_regression_risk ===")
    print(json.dumps(risk, indent=2))


if __name__ == "__main__":
    main()