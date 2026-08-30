"""
scripts/verify_judge_runner.py

Smoke test for evalops/judge/runner.py -- confirms the full Step B chain
(prompt.py -> client.py -> schema.py) works end to end against a real case,
not synthetic data. Not a pytest file.

Uses the same real fixture already verified elsewhere: issue 22068's
worktree at data/repo_cache/worktree/22068, and its real issue body from
data/raw_issues.jsonl. regression_results comes from a real
run_regression_checks() call against that same worktree, not a stub --
run_regression_checks was already verified against this exact worktree/issue
body combination in regression/orchestrator.py's own verification pass.

top_chunks is recomputed fresh via chunk_repo + build_query + retrieve,
the same call sequence the real pipeline uses in orchestrator.py -- not
stored anywhere from the original resolve run, so it has to be rebuilt
here for context_faithfulness/reasoning_relevancy to have something to
judge against.
"""

import json
import uuid

from dotenv import load_dotenv
load_dotenv()

from evalops.regression.orchestrator import run_regression_checks
from evalops.judge.runner import run_judge
from issue_worker.retrieval.chunker import chunk_repo
from issue_worker.retrieval.query_builder import build_query
from issue_worker.retrieval.retriever import retrieve

SOURCE_ID = "22068"
WORKTREE_PATH = f"data/repo_cache/worktree/{SOURCE_ID}"


def _load_issue(source_id: str) -> dict:
    with open("data/raw_issues.jsonl") as f:
        for line in f:
            record = json.loads(line)
            if str(record.get("source_id")) == source_id:
                return record
    raise ValueError(f"source_id {source_id} not found in raw_issues.jsonl")


print(f"Loading issue for {SOURCE_ID}...")
issue = _load_issue(SOURCE_ID)
issue_body = issue["body"]

print("Running Step A regression checks...")
regression_results = run_regression_checks(WORKTREE_PATH, issue_body)
print("Step A results:")
print(json.dumps(regression_results, indent=2, default=str))

print("Rebuilding retrieved context (chunk_repo -> build_query -> retrieve)...")
full_chunks = chunk_repo(WORKTREE_PATH, SOURCE_ID)
query = build_query(issue, full_chunks)
top_chunks = retrieve(query, full_chunks, top_k=5)

run_id = str(uuid.uuid4())
print(f"\nRunning Step B judge (run_id={run_id})...")
judge_result = run_judge(
    worktree_path=WORKTREE_PATH,
    issue_body=issue_body,
    regression_results=regression_results,
    top_chunks=top_chunks,
    source_id=SOURCE_ID,
    run_id=run_id,
)

print("\nJudge result:")
print(json.dumps(judge_result, indent=2))