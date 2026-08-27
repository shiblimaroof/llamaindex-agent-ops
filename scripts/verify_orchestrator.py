import json
from evalops.regression.orchestrator import run_regression_checks

worktree_path = "data/repo_cache/worktree/22068"

with open("data/raw_issues.jsonl") as f:
    for line in f:
        record = json.loads(line)
        if record["source_id"] == "22068":
            issue_body = record["body"]
            break

results = run_regression_checks(worktree_path, issue_body)

for key, value in results.items():
    print(f"{key}: {value}")