from __future__ import annotations
import subprocess
from pathlib import Path

REPO_URL = "https://github.com/run-llama/llama_index.git"

def _run_git(args : list[str], cwd :str | None = None) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd = cwd,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"git {' '.join(args)} failed:\n{result.stderr.strip()}"
        )
    return result.stdout.strip()

def ensure_base_clone(cache_dir : str="data/repo_cache") -> str:
    repo_path = str(Path(cache_dir)/"llama_index")

    if Path(repo_path, ".git").exists():
        return repo_path
    
    Path(cache_dir).mkdir(parents=True , exist_ok=True)
    _run_git(["clone", REPO_URL, repo_path])
    return repo_path

def get_commit_before(repo_path : str, before_date : str, branch: str = "main")->str:

    commit_hash = _run_git(
        [
            "log",
            branch,
            f"--before={before_date}",
            "-1",
            "--format=%H",
        ],
        cwd = repo_path,
    )

    if not commit_hash:
        raise RuntimeError(
            f"No commit found on {branch} before {before_date}"
            "the repo history may not reach back this far, or the branch"
            "name is wrong"
        )
    return commit_hash

def checkout_worktree(
        repo_path : str,
        commit_hash : str,
        source_id : str,
        worktree_dir : str = "data/repo_cache/worktree",
    ) -> str:

    worktree_path = str(Path(worktree_dir).resolve() / source_id)

    if Path(worktree_path).exists():
        return worktree_path
    
    Path(worktree_dir).mkdir(parents=True, exist_ok=True)
    _run_git(["worktree", "add" , "--detach", worktree_path, commit_hash], cwd=repo_path)
    return worktree_path

def get_repo_at_commit(
        source_id : str,
        created_at : str,
        cache_dir : str = "data/repo_cache",
    ) -> str:
    
    repo_path = ensure_base_clone(cache_dir)
    commit_hash = get_commit_before(repo_path , created_at)
    return checkout_worktree(repo_path, commit_hash, source_id)

if __name__ == "__main__":
    import json
    with open("data/raw_issues.jsonl", "r") as f:
        first_issue = json.loads(f.readline())

    path = get_repo_at_commit(
        source_id = first_issue["source_id"],
        created_at=first_issue["created_at"],
    )
    print(f"source_id: {first_issue['source_id']}")
    print(f"created_at: {first_issue['created_at']}")
    print(f"checked out at: {path}")
    print(_run_git(["log", "-1", "--format=%H %ad", "--date=iso"], cwd=path))