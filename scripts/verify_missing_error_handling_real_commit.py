"""
Finds real llama_index commits where a brand-new function was added
containing a risky call (network/file/subprocess/parsing) with no
try/except anywhere in its body -- candidates to verify
missing_error_handling_new_code against real data, same verification
bar as removed_raise was checked against commit 83a0deceb.

Usage:
    python3 verify_missing_error_handling_real_commit.py <repo_path> [--since N] [--limit M]

Strategy: walk recent commits' diffs, for each touched .py file compare
old/new source via check_error_handling_weakened, and print any commit
where missing_error_handling_new_code actually fires. Manual inspection
of the printed commit + finding still required -- this only narrows down
candidates, it doesn't replace eyeballing the real diff.
"""

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # adjust if needed

from evalops.regression.error_handling import check_error_handling_weakened


def _run_git(repo_path: str, args: list[str]) -> str:
    result = subprocess.run(["git", *args], cwd=repo_path, capture_output=True, text=True, timeout=30)
    if result.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed: {result.stderr.strip()}")
    return result.stdout


def _recent_commits(repo_path: str, limit: int) -> list[str]:
    out = _run_git(repo_path, ["log", f"-{limit}", "--pretty=%H"])
    return [line.strip() for line in out.splitlines() if line.strip()]


def _changed_py_files(repo_path: str, commit: str) -> list[str]:
    out = _run_git(repo_path, ["diff-tree", "--no-commit-id", "--name-only", "-r", commit])
    return [f for f in out.splitlines() if f.strip().endswith(".py")]


def _file_at_commit(repo_path: str, commit: str, file_path: str) -> str | None:
    try:
        return _run_git(repo_path, ["show", f"{commit}:{file_path}"])
    except RuntimeError:
        return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("repo_path")
    parser.add_argument("--limit", type=int, default=300, help="how many recent commits to scan")
    args = parser.parse_args()

    commits = _recent_commits(args.repo_path, args.limit)
    print(f"Scanning {len(commits)} commits in {args.repo_path}...")

    hits = 0
    for commit in commits:
        parent = f"{commit}^"
        try:
            changed_files = _changed_py_files(args.repo_path, commit)
        except RuntimeError:
            continue  # e.g. root commit with no parent

        for file_path in changed_files:
            if "/tests/" in file_path or file_path.split("/")[-1].startswith("test_"):
                continue

            old_source = _file_at_commit(args.repo_path, parent, file_path)
            if old_source is None:
                old_source = ""  # new file

            new_source = _file_at_commit(args.repo_path, commit, file_path)
            if new_source is None:
                continue  # deleted file

            try:
                diff_text = _run_git(args.repo_path, ["diff", parent, commit, "--", file_path])
            except RuntimeError:
                continue

            try:
                result = check_error_handling_weakened(old_source, new_source, diff_text, file_path)
            except SyntaxError:
                continue  # old or new side didn't parse cleanly, skip

            new_code_findings = [f for f in result["findings"] if f["kind"] == "missing_error_handling_new_code"]
            if new_code_findings:
                hits += 1
                print(f"\n--- HIT #{hits} ---")
                print(f"commit: {commit}")
                print(f"file:   {file_path}")
                for f in new_code_findings:
                    print(f"  {f['location']}: {f['detail']}")
                if hits >= 10:
                    print("\n(stopping after 10 hits -- enough candidates to inspect manually)")
                    return

    if hits == 0:
        print("\nNo hits found in scanned range. Try increasing --limit.")


if __name__ == "__main__":
    main()