"""
scripts/verify_error_handling.py

Ad hoc verification script for check_error_handling_weakened, same pattern
used to spot-verify the other regression-tier mechanical checks against
data/repo_cache/llama_index. Not part of the shipped evalops package --
scratch/dev-only.

Usage:
    python3 scripts/verify_error_handling.py <old_commit> <new_commit> <file_path>

Example:
    python3 scripts/verify_error_handling.py abc123 def456 llama_index/core/config.py
"""

import json
import subprocess
import sys
from pathlib import Path

# Adjust if your repo root / package layout differs
REPO_ROOT = Path(__file__).resolve().parent.parent
LLAMA_INDEX_REPO = REPO_ROOT / "data" / "repo_cache" / "llama_index"

sys.path.insert(0, str(REPO_ROOT))
from evalops.regression.error_handling import check_error_handling_weakened


def git_show(repo: Path, commit: str, file_path: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), "show", f"{commit}:{file_path}"],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"git show failed for {commit}:{file_path}\n{result.stderr}")
    return result.stdout


def git_diff(repo: Path, old_commit: str, new_commit: str, file_path: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), "diff", old_commit, new_commit, "--", file_path],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"git diff failed\n{result.stderr}")
    return result.stdout


def main():
    if len(sys.argv) != 4:
        print(__doc__)
        sys.exit(1)

    old_commit, new_commit, file_path = sys.argv[1], sys.argv[2], sys.argv[3]

    old_source = git_show(LLAMA_INDEX_REPO, old_commit, file_path)
    new_source = git_show(LLAMA_INDEX_REPO, new_commit, file_path)
    diff_text = git_diff(LLAMA_INDEX_REPO, old_commit, new_commit, file_path)

    result = check_error_handling_weakened(old_source, new_source, diff_text, file_path)

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()