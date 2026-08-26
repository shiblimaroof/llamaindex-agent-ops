"""
Step A orchestration for the removes_weakens_error_handling hybrid check.

Wires check_error_handling_weakened (regression/error_handling.py) into the
worktree-based world regression/mechanical.py already operates in:
worktree_path + base_ref, no old_sha/new_sha. Diff text is fetched per-file
directly via git (git diff base_ref -- file_path), same pattern as
mechanical.py's dependency_changed and _added_line_ranges — this avoids
needing to split one combined multi-file diff, since each git call is
already scoped to a single file.

Reuses _changed_python_files, _file_at_ref, and _run_git from
regression/mechanical.py instead of re-deriving worktree access.
"""

from pathlib import Path

from evalops.regression.mechanical import (
    _changed_python_files,
    _file_at_ref,
    _run_git,
)
from evalops.regression.error_handling import (
    _is_test_file,
    check_error_handling_weakened,
)


def _file_diff(worktree_path: str, base_ref: str, file_path: str) -> str:
    """Diff text scoped to one file — no multi-file hunk-header ambiguity."""
    return _run_git(worktree_path, ["diff", base_ref, "--", file_path])


def run_error_handling_check(worktree_path: str, base_ref: str = "HEAD") -> dict:
    """Runs check_error_handling_weakened over every changed, non-test
    Python file and merges results into one {"signal": bool, "findings": [...]}.

    signal is True if any file produced a finding. findings is the flat
    concatenation across files. Known limitation carried over unchanged:
    locations stay func_name:Lxxx, not file-qualified — see the ambiguity
    note in error_handling.py's docstring. Tracked as a separate follow-up,
    not fixed here.
    """
    changed_files = [
        f for f in _changed_python_files(worktree_path, base_ref)
        if not _is_test_file(f)
    ]

    all_findings = []
    for file_path in changed_files:
        old_source = _file_at_ref(worktree_path, base_ref, file_path)
        if old_source is None:
            # New file at this ref (not deleted — that's caught below).
            # Empty old_source means every function in it has no old-side
            # match, correctly routing through the missing_error_handling_
            # new_code branch instead of being skipped.
            old_source = ""

        new_path = Path(worktree_path) / file_path
        if not new_path.exists():
            continue  # deleted file, nothing to check

        new_source = new_path.read_text()
        diff_text = _file_diff(worktree_path, base_ref, file_path)

        result = check_error_handling_weakened(old_source, new_source, diff_text, file_path)
        all_findings.extend(result["findings"])

    return {
        "signal": len(all_findings) > 0,
        "findings": all_findings,
    }





