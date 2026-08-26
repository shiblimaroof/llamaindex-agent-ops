"""
Tests for regression/error_handling_runner.py.

Uses a real temp git repo (init + commit + modify), not mocks — same
verification style as error_handling.py's own fixes, which were checked
against a real llama_index commit rather than synthetic data alone. Git
behavior (ref resolution, diff formatting, new/deleted file handling) is
exactly what's risky to get wrong here, so exercising real git catches
what a mocked _run_git would hide.
"""

import subprocess
from pathlib import Path

import pytest

from evalops.regression.error_handling_runner import run_error_handling_check


def _git(worktree_path: str, *args: str) -> None:
    subprocess.run(["git", *args], cwd=worktree_path, check=True, capture_output=True)


def _init_repo(tmp_path: Path) -> str:
    worktree_path = str(tmp_path)
    _git(worktree_path, "init", "-q")
    _git(worktree_path, "config", "user.email", "test@test.com")
    _git(worktree_path, "config", "user.name", "test")
    return worktree_path


def _write_and_commit(worktree_path: str, file_path: str, content: str, message: str) -> None:
    full_path = Path(worktree_path) / file_path
    full_path.parent.mkdir(parents=True, exist_ok=True)
    full_path.write_text(content)
    _git(worktree_path, "add", file_path)
    _git(worktree_path, "commit", "-q", "-m", message)


# Fixtures ---------------------------------------------------------------------------


@pytest.fixture
def repo(tmp_path: Path) -> str:
    return _init_repo(tmp_path)


# Tests --------------------------------------------------------------------------------


def test_no_changes_no_signal(repo: str) -> None:
    """Committed once, nothing changed since — no findings, no signal."""
    _write_and_commit(repo, "handler.py", "def foo():\n    return 1\n", "initial")

    result = run_error_handling_check(repo, base_ref="HEAD")

    assert result == {"signal": False, "findings": []}


def test_removed_try_block_detected(repo: str) -> None:
    """A real regression (whole try/except removed) surfaces as a finding."""
    old = (
        "def foo():\n"
        "    try:\n"
        "        risky()\n"
        "    except ValueError:\n"
        "        log.error('failed')\n"
        "        raise\n"
    )
    _write_and_commit(repo, "handler.py", old, "initial")

    new = "def foo():\n    risky()\n"
    Path(repo, "handler.py").write_text(new)

    result = run_error_handling_check(repo, base_ref="HEAD")

    assert result["signal"] is True
    kinds = {f["kind"] for f in result["findings"]}
    assert "removed_try_block" in kinds


def test_new_file_with_risky_unguarded_call(repo: str) -> None:
    """New file at this ref — old_source resolves to "", so the function
    has no old-side match and routes through missing_error_handling_new_code
    instead of being skipped. This is the empty-old_source path flagged as
    worth testing when the runner was written."""
    _write_and_commit(repo, "existing.py", "def unrelated():\n    pass\n", "initial")

    new_file = (
        "def fetch_data():\n"
        "    resp = requests.get('http://example.com')\n"
        "    return resp.json()\n"
    )
    Path(repo, "new_module.py").write_text(new_file)
    _git(repo, "add", "new_module.py")
    _git(repo, "commit", "-q", "-m", "add new_module")

    result = run_error_handling_check(repo, base_ref="HEAD~1")

    assert result["signal"] is True
    kinds = {f["kind"] for f in result["findings"]}
    assert "missing_error_handling_new_code" in kinds


def test_deleted_file_skipped_not_crashed(repo: str) -> None:
    """A file present at base_ref but deleted on disk now must be skipped
    cleanly, not raise (new_path.exists() check in the runner)."""
    _write_and_commit(repo, "gone.py", "def foo():\n    pass\n", "initial")

    Path(repo, "gone.py").unlink()
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "delete gone.py")

    result = run_error_handling_check(repo, base_ref="HEAD~1")

    assert result == {"signal": False, "findings": []}


def test_test_files_excluded(repo: str) -> None:
    """A change under tests/ that would otherwise trigger a finding is
    filtered out entirely before check_error_handling_weakened ever runs."""
    _write_and_commit(repo, "tests/test_thing.py", "def helper():\n    pass\n", "initial")

    new_content = (
        "def helper():\n"
        "    resp = requests.get('http://example.com')\n"
        "    return resp\n"
    )
    Path(repo, "tests/test_thing.py").write_text(new_content)

    result = run_error_handling_check(repo, base_ref="HEAD")

    assert result == {"signal": False, "findings": []}


def test_multiple_files_findings_merged(repo: str) -> None:
    """Two separate files each contribute a finding — confirms the loop
    merges across files into one flat list rather than only keeping the
    last file's result."""
    old_a = "def a():\n    try:\n        risky()\n    except ValueError:\n        raise\n"
    old_b = "def b():\n    try:\n        risky()\n    except TypeError:\n        raise\n"
    # Both files committed together so both exist at the same base_ref —
    # two separate commits would mean HEAD~1 predates b.py's own commit,
    # making b.py look like a new file instead of a modified one.
    Path(repo, "a.py").write_text(old_a)
    Path(repo, "b.py").write_text(old_b)
    _git(repo, "add", "a.py", "b.py")
    _git(repo, "commit", "-q", "-m", "initial a and b")

    Path(repo, "a.py").write_text("def a():\n    risky()\n")
    Path(repo, "b.py").write_text("def b():\n    risky()\n")

    result = run_error_handling_check(repo, base_ref="HEAD")

    assert result["signal"] is True
    locations = [f["location"] for f in result["findings"]]
    assert any(loc.startswith("a.py:") for loc in locations)
    assert any(loc.startswith("b.py:") for loc in locations)