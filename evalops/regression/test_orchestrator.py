"""
Tests for regression/orchestrator.py.

Unlike the other regression test files, this one uses monkeypatched stand-ins
for the eight check functions instead of a real git repo. The other test
files (test_mechanical.py, test_scope_check.py, test_error_handling_runner.py)
already exercise real git/AST/diff behavior for each individual check --
that's where getting real repo behavior right actually matters. This file's
only job is the orchestrator's own logic: does the merge use the right key
for the right function, call each with the right arguments, and stay clear
of computing introduces_regression_risk. None of that requires real git, and
building one fixture that triggers a specific result from all eight checks
at once would be fragile and would just re-test coverage that already exists
elsewhere. So each check is replaced with a sentinel stand-in here on purpose.
"""

from typing import Any

import pytest

import evalops.regression.orchestrator as orchestrator_module
from evalops.regression.orchestrator import run_regression_checks


# Helpers ------------------------------------------------------------------------------


def _stub(return_value: Any, calls: list) -> Any:
    """Returns a function that records its call args and returns a fixed value."""

    def _fn(*args, **kwargs):
        calls.append((args, kwargs))
        return return_value

    return _fn


# Fixtures -------------------------------------------------------------------------------


@pytest.fixture
def call_log() -> dict:
    """One list per patched function, so each call site's args can be checked
    independently rather than lumped into a single shared log."""
    return {
        "signature_unchanged": [],
        "dependency_changed": [],
        "syntax_valid": [],
        "imports_valid": [],
        "test_passed": [],
        "no_unused_code_introduced": [],
        "files_modified_outside_issue_scope": [],
        "run_error_handling_check": [],
    }


@pytest.fixture
def patched(monkeypatch: pytest.MonkeyPatch, call_log: dict) -> dict:
    """Patches all eight check functions at the orchestrator module's own
    namespace (not their source modules) since that's where
    run_regression_checks looks them up after the `from ... import` at the
    top of the file. Each returns a distinct, recognizable sentinel so a
    mixup between two checks would fail loudly rather than silently."""
    sentinels = {
        "signature_unchanged": {"passed": True, "detail": "sig-sentinel"},
        "dependency_changed": {"passed": False, "detail": "dep-sentinel"},
        "syntax_valid": {"passed": True, "detail": "syntax-sentinel"},
        "imports_valid": {"passed": False, "detail": "imports-sentinel"},
        "test_passed": {"passed": None, "detail": "test-sentinel"},
        "no_unused_code_introduced": {"passed": True, "detail": "unused-sentinel"},
        "files_modified_outside_issue_scope": {"passed": True, "detail": "scope-sentinel"},
        "run_error_handling_check": {"signal": True, "findings": ["error-handling-sentinel"]},
    }

    for name, value in sentinels.items():
        monkeypatch.setattr(orchestrator_module, name, _stub(value, call_log[name]))

    return sentinels


# Tests --------------------------------------------------------------------------------


def test_all_eight_keys_present_and_correctly_mapped(patched: dict) -> None:
    """Each result key must hold the return value of its own check, not a
    neighbor's -- catches a copy-paste key/function mismatch in the dict
    literal."""
    result = run_regression_checks("worktree", "issue body", base_ref="HEAD")

    assert result == {
        "signature_unchanged": patched["signature_unchanged"],
        "dependency_changed": patched["dependency_changed"],
        "syntax_valid": patched["syntax_valid"],
        "imports_valid": patched["imports_valid"],
        "test_passed": patched["test_passed"],
        "no_unused_code_introduced": patched["no_unused_code_introduced"],
        "files_modified_outside_issue_scope": patched["files_modified_outside_issue_scope"],
        "removes_weakens_error_handling": patched["run_error_handling_check"],
    }


def test_exactly_eight_keys_no_more_no_less(patched: dict) -> None:
    """Guards the count directly, separate from the equality check above --
    a stray extra key would still pass an == comparison against a dict
    literal typo but not this."""
    result = run_regression_checks("worktree", "issue body", base_ref="HEAD")

    assert len(result) == 8


def test_introduces_regression_risk_not_computed(patched: dict) -> None:
    """Locked design decision from the module docstring: a derived risk
    field must not appear here, even though run_error_handling_check
    returns signal: True in this fixture (a value that, if the orchestrator
    wrongly aggregated it, would tempt introduces_regression_risk into
    existing as True)."""
    result = run_regression_checks("worktree", "issue body", base_ref="HEAD")

    assert "introduces_regression_risk" not in result


def test_worktree_and_base_ref_passed_to_plain_mechanical_checks(
    patched: dict, call_log: dict
) -> None:
    """The six plain mechanical/hybrid-adjacent checks that don't need
    issue_body should each receive exactly (worktree_path, base_ref)."""
    run_regression_checks("worktree-x", "issue body", base_ref="ref-y")

    for name in (
        "signature_unchanged",
        "dependency_changed",
        "syntax_valid",
        "imports_valid",
        "test_passed",
        "no_unused_code_introduced",
    ):
        args, kwargs = call_log[name][0]
        assert args == ("worktree-x", "ref-y")
        assert kwargs == {}


def test_scope_check_receives_issue_body(call_log: dict, patched: dict) -> None:
    """files_modified_outside_issue_scope is the one plain check that also
    needs issue_body -- confirms it isn't silently dropped or swapped with
    base_ref."""
    run_regression_checks("worktree-x", "issue body text", base_ref="ref-y")

    args, kwargs = call_log["files_modified_outside_issue_scope"][0]
    assert args == ("worktree-x", "ref-y", "issue body text")
    assert kwargs == {}


def test_error_handling_check_receives_worktree_and_base_ref_only(
    call_log: dict, patched: dict
) -> None:
    """run_error_handling_check takes (worktree_path, base_ref) -- no
    issue_body -- confirming the orchestrator doesn't pass it an argument
    its real signature doesn't accept."""
    run_regression_checks("worktree-x", "issue body text", base_ref="ref-y")

    args, kwargs = call_log["run_error_handling_check"][0]
    assert args == ("worktree-x", "ref-y")
    assert kwargs == {}


def test_default_base_ref_is_head(patched: dict, call_log: dict) -> None:
    """base_ref defaults to "HEAD" per the function signature -- confirmed
    by omitting it and checking what actually got passed through."""
    run_regression_checks("worktree-x", "issue body")

    args, _ = call_log["signature_unchanged"][0]
    assert args == ("worktree-x", "HEAD")