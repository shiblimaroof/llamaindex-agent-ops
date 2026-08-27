"""
evalops/regression/orchestrator.py

Step A orchestration for the regression tier. Runs every regression-level
mechanical/hybrid check for one patch/commit and merges results into a
single dict for Step B's judge prompt. No new checking logic lives here --
pure coordination, same pattern as error_handling_runner.py's own combiner.

Deliberately does NOT compute introduces_regression_risk. That field is
derived AFTER Step B's judge has turned the hybrid check's raw signal
into a real judged verdict (e.g. weakens_error_handling: bool) -- a raw
signal is not itself a verdict (a rewritten function can legitimately
trip signal: True without being a regression), so aggregating "any risk"
here would bake in a false positive every time the hybrid check's
mechanical signal fires on a legitimate change. See error_handling.py's
module docstring for why {"signal": bool, "findings": [...]} is kept
structurally separate from {"passed": bool, "detail": str} in the first
place -- computing a risk flag from signal alone here would erase exactly
that distinction one call site later.

introduces_regression_risk belongs in whatever consumes Step B's judged
output, not in this file.
"""


from evalops.regression.mechanical import (
    signature_unchanged,
    dependency_changed,
    syntax_valid,
    imports_valid,
    test_passed,
    no_unused_code_introduced,
)
from evalops.regression.scope_check import files_modified_outside_issue_scope
from evalops.regression.error_handling_runner import run_error_handling_check



def run_regression_checks(worktree_path : str, issue_body : str, base_ref : str = "HEAD") ->dict:
    """
    Runs all regression-tier checks against worktree_path and merges them
    into one dict keyed by check name. Feeds directly into Step B's judge
    prompt with no further transformation -- no derived/aggregate field is
    computed here (see module docstring).

    run_error_handling_check is one call that internally covers all seven
    finding kinds (including missing_error_handling_new_code) and returns
    one merged {"signal": bool, "findings": [...]} -- not two separate
    hybrid checks, confirmed against the real function signature.

    Not yet verified against the real mechanical.py/scope_check.py source
    beyond signature_unchanged and files_modified_outside_issue_scope's
    confirmed signatures -- run this against a real worktree (e.g. issue
    22068) and fix whichever call site's TypeError comes back before
    trusting the rest of this list.
    """
    results = {
        "signature_unchanged": signature_unchanged(worktree_path, base_ref),
        "dependency_changed": dependency_changed(worktree_path, base_ref),
        "syntax_valid": syntax_valid(worktree_path, base_ref),
        "imports_valid": imports_valid(worktree_path, base_ref),
        "test_passed": test_passed(worktree_path, base_ref),
        "no_unused_code_introduced": no_unused_code_introduced(worktree_path, base_ref),
        "files_modified_outside_issue_scope": files_modified_outside_issue_scope(
            worktree_path, base_ref, issue_body
        ),
        "removes_weakens_error_handling": run_error_handling_check(worktree_path, base_ref),

    }
    return results