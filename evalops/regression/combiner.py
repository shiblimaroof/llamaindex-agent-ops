"""
Step A + Step B combiner: computes introduces_regression_risk from
regression_results (Step A's run_regression_checks output) and
judge_output (Step B's validated judge output).

Deliberately scoped to regression-shaped signals only. Does NOT fold in
judge_output["resolves_issue"] — resolves_issue answers "did the patch
solve the reported problem," introduces_regression_risk answers "did the
patch risk breaking existing behavior." Kept as separate concerns.

None on any mechanical/scope check means "couldn't verify," not "risk
detected" — it is excluded from the risk boolean, same honesty principle
as elsewhere in this codebase (tests_passed, files_modified_outside_issue_scope).
A None is a coverage gap, not a signal, so it's surfaced separately in
coverage_gaps rather than silently counted either way.
"""

# Keys in regression_results whose value is a plain {"passed": bool | None, "detail": str}
# dict. Excludes "error_handling", which has its own {"signal": bool, "findings": [...]} shape.
_MECHANICAL_KEYS = [
    "signature_unchanged",
    "dependency_changed",
    "no_unused_code_introduced",
    "syntax_valid",
    "imports_valid",
    "test_passed",
    "files_modified_outside_issue_scope",
]


def compute_regression_risk(regression_results: dict, judge_output: dict) -> dict:
    reasons = []
    coverage_gaps = []

    for key in _MECHANICAL_KEYS:
        check = regression_results.get(key)
        if check is None:
            continue
        passed = check.get("passed")
        if passed is False:
            reasons.append(key)
        elif passed is None:
            coverage_gaps.append(key)

    error_handling = regression_results.get("removes_weakens_error_handling")
    if error_handling is not None and error_handling.get("signal") is True:
        reasons.append("removes_weakens_error_handling")

    if judge_output.get("unexplained_concern") is True:
        reasons.append("unexplained_concern")

    return {
        "introduces_regression_risk": bool(reasons),
        "reasons": reasons,
        "coverage_gaps": coverage_gaps,
    }