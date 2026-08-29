"""
Unit tests for regression/combiner.py's compute_regression_risk.

Uses constructed dicts directly, not real fixtures/monkeypatching — unlike
orchestrator.py, combiner.py calls no other functions, it only processes
the shapes those functions already produce. Real-data verification already
happened via scripts/verify_combiner.py against issue 22068; these tests
exist to exercise branches that one real run doesn't hit (False on each
mechanical key, zero-reasons case, resolves_issue exclusion in isolation).
"""

from evalops.regression.combiner import compute_regression_risk

ALL_PASS_REGRESSION_RESULTS = {
    "signature_unchanged": {"passed": True, "detail": ""},
    "dependency_changed": {"passed": True, "detail": ""},
    "syntax_valid": {"passed": True, "detail": ""},
    "imports_valid": {"passed": True, "detail": ""},
    "test_passed": {"passed": True, "detail": ""},
    "no_unused_code_introduced": {"passed": True, "detail": ""},
    "files_modified_outside_issue_scope": {"passed": True, "detail": ""},
    "removes_weakens_error_handling": {"signal": False, "findings": []},
}

CLEAN_JUDGE_OUTPUT = {
    "addresses_root_cause": True,
    "handles_reported_case": True,
    "avoids_described_failure_mode": True,
    "weakens_error_handling": False,
    "unexplained_concern": False,
    "unexplained_concern_note": "",
    "reasoning": "",
    "resolves_issue": True,
}


def _regression_results(**overrides):
    results = {k: dict(v) for k, v in ALL_PASS_REGRESSION_RESULTS.items()}
    results["removes_weakens_error_handling"] = dict(
        ALL_PASS_REGRESSION_RESULTS["removes_weakens_error_handling"]
    )
    for key, value in overrides.items():
        results[key] = value
    return results


def _judge_output(**overrides):
    output = dict(CLEAN_JUDGE_OUTPUT)
    output.update(overrides)
    return output


def test_all_clean_no_risk():
    risk = compute_regression_risk(_regression_results(), _judge_output())
    assert risk == {
        "introduces_regression_risk": False,
        "reasons": [],
        "coverage_gaps": [],
    }


def test_each_mechanical_false_flags_as_reason():
    for key in [
        "signature_unchanged",
        "dependency_changed",
        "syntax_valid",
        "imports_valid",
        "test_passed",
        "no_unused_code_introduced",
        "files_modified_outside_issue_scope",
    ]:
        results = _regression_results(**{key: {"passed": False, "detail": "x"}})
        risk = compute_regression_risk(results, _judge_output())
        assert risk["introduces_regression_risk"] is True
        assert risk["reasons"] == [key]
        assert risk["coverage_gaps"] == []


def test_each_mechanical_none_goes_to_coverage_gaps_not_reasons():
    for key in [
        "signature_unchanged",
        "dependency_changed",
        "syntax_valid",
        "imports_valid",
        "test_passed",
        "no_unused_code_introduced",
        "files_modified_outside_issue_scope",
    ]:
        results = _regression_results(**{key: {"passed": None, "detail": "x"}})
        risk = compute_regression_risk(results, _judge_output())
        assert risk["introduces_regression_risk"] is False
        assert risk["reasons"] == []
        assert risk["coverage_gaps"] == [key]


def test_error_handling_signal_true_flags_correct_key():
    results = _regression_results(
        removes_weakens_error_handling={"signal": True, "findings": ["x"]}
    )
    risk = compute_regression_risk(results, _judge_output())
    assert risk["introduces_regression_risk"] is True
    assert risk["reasons"] == ["removes_weakens_error_handling"]


def test_unexplained_concern_true_flags_reason():
    risk = compute_regression_risk(
        _regression_results(), _judge_output(unexplained_concern=True)
    )
    assert risk["introduces_regression_risk"] is True
    assert risk["reasons"] == ["unexplained_concern"]


def test_resolves_issue_false_never_counted():
    # resolves_issue is explicitly out of scope for regression risk, even
    # when everything else is clean.
    risk = compute_regression_risk(
        _regression_results(), _judge_output(resolves_issue=False)
    )
    assert risk["introduces_regression_risk"] is False
    assert risk["reasons"] == []


def test_multiple_reasons_combine():
    results = _regression_results(
        imports_valid={"passed": False, "detail": "x"},
        test_passed={"passed": None, "detail": "x"},
    )
    risk = compute_regression_risk(
        results, _judge_output(unexplained_concern=True)
    )
    assert risk["introduces_regression_risk"] is True
    assert set(risk["reasons"]) == {"imports_valid", "unexplained_concern"}
    assert risk["coverage_gaps"] == ["test_passed"]