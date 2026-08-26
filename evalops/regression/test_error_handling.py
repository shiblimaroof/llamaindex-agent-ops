"""
Unit tests for regression/error_handling.py.

Calls check_error_handling_weakened directly with source strings — no git,
no worktree — since old_source/new_source/diff_text/file_path are the
function's actual inputs. Runner-level (git, file discovery) behavior is
covered separately in test_error_handling_runner.py.

diff_text uses a wide synthetic @@ hunk header (covering the whole file)
rather than a real diff, since only the line-range parsing matters here,
not diff correctness itself.
"""

from evalops.regression.error_handling import check_error_handling_weakened


def _wide_diff(num_lines: int = 50) -> str:
    """A synthetic unified diff hunk header claiming every line as added,
    wide enough to cover any function used in these tests."""
    return f"@@ -1,{num_lines} +1,{num_lines} @@\n"


def _findings_of_kind(result: dict, kind: str) -> list[dict]:
    return [f for f in result["findings"] if f["kind"] == kind]


# removed_try_block / removed_except_clause --------------------------------------------


def test_removed_try_block() -> None:
    old_source = (
        "def foo():\n"
        "    try:\n"
        "        risky()\n"
        "    except ValueError:\n"
        "        raise\n"
    )
    new_source = "def foo():\n    risky()\n"

    result = check_error_handling_weakened(old_source, new_source, _wide_diff(), "mod.py")

    assert result["signal"] is True
    assert _findings_of_kind(result, "removed_try_block")


def test_removed_except_clause() -> None:
    old_source = (
        "def foo():\n"
        "    try:\n"
        "        risky()\n"
        "    except ValueError:\n"
        "        raise\n"
        "    except TypeError:\n"
        "        raise\n"
    )
    new_source = (
        "def foo():\n"
        "    try:\n"
        "        risky()\n"
        "    except ValueError:\n"
        "        raise\n"
    )

    result = check_error_handling_weakened(old_source, new_source, _wide_diff(), "mod.py")

    assert result["signal"] is True
    assert _findings_of_kind(result, "removed_except_clause")


# removed_raise (handler and try-body variants) -----------------------------------------


def test_removed_raise_in_handler() -> None:
    old_source = (
        "def foo():\n"
        "    try:\n"
        "        risky()\n"
        "    except ValueError:\n"
        "        raise\n"
    )
    new_source = (
        "def foo():\n"
        "    try:\n"
        "        risky()\n"
        "    except ValueError:\n"
        "        pass\n"
    )

    result = check_error_handling_weakened(old_source, new_source, _wide_diff(), "mod.py")

    assert result["signal"] is True
    assert _findings_of_kind(result, "removed_raise")


def test_removed_raise_in_try_body() -> None:
    """The gap fixed via _detect_removed_raise_in_try_body: a raise sitting
    directly in the try body (not a handler), removed."""
    old_source = (
        "def foo():\n"
        "    try:\n"
        "        if bad:\n"
        "            raise ValueError('bad')\n"
        "        return risky()\n"
        "    except TypeError:\n"
        "        pass\n"
    )
    new_source = (
        "def foo():\n"
        "    try:\n"
        "        if bad:\n"
        "            return None\n"
        "        return risky()\n"
        "    except TypeError:\n"
        "        pass\n"
    )

    result = check_error_handling_weakened(old_source, new_source, _wide_diff(), "mod.py")

    assert result["signal"] is True
    assert _findings_of_kind(result, "removed_raise")


# error_logging_removed / error_logging_downgraded (handler variant) --------------------


def test_error_logging_removed_in_handler() -> None:
    old_source = (
        "def foo():\n"
        "    try:\n"
        "        risky()\n"
        "    except ValueError:\n"
        "        log.error('failed')\n"
        "        raise\n"
    )
    new_source = (
        "def foo():\n"
        "    try:\n"
        "        risky()\n"
        "    except ValueError:\n"
        "        raise\n"
    )

    result = check_error_handling_weakened(old_source, new_source, _wide_diff(), "mod.py")

    assert result["signal"] is True
    assert _findings_of_kind(result, "error_logging_removed")


def test_error_logging_downgraded_in_handler() -> None:
    old_source = (
        "def foo():\n"
        "    try:\n"
        "        risky()\n"
        "    except ValueError:\n"
        "        log.error('failed')\n"
    )
    new_source = (
        "def foo():\n"
        "    try:\n"
        "        risky()\n"
        "    except ValueError:\n"
        "        log.debug('failed')\n"
    )

    result = check_error_handling_weakened(old_source, new_source, _wide_diff(), "mod.py")

    assert result["signal"] is True
    assert _findings_of_kind(result, "error_logging_downgraded")


# error_logging_removed / error_logging_downgraded (try-body variant -- the new fix) ----


def test_error_logging_removed_in_try_body() -> None:
    """The gap fixed via _detect_error_log_changed_in_try_body: a defensive
    log.error(...) sitting directly in the try body (not a handler),
    removed. Same class of bug as removed_raise_in_try_body -- confirms
    the handler-only detector alone would have missed this."""
    old_source = (
        "def foo():\n"
        "    try:\n"
        "        if not resp.ok:\n"
        "            log.error('request failed')\n"
        "            return None\n"
        "        return resp.json()\n"
        "    except ValueError:\n"
        "        pass\n"
    )
    new_source = (
        "def foo():\n"
        "    try:\n"
        "        if not resp.ok:\n"
        "            return None\n"
        "        return resp.json()\n"
        "    except ValueError:\n"
        "        pass\n"
    )

    result = check_error_handling_weakened(old_source, new_source, _wide_diff(), "mod.py")

    assert result["signal"] is True
    assert _findings_of_kind(result, "error_logging_removed")


def test_error_logging_downgraded_in_try_body() -> None:
    old_source = (
        "def foo():\n"
        "    try:\n"
        "        if not ok:\n"
        "            log.error('bad')\n"
        "            return None\n"
        "        return 1\n"
        "    except ValueError:\n"
        "        pass\n"
    )
    new_source = (
        "def foo():\n"
        "    try:\n"
        "        if not ok:\n"
        "            log.debug('bad')\n"
        "            return None\n"
        "        return 1\n"
        "    except ValueError:\n"
        "        pass\n"
    )

    result = check_error_handling_weakened(old_source, new_source, _wide_diff(), "mod.py")

    assert result["signal"] is True
    assert _findings_of_kind(result, "error_logging_downgraded")


def test_try_body_log_call_unchanged_no_finding() -> None:
    """A try-body log call that stays exactly the same across old/new must
    not produce a finding -- guards against a false positive from the new
    detector."""
    source = (
        "def foo():\n"
        "    try:\n"
        "        if not ok:\n"
        "            log.error('bad')\n"
        "            return None\n"
        "        return 1\n"
        "    except ValueError:\n"
        "        pass\n"
    )

    result = check_error_handling_weakened(source, source, _wide_diff(), "mod.py")

    assert _findings_of_kind(result, "error_logging_removed") == []
    assert _findings_of_kind(result, "error_logging_downgraded") == []


# broadened_except ------------------------------------------------------------------------


def test_broadened_except() -> None:
    old_source = (
        "def foo():\n"
        "    try:\n"
        "        risky()\n"
        "    except ValueError:\n"
        "        pass\n"
    )
    new_source = (
        "def foo():\n"
        "    try:\n"
        "        risky()\n"
        "    except Exception:\n"
        "        pass\n"
    )

    result = check_error_handling_weakened(old_source, new_source, _wide_diff(), "mod.py")

    assert result["signal"] is True
    assert _findings_of_kind(result, "broadened_except")


# missing_error_handling_new_code ----------------------------------------------------------


def test_missing_error_handling_new_code() -> None:
    old_source = "def unrelated():\n    pass\n"
    new_source = (
        "def fetch_data():\n"
        "    resp = requests.get('http://example.com')\n"
        "    return resp.json()\n"
    )

    result = check_error_handling_weakened(old_source, new_source, _wide_diff(), "mod.py")

    assert result["signal"] is True
    assert _findings_of_kind(result, "missing_error_handling_new_code")


def test_missing_error_handling_new_code_skipped_for_test_files() -> None:
    old_source = "def unrelated():\n    pass\n"
    new_source = (
        "def fetch_data():\n"
        "    resp = requests.get('http://example.com')\n"
        "    return resp.json()\n"
    )

    result = check_error_handling_weakened(old_source, new_source, _wide_diff(), "tests/test_mod.py")

    assert result == {"signal": False, "findings": []}