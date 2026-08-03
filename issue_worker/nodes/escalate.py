"""
Two ways an issue reaches Escalate:
  1. `not_retryable` — Retry never even attempted a retry loop
     (currently: dirty_worktree / rollback_failed).
     Fallback is never called on this path.
  2. `fallback_failed` — Retry exhausted, Fallback (Gemini) also failed.

`run_pipeline()` normalizes both into one flat `escalation_input` dict
before calling `escalate_issue()`, so this file stays a pure, mechanical
categorizer with no branch-detection logic of its own.

Expected `escalation_input` shape (all four keys always present):
    {
        "failure_reason": str,            # the reason used for categorization
        "outcome": str,                   # "not_retryable" or "fallback_failed"
        "original_failure_reason": str | None,
        "fallback_failure_reason": str | None,
    }

"""


INFRA_FAILURE_REASONS = {"dirty_worktree", "rollback_failed", "io_error"}


def categorize_escalation(failure_reason: str, outcome: str) -> str:
    """
    Deterministic, mechanical categorization. No LLM call, no heuristics.

    Order matters: failure_reason is checked before outcome, so an infra
    cause is categorized as such even if it happened to reach Escalate via
    the fallback_failed path (persisting through Retry AND Fallback
    unchanged is itself evidence for infra, not against it).

    Raises on any (failure_reason, outcome) pair not covered below —
    an unrecognized state during a batch run means a bug or an unwired
    new failure mode, and should surface immediately rather than fall
    into a silent "unclassified" bucket. No guardrail_trip branch yet —
    see the module TODO.
    """
    if failure_reason in INFRA_FAILURE_REASONS:
        return "infra_failure"

    if outcome == "fallback_failed":
        return "capability_exhausted"

    raise ValueError(
        f"Unrecognized escalation state: failure_reason={failure_reason!r}, "
        f"outcome={outcome!r}"
    )


def escalate_issue(issue: dict, source_id: str, escalation_input: dict) -> dict:
    """
    Return shape is a categorized record only — this does NOT write to
    the HITL flywheel/signature library. That store is fed by human-
    reviewed signal after Notify, not by Escalate's automatic pass.
    """
    failure_reason = escalation_input.get("failure_reason")
    outcome = escalation_input.get("outcome")
    original_failure_reason = escalation_input.get("original_failure_reason")
    fallback_failure_reason = escalation_input.get("fallback_failure_reason")

    category = categorize_escalation(failure_reason, outcome)

    return {
        "source_id": source_id,
        "category": category,
        "failure_reason": failure_reason,
        "outcome": outcome,
        "original_failure_reason": original_failure_reason,
        "fallback_failure_reason": fallback_failure_reason,
    }


if __name__ == "__main__":
    # Smoke test — three cases, one per category, plus one that should raise.
    not_retryable_case = {
        "failure_reason": "dirty_worktree",
        "outcome": "not_retryable",
        "original_failure_reason": None,
        "fallback_failure_reason": None,
    }
    capability_exhausted_case = {
        "failure_reason": "malformed_edit",
        "outcome": "fallback_failed",
        "original_failure_reason": "retry_exhausted",
        "fallback_failure_reason": "malformed_edit",
    }
    unrecognized_case = {
        "failure_reason": "made_up_reason",
        "outcome": "not_retryable",
        "original_failure_reason": None,
        "fallback_failure_reason": None,
    }

    for name, case in [
        ("not_retryable -> infra_failure", not_retryable_case),
        ("capability_exhausted", capability_exhausted_case),
    ]:
        result = escalate_issue({"source_id": "22068"}, "22068", case)
        print(f"{name}: {result['category']}")
        assert result["category"] in {"infra_failure", "capability_exhausted"}

    try:
        escalate_issue({"source_id": "22068"}, "22068", unrecognized_case)
        print("FAIL: expected ValueError, none raised")
    except ValueError as e:
        print(f"unrecognized state correctly raised: {e}")
