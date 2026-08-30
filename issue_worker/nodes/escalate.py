"""
Two ways an issue reaches Escalate:
  1. `not_retryable` — Retry never even attempted a retry loop
     (currently: dirty_worktree / rollback_failed).
     Fallback is never called on this path.
  2. `fallback_failed` — Retry exhausted, Fallback (Gemini) also failed.

`run_pipeline()` normalizes both into one flat `escalation_input` dict
before calling `escalate_issue()`, so this file stays a pure, mechanical
categorizer with no branch-detection logic of its own.

Expected `escalation_input` shape (all fire keys always present)

"""
from issue_worker.notify import notify_escalation
import time
from issue_worker.nodes.log import log_event

INFRA_FAILURE_REASONS = {"dirty_worktree", "rollback_failed", "io_error"}


def categorize_escalation(failure_reason: str, outcome: str) -> str:
    """
    Deterministic, mechanical categorization. No LLM call, no heuristics.

    Order matters: guardrail_trip is checked first (a proposed dangerous
    pattern is categorically distinct from an infra failure, and should
    never be miscategorized even if it somehow reached Escalate via a
    path that also looks infra-like). failure_reason is then checked
    before outcome, so an infra cause is categorized as such even if it
    happened to reach Escalate via the fallback_failed path (persisting
    through Retry AND Fallback unchanged is itself evidence for infra,
    not against it).

    Raises on any (failure_reason, outcome) pair not covered below —
    an unrecognized state during a batch run means a bug or an unwired
    new failure mode, and should surface immediately rather than fall
    into a silent "unclassified" bucket.
    """
    if failure_reason == "unsafe_pattern_detected":
        return "guardrail_trip"

    if failure_reason in INFRA_FAILURE_REASONS:
        return "infra_failure"

    if outcome == "fallback_failed":
        return "capability_exhausted"

    if failure_reason == "insufficient_context" and outcome == "not_retryable":
        return "context_exhausted"

    raise ValueError(
        f"Unrecognized escalation state: failure_reason={failure_reason!r}, "
        f"outcome={outcome!r}"
    )

def escalate_issue(issue: dict, source_id: str, run_id: str, escalation_input: dict) -> dict:
    """
    Return shape is a categorized record only — this does NOT write to
    the HITL flywheel/signature library. That store is fed by human-
    reviewed signal after Notify, not by Escalate's automatic pass.
    """
    start = time.perf_counter()

    failure_reason = escalation_input.get("failure_reason")
    outcome = escalation_input.get("outcome")
    original_failure_reason = escalation_input.get("original_failure_reason")
    fallback_failure_reason = escalation_input.get("fallback_failure_reason")
    detail = escalation_input.get("detail")

    category = categorize_escalation(failure_reason, outcome)

    record = {
        "source_id": source_id,
        "category": category,
        "failure_reason": failure_reason,
        "outcome": outcome,
        "original_failure_reason": original_failure_reason,
        "fallback_failure_reason": fallback_failure_reason,
        "detail": detail,
    }
    notify_escalation(record)

    duration_ms = (time.perf_counter() - start) * 1000
    log_event(
        node_name="escalate",
        source_id=source_id,
        run_id= run_id,
        outcome="success",
        failure_reason=failure_reason,
        duration_ms=duration_ms,
    )

    return record
