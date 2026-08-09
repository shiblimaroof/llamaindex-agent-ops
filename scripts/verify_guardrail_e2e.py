"""
scripts/verify_guardrail_e2e.py

One-shot end-to-end verification that a deliberately unsafe patch is
caught by guardrail.py and correctly routed: Patch Application (trip)
-> Retry (not_retryable) -> Escalate (guardrail_trip category, Slack
notification fires with real detail).

Uses issue 22068's real worktree/chunk cache (already checked out from
prior sessions) so the grounding check passes on real old_source content
before the guardrail check ever runs -- this test exists to prove the
guardrail path, not to re-prove grounding.

Kept permanently, same convention as verify_patch_application_edge_cases.py.
"""


import json
from dotenv import load_dotenv

load_dotenv()

from issue_worker.nodes.patch_application import apply_patch
from issue_worker.nodes.retry import retry_issue
from issue_worker.nodes.escalate import escalate_issue

SOURCE_ID = "22068"


def _load_first_chunk():
    with open(f"data/chunk_cache/{SOURCE_ID}.jsonl") as f:
        first_line = f.readline()
    return json.loads(first_line)


def main():
    chunk = _load_first_chunk()
    full_chunks = [chunk]

    # Verbatim match required for grounding to pass.
    old_source = chunk["source"]
    file_path = chunk["file_path"]
    chunk_id = chunk["chunk_id"]

    unsafe_new_source = old_source + '\n\nos.system("rm -rf /")\n'

    forced_resolve_output = {
        "insufficient_context": False,
        "edits": [
            {
                "chunk_id": chunk_id,
                "file_path": file_path,
                "old_source": old_source,
                "new_source": unsafe_new_source,
            }
        ],
    }

    print("=== Step 1: apply_patch (expect unsafe_pattern_detected) ===")
    patch_result = apply_patch(forced_resolve_output, full_chunks, SOURCE_ID)
    print(json.dumps(patch_result, indent=2))
    assert patch_result["failure_reason"] == "unsafe_pattern_detected", \
        f"Expected unsafe_pattern_detected, got {patch_result['failure_reason']}"

    print("\n=== Step 2: retry_issue (expect not_retryable) ===")
    retry_result = retry_issue(
        issue={"source_id": SOURCE_ID},
        full_chunks=full_chunks,
        prev_chunks=full_chunks,
        resolve_output=forced_resolve_output,
        patch_result=patch_result,
        source_id=SOURCE_ID,
        attempt=1,
    )
    print(json.dumps(retry_result, indent=2))
    assert retry_result["outcome"] == "not_retryable", \
        f"Expected not_retryable, got {retry_result['outcome']}"

    print("\n=== Step 3: escalate_issue (expect category=guardrail_trip) ===")
    escalation_input = {
        "failure_reason": retry_result.get("failure_reason", "unsafe_pattern_detected"),
        "outcome": retry_result["outcome"],
        "original_failure_reason": None,
        "fallback_failure_reason": None,
        "detail": patch_result.get("detail"),
    }
    record = escalate_issue({"source_id": SOURCE_ID}, SOURCE_ID, escalation_input)
    print(json.dumps(record, indent=2))
    assert record["category"] == "guardrail_trip", \
        f"Expected guardrail_trip, got {record['category']}"

    print("\n=== ALL ASSERTIONS PASSED ===")
    print("Check Slack for the guardrail_trip notification.")


if __name__ == "__main__":
    main()