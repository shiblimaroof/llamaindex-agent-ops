"""

Top-level pipeline entry point. Wires the individually-verified nodes
together into one end-to-end run per issue: Classify -> Retrieve ->
Resolve -> Retry -> Fallback -> Escalate -> Notify -> Log.


Deliberately does NOT extract shared checkout/chunk/retrieve boilerplate
out of the other nodes' __main__ blocks -- that's a separate refactor of
already-verified files, not part of building this new one. This file
has its own copy.
"""

import json
import os
import time
from dotenv import load_dotenv
from groq import Groq
from issue_worker.nodes.log import log_event
import uuid

 
load_dotenv()

from issue_worker.nodes.classify import classify_issue
from issue_worker.nodes.resolve import resolve_issue
from issue_worker.nodes.patch_application import apply_patch
from issue_worker.nodes.retry import retry_issue
from issue_worker.retrieval.checkout import get_repo_at_commit
from issue_worker.retrieval.chunker import chunk_repo
from issue_worker.retrieval.query_builder import build_query
from issue_worker.retrieval.retriever import retrieve
from issue_worker.nodes.multi_provider_router import fallback_issue
from issue_worker.nodes.escalate import escalate_issue

RAW_ISSUES_PATH = "data/raw_issues.jsonl"
CHUNK_CACHE_DIR = "data/chunk_cache"


def _load_issue(source_id: str) -> dict:
    with open(RAW_ISSUES_PATH, "r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            if record["source_id"] == source_id:
                return record
    raise ValueError(f"source_id {source_id} not found in {RAW_ISSUES_PATH}")


def _load_or_build_chunks(source_id: str, issue: dict) -> list[dict]:
    """
    reuse the cache if it exists, otherwise checkout the repo at the issue's
    created_at and chunk it fresh.
    """
    cache_path = f"{CHUNK_CACHE_DIR}/{source_id}.jsonl"
    if os.path.exists(cache_path):
        with open(cache_path, "r", encoding="utf-8") as f:
            return [json.loads(line) for line in f]

    worktree_path = get_repo_at_commit(source_id, issue["created_at"])
    return chunk_repo(worktree_path, source_id)


def _run_escalate(issue: dict, source_id: str, run_id: str, escalation_input: dict) -> dict:
    escalation_result = escalate_issue(issue, source_id, run_id, escalation_input)
    return {
        "outcome": "escalated",
        "escalation_result": escalation_result,
    }


def _run_fallback(issue, top_chunks, retry_result, source_id,run_id):
    print(f"--- FALLBACK TRIGGERED: switching to Gemini (source_id={source_id}, "
          f"reason={retry_result['outcome']}) ---")

    chunks = retry_result.get("chunks_used", top_chunks)
    original_failure_reason = retry_result.get("failure_reason")

    start = time.perf_counter()
    fallback_result = fallback_issue(issue, chunks, source_id,run_id, original_failure_reason)
    duration_ms = (time.perf_counter() - start) * 1000

    log_event(
        node_name="fallback",
        source_id=source_id,
        run_id=run_id,
        outcome="success" if fallback_result["outcome"] == "applied" else "failure",
        failure_reason=fallback_result.get("fallback_failure_reason"),
        duration_ms=duration_ms,
    )

    if fallback_result["outcome"] == "applied":
        return {
            "outcome": "applied",
            "resolve_output": fallback_result["resolve_output"],
            "patch_result": fallback_result["patch_result"],
            "fallback_triggered": True,
        }

    escalation_input = {
        "failure_reason": fallback_result.get("fallback_failure_reason"),
        "outcome": "fallback_failed",
        "original_failure_reason": fallback_result.get("original_failure_reason"),
        "fallback_failure_reason": fallback_result.get("fallback_failure_reason"),
        "detail": fallback_result.get("patch_result", {}).get("detail"),
    }
    return _run_escalate(issue, source_id, run_id, escalation_input)


def run_pipeline(source_id: str) -> dict:
    """
    Entry point for the full pipeline on one issue. Returns a result dict
    whose "outcome" is one of:
      "not_actionable"            - Classify routed it away, nothing else ran
      "applied"                   - resolved and patched, first pass or retry
      "fallback_not_implemented"  - would route to Fallback (provider_error
                                     or retry_exhausted), stubbed for now
      "escalate_not_implemented"  - would route to Escalate (not_retryable),
                                     stubbed for now
    """
    run_id = str(uuid.uuid4())
    issue = _load_issue(source_id)
    groq_client = Groq(api_key=os.environ["GROQ_API_KEY"])

    start = time.perf_counter()
    classification = classify_issue(
        source_id=source_id,
        title=issue["title"],
        body=issue["body"],
        labels=issue.get("labels", []),
        client=groq_client,
    )
    duration_ms = (time.perf_counter() - start)*1000

    log_event(
        node_name="classify",
        source_id=source_id,
        run_id=run_id,
        outcome="success",
        duration_ms=duration_ms,
    )
    if not classification.is_actionable:
        return {
            "outcome": "not_actionable",
            "category": classification.category.value,
            "reasoning": classification.reasoning,
        }

    full_chunks = _load_or_build_chunks(source_id, issue)
    query = build_query(issue, full_chunks)

    start = time.perf_counter()
    top_chunks = retrieve(query, full_chunks, top_k=5)
    duration_ms = (time.perf_counter() - start) * 1000


    log_event(
        node_name="retrieve",
        source_id=source_id,
        run_id=run_id,
        outcome="success",
        duration_ms=duration_ms,
        )

    start = time.perf_counter()
    resolve_output = resolve_issue(issue, top_chunks, source_id,run_id)
    duration_ms = (time.perf_counter() - start) * 1000

    if resolve_output.get("failure_reason"):
        outcome = "failure"
    elif resolve_output.get("insufficient_context"):
        outcome = "insufficient_context"
    else:
        outcome = "success"

    log_event(
        node_name="resolve",
        source_id=source_id,
        run_id=run_id,
        outcome=outcome,
        failure_reason=resolve_output.get("failure_reason"),
        duration_ms=duration_ms,
        )

    start = time.perf_counter()
    patch_result = apply_patch(resolve_output, top_chunks, source_id)
    duration_ms = (time.perf_counter() -start) *1000

    log_event(
        node_name="patch_application",
        source_id=source_id,
        run_id=run_id,
        outcome="success" if patch_result["applied"] else "failure",
        failure_reason=patch_result.get("failure_reason"),
        duration_ms=duration_ms,
        )


    if patch_result["applied"]:
        return {
            "outcome": "applied",
            "resolve_output": resolve_output,
            "patch_result": patch_result,
            "attempts_used": 0,
        }

    retry_result = retry_issue(
        issue,
        full_chunks,
        top_chunks,
        resolve_output,
        patch_result,
        source_id,
        run_id,
        attempt=1,
    )


    if retry_result["outcome"] == "applied":
        return {
            "outcome": "applied",
            "resolve_output": retry_result["resolve_output"],
            "patch_result": retry_result["patch_result"],
            "attempts_used": retry_result["attempts_used"],
        }

    if retry_result["outcome"] in ("provider_error", "retry_exhausted"):
        return _run_fallback(issue, top_chunks, retry_result, source_id,run_id)

    # "not_retryable"
    escalation_input = {
        "failure_reason": retry_result.get("failure_reason"),
        "outcome": "not_retryable",
        "original_failure_reason": None,
        "fallback_failure_reason": None,
        "detail": retry_result.get("patch_result", {}).get("detail"),
    }
    return _run_escalate(issue, source_id, run_id, escalation_input)


if __name__ == "__main__":
    # Smoke test against the already-verified issue 22068.
    SOURCE_ID = "22068"
    result = run_pipeline(SOURCE_ID)

    print("--- PIPELINE RESULT ---")
    print(json.dumps(result, indent=2, default=str))