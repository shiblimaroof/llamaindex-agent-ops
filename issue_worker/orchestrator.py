"""

Top-level pipeline entry point. Wires the individually-verified nodes
together into one end-to-end run per issue: Classify -> Retrieve ->
Resolve -> Retry -> Fallback -> Escalate -> Notify -> Log.

Fallback, Escalate, Notify, and Log are not built yet (see
/areas/llamaindex-agent-ops.md). This file gives each of those a
placeholder-dict stub rather than raising, so run_pipeline() always
returns cleanly regardless of which path an issue takes -- the point is
to get real, end-to-end signal on where issues land (applied cleanly,
need retry, would need fallback, not_retryable) before those stages
are built for real.

Deliberately does NOT extract shared checkout/chunk/retrieve boilerplate
out of the other nodes' __main__ blocks -- that's a separate refactor of
already-verified files, not part of building this new one. This file
has its own copy.
"""

import json
import os

from dotenv import load_dotenv
from groq import Groq
 
load_dotenv()

from issue_worker.nodes.classify import classify_issue
from issue_worker.nodes.resolve import resolve_issue
from issue_worker.nodes.patch_application import apply_patch
from issue_worker.nodes.retry import retry_issue
from issue_worker.retrieval.checkout import get_repo_at_commit
from issue_worker.retrieval.chunker import chunk_repo
from issue_worker.retrieval.query_builder import build_query
from issue_worker.retrieval.retriever import retrieve

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


def _fallback_stub(retry_result: dict) -> dict:
    # Fallback (multi_provider_router.py / Gemini) not built yet.
    return {
        "outcome": "fallback_not_implemented",
        "retry_result": retry_result,
    }


def _escalate_stub(retry_result: dict) -> dict:
    # Escalate -> Notify -> Log not built yet.
    return {
        "outcome": "escalate_not_implemented",
        "retry_result": retry_result,
    }


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
    issue = _load_issue(source_id)
    groq_client = Groq(api_key=os.environ["GROQ_API_KEY"])

    classification = classify_issue(
        source_id=source_id,
        title=issue["title"],
        body=issue["body"],
        labels=issue.get("labels", []),
        client=groq_client,
    )
    if not classification.is_actionable:
        return {
            "outcome": "not_actionable",
            "category": classification.category.value,
            "reasoning": classification.reasoning,
        }

    full_chunks = _load_or_build_chunks(source_id, issue)
    query = build_query(issue, full_chunks)
    top_chunks = retrieve(query, full_chunks, top_k=5)

    resolve_output = resolve_issue(issue, top_chunks)
    patch_result = apply_patch(resolve_output, top_chunks, source_id)

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
        return _fallback_stub(retry_result)

    # "not_retryable"
    return _escalate_stub(retry_result)


if __name__ == "__main__":
    # Smoke test against the already-verified issue 22068.
    SOURCE_ID = "22068"
    result = run_pipeline(SOURCE_ID)

    print("--- PIPELINE RESULT ---")
    print(json.dumps(result, indent=2, default=str))