"""
Retry node: bounded, same-provider re-attempt after a failed Patch
Application. Only fires for failure_reasons where retry's mechanism
(reset + rerun, or rerun with fresh retrieval) actually addresses the
cause. dirty_worktree, rollback_failed, unsafe_pattern_detected, and
insufficient_context skip this node entirely and route straight to
Escalate -- see /areas/resolve-patch-application.md for why those are
excluded. insufficient_context specifically: Resolve already exhausts
its own smarter retry (follow_up_query) before returning this reason,
so there's nothing left for this node's cruder retrieval to add.

"""

import subprocess
from pathlib import Path

from issue_worker.nodes.resolve import resolve_issue
from issue_worker.nodes.patch_application import apply_patch
from issue_worker.retrieval.retriever import retrieve
from issue_worker.retrieval.query_builder import build_query

WORKTREE_ROOT = Path("data/repo_cache/worktree")
MAX_RETRY_ATTEMPTS = 3

# failure_reasons where retry's mechanism actually addresses the cause.
RETRYABLE_REASONS = {"malformed_edit", "io_error", "stale_chunks"}

# needs fresh retrieval, not just a fresh Resolve call -- the narrowed
# chunk set already seen is out of date relative to what's now needed.
# insufficient_context is deliberately NOT here: Resolve's own
# follow_up_query loop (up to 2 rounds) already tries refined queries
# and merges newly retrieved chunks before ever returning this reason,
# so a naive re-retrieve here would be strictly dumber than what
# already failed. Treated as not_retryable instead -- see below.
NEEDS_FRESH_RETRIEVAL = {"stale_chunks"}

def _worktree_path(source_id : str)->Path:
    return WORKTREE_ROOT/source_id

def _reset_worktree(source_id : str, touched_files : list[str]) -> tuple[bool, str]:
    """
    git checkout -- <files> before each new Resolve attempt, per the
    locked reset-before-retry design. Resets every file_path the previous
    edits targeted (conservative superset of what Patch Application may
    have actually written -- checkout on an untouched file is a no-op).

    A failed reset here is distinct from Patch Application's own
    rollback_failed -- this is Retry's own precondition step, surfaced
    to the caller as a reason to escalate rather than retry blind.
    """
    if not touched_files:
        return True, ""
    worktree_path = _worktree_path(source_id)
    result = subprocess.run(
        ["git", "checkout","--"] + touched_files,
        cwd = worktree_path,
        capture_output= True,
        text =True,
    )
    if result.returncode !=0:
        return False, result.stderr.strip()
    return True, ""

def _build_retry_context(failure_reason : str, detail : str, attempt : int) -> dict:
    """Fed into Resolve's prompt so it sees what went wrong last time."""
    return {
        "previous_failure_reason" : failure_reason,
        "previous_detail" : detail,
        "attempt_number" : attempt,
    }

def retry_issue(
        issue : dict,
        full_chunks : list[dict],
        prev_chunks : list[dict],
        resolve_output : dict,
        patch_result : dict,
        source_id : str,
        attempt : int = 1,
    ) -> dict:
    """
    Entry point, called after a failed apply_patch() result.

    full_chunks: the entire chunked corpus for this source_id, needed so
    stale_chunks can re-retrieve outside the narrow set that already
    failed. prev_chunks: the top_k set Resolve/Patch Application already
    saw, reused as-is for malformed_edit/io_error.

    Returns one of:
      {"outcome": "applied", "resolve_output", "patch_result",
       "chunks_used", "attempts_used"}
      {"outcome": "retry_exhausted", "last_patch_result", "attempts_used"}
      {"outcome": "not_retryable", "failure_reason", "patch_result"}

    "not_retryable" (dirty_worktree, rollback_failed, unsafe_pattern_detected,
    insufficient_context, or a failed reset) means route to Escalate, no
    attempt consumed. "retry_exhausted" means MAX_RETRY_ATTEMPTS was hit --
    route to Fallback."""

    failure_reason = patch_result.get("failure_reason")

    if failure_reason not in RETRYABLE_REASONS:
        return {
            "outcome" : "not_retryable",
            "failure_reason" : failure_reason,
            "patch_result" : patch_result
        }
    if attempt > MAX_RETRY_ATTEMPTS:
        return {
            "outcome" : "retry_exhausted",
            "last_patch_result" : patch_result,
            "attempts_used" : attempt -1, 
        }

    touched_files = sorted({e["file_path"] for e in resolve_output.get("edits",[])})
    reset_ok, reset_detail = _reset_worktree(source_id, touched_files)
    if not reset_ok:
        return {
            "outcome": "not_retryable",
            "failure_reason": "reset_failed",
            "patch_result": {
                "applied": False,
                "failure_reason": "reset_failed",
                "detail": reset_detail,
            },
        }
    if failure_reason in NEEDS_FRESH_RETRIEVAL:
        query = build_query(issue, full_chunks)
        chunks_for_resolve = retrieve(query, full_chunks, top_k=5)
    else : 
        chunks_for_resolve = prev_chunks

    retry_context = _build_retry_context(failure_reason, patch_result.get("detail",""), attempt)
    new_resolve_output = resolve_issue(issue, chunks_for_resolve, retry_context=retry_context)
    new_patch_result = apply_patch(new_resolve_output, chunks_for_resolve, source_id)

    if new_patch_result["applied"]:
        return {
            "outcome" : "applied",
            "resolve_output" : new_resolve_output,
            "patch_result" : new_patch_result,
            "chunks_used" : chunks_for_resolve,
            "attempts_used" : attempt,
        }

    return retry_issue(
        issue,
        full_chunks,
        chunks_for_resolve,
        new_resolve_output,
        new_patch_result,
        source_id,
        attempt=attempt +1,
    )


if __name__ == "__main__":
    import argparse
    import json
    from issue_worker.retrieval.checkout import get_repo_at_commit
    from issue_worker.retrieval.chunker import chunk_repo

    parser = argparse.ArgumentParser()
    parser.add_argument("--source_id", default="22068")
    args = parser.parse_args()
    SOURCE_ID = args.source_id

    def _load_issue(source_id: str) -> dict:
        with open("data/raw_issues.jsonl", "r", encoding="utf-8") as f:
            for line in f:
                record = json.loads(line)
                if record["source_id"] == source_id:
                    return record
        raise ValueError(f"source_id {source_id} not found in raw_issues.jsonl")

    def _load_full_chunks(source_id: str) -> list[dict]:
        cache_path = f"data/chunk_cache/{source_id}.jsonl"
        if not Path(cache_path).exists():
            issue = _load_issue(source_id)
            worktree_path = get_repo_at_commit(source_id, issue["created_at"])
            chunk_repo(worktree_path, source_id)
        with open(cache_path, "r", encoding="utf-8") as f:
            return [json.loads(line) for line in f]

    issue = _load_issue(SOURCE_ID)
    full_chunks = _load_full_chunks(SOURCE_ID)
    query = build_query(issue, full_chunks)
    prev_chunks = retrieve(query, full_chunks, top_k=5)

    # first Resolve + Patch Application pass, no retry_context yet --
    # this is expected to produce the known malformed_edit failure on
    # 22068 (Resolve injecting a comment into old_source)
    resolve_output = resolve_issue(issue, prev_chunks)
    print("--- FIRST RESOLVE OUTPUT ---")
    print(json.dumps(resolve_output, indent=2))

    patch_result = apply_patch(resolve_output, prev_chunks, SOURCE_ID)
    print("\n--- FIRST PATCH APPLICATION RESULT ---")
    print(json.dumps(patch_result, indent=2))

    if patch_result["applied"]:
        print("\nFirst attempt already applied cleanly -- nothing for retry to do.")
    else:
        print(f"\n--- RUNNING retry_issue() (failure_reason={patch_result['failure_reason']!r}) ---")
        retry_result = retry_issue(
            issue,
            full_chunks,
            prev_chunks,
            resolve_output,
            patch_result,
            SOURCE_ID,
        )
        print("\n--- RETRY RESULT ---")
        # chunks are large -- summarize rather than dump verbatim
        summary = {k: v for k, v in retry_result.items() if k != "chunks_used"}
        if "chunks_used" in retry_result:
            summary["chunks_used"] = [c["chunk_id"] for c in retry_result["chunks_used"]]
        print(json.dumps(summary, indent=2))
