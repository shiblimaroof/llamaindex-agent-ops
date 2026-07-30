
"""
issue_worker/nodes/patch_application.py
 
Patch Application: takes Resolve's output + the chunks Resolve saw, validates
against the real worktree, and applies the edits if everything checks out.
"""
 
import subprocess
from pathlib import Path
 
WORKTREE_ROOT = Path("data/repo_cache/worktree")
 
REQUIRED_EDIT_KEYS = {"chunk_id", "file_path", "old_source", "new_source"}
 
 
def _result(applied: bool, failure_reason: str | None = None, detail: str = "") -> dict:
    return {"applied": applied, "failure_reason": failure_reason, "detail": detail}
 
 
def _worktree_path(source_id: str) -> Path:
    return WORKTREE_ROOT / source_id
 
 
def _validate_edit_shapes(edits: list[dict]) -> str | None:
    """
    Checks every edit has all keys Patch Application needs before touching
    the filesystem. Patch Application validates shape (not Resolve) since
    it's the stage about to consume these dicts directly — Resolve already
    validates chunk_id membership, this is the complementary check.
    Returns a detail string if anything's missing, else None.
    """
    for i, edit in enumerate(edits):
        missing = REQUIRED_EDIT_KEYS - edit.keys()
        if missing:
            return f"edit index {i} missing required key(s): {sorted(missing)}"
    return None
 
 
def _is_clean(worktree_path: Path) -> tuple[bool, str]:
    """
    Runs `git status --porcelain` in the worktree. Empty output means clean.
    Returns (is_clean, raw_output) so a dirty_worktree failure can report
    exactly what git saw, since this one is meant to be investigated rather
    than silently retried.
 
    Raises on subprocess/path failures (missing worktree, git not on PATH,
    not a git repo) — caller wraps this so it resolves as io_error rather
    than crashing the pipeline.
    """
    proc = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=worktree_path,
        capture_output=True,
        text=True,
        check=True,
    )
    output = proc.stdout.strip()
    return (output == "", output)
 
 
def _chunk_by_id(chunks: list[dict], chunk_id: str) -> dict | None:
    for c in chunks:
        if c["chunk_id"] == chunk_id:
            return c
    return None

 
def _check_grounding(edits: list[dict], chunks: list[dict]) -> str | None:
    """
    For each edit, confirms old_source actually appears verbatim inside the
    chunk Resolve saw. Catches a hallucinated/invented snippet that was never
    really in the retrieved code — this is the check that replaces the old
    chunk-boundary math entirely, since old_source can now be any sub-range
    of a chunk, not just the whole thing. Returns a detail string on
    failure, else None.
    """
    for i, edit in enumerate(edits):
        chunk = _chunk_by_id(chunks, edit["chunk_id"])
        if chunk is None:
            # Resolve already validates chunk_id membership, but don't trust
            # that blindly here either.
            return f"edit index {i}: chunk_id not found in input chunks"
 
        if edit["old_source"] not in chunk["source"]:
            print("--- GROUNDING MISMATCH DEBUG ---")
            print(f"edit index {i}, chunk_id: {edit['chunk_id']}")
            print("old_source repr:")
            print(repr(edit["old_source"]))
            print("chunk source repr:")
            print(repr(chunk["source"]))
            return f"edit index {i}: old_source not found verbatim in its chunk's source"
 
    return None
 
 
def _check_staleness(edits: list[dict], worktree_path: Path) -> str | None:
    """
    Uniqueness check on the CURRENT file content — for each edit, counts
    occurrences of old_source in the current worktree file:
      - 0 matches: code moved/changed since Resolve looked at it (stale)
      - 2+ matches: ambiguous, can't safely tell which occurrence is the
        one Resolve meant (also treated as stale — no safe way to apply)
      - exactly 1 match: safe to apply
    Replaces the old line-offset comparison entirely — no line numbers or
    chunk boundaries involved, just a text search against the real file.
    Returns a detail string if any edit fails, else None.
    """
    for i, edit in enumerate(edits):
        full_path = worktree_path / edit["file_path"]
        text = full_path.read_text(encoding="utf-8")
        count = text.count(edit["old_source"])
 
        if count == 0:
            return f"edit index {i}: old_source not found in {edit['file_path']} (stale)"
        if count > 1:
            return (
                f"edit index {i}: old_source found {count} times in "
                f"{edit['file_path']}, ambiguous which to replace (stale)"
            )
 
    return None
 
 
def _group_by_file(edits: list[dict]) -> dict[str, list[dict]]:
    by_file: dict[str, list[dict]] = {}
    for edit in edits:
        by_file.setdefault(edit["file_path"], []).append(edit)
    return by_file
 
 
def _apply_file_edits(worktree_path: Path, file_path: str, file_edits: list[dict]) -> None:
    """
    Writes all edits for a single file using exact-text replacement. No line
    numbers or ordering tricks needed — each old_source was already confirmed
    unique in the current file by the staleness check, so replace(..., 1) is
    unambiguous regardless of what order the edits are applied in.
    """
    full_path = worktree_path / file_path
    text = full_path.read_text(encoding="utf-8")
 
    for edit in file_edits:
        text = text.replace(edit["old_source"], edit["new_source"], 1)
 
    full_path.write_text(text, encoding="utf-8")
 
 
def _rollback(worktree_path: Path, touched_files: list[str]) -> None:
    """
    Resets any files already written before a mid-batch failure, restoring
    the worktree to clean state. Turns "half-applied and undetected" into
    "attempted, failed cleanly, safe to retry" — the precondition check
    only guards state going INTO a batch, this guards state coming OUT of
    a failed one.
 
    Raises RuntimeError if the rollback itself fails, since a failed
    rollback means the worktree may still be left in a corrupted,
    half-applied state — the caller needs to know that, not have it
    silently swallowed.
    """
    if not touched_files:
        return
    result = subprocess.run(
        ["git", "checkout", "--"] + touched_files,
        cwd=worktree_path,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Rollback failed for {touched_files!r} in {worktree_path}: "
            f"{result.stderr.strip()}"
        )
 
 
def apply_patch(resolve_output: dict, chunks: list[dict], source_id: str) -> dict:
    """
    Entry point. Takes Resolve's output dict, the same chunks list Resolve
    saw, and the source_id to locate the worktree. Returns:
    {"applied": bool, "failure_reason": None | "insufficient_context" |
     "dirty_worktree" | "stale_chunks" | "malformed_edit" | "io_error" |
     "rollback_failed", "detail": "..."}
    """
    if resolve_output.get("insufficient_context"):
        return _result(False, "insufficient_context", resolve_output.get("summary", ""))
 
    edits = resolve_output.get("edits", [])
    worktree_path = _worktree_path(source_id)
 
    shape_error = _validate_edit_shapes(edits)
    if shape_error:
        return _result(False, "malformed_edit", shape_error)
 
    grounding_error = _check_grounding(edits, chunks)
    if grounding_error:
        return _result(False, "malformed_edit", grounding_error)
 
    try:
        clean, git_output = _is_clean(worktree_path)
    except (subprocess.CalledProcessError, OSError) as e:
        return _result(False, "io_error", f"git status failed: {e}")
 
    if not clean:
        return _result(False, "dirty_worktree", git_output)
 
    try:
        staleness_error = _check_staleness(edits, worktree_path)
    except OSError as e:
        return _result(False, "io_error", f"staleness check failed: {e}")
 
    if staleness_error:
        return _result(False, "stale_chunks", staleness_error)
 
    by_file = _group_by_file(edits)
    touched: list[str] = []
    try:
        for file_path, file_edits in by_file.items():
            _apply_file_edits(worktree_path, file_path, file_edits)
            touched.append(file_path)
    except OSError as e:
        try:
            _rollback(worktree_path, touched)
        except RuntimeError as rollback_error:
            return _result(
                False,
                "rollback_failed",
                f"write failed ({e}) AND rollback failed ({rollback_error}) — "
                f"worktree may be left in a corrupted state, needs manual investigation",
            )
        return _result(False, "io_error", f"write failed partway through, rolled back: {e}")
 
    return _result(True)
 
 
if __name__ == "__main__":
    import argparse
    import json
    from issue_worker.nodes.resolve import resolve_issue
    from issue_worker.retrieval.retriever import retrieve
    from issue_worker.retrieval.query_builder import build_query
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

    def _load_chunks(source_id: str) -> list[dict]:
        cache_path = f"data/chunk_cache/{source_id}.jsonl"
        if not Path(cache_path).exists():
            issue = _load_issue(source_id)
            worktree_path = get_repo_at_commit(source_id, issue["created_at"])
            chunk_repo(worktree_path, source_id)
        with open(cache_path, "r", encoding="utf-8") as f:
            return [json.loads(line) for line in f]

    issue = _load_issue(SOURCE_ID)
    chunks = _load_chunks(SOURCE_ID)
    query = build_query(issue, chunks)
    top_chunks = retrieve(query, chunks, top_k=5)

    resolve_output = resolve_issue(issue, top_chunks)
    print("--- RESOLVE OUTPUT ---")
    print(json.dumps(resolve_output, indent=2))

    patch_result = apply_patch(resolve_output, top_chunks, SOURCE_ID)
    print("\n--- PATCH APPLICATION RESULT ---")
    print(json.dumps(patch_result, indent=2))
 
