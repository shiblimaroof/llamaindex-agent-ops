"""
Manual verification script for Patch Application's edge-case paths not
naturally reachable by picking golden-set issues: dirty_worktree,
stale_chunks, io_error (missing worktree), rollback_failed. Uses synthetic
edits/chunks against the already-checked-out 21015 worktree — no Resolve
call needed for the first three; rollback_failed is a direct unit test of
_rollback() in isolation since forcing it through the real apply_patch path
would require corrupting real git state.
"""

import subprocess
from pathlib import Path
from unittest.mock import patch as mock_patch

from issue_worker.nodes.patch_application import apply_patch, _rollback

SOURCE_ID = "21015"
WORKTREE = Path("data/repo_cache/worktree") / SOURCE_ID
TARGET_FILE = "llama-index-core/llama_index/core/output_parsers/pydantic.py"
TARGET_PATH = WORKTREE /TARGET_FILE

OLD_SOURCE = "        schema_str = json.dumps(schema_dict)"
NEW_SOURCE = "        schema_str = json.dumps(schema_dict, ensure_ascii=False)"
CHUNK_ID = "llama-index-core/llama_index/core/output_parsers/pydantic.py ::PydanticOutputParser.get_format_string"

def _reset_worktree():
    subprocess.run(["git","checkout","--","."],cwd = WORKTREE, check=True)
    subprocess.run(["git","clean","-fd"],cwd = WORKTREE,check=True)


def _base_edit_and_chunks():
    chunk_source = TARGET_PATH.read_text(encoding="utf-8")
    edit = {
        "chunk_id" : CHUNK_ID,
        "file_path" : TARGET_FILE,
        "old_source" : OLD_SOURCE,
        "new_source" : NEW_SOURCE,
    }
    chunks = [{"chunk_id" : CHUNK_ID, "source" : chunk_source}]
    return edit , chunks

def verify_dirty_worktree():
    print("\n=== dirty_worktree ===")
    _reset_worktree()
    scratch = WORKTREE / "SCRATCH_DIRTY.txt"
    scratch.write_text("uncommitted\n", encoding="utf-8")

    edit,chunks = _base_edit_and_chunks()
    result = apply_patch({"edits": [edit], "insufficient_context": False}, chunks, SOURCE_ID)
    print(result)
    assert result["failure_reason"] == "dirty_worktree", result 

    scratch.unlink()
    _reset_worktree()

def _git(args):
    return subprocess.run(
        ["git"] + args, cwd=WORKTREE, capture_output=True, text=True, check=True
    )


def verify_stale_chunks():
    print("\n=== stale_chunks (ambiguous, 2+ matches) ===")
    _reset_worktree()
    edit, chunks = _base_edit_and_chunks()

    original_head = _git(["rev-parse", "HEAD"]).stdout.strip()

    text = TARGET_PATH.read_text(encoding="utf-8")
    text = text.replace(OLD_SOURCE, OLD_SOURCE + "\n" + OLD_SOURCE, 1)
    TARGET_PATH.write_text(text, encoding="utf-8")

    # Commit the duplication so git status is clean going into apply_patch —
    # dirty_worktree is checked before stale_chunks, so an uncommitted
    # change here would trip that check first instead of the one we want.
    _git(["add", TARGET_FILE])
    _git(["commit", "-m", "TEMP: duplicate line for stale_chunks test"])

    try:
        result = apply_patch({"edits": [edit], "insufficient_context": False}, chunks, SOURCE_ID)
        print(result)
        assert result["failure_reason"] == "stale_chunks", result
    finally:
        _git(["reset", "--hard", original_head])

def verify_io_error_missing_worktree():
    print("\n=== io_error (worktree path doesn't exist) ===")
    edit, chunks = _base_edit_and_chunks()
    result = apply_patch(
        {"edits" : [edit], "insufficient_context" : False}, chunks , "9999-nonexistent"
    )
    print(result)
    assert result["failure_reason"]== "io_error", result

def verify_rollback_failed():
    print("\n=== rollback_failed (unit test on _rollback directly) ===")
    fake_result = subprocess.CompletedProcess(
        args=[], returncode=1, stdout="", stderr="fatal: pathspec did not match"
    )
    with mock_patch("subprocess.run", return_value = fake_result):
        try:
            _rollback(Path("data/repo_cache/worktree/21015"), ["some_file.py"])
            assert False, "expected RuntimeError"
        except RuntimeError as e:
            print("Correctly raised:", e)

if __name__ == "__main__":
    verify_dirty_worktree()
    verify_stale_chunks()
    verify_io_error_missing_worktree()
    verify_rollback_failed()
    print("\nAll manual edge-case verifications completed.")
