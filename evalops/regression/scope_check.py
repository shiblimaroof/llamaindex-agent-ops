import re
from pathlib import Path
from evalops.regression.mechanical import _changed_python_files

def _extract_referenced_paths(worktree_path: str, issue_text: str) -> set[str]:
    """Extract .py file paths referenced in an issue's body/traceback.

    Merges three sources into one flat set, each canonicalized to a
    llama_index/-rooted repo-relative path where possible:
      1. Traceback frames: File "path/to/file.py", line N, in func
      2. Inline backtick paths: `llama_index/core/x.py`
      3. Markdown code-fence paths: same shape, inside ``` blocks

    Only paths that resolve to a real file under worktree_path are kept.
    Traceback frames outside the repo (stdlib, site-packages, venv) are
    dropped at extraction time, not left for the caller to filter.
    """
    root = Path(worktree_path).resolve()
    candidates: set[str] = set()

    # 1. Traceback frames
    for match in re.finditer(r'File "([^"]+\.py)"', issue_text):
        candidates.add(match.group(1))

    # 2 & 3. Backtick-quoted and fenced paths — same shape, one pass.
    # Matches any backtick-delimited (single or triple) token that looks
    # like a path: contains a path separator and ends in .py
    for match in re.finditer(r'`{1,3}([^`\s]+\.py)`{1,3}', issue_text):
        candidates.add(match.group(1))

    referenced: set[str] = set()
    for raw_path in candidates:
        resolved = _resolve_candidate_path(root, raw_path)
        if resolved is not None:
            referenced.add(resolved)

    return referenced


def _resolve_candidate_path(root: Path, raw_path: str) -> str | None:
    raw = raw_path.strip()

    if "/" in raw:
        parts = raw.split("/")
        if "llama_index" not in parts:
            return None
        idx = len(parts) - 1 - parts[::-1].index("llama_index")  # last occurrence
        candidate_parts = tuple(parts[idx:])
    else:
        candidate_parts = (raw,)

    matches = [
        f for f in root.rglob("*.py")
        if f.parts[-len(candidate_parts):] == candidate_parts
    ]

    if len(matches) == 1:
        return str(matches[0].relative_to(root))
    return None  # zero or ambiguous matches — can't resolve safely


def files_modified_outside_issue_scope(
    worktree_path: str, base_ref: str, issue_text: str
) -> dict:
    """True (passed) if every file the patch changed was referenced in the
    issue text (traceback, backtick paths, code fences). Flags files touched
    but never mentioned as potential scope creep.

    Matching is strict/exact — a changed file must resolve to the same real
    file as one of the extracted referenced paths. No same-directory
    leniency: "related but not referenced" is a judgment call, routed to
    unexplained_concern, not this check.
    """
    changed_files = _changed_python_files(worktree_path, base_ref)
    referenced_files = _extract_referenced_paths(worktree_path, issue_text)

    if not referenced_files:
        return {
            "passed": None,
            "detail": "No file paths could be extracted from the issue text — cannot judge scope.",
        }

    out_of_scope = [f for f in changed_files if f not in referenced_files]

    if out_of_scope:
        return {
            "passed": False,
            "detail": "File(s) changed but never referenced in the issue: " + ", ".join(out_of_scope),
        }
    return {
        "passed": True,
        "detail": "All changed files were referenced in the issue.",
    }