from __future__ import annotations
import re

# Python builtins to exclude from identifier matches - too common to
# be useful search signal (would match nearly every chunk)
_BUILTIN_BLOCKLIST = {
    "print", "len", "str", "int", "dict", "list", "range", "type",
    "isinstance", "super", "getattr", "setattr", "format", "open",
    "sorted", "enumerate", "zip", "map", "filter", "input", "float",
    "bool", "set", "tuple", "sum", "min", "max", "abs", "round",
    "repr", "hash", "id", "next", "iter", "vars", "dir", "help",
}

# Backtick-wrapped code spans, e.g. `foo.bar` - higher precision than
# scanning raw prose since reporters backtick real code by convention.
# Dotted spans are captured whole; last segment is used as the name.
_IDENTIFIER_PATTERN = re.compile(r"`([\w\.]+)`")

# Plain-text snake_case mentions, e.g. "...builds its schema with
# create_schema_from_function, which..." - not every reporter backticks
# code in prose. Requires at least one underscore so this doesn't match
# ordinary English words; real-chunk-name validation downstream is the
# actual safety net against false positives.
_PLAIN_SNAKE_CASE_PATTERN = re.compile(r"\b[a-z][a-z0-9]*(?:_[a-z0-9]+)+\b")


# File-path-looking tokens ending in .py, backticks optional - paths
# often appear bare in prose/tracebacks, unlike function/class names.
_FILE_PATH_PATTERN = re.compile(r"`?([\w\-/]+\.py)`?")

# CamelCase + colon, matching a traceback's exception line (e.g.
# "KeyError: ..."). Trailing colon filters out prose mentions like
# "should raise a ValueError if...". No further validation needed -
# most exception types are builtin/third-party, not repo-defined.
_EXCEPTION_PATTERN = re.compile(r"\b([A-Z][a-zA-Z0-9]*(?:Error|Exception|Warning))\b\s*:")

# Markdown ATX headers (#, ##, ...) - GitHub issue template section
# markers. MULTILINE since headers land mid-body, not just at start.
_TEMPLATE_HEADER_PATTERN = re.compile(r"^#{1,6}\s", re.MULTILINE)

def _clean_description(body:str) ->str:

    if not body:
        return ""
    headers = list(_TEMPLATE_HEADER_PATTERN.finditer(body))
    if not headers:
        head = body
    elif headers[0].start() > 0:
        head = body[:headers[0].start()]
    else:
        newline = body.find("\n", headers[0].end())
        start = newline + 1 if newline != -1 else headers[0].end()
        end = headers[1].start() if len(headers) > 1 else len(body)
        head = body[start:end]

    
    head = re.sub(r"```.*?```", "", head, flags=re.DOTALL)

    return head.strip()

def _extract_identifiers(text : str, chunk_names : set[str]) -> list[str]:
#Find everything inside backticks (...), clean it up, throw away obvious junk, and 
# finally keep only names that actually exist as chunks in this repository.
    candidates = set()
    for match in _IDENTIFIER_PATTERN.findall(text):
        name = match.split(".")[-1]
        if name in _BUILTIN_BLOCKLIST:
            continue
        candidates.add(name)

    for match in _PLAIN_SNAKE_CASE_PATTERN.findall(text):
        if match in _BUILTIN_BLOCKLIST:
            continue
        candidates.add(match)

    return sorted(c for c in candidates if c in chunk_names)

def _extract_file_path(text:str, chunk_file_paths : set[str])-> list[str]:

    candidates = set(_FILE_PATH_PATTERN.findall(text))

    validated = set()
    for candidate in candidates:
        for known_path in chunk_file_paths:
            if known_path.endswith(candidate) or candidate.endswith(known_path):
                validated.add(known_path)
                break
    return sorted(validated)


def _extract_exception_types(text:str) ->list[str]:
    """ Extract Exception error not message"""
    return sorted(set(_EXCEPTION_PATTERN.findall(text)))

def build_query(issue : dict, chunks : list[dict]) -> dict:
    
    title = issue.get("title", "") or ""
    body = issue.get("body","") or ""

    semantic = f"{title}\n\n{_clean_description(body)}".strip()

    chunk_names = {c["name"] for c in chunks if c.get("name")}
    chunk_file_paths = {c["file_path"] for c in chunks if c.get("file_path")}

    full_text = f"{title}\n{body}"

    return {
        "semantic": semantic,
        "identifiers": _extract_identifiers(full_text, chunk_names),
        "file_paths": _extract_file_path(full_text, chunk_file_paths),
        "exception_types": _extract_exception_types(full_text),
        "source_id": issue["source_id"],
        "created_at": issue["created_at"],
    }

if __name__ == "__main__":
    import json

    from issue_worker.retrieval.checkout import get_repo_at_commit
    from issue_worker.retrieval.chunker import chunk_repo

    source_id = "22068"
    created_at = "2026-06-22T01:33:30Z"

    with open("data/raw_issues.jsonl") as f:
        issues = [json.loads(line) for line in f]
    issue = next(i for i in issues if i['source_id']==source_id)

    repo_path = get_repo_at_commit(source_id, created_at)
    chunks = chunk_repo(repo_path, source_id)

    query = build_query(issue, chunks)

    print("semantic:", query["semantic"][:200])
    print("identifiers:", query["identifiers"])
    print("file_paths:", query["file_paths"])
    print("exception_types:", query["exception_types"])
