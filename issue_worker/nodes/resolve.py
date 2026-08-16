"""
issue_worker/nodes/resolve.py

Resolve node: takes an issue + retriever's top reranked chunks, produces a
structured fix (exact-text old_source/new_source edits) or flags
insufficient_context, optionally with a follow_up_query describing what
context was missing — enables one caller-driven bounded retry round back
through the retriever before the issue is treated as failed. Optionally
takes retry_context from a previous failed attempt so it can avoid
repeating the same mistake.
"""

import json
from groq import Groq
from issue_worker.usage_logger import log_usage

client = Groq()

MODEL = "llama-3.3-70b-versatile"

RESOLVE_SYSTEM_PROMPT = """You are a senior Python engineer fixing a real bug in the llama_index codebase.

You will be given:
1. An issue title and description (the reported bug)
2. A set of candidate code chunks retrieved from the repository that MAY contain the bug
3. Optionally, details of a previous failed attempt to fix this same issue

Your job: identify the exact root cause and produce a fix, or admit the
provided chunks don't contain enough information to fix it correctly.

Rules:
- Only use the chunks given to you. Do not invent code or file paths.
- If you can fix it, output one edit per distinct change needed. A real fix
  can span multiple locations/files — do not force everything into one edit.
- Each edit's old_source must be an EXACT, VERBATIM copy of the code you are
  replacing — copy it character-for-character from the chunk's source, do
  not paraphrase or reformat it. It can be as small as a few lines or as
  large as the whole chunk, whichever is the smallest change that correctly
  fixes the bug. old_source must actually appear in the chunk's source.
- Common mistakes that WILL cause your edit to be rejected: collapsing
  multiple lines onto one line, changing indentation, or adding any comment,
  explanation, or annotation inline in the code that is not present in the
  original source, however small. If the original spans three lines,
  old_source must span exactly those three lines, unchanged, with nothing
  added or removed. Do not annotate, explain, or clean up the code inside
  old_source — save all explanation for the reasoning field.
- A specific, common version of this mistake: adding a comment that names
  or describes the bug you are fixing, inside old_source. Example of a
  REJECTED old_source:
      tool_call_id=message.additional_kwargs.get(
          "tool_call_id", str(uuid.uuid4())  # silent fabrication
      )
  This is wrong because "# silent fabrication" was written by you to
  describe the bug — it does not exist in the real file. The correct
  old_source is the exact original text with no added comment:
      tool_call_id=message.additional_kwargs.get(
          "tool_call_id", str(uuid.uuid4())
      )
  Your explanation of the bug belongs in the reasoning field, never as a
  comment inside old_source, even when the comment seems purely explanatory.
- If the chunks don't actually contain the bug, or you're not confident the
  fix is correct, set insufficient_context to true and return an EMPTY edits
  list. Do not guess. A wrong fix is worse than an honest "not enough context."
- insufficient_context and edits are mutually exclusive: if
  insufficient_context is true, edits MUST be an empty list.
- If insufficient_context is true, also set follow_up_query to a short
  search query describing the specific missing information — what you'd
  search for to find it (e.g. "async middleware request lifecycle handling",
  not "more context needed"). Be specific enough that a code search on this
  query would plausibly surface the missing piece. If insufficient_context
  is false, omit follow_up_query or set it to null.
- If you are told about a previous failed attempt, treat that as a
  constraint: do not reproduce the exact same old_source that was already
  rejected, and address the specific reason it failed before proposing
  anything else.

Respond with ONLY a JSON object, no markdown fences, no preamble, in this
exact shape:
{
  "edits": [
    {
      "chunk_id": "...",
      "file_path": "...",
      "old_source": "...",
      "new_source": "...",
      "reasoning": "..."
    }
  ],
  "summary": "...",
  "insufficient_context": false,
  "follow_up_query": null
}
"""


def _build_retry_block(retry_context: dict | None) -> str:
    """
    Renders retry_context into a prompt section. Returns "" when there's
    no retry context, so the first-attempt prompt is byte-identical to
    before this change — retry is additive, not a rewrite of the base
    prompt path.
    """
    if not retry_context:
        return ""

    return (
        "\n\n=== PREVIOUS ATTEMPT FAILED ===\n"
        f"This is retry attempt {retry_context['attempt_number']}.\n"
        f"Failure reason: {retry_context['previous_failure_reason']}\n"
        f"Detail: {retry_context['previous_detail']}\n"
        "Do not repeat the same old_source that was rejected. Address the "
        "specific reason above before proposing a fix."
    )


def _build_user_prompt(issue: dict, chunks: list[dict], retry_context: dict | None = None) -> str:
    """
    Builds the user-turn prompt from agent-visible issue fields and the
    retriever's chunk list, plus an optional retry block.

    Only title/body (and other agent-visible fields) are read from `issue` —
    same leakage discipline as every earlier file: no grading_key,
    golden_set, or linked_fix_diff_url fields are ever passed in here.
    """
    issue_block = f"ISSUE TITLE: {issue['title']}\n\nISSUE BODY:\n{issue['body']}"

    chunk_blocks = []
    for c in chunks:
        meta_lines = [
            f"chunk_id: {c['chunk_id']}",
            f"file_path: {c['file_path']}",
            f"lines: {c['start_line']}-{c['end_line']}",
        ]
        if c.get("class_name"):
            meta_lines.append(f"class_name: {c['class_name']}")
        if c.get("imports"):
            meta_lines.append(f"imports: {', '.join(c['imports'])}")
        if c.get("class_context"):
            meta_lines.append(f"class __init__ context:\n{c['class_context']}")
        if c.get("decorators"):
            meta_lines.append(f"decorators: {', '.join(c['decorators'])}")

        block = "\n".join(meta_lines) + f"\n\nsource:\n{c['source']}"
        chunk_blocks.append(block)

    chunks_section = "\n\n---\n\n".join(chunk_blocks)
    retry_block = _build_retry_block(retry_context)

    return f"{issue_block}\n\n=== CANDIDATE CHUNKS ===\n\n{chunks_section}{retry_block}"


def _strip_markdown_fence(raw: str) -> str:
    """
    Models sometimes wrap JSON in ```json ... ``` fences despite explicit
    instructions not to (confirmed happening with llama-3.3-70b-versatile
    on this prompt). Strip the fence if present rather than relying on
    prompt compliance alone — this is cheap and the response is otherwise
    well-formed JSON.
    """
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1] if "\n" in raw else raw[3:]
        if raw.endswith("```"):
            raw = raw[:-3]
        raw = raw.strip()
    return raw


def _honest_failure(reason: str, failure_reason: str = "insufficient_context") -> dict:
    """
    Shared shape for any failure that should resolve honestly rather than
    crash the pipeline — parse errors, API errors, chunk_id hallucination.
    Same principle as insufficient_context: a controlled "couldn't do it"
    beats an unhandled exception or a silently wrong result.

    failure_reason distinguishes *why* insufficient_context is True.
    Defaults to "insufficient_context" itself, so the two existing call
    sites (JSON parse failure, hallucinated chunk_id) are unchanged — both
    are genuinely model-quality failures, not provider failures. Only the
    API-call except block overrides this, passing "api_error", so
    downstream (Patch Application, Retry) can route provider outages to
    Fallback instead of Escalate without retrying against the same
    provider that just failed.

    follow_up_query is always None here — a parse/API failure means there's
    no model output to pull a follow-up query from, so there's nothing
    meaningful to loop with. Explicit None, not omitted, so callers can
    rely on the key always being present when insufficient_context is True.
    """
    return {
        "edits": [],
        "summary": reason,
        "insufficient_context": True,
        "follow_up_query": None,
        "failure_reason": failure_reason,
    }


def _validate_response(parsed: dict, valid_chunk_ids: set[str]) -> dict:
    """
    Enforces the locked either/or contract: insufficient_context=True must
    come with an empty edits list. If the model violates this, we don't
    trust either field blindly — we force insufficient_context and drop
    edits, since a wrong fix is the worse failure mode.

    Also cross-checks every edit's chunk_id against the actual chunks that
    were sent in. A hallucinated chunk_id (typo or invented) is caught here
    rather than left for Patch Application, which only checks file content,
    not chunk identity — Resolve is the one place that actually has the
    input chunk list to check against.

    Bounded iterative retrieval extension: when insufficient_context is
    (or becomes) True, follow_up_query is normalized to either a usable
    string or None. A missing/empty follow_up_query is NOT treated as a
    contract violation — insufficient_context alone is still a valid,
    expected outcome on its own, and follow_up_query is an addition on
    top of that, not a replacement for it. If it's None, the caller
    simply has nothing to loop with and falls through without a retry
    round, same as before this field existed.
    """
    insufficient = parsed.get("insufficient_context", False)
    edits = parsed.get("edits", [])

    if insufficient and edits:
        parsed["edits"] = []
        edits = []

    if not insufficient and not edits:
        # Model said it has a fix but gave no edits — treat as insufficient
        # context rather than silently passing an empty-but-"successful" result.
        # Deliberate contract extension beyond the original lock: when the
        # model's two fields are internally inconsistent, insufficient_context
        # wins. Logged in /areas/resolve-patch-application.md.
        parsed["insufficient_context"] = True
        insufficient = True

    if insufficient:
        follow_up = parsed.get("follow_up_query")
        parsed["follow_up_query"] = follow_up if follow_up else None
        return parsed

    bad_ids = [e.get("chunk_id") for e in edits if e.get("chunk_id") not in valid_chunk_ids]
    if bad_ids:
        return _honest_failure(
            f"model returned edit(s) with chunk_id not in the input chunk list: {bad_ids}"
        )

    return parsed


def resolve_issue(issue: dict, chunks: list[dict], source_id: str, retry_context: dict | None = None) -> dict:

    """
    Entry point. Takes an issue dict (agent-visible fields only), the
    retriever's top 3-5 reranked chunks, and optionally retry_context from
    a previous failed attempt (see retry.py). Returns the locked output
    shape: {"edits": [{"chunk_id", "file_path", "old_source", "new_source",
    "reasoning"}, ...], "summary": "...", "insufficient_context": bool,
    "follow_up_query": str | None}
    """
    user_prompt = _build_user_prompt(issue, chunks, retry_context)
    valid_chunk_ids = {c["chunk_id"] for c in chunks}

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": RESOLVE_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0,
        )
    except Exception as e:
        return _honest_failure(f"API call failed: {e}", failure_reason="api_error")

    log_usage(
        node_name="resolve",
        provider="groq",
        model=MODEL,
        source_id=issue.get("source_id", "unknown"),
        prompt_tokens=response.usage.prompt_tokens,
        completion_tokens=response.usage.completion_tokens,
    )


    raw = response.choices[0].message.content
    raw = _strip_markdown_fence(raw)

    try:
        parsed = json.loads(raw, strict=False)
    except json.JSONDecodeError as e:
        return _honest_failure(f"failed to parse model response as JSON: {e}")

    return _validate_response(parsed, valid_chunk_ids)


if __name__ == "__main__":
    # End-to-end smoke test against the already-verified issue 22068.
    # Uses the cached chunks (data/chunk_cache/22068.jsonl) rather than
    # re-chunking the repo, and pulls the real issue title/body from
    # data/raw_issues.jsonl by source_id.
    from issue_worker.retrieval.retriever import retrieve
    from issue_worker.retrieval.query_builder import build_query

    SOURCE_ID = "22068"

    def _load_issue(source_id: str) -> dict:
        with open("data/raw_issues.jsonl", "r", encoding="utf-8") as f:
            for line in f:
                record = json.loads(line)
                if record["source_id"] == source_id:
                    return record
        raise ValueError(f"source_id {source_id} not found in raw_issues.jsonl")

    def _load_chunks(source_id: str) -> list[dict]:
        with open(f"data/chunk_cache/{source_id}.jsonl", "r", encoding="utf-8") as f:
            return [json.loads(line) for line in f]

    issue = _load_issue(SOURCE_ID)
    chunks = _load_chunks(SOURCE_ID)

    query = build_query(issue, chunks)
    top_chunks = retrieve(query, chunks, top_k=5)

    print(f"retrieved {len(top_chunks)} chunks:")
    for c in top_chunks:
        print(f"  - {c['chunk_id']}")

    result = resolve_issue(issue, top_chunks)

    print("\n--- RESOLVE RESULT ---")
    print(json.dumps(result, indent=2))