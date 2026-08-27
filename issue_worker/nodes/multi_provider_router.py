
"""
Fires when retry_issue() returns "provider_error"
(Groq itself failed) or "retry_exhausted" (Groq's retries ran out) --
switches to Gemini free tier via its OpenAI-compatible endpoint, attempts
one resolution + patch application pass, no retry loop of its own.
 
Reuses resolve.py's prompt-building and response-parsing logic directly
(RESOLVE_SYSTEM_PROMPT, _build_user_prompt, _strip_markdown_fence,
_validate_response, _honest_failure) rather than duplicating it -- only
the client construction and the model call differ from resolve_issue().
resolve.py itself is untouched by this file.
"""
 
import json
import os
 
from openai import OpenAI

from issue_worker.usage_logger import log_usage
 
from issue_worker.nodes.resolve import (
    RESOLVE_SYSTEM_PROMPT,
    _build_user_prompt,
    strip_markdown_fence,
    _validate_response,
    _honest_failure,
)
from issue_worker.nodes.patch_application import apply_patch
 
GEMINI_MODEL =  "gemini-3.6-flash"
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
 
 
def resolve_issue_gemini(issue: dict, chunks: list[dict], source_id: str, run_id: str,retry_context: dict | None = None) -> dict:
    """
    Same input/output contract as resolve.py's resolve_issue() -- same
    locked return shape (edits/summary/insufficient_context/follow_up_query,
    plus the failure_reason key added for the api_error fix). Only the
    provider differs: Gemini via its OpenAI-compatible endpoint instead of
    Groq directly.
    """
    client = OpenAI(api_key=os.environ["GEMINI_API_KEY"], base_url=GEMINI_BASE_URL)
    user_prompt = _build_user_prompt(issue, chunks, retry_context)
    valid_chunk_ids = {c["chunk_id"] for c in chunks}
 
    try:
        response = client.chat.completions.create(
            model=GEMINI_MODEL,
            messages=[
                {"role": "system", "content": RESOLVE_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0,
        )
    except Exception as e:
        return _honest_failure(f"Gemini API call failed: {e}", failure_reason="api_error")

    log_usage(
        node_name="resolve_gemini",
        provider="gemini",
        model=GEMINI_MODEL,
        source_id=issue.get("source_id", "unknown"),
        run_id = run_id,
        prompt_tokens=response.usage.prompt_tokens,
        completion_tokens=response.usage.completion_tokens,
    )

 
    raw = response.choices[0].message.content
    raw = strip_markdown_fence(raw)
 
    try:
        parsed = json.loads(raw, strict=False)
    except Exception as e:
        return _honest_failure(f"failed to parse Gemini response as JSON: {e}")
 
    return _validate_response(parsed, valid_chunk_ids)
 
 
def fallback_issue(
    issue: dict,
    chunks: list[dict],
    source_id: str,
    run_id: str,
    original_failure_reason: str,
) -> dict:
    """
    Entry point, called when retry_issue() returns outcome "provider_error"
    or "retry_exhausted". One-shot: no retry loop against Gemini, matches
    the locked design (Fallback is a single second attempt with a
    different provider, not a second bounded retry sequence).

    original_failure_reason is why Retry gave up (the outcome/reason that
    triggered this call) -- preserved on failure so downstream Escalate
    categorization isn't blind to what actually failed.

    """
    new_resolve_output = resolve_issue_gemini(issue, chunks, source_id,run_id)
    new_patch_result = apply_patch(new_resolve_output, chunks, source_id)

    if new_patch_result["applied"]:
        return {
            "outcome": "applied",
            "resolve_output": new_resolve_output,
            "patch_result": new_patch_result,
        }

    fallback_failure_reason = (
        new_resolve_output.get("failure_reason")
        if new_resolve_output.get("insufficient_context")
        else new_patch_result.get("failure_reason")
    )

    return {
        "outcome": "fallback_failed",
        "resolve_output": new_resolve_output,
        "patch_result": new_patch_result,
        "original_failure_reason": original_failure_reason,
        "fallback_failure_reason": fallback_failure_reason,
    }
 
