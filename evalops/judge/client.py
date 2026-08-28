"""

Thin OpenRouter API wrapper for Step B judge calls. Sends a system/user
prompt pair to nvidia/nemotron-3-ultra-550b-a55b:free via OpenRouter's
OpenAI-compatible endpoint and returns parsed, schema-validated JSON.

Judge model deliberately shares no family with either resolver model
(llama-3.3-70b via Groq, gemini-3.6-flash via Gemini Fallback) -- avoids
self-preference bias, where a model rates its own family's output more
favorably.

Follows fallback.py's resolve_issue_gemini pattern (OpenAI client + custom
base_url, log_usage after the call, strip_markdown_fence before parsing).

Deliberately does NOT return a failure dict like resolve_gemini does.
A resolver failure has designed downstream handling (Retry/Fallback); a
judge failure has none, so this raises JudgeCallError instead -- a silent
failure dict here would undercut why unexplained_concern exists. Locked.
"""


import json
import os

from openai import OpenAI

from issue_worker.usage_logger import log_usage
from issue_worker.nodes.resolve import strip_markdown_fence
from evalops.judge.schema import validate, JudgeSchemaError

OPENROUTER_MODEL = "nvidia/nemotron-3-ultra-550b-a55b:free"
BASE_URL = "https://openrouter.ai/api/v1"

# Malformed-JSON and wrong-shape retries only -- one retry, same API call
# repeated. Does not cover API-level exceptions (network/auth/rate-limit
# errors), which raise immediately with no retry since a second identical
# call is unlikely to succeed where the first failed for a non-parsing
# reason.
MAX_JSON_PARSE_ATTEMPTS = 2


class JudgeCallError(RuntimeError):
    """Raised on any Nvidia judge-call failure -- API error, malformed JSON,
    or a schema-invalid response that didn't resolve within
    MAX_JSON_PARSE_ATTEMPTS. Never caught and converted into a fallback
    value inside this file; the caller (Step B runner) decides what to do
    when a case can't be judged."""


def call_judge(
    system_prompt: str,
    user_prompt: str,
    source_id: str,
    run_id: str,
) -> dict:
    """
    Sends one system/user prompt pair to Nvidia's judge model and returns
    the parsed, schema-validated JSON response body (including the derived
    resolves_issue field -- see schema.py).

    Raises JudgeCallError on:
      - the API call itself failing (network, auth, rate limit, etc.)
      - the response not being valid JSON after MAX_JSON_PARSE_ATTEMPTS
        attempts
      - the response being valid JSON but failing schema validation
        (missing/mistyped fields, or a judge-supplied resolves_issue)
        after MAX_JSON_PARSE_ATTEMPTS attempts

    source_id and run_id are passed through only for usage logging -- same
    as every other node_name in usage_logger.py's convention.
    """
    client = OpenAI(api_key=os.environ["OPENROUTER_API_KEY"], base_url=BASE_URL)

    last_error: Exception | None = None
    last_raw: str | None = None

    for attempt in range(1, MAX_JSON_PARSE_ATTEMPTS + 1):
        try:
            response = client.chat.completions.create(
                model=OPENROUTER_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0,
            )
        except Exception as e:
            raise JudgeCallError(f"nvidia API call failed: {e}") from e

        log_usage(
            node_name="judge_nvidia",
            provider="nvidia",
            model=OPENROUTER_MODEL,
            source_id=source_id,
            run_id=run_id,
            prompt_tokens=response.usage.prompt_tokens,
            completion_tokens=response.usage.completion_tokens,
        )

        raw = strip_markdown_fence(response.choices[0].message.content)
        last_raw = raw

        try:
            parsed = json.loads(raw, strict=False)
        except Exception as e:
            last_error = e
            continue

        try:
            return validate(parsed)
        except JudgeSchemaError as e:
            last_error = e
            continue

    raise JudgeCallError(
        f"failed to obtain a schema-valid Nvidia judge response after "
        f"{MAX_JSON_PARSE_ATTEMPTS} attempts: {last_error}. "
        f"Last raw response: {last_raw!r}"
    )