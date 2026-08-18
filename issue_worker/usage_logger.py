"""
Writes one JSONL line per LLM call to data/llm_usage.jsonl, tracking token
usage per node/provider/source_id. Locked schema (see llamaindex-agent-ops
project notes): {node_name, provider, source_id, timestamp}, extended here
with prompt_tokens/completion_tokens/total_tokens.

This file only writes -- it does not call any LLM API itself. Callers
(resolve.py's Groq call site, multi_provider_router.py's Gemini call site)
extract prompt_tokens/completion_tokens from their own response object and
pass them in here, same reason log.py stays a pure formatter/writer rather
than reaching into caller state: keeps this module provider-agnostic, since
Groq and Gemini's response.usage shapes aren't guaranteed identical.

"""


import json
import time
from pathlib import Path
import uuid

USAGE_LOG_PATH = Path("data/llm_usage.jsonl")

def log_usage(
        node_name : str,
        provider : str,
        model : str,
        source_id : str,
        run_id : str,

        prompt_tokens : int,
        completion_tokens : int,
        ) -> None:

    """
    Appends one usage record to data/llm_usage.jsonl.

    prompt_tokens/completion_tokens are whatever the caller's SDK response
    reports -- this function does no validation beyond int conversion, since
    a malformed usage object at the call site is that call site's bug, not
    this writer's job to catch.
    """

    record = {
        "node_name" : node_name,
        "provider" : provider,
        "model" : model,
        "source_id" : source_id,
        "run_id" : run_id,
        "timestamp" : time.time(),
        "prompt_tokens" : int(prompt_tokens),
        "completion_tokens" : int(completion_tokens),
        "total_tokens" : int(prompt_tokens) + int(completion_tokens),
    }

    USAGE_LOG_PATH.parent.mkdir(parents = True, exist_ok= True)
    with open(USAGE_LOG_PATH, "a") as f:
        f.write(json.dumps(record) +"\n")


if __name__ == "__main__":
    log_usage(
        node_name="resolve",
        provider="groq",
        model="llama-3.3-70b-versatile",
        source_id="test_123",
        prompt_tokens=150,
        completion_tokens=40,
    )
    print("Wrote test record to", USAGE_LOG_PATH)

    with open(USAGE_LOG_PATH) as f:
        last_line = f.readlines()[-1]
    print(last_line)
