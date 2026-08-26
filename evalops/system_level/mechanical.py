
"""
System-level evaluation tier for EvalOps. Answers "how did the agent
perform" (operational metrics), not "is the patch correct" (that's
regression-tier mechanical.py). All five metrics are pulled straight from
trace logs (data/pipeline_log.jsonl, data/llm_usage.jsonl) -- no LLM
judgment, consistent with mechanical-first design. """

import json
from pathlib import Path

PIPELINE_LOG_PATH = Path("data/pipeline_log.jsonl")
USAGE_LOG_PATH = Path("data/llm_usage.jsonl")

# Locked price table, USD per 1M tokens (input, output). Keyed by the exact
# model string written into llm_usage.jsonl's "model" field. Unknown models
# fall through to cost's honesty branch below rather than guessing a price.
PRICE_TABLE = {
    "llama-3.3-70b-versatile": {"input": 0.59, "output": 0.79},
    "gemini-3.6-flash": {"input": 1.50, "output": 7.50},
}

#Terminal node names whose presence in the trace means the pipeline
# reached an end state. Matches task_success's locked design: reaching
# escalate at all means Retry+Fallback both failed, even though escalate's
# own log line always shows outcome:"success" (its job is to categorize
# and notify, not to indicate pipeline success).
SUCCESS_TERMINAL_NODES = {"patch_application", "fallback"}
FAILURE_TERMINAL_NODE = "escalate"

def _read_log_lines(path : Path, run_id : str) -> list[dict]:
    """
    Reads a JSONL log file and returns all records matching run_id, in
    file order. run_id (not source_id) is the join key -- source_id alone
    isn't unique per execution, since the same issue gets rerun many times
    across dev/verification sessions.
    """
    if not path.exists():
        return []

    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if record.get("run_id") == run_id:
                records.append(record)
    return records

def task_success(run_id : str)->dict:
    """
    Derives pipeline success from the terminal node reached, not from any
    single node's own outcome field -- escalate's log line always shows
    outcome:"success" for itself (it always completes its categorize+notify
    job), which is not the same as the pipeline having resolved the issue.
 
    passed: True  -- patch_application or fallback logged outcome:"success"
    passed: False -- escalate was reached (Retry + Fallback both failed)
    passed: None  -- no terminal node found (crash/hang, orphaned partial
                     trace) -- same honesty principle as regression-tier's
                     test_passed not-applicable case, no guessed verdict
    """
    records = _read_log_lines(PIPELINE_LOG_PATH, run_id)

    for record in records:
        node_name = record.get("node_name")
        if node_name in SUCCESS_TERMINAL_NODES and record.get("outcome") == "success":
            return {
                "passed" : True,
                "detail" : f"terminal node '{node_name}' logged outcome:success"
            }
        if node_name == FAILURE_TERMINAL_NODE:
            return {
                "passed" : False,
                "detail" : "escalate reached -- retry and fallback both failed",
            }

    return {
        "passed" : None,
        "detail" : "no terminal node (patch_application/fallback/escalate) found in trace"
    }


def retries(run_id : str) -> dict:
    """
    Counts real retry attempts for this run, excluding the known
    zero-duration bookkeeping line -- the accepted quirk where a fully
    exhausted MAX_RETRY_ATTEMPTS=3 sequence produces 4 log lines (3 real
    attempts + 1 near-zero-duration bookkeeping call whose limit check
    happens at the top of the next recursive call). Counting the
    bookkeeping line would overstate real retry cost.
    """
    records = _read_log_lines(PIPELINE_LOG_PATH, run_id)

    retry_records = [r for r in records if r.get("node_name") == "retry"]
    real_attempts = [r for r in retry_records if r.get("duration_ms", 0) > 1]

    return {
        "count": len(real_attempts),
        "detail": f"{len(real_attempts)} real retry attempt(s) "
        f"({len(retry_records) - len(real_attempts)} bookkeeping line(s) excluded)",
    }


def latency(run_id: str) -> dict:
    """
    Sums duration_ms across every log line for this run -- node-level
    timing (orchestrator-owned, written by log.py), not per-LLM-call
    timing. Deliberately not split from cost's per-call token data in
    llm_usage.jsonl
    """
    records = _read_log_lines(PIPELINE_LOG_PATH, run_id)
 
    total_ms = sum(r.get("duration_ms", 0) for r in records)
 
    return {
        "count": total_ms,
        "detail": f"{total_ms:.1f}ms total across {len(records)} logged node(s)",
    }


def cost(run_id : str)-> dict:
    """
    Sums USD cost across every LLM call for this run, using PRICE_TABLE
    keyed by the exact model string in llm_usage.jsonl. Raises ValueError
    on any call whose model isn't in PRICE_TABLE, rather than silently
    excluding it -- an unpriced call would otherwise still produce a
    plausible-looking total that's quietly wrong (unlike task_success's
    None case, a wrong float has no visible signal something's off).
    Same reasoning as categorize_escalation()'s hard raise on unrecognized
    state: a stale PRICE_TABLE is a bug to surface immediately, not a
    data-quality blip to average away. Caller (e.g. run_batch.py) is
    expected to wrap per-issue calls in try/except, same isolation it
    already has for other per-issue failures.
    """

    records = _read_log_lines(USAGE_LOG_PATH, run_id)

    total_cost = 0.0

    for record in records:
        model = record.get("model")
        prices = PRICE_TABLE.get(model)
        if prices is None:
            raise ValueError(
                f"no price entry for model '{model}' (run_id={run_id}) -- "
                f"add it to PRICE_TABLE before computing cost"
            )

        prompt_tokens = record.get("prompt_tokens", 0)
        completion_tokens = record.get("completion_tokens", 0)
        total_cost += (prompt_tokens / 1_000_000) * prices["input"]
        total_cost += (completion_tokens / 1_000_000) * prices["output"]

    return {
        "count" : total_cost,
        "detail": f"${total_cost:.6f} across {len(records)} LLM call(s)",
    }

if __name__ == "__main__":
    test_run_id = "52fa6391-255e-4dc0-9279-f55299e23bbb"
    print("task_success:", task_success(test_run_id))
    print("retries:", retries(test_run_id))
    print("latency:", latency(test_run_id))
    print("cost:", cost(test_run_id))



