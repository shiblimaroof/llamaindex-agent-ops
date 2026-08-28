
"""
Step B runner. Thin coordinator tying prompt.py + client.py + schema.py
together -- mirrors orchestrator.py's shape for Step A: build inputs, call
out, return the result. No error handling of its own; JudgeCallError from
client.py (which already covers API failures, malformed JSON, and
schema-invalid responses) propagates uncaught. A judge failure has no
designed downstream handling at this layer either -- same reasoning as
client.py's decision not to return a failure dict.
"""
 
from evalops.judge.prompt import build_judge_prompt
from evalops.judge.client import call_judge
 
 
def run_judge(
    worktree_path: str,
    issue_body: str,
    regression_results: dict,
    source_id: str,
    run_id: str,
    base_ref: str = "HEAD",
) -> dict:
    """
    Runs the full Step B judge call for one case: builds the judge prompt
    from the worktree diff + issue body + Step A's regression_results, then
    calls the judge and returns its schema-validated response (including
    the derived resolves_issue field).
 
    Raises JudgeCallError (from client.py) if the judge call fails for any
    reason -- caller decides what to do when a case can't be judged.
    """
    system_prompt, user_prompt = build_judge_prompt(
        worktree_path=worktree_path,
        issue_body=issue_body,
        regression_results=regression_results,
        base_ref=base_ref,
    )
 
    return call_judge(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        source_id=source_id,
        run_id=run_id,
    )
 
