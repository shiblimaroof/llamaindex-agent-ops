"""
Builds the system/user prompt pair for Step B judge calls.
 
Derives the patch diff itself rather than depending on Patch Application to
supply one -- Patch Application applies changes directly to the worktree and
does not store diff text anywhere. Reuses the same git-diff approach
run_error_handling_check already relies on (regression/mechanical.py's
_changed_python_files + _run_git), so this file introduces no new way of
reading the worktree.
 
regression_results (Step A's run_regression_checks output) is passed through
in full, not filtered to only the fields the three correctness booleans and
weakens_error_handling need -- unexplained_concern exists precisely to catch
things a fixed taxonomy of questions doesn't anticipate, so hiding any of
Step A's findings from the judge would undercut that.
"""

import json

from evalops.regression.mechanical import _changed_python_files, _run_git

SYSTEM_PROMPT = """You are a code review judge evaluating whether an automated patch correctly resolves a reported GitHub issue.
 
You will be given:
- The original issue text
- The diff of the patch that was applied
- The retrieved context: code chunks that were available to the resolver when it wrote the patch
- Results from a set of mechanical and hybrid checks that already ran against the patch (regression_results)
 
Answer the following as a single JSON object, with no markdown code fence and no text outside the JSON object:
 
{
  "addresses_root_cause": bool,
  "handles_reported_case": bool,
  "avoids_described_failure_mode": bool,
  "weakens_error_handling": bool,
  "unexplained_concern": bool,
  "unexplained_concern_note": string,
  "context_faithfulness": bool,
  "reasoning_relevancy": bool,
  "reasoning": string
}
 
Field meanings:
- addresses_root_cause: does the patch fix the underlying cause of the issue, not just a symptom?
- handles_reported_case: does the patch handle the specific case described in the issue?
- avoids_described_failure_mode: does the patch avoid reintroducing or leaving open the failure mode described in the issue?
- weakens_error_handling: does the patch remove, weaken, or downgrade any error handling (raises, logging, validation)? Use regression_results' error-handling findings as a starting signal, but judge independently.
- unexplained_concern: is there anything about this patch that concerns you which is not captured by the fields above? This is a deliberately open-ended catch-all -- do not leave it false by default. Set it true whenever something feels off, even if you can't fully articulate why.
- unexplained_concern_note: always include this key. If unexplained_concern is false, use an empty string. If true, explain the concern in plain language.
- context_faithfulness: is the patch grounded in the retrieved context? Set this false if the patch references functions, files, or behavior that are not present anywhere in the retrieved context -- that indicates the resolver invented something rather than working from what it was given.
- reasoning_relevancy: was the retrieved context actually relevant to resolving this issue? Set this false if the retrieved chunks are mostly unrelated to the issue's root cause -- this reflects retrieval quality, not the patch itself.
- reasoning: your freeform reasoning for all of the above, in a few sentences.
 
Do not include a "resolves_issue" field -- it is computed separately from your other answers and must not be supplied by you.
 
Base your judgment on the issue text, the diff, and the retrieved context. Use regression_results as supporting evidence, not as a substitute for your own reasoning."""


def _build_patch_diff(worktree_path : str, base_ref : str)-> str:

    """Concatenates per-file diffs for every changed Python file in the
    worktree, same file-scoped git diff approach run_error_handling_check
    already uses. Returns an explanatory string instead of raising when
    there are no changed files (a genuinely empty diff is worth surfacing
    to the judge, not treated as an error)."""

    changed_files = _changed_python_files(worktree_path,base_ref)
    if not changed_files:
        return "(no changed python files found in worktree)"

    diffs = []
    for file_path in changed_files:
        diff_text = _run_git(worktree_path,["diff", base_ref,"--",file_path])
        diffs.append(f"---{file_path}---\n{diff_text}")

    return "\n\n".join(diffs)

def _summarize_chunk(chunk: dict) -> dict:
    return {
        "file_path": chunk["file_path"],
        "name": chunk["name"],
        "source": chunk["source"],
    }
   

def build_judge_prompt(
        worktree_path: str,
        issue_body: str,
        regression_results: dict,
        top_chunks: list[dict],
        base_ref: str = "HEAD",
    ) -> tuple[str, str]:

    patch_diff = _build_patch_diff(worktree_path, base_ref)
    regression_results_json = json.dumps(regression_results, indent=2)
    context_json = json.dumps([_summarize_chunk(c) for c in top_chunks], indent=2)

    user_prompt = f"""## Issue
    
    {issue_body}
    

    ##Patch diff

    {patch_diff}

    ## Retrieved context (chunks the resolver had available)

    {context_json}

    ## Regression check results (from Step A)

    {regression_results_json}"""

    return SYSTEM_PROMPT, user_prompt