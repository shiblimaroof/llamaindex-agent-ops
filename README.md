# llamaindex-agent-ops

An agentic pipeline that resolves real GitHub issues from the [llama_index](https://github.com/run-llama/llama_index) repository end-to-end — classifying, retrieving relevant code, generating a fix, applying it, retrying on failure, falling back to a second LLM provider, and escalating to a human when it genuinely can't solve the issue. Paired with EvalOps, a separate system that audits the pipeline's own execution traces for correctness — "ESLint for agent traces."

Built entirely on free-tier infrastructure: Groq (Llama 3.3 70B) as the primary model, Google Gemini as a fallback provider, FAISS/BM25/sentence-transformers for retrieval, and Slack for human notification.

## Why this exists

Most agentic-coding demos show the happy path: issue in, patch out. This project is built around the opposite question — what does an agent do when it *can't* solve something? Every failure mode here is handled explicitly and traced, not swallowed. The pipeline is designed so that when it fails, it fails legibly: you can always answer *why* a given issue didn't get resolved, using the same execution trace whether it succeeded or not.

A second, harder question sits behind that: how do you know the pipeline is telling the truth about succeeding? A pipeline that grades its own homework can report `applied` on a patch that's actually wrong. EvalOps exists to answer that question independently — see below.

## Pipeline architecture

```
Classify → Retrieve → Resolve → Patch Application → Retry → Fallback → Escalate → Notify
    ↓          ↓          ↓             ↓              ↓        ↓          ↓         ↓
    └──────────┴──────────┴─────────────┴──────────────┴────────┴──────────┴─────────┘
                                         ↓
                                   pipeline_log.jsonl
```

Every stage writes a structured trace line to `data/pipeline_log.jsonl`, so the full path any issue took through the pipeline can be reconstructed after the fact.

| Stage | What it does |
|---|---|
| **Classify** | Determines whether an issue is actionable (a real code-level bug/fix) or should be routed away (feature request, question, duplicate, etc). |
| **Retrieve** | Builds a query from the issue and pulls the top-k most relevant code chunks from the repo, using dense (jina code-embeddings) + sparse (BM25) retrieval, with a cross-encoder reranker on top. |
| **Resolve** | The primary LLM (Groq/Llama 3.3 70B) proposes a fix as a set of edits, grounded against the retrieved chunks. Can report `insufficient_context` rather than guess. |
| **Patch Application** | Applies proposed edits to a git worktree. Verifies each edit's `old_source` matches the real file byte-for-byte before writing — catches hallucinated or malformed edits before they touch disk. Includes a guardrail scan (regex + AST) for unsafe patterns like `os.system()` or `shell=True`, and automatic rollback on partial-write failure. |
| **Retry** | If the patch fails for a retryable reason (`malformed_edit`, `stale_chunks`, `io_error`), resets the worktree and retries with the failure reason fed back into the prompt. Bounded (`MAX_RETRY_ATTEMPTS = 3`). Non-retryable failures (`dirty_worktree`, `rollback_failed`, `insufficient_context`, `unsafe_pattern_detected`) route straight to Escalate. |
| **Fallback** | If Retry exhausts, or the primary provider itself fails (`provider_error`), a second model (Gemini) gets one attempt at the same issue — no retry loop of its own. This is deliberately a *different model*, not just a second try, on the theory that a different provider's failure modes are more likely to be uncorrelated. |
| **Escalate** | Categorizes any unresolved issue — `infra_failure`, `guardrail_trip` (an unsafe code pattern was caught before ever touching disk), `capability_exhausted`, or `context_exhausted` (retrieval never found enough to act on) — and produces a record for human review. Never silently drops a failure into an "unclassified" bucket — an unrecognized failure state raises loudly instead. |
| **Notify** | Sends the escalation record to Slack. Pluggable notification channel (currently Slack, designed so other channels like email could be added without touching Escalate). |
| **Log** | Every node writes one structured JSONL line per call — `node_name`, `source_id`, `outcome`, `failure_reason`, `timestamp`, `duration_ms`, plus `attempt` for Retry's per-attempt calls. This is the audit trail EvalOps reads. |

## EvalOps

Issue Worker's own reported outcome (`applied`, `escalated`, etc.) is not treated as ground truth. EvalOps independently re-checks every patch the pipeline claims to have applied, using a mix of deterministic checks and an LLM judge — the design principle being that anything code can verify deterministically should never be left to an LLM's judgment.

**Three tiers of checks, run as peers:**

| Tier | What it checks | How |
|---|---|---|
| **Regression** | Does the patch break anything? Signature changes, dependency changes, syntax validity, import validity, unused code, files touched outside the issue's stated scope, and weakened error handling. | Mostly mechanical (no LLM). One check (`removes_weakens_error_handling`) is hybrid — mechanical detection feeds into the judge's reasoning. |
| **Correctness** | Does the patch actually fix the issue? Root cause addressed, reported case handled, described failure mode avoided — combined into a single `resolves_issue` verdict. Also checks whether the fix is grounded in the retrieved code (`context_faithfulness`) and whether the model's stated reasoning matches what it actually did (`reasoning_relevancy`). | Fully LLM-judged — this requires semantic understanding, not pattern matching. |
| **System-level** | Operational health: task success rate, latency, cost, tool calls, retries. | Mechanical, no LLM. |

A fourth field, `unexplained_concern`, is a deliberately open-ended catch-all — the judge can flag something as worth a human's attention even if it doesn't fit any of the structured categories above. It routes straight to human review regardless of how confident the rest of the verdict is.

**Real result:** on issue 22068, Issue Worker reported `outcome: applied`. EvalOps's judge disagreed — `resolves_issue: false`. The shipped patch had over-corrected: it removed an entire conditional the original bug report never asked to touch, breaking round-trip message conversion for a valid case the pipeline never tested against. This is the exact failure mode the two-stage design exists to catch — a pipeline grading its own homework will miss a regression like this every time, because from the inside, the patch *looked* like a clean success.

A second real gap was found the same way on issue 21582: the shipped patch only fixed half of what the issue asked for (the non-streaming code path), silently leaving the streaming path — which the issue explicitly named — untouched.

**Current coverage:** running against a growing slice of the 59-issue golden set. 20 issues have real worktrees so far, 10 of those have been judged, and both real bugs above were caught in that first batch. Validation labeling (an independent human review of the judge's verdicts) has been completed on all 10 — full agreement, no overturns.

## Design decisions worth knowing about

- **Two-provider fallback is a real design choice, not redundancy.** In testing, Gemini successfully resolved an issue that Groq failed on 4 retry attempts in a row — not by trying harder, but by finding a different, better fix that avoided the exact code path Groq kept getting stuck on. This is the kind of result you only get from an actually different model, not a second attempt with the same one.
- **The grounding check in Patch Application is intentionally strict** (byte-for-byte match against the real source) rather than fuzzy. A fuzzy match risks silently applying a patch to the wrong location. When Resolve's proposed edit doesn't match, that's treated as Resolve's problem to fix on retry, not Patch Application's problem to work around.
- **`insufficient_context` never gets retried** — Resolve already runs its own internal follow-up-retrieval loop before giving up, so by the time it reports `insufficient_context`, retrying with the same retrieval strategy would be strictly worse than what already happened internally. It routes straight to Escalate instead.
- **Explicit dicts instead of framework "memory."** Retry context (what failed, why, which attempt) is threaded through as plain dictionaries, not a memory abstraction from an agent framework. This keeps every prompt's exact input auditable — you can always see precisely what the model was told on any given attempt.
- **EvalOps deliberately keeps its LLM judge small.** Every check that code can answer deterministically (syntax, imports, signatures, scope) is mechanical, not an LLM call. The judge is reserved for genuinely semantic questions — is this the right fix, not just a valid-looking one.
- **This is a batch evaluation system, not live production monitoring.** It runs against a fixed golden set of real llama_index issues with held-out unseen slices for scoring — not a continuously running service watching a live issue tracker.

## Project structure

```
issue_worker/
  orchestrator.py          # run_pipeline(source_id) — wires every node together
  nodes/
    classify.py
    resolve.py
    patch_application.py
    retry.py
    multi_provider_router.py   # Fallback (Gemini)
    escalate.py
    log.py
  retrieval/
    checkout.py
    chunker.py
    query_builder.py
    retriever.py
  notify.py                 # Slack notification channel
  config.py
evalops/
  regression/               # mechanical + hybrid checks (scope, error handling, etc.)
  judge/                    # LLM judge — client, schema, prompt, runner
  system_level/             # task_success, latency, cost, retries
data/
  golden_set.jsonl          # the fixed evaluation set
  pipeline_log.jsonl        # full execution trace, one line per node call
  batch_results.jsonl       # EvalOps judge output per issue
  validation_labels.jsonl   # human-reviewed labels for judge verdicts
scripts/
  run_batch.py               # runs Issue Worker across the golden set
  batch_run_evalops.py       # runs EvalOps against all worktrees with real patches
  verify_patch_application_edge_cases.py
  golden_set_breakdown.py
```
