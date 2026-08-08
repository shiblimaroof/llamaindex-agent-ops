# llamaindex-agent-ops

An agentic pipeline that resolves real GitHub issues from the [llama_index](https://github.com/run-llama/llama_index) repository end-to-end — classifying, retrieving relevant code, generating a fix, applying it, retrying on failure, falling back to a second LLM provider, and escalating to a human when it genuinely can't solve the issue. Paired with EvalOps, a separate system that audits the pipeline's own execution traces for correctness — "ESLint for agent traces."

Built entirely on free-tier infrastructure: Groq (Llama 3.3 70B) as the primary model, Google Gemini as a fallback provider, FAISS/BM25/sentence-transformers for retrieval, and Slack for human notification.

## Why this exists

Most agentic-coding demos show the happy path: issue in, patch out. This project is built around the opposite question — what does an agent do when it *can't* solve something? Every failure mode here is handled explicitly and traced, not swallowed. The pipeline is designed so that when it fails, it fails legibly: you can always answer *why* a given issue didn't get resolved, using the same execution trace whether it succeeded or not.

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
| **Retrieve** | Builds a query from the issue and pulls the top-k most relevant code chunks from the repo, using dense (jina code-embeddings) + sparse (BM25) retrieval. |
| **Resolve** | The primary LLM (Groq/Llama 3.3 70B) proposes a fix as a set of edits, grounded against the retrieved chunks. Can report `insufficient_context` rather than guess. |
| **Patch Application** | Applies proposed edits to a git worktree. Verifies each edit's `old_source` matches the real file byte-for-byte before writing — catches hallucinated or malformed edits before they touch disk. Includes automatic rollback on partial-write failure. |
| **Retry** | If the patch fails for a retryable reason (`malformed_edit`, `stale_chunks`, `io_error`), resets the worktree and retries with the failure reason fed back into the prompt. Bounded (`MAX_RETRY_ATTEMPTS = 3`). Non-retryable failures (`dirty_worktree`, `rollback_failed`, `insufficient_context`) route straight to Escalate. |
| **Fallback** | If Retry exhausts, or the primary provider itself fails (`provider_error`), a second model (Gemini) gets one attempt at the same issue — no retry loop of its own. This is deliberately a *different model*, not just a second try, on the theory that a different provider's failure modes are more likely to be uncorrelated. |
| **Escalate** | Categorizes any unresolved issue (`infra_failure`, `guardrail_trip`, `capability_exhausted`) and produces a record for human review. Never silently drops a failure into an "unclassified" bucket — an unrecognized failure state raises loudly instead. |
| **Notify** | Sends the escalation record to Slack. Pluggable notification channel (currently Slack, designed so other channels like email could be added without touching Escalate). |
| **Log** | Every node writes one structured JSONL line per call — `node_name`, `source_id`, `outcome`, `failure_reason`, `timestamp`, `duration_ms`, plus `attempt` for Retry's per-attempt calls. This is the audit trail EvalOps reads. |

*Note: Escalate's `guardrail_trip` category (e.g. `unsafe_pattern_detected` from the patch safety check) isn't documented in this table yet — to be added.*

## Design decisions worth knowing about

- **Two-provider fallback is a real design choice, not redundancy.** In testing, Gemini successfully resolved an issue that Groq failed on 4 retry attempts in a row — not by trying harder, but by finding a different, better fix that avoided the exact code path Groq kept getting stuck on. This is the kind of result you only get from an actually different model, not a second attempt with the same one.
- **The grounding check in Patch Application is intentionally strict** (byte-for-byte match against the real source) rather than fuzzy. A fuzzy match risks silently applying a patch to the wrong location. When Resolve's proposed edit doesn't match, that's treated as Resolve's problem to fix on retry, not Patch Application's problem to work around.
- **`insufficient_context` never gets retried** — Resolve already runs its own internal follow-up-retrieval loop before giving up, so by the time it reports `insufficient_context`, retrying with the same retrieval strategy would be strictly worse than what already happened internally. It routes straight to Escalate instead.
- **Explicit dicts instead of framework "memory."** Retry context (what failed, why, which attempt) is threaded through as plain dictionaries, not a memory abstraction from an agent framework. This keeps every prompt's exact input auditable — you can always see precisely what the model was told on any given attempt.
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
data/
  golden_set.jsonl          # the fixed evaluation set
  pipeline_log.jsonl        # full execution trace, one line per node call
scripts/
  run_batch.py               # runs the pipeline across the whole golden set
  verify_patch_application_edge_cases.py
  golden_set_breakdown.py
```

## Status

Issue Worker's full pipeline (Classify through Log) is built, wired, and verified end-to-end — every stage, including Retry, Fallback, and Escalate, writes to `pipeline_log.jsonl`. EvalOps — the companion system that scores the traces this pipeline produces — is in progress.
