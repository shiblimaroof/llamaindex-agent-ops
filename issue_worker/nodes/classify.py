"file 19 - classify node"

from __future__ import annotations
import json
import os
from dataclasses import dataclass
from enum import Enum
from groq import Groq

from issue_worker.config import load_resolver_config

# wherever the call site is:
MODEL = load_resolver_config().groq_model
class IssueCategory(str, Enum):
    BUG = "bug"
    FEATURE_REQUEST = "feature_request"
    DOCUMENTATION = "documentation"
    QUESTION = "question"
    OTHER = "other"


# Only these categories get a resolution attempt in Phase 1. Everything
# else routes straight to Escalate 
ACTIONABLE_CATEGORIES = {IssueCategory.BUG, IssueCategory.DOCUMENTATION}

CLASSIFY_SYSTEM_PROMPT = """you are classifying a Github issue for an automated resolution pipeline.

Read only the issue title and body below. Do not assume anything that is not stated in them.

Classify the issue into exactly one category:
- bug: something is broken, erroring, or behaving incorrectly
- feature_request: a request for new functionality that does not exist yet
- documentation: docs are missing, unclear, or incorrect
- question: the reporter is asking how to do something, not reporting a defect
- other: none of the above fit

Existing Github labels may be provided for reference only. 
They are sometimes wrong or missing - classify from the title and body, not the labels.

Respond with only a JSON object, no other text, no markdown fences:
{"category" : "<one of the five values above>", "reasoning" : "<one sentence, why this category>"}"""

@dataclass
class ClassificationResult:
    source_id: str
    category : IssueCategory
    is_actionable : bool
    reasoning : str

def classify_issue(source_id : str, title : str, body : str, labels : list[str], client : Groq | None = None) -> ClassificationResult:
    """
        Classify a single issue using Groq. Raises on malformed model output
        rather than silently defaulting to "other" - a silent default would
        hide classification failures instead of surfacing them for the
        Retry node to handle.
        """
    if client is None:
        client = Groq(api_key=os.environ["GROQ_API_KEY"])
        
    user_content = (
        f"Title: {title}\n\n"
        f"Body: {body}\n\n"
        f"Existing labels (reference only, may be inaccurate): {labels}"
    )

    response = client.chat.completions.create(
        model = MODEL,
        messages= [
            {"role" : "system", "content" : CLASSIFY_SYSTEM_PROMPT},
            {"role" : "user", "content" : user_content},
        ],
        temperature= 0,
        response_format= {"type" : "json_object"}
    )
    raw = response.choices[0].message.content
    parsed = json.loads(raw)

    category = IssueCategory(parsed["category"])
    is_actionable = category in ACTIONABLE_CATEGORIES

    return ClassificationResult(
        source_id= source_id,
        category=category,
        is_actionable=is_actionable,
        reasoning=parsed["reasoning"],
    )

def run_classification(
        golden_set_path: str = "data/golden_set.jsonl",
        raw_issues_path: str = "data/raw_issues.jsonl",
        output_path: str = "data/classifications.jsonl",
    ) -> None:

    raw_by_id = {}
    with open(raw_issues_path, "r") as f:
        for line in f:
            record = json.loads(line)
            raw_by_id[record["source_id"]] = record

    client = Groq(api_key=os.environ["GROQ_API_KEY"])

    with open(golden_set_path, "r") as gf, open(output_path, "a") as out:
        for line in gf:
            entry = json.loads(line)
            if not entry["passes_golden_filter"]:
                continue

            source_id = entry["source_id"]
            issue = raw_by_id[source_id]

            result = classify_issue(
                source_id=source_id,
                title=issue["title"],
                body=issue["body"],
                labels = issue.get("labels",[]),
                client=client,
            )  

            out.write(
                json.dumps(
                    {
                        "source_id" : result.source_id,
                        "category" : result.category.value,
                        "is_actionable" : result.is_actionable,
                        "reasoning" : result.reasoning,
                    }
                )
                +"\n"
            )  

if __name__ == "__main__":
    run_classification()

