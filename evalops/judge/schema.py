"""

Single source of truth for what a valid judge response looks like.
Used by judge/client.py's JSON-parse retry loop to distinguish
"syntactically valid JSON" from "the right JSON."
"""

from typing import Any, Dict


class JudgeSchemaError(Exception):
    """Raised when a parsed judge response doesn't match the locked schema.

    Distinct from JudgeCallError (client.py) so logs/debugging can tell
    apart "API/JSON layer failed" from "valid JSON, wrong shape."
    """
    pass


# Fields the judge itself must produce. resolves_issue is deliberately
# excluded here — it's derived, not judge-supplied (see validate()).
REQUIRED_FIELDS = {
    "addresses_root_cause": bool,
    "handles_reported_case": bool,
    "avoids_described_failure_mode": bool,
    "weakens_error_handling": bool,
    "unexplained_concern": bool,
    "unexplained_concern_note": str,
    "reasoning": str,
}

DERIVED_FIELD = "resolves_issue"


def validate(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Validate a parsed judge response against the locked schema.

    Checks all required keys are present with correct types, rejects
    the response outright if the judge also supplied its own
    `resolves_issue` (that field must only ever be computed here — a
    judge-supplied value signals prompt drift, not something to
    silently overwrite), then returns a new dict with `resolves_issue`
    added.

    Raises JudgeSchemaError on any violation. Does not mutate `raw`.
    """
    if not isinstance(raw, dict):
        raise JudgeSchemaError(f"Judge response is not a JSON object: {type(raw).__name__}")

    missing = [field for field in REQUIRED_FIELDS if field not in raw]
    if missing:
        raise JudgeSchemaError(f"Judge response missing required fields: {missing}")

    wrong_type = []
    for field, expected_type in REQUIRED_FIELDS.items():
        if not isinstance(raw[field], expected_type):
            wrong_type.append(
                f"{field} (expected {expected_type.__name__}, got {type(raw[field]).__name__})"
            )
    if wrong_type:
        raise JudgeSchemaError(f"Judge response has wrong field types: {wrong_type}")

    if DERIVED_FIELD in raw:
        raise JudgeSchemaError(
            f"Judge response must not include '{DERIVED_FIELD}' — it is derived, "
            f"not judge-supplied. Its presence indicates prompt drift."
        )

    result = dict(raw)
    result[DERIVED_FIELD] = (
        raw["addresses_root_cause"]
        and raw["handles_reported_case"]
        and raw["avoids_described_failure_mode"]
    )
    return result