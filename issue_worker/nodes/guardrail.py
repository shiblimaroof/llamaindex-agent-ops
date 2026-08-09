"""

Hooked in right after the grounding check passes and before the patch is
written to disk. Scans the proposed `new_source` for dangerous patterns
using two layers, both always run (neither is a fallback for the other --
they catch different failure shapes):

  1. Regex layer   -- textual presence (fast, catches literal/obfuscated
                       string forms, works even on unparseable text).
  2. AST layer      -- structural matching (catches multi-line calls,
                       reordered kwargs, anything regex misses because
                       it's spread across syntax rather than one line).
                       This mirrors how Semgrep/Bandit actually operate in
                       production: AST *pattern* matching, not full
                       dataflow/taint tracking. A `subprocess.run(cmd,
                       shell=True)` call is matched by walking Call nodes
                       and checking for a `shell=True` keyword regardless
                       of argument order or line breaks -- no tracking of
                       where `cmd`'s value came from.

Both layers are deterministic and auditable -- no LLM judge, no dataflow
analysis -- so EvalOps can re-verify the same result on the same input
every time.

On a match, Patch Application should return failure_reason="unsafe_pattern_detected",
non-retryable (same bucket as dirty_worktree/rollback_failed), routing to
Escalate -> guardrail_trip.
"""

import ast
import re

# Priority order matters: first category (top of this list) wins if more
# than one rule trips on the same patch, regardless of which layer (regex
# or AST) caught it. Credential exfiltration ranked above generic
# destructive commands, since it's the higher-urgency case for whoever's
# triaging in Slack.

_CATEGORY_PRIORITY = [
    "credential_exfiltration",
    "arbitrary_code_execution",
    "destructive_shell_command",
]

# --- Layer 1: regex rules (textual presence) ---

_SINGLE_PATTERN_RULES = [
    # destructive_shell_command
    # Each pattern tolerates both shell-string form ("rm -rf path") and
    # list-arg form (["rm", "-rf", path]) via a [\s"',]+ separator between
    # command and flag. Leading \b anchors the command name so it can't
    # match as a substring of an unrelated identifier (e.g. "confirm-rf").
    ("destructive_shell_command", r"\brm[\s\"',]+-rf\b"),
    ("destructive_shell_command", r"\brm[\s\"',]+-f\b.*\*"),
    ("destructive_shell_command", r"\bgit[\s\"',]*push[\s\"',]+(--force|-f)\b"),
    ("destructive_shell_command", r"\bgit[\s\"',]*reset[\s\"',]+--hard\b"),
    ("destructive_shell_command", r"\bdd[\s\"',]*if="),
    ("destructive_shell_command", r"\bmkfs\b"),
    ("destructive_shell_command", r">\s*/dev/sd[a-z]"),
    ("destructive_shell_command", r"\bchmod[\s\"',]+-R[\s\"',]+777\b"),
    ("destructive_shell_command", r"\bchown[\s\"',]+-R\b"),

    # credential_exfiltration (hardcoded secret shapes)
    ("credential_exfiltration", r"AKIA[0-9A-Z]{16}"),
    ("credential_exfiltration", r"sk-[a-zA-Z0-9]{20,}"),
    ("credential_exfiltration", r"api_key\s*=\s*[\"'][^\"']{20,}[\"']"),

    # arbitrary_code_execution
    ("arbitrary_code_execution", r"\beval\s*\("),
    ("arbitrary_code_execution", r"\bexec\s*\("),
    ("arbitrary_code_execution", r"\bos\.system\s*\("),
    ("arbitrary_code_execution", r"\bpickle\.loads\s*\("),
    ("arbitrary_code_execution", r"__import__\s*\("),
]

# (category, pattern_a, pattern_b, window_lines)
_CO_OCCURRENCE_RULES = [
    (
        "credential_exfiltration",
        r"os\.environ|os\.getenv|\.env\b",
        r"requests\.(get|post)|urllib|socket\.|print\s*\(|log\w*\.\w+\s*\(",
        10,
    ),
    (
        "arbitrary_code_execution",
        r"\bsubprocess\.",
        r"shell\s*=\s*True",
        3,
    ),
]

# --- Layer 2: AST rules (structural presence) ---
# Dangerous call names, keyed by category. Matched against Call nodes'
# resolved function name (handles both `eval(...)` and `os.system(...)`
# shapes), regardless of line breaks or argument order.

_AST_DANGEROUS_CALLS = {
    "eval": "arbitrary_code_execution",
    "exec": "arbitrary_code_execution",
    "os.system": "arbitrary_code_execution",
    "pickle.loads": "arbitrary_code_execution",
    "__import__": "arbitrary_code_execution",
}

_AST_ENVIRON_NAMES = {"os.environ", "os.getenv"}
_AST_NETWORK_OR_LOG_CALLS = {
    "requests.get", "requests.post", "urllib.request.urlopen",
    "socket.socket", "print", "logging.info", "logging.warning",
    "logging.error", "logging.debug",
}


def _call_name(node: ast.Call) -> str | None:
    """Resolve a Call node's function to a dotted name string, e.g.
    'os.system' for `os.system(...)`, or 'eval' for `eval(...)`.
    Returns None for call shapes we don't resolve (e.g. calling the
    result of another call)."""
    func = node.func
    parts = []
    while isinstance(func, ast.Attribute):
        parts.append(func.attr)
        func = func.value
    if isinstance(func, ast.Name):
        parts.append(func.id)
        return ".".join(reversed(parts))
    return None


def _attr_name(node: ast.Attribute) -> str:
    """Resolve an Attribute node to a dotted name, e.g. 'os.environ'
    for the `os.environ` access (not a call -- just the attribute)."""
    parts = [node.attr]
    val = node.value
    while isinstance(val, ast.Attribute):
        parts.append(val.attr)
        val = val.value
    if isinstance(val, ast.Name):
        parts.append(val.id)
    return ".".join(reversed(parts))


def _has_shell_true_kwarg(node: ast.Call) -> bool:
    for kw in node.keywords:
        if kw.arg == "shell" and isinstance(kw.value, ast.Constant) and kw.value.value is True:
            return True
    return False


def _ast_scan(new_source: str) -> tuple[dict, bool]:
    """
    Structural scan via the AST. Returns (hits, parse_ok).
    parse_ok=False means new_source wasn't parseable Python -- the AST
    layer is skipped for this patch, but that's reported rather than
    silently swallowed, since it's a real coverage gap worth knowing about.
    """
    try:
        tree = ast.parse(new_source)
    except SyntaxError:
        return {}, False

    hits = {}
    environ_lines = []
    network_lines = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute):
            if _attr_name(node) in _AST_ENVIRON_NAMES:
                environ_lines.append(node.lineno)
            continue

        if not isinstance(node, ast.Call):
            continue

        name = _call_name(node)
        if name is None:
            continue

        if name in _AST_ENVIRON_NAMES:
            environ_lines.append(node.lineno)

        if name in _AST_DANGEROUS_CALLS:
            category = _AST_DANGEROUS_CALLS[name]
            hits.setdefault(
                category,
                f"AST match: call to '{name}' at line {node.lineno}",
            )

        if name.startswith("subprocess.") and _has_shell_true_kwarg(node):
            hits.setdefault(
                "arbitrary_code_execution",
                f"AST match: '{name}' with shell=True kwarg at line {node.lineno} "
                f"(detected regardless of argument order/line breaks)",
            )

        if name in _AST_NETWORK_OR_LOG_CALLS:
            network_lines.append(node.lineno)

    if "credential_exfiltration" not in hits and environ_lines and network_lines:
        for le in sorted(environ_lines):
            for ln in sorted(network_lines):
                if le <= ln <= le + 10:
                    hits["credential_exfiltration"] = (
                        f"AST co-occurrence: environ read at line {le} followed by "
                        f"network/log call at line {ln} (within 10 lines)"
                    )
                    break
            if "credential_exfiltration" in hits:
                break

    return hits, True


def _find_line_matches(pattern: str, lines: list) -> list:
    regex = re.compile(pattern)
    return [i for i, line in enumerate(lines) if regex.search(line)]


def _regex_scan(new_source: str) -> dict:
    lines = new_source.splitlines()
    hits = {}

    for category, pattern in _SINGLE_PATTERN_RULES:
        if category in hits:
            continue
        if re.search(pattern, new_source):
            hits[category] = f"matched pattern: {pattern}"

    for category, pattern_a, pattern_b, window in _CO_OCCURRENCE_RULES:
        if category in hits:
            continue
        lines_a = _find_line_matches(pattern_a, lines)
        lines_b = _find_line_matches(pattern_b, lines)
        if not lines_a or not lines_b:
            continue
        for la in lines_a:
            for lb in lines_b:
                if la <= lb <= la + window:
                    hits[category] = (
                        f"co-occurrence: '{pattern_a}' at line {la + 1} "
                        f"followed by '{pattern_b}' at line {lb + 1} "
                        f"(within {window} lines)"
                    )
                    break
            if category in hits:
                break

    return hits


def scan_for_guardrail_violations(new_source: str) -> dict:
    """
    Scan proposed patch content for dangerous patterns using both the
    regex layer and the AST layer -- both always run; neither is a
    fallback for the other, since they catch different failure shapes.

    Returns:
      {"tripped": False, "ast_parse_ok": bool}
      or
      {"tripped": True, "category": str, "detail": str, "ast_parse_ok": bool}

    ast_parse_ok=False means new_source wasn't syntactically parseable,
    so only the regex layer ran for this patch -- surfaced explicitly
    rather than silently skipped, since it's a real coverage gap.

    If multiple rules trip (across either layer), the highest-priority
    category wins (see _CATEGORY_PRIORITY). Only one violation is ever
    reported.
    """
    regex_hits = _regex_scan(new_source)
    ast_hits, ast_parse_ok = _ast_scan(new_source)

    hits = dict(regex_hits)
    for category, detail in ast_hits.items():
        hits.setdefault(category, detail)

    if not hits:
        return {"tripped": False, "ast_parse_ok": ast_parse_ok}

    for category in _CATEGORY_PRIORITY:
        if category in hits:
            return {
                "tripped": True,
                "category": category,
                "detail": hits[category],
                "ast_parse_ok": ast_parse_ok,
            }

    raise ValueError(f"Guardrail hit in unranked category: {list(hits.keys())}")
