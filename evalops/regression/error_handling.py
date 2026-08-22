"""

Hybrid mechanical check: removes_weakens_error_handling.

Produces {"signal": bool, "findings": [...]} -- deliberately NOT the
{"passed": bool | None, "detail": str} shape used by the other regression-tier
mechanical checks. This check only surfaces evidence for the LLM judge
(Step B) to reason over; it never asserts a pass/fail fact on its own.
Reusing "passed" with None as a third state would let a downstream
`if result["passed"]:` silently merge "no signal, clean" with "signal,
needs judgment" -- different contract, different shape.

Scope: diff-touched functions only (same principle as
no_unused_code_introduced -- a patch isn't blamed for weak error handling
that pre-dates the diff).

Seven finding kinds:
  - removed_try_block        (whole try/except gone, body now unguarded)
  - removed_except_clause    (one handler gone, others on the same try remain)
  - removed_raise            (pure removal only -- a raise X -> raise Y swap
                               is lateral, same rule as broadened_except)
  - error_logging_removed
  - error_logging_downgraded
  - broadened_except         (strictly-broader catches only; lateral changes,
                               e.g. except ValueError -> except OSError, are
                               out of scope -- routed to unexplained_concern)
  - missing_error_handling_new_code
                              (function has no old-side match by name --
                               either genuinely new or renamed -- contains a
                               risky call, and has no try anywhere in its
                               body; skipped for test files)

The first six are delta detectors: they require an old-side function match
by name and compare old vs new. missing_error_handling_new_code is the
mirror case -- it only runs when no old-side match exists, so a function
is checked by exactly one of "was handling weakened" or "is handling
present at all," never both, and never neither. A renamed function used to
fall through both with zero findings; it's now covered by the new-code
detector, since a name-based match treats it identically to a brand-new
function.

Known limitation (intentional, documented, not an oversight):
  Handler correspondence is by source-order position within a matched try
  block, not by exception-type or bound-variable name (see `_paired_handlers`
  docstring for why -- a signature-based design was tried first and broke
  on retyped handlers). Position pairing is stable against a same-slot
  retype or bound-name rename, but not against a handler being inserted or
  deleted from the middle of a multi-except try, which shifts every later
  handler's position. Accepted as noise, same reasoning as try-block
  reordering being accepted as noise.

Known limitation on broadened_except: only builtin exception ancestry is
resolved, via `issubclass()` against the real `builtins` module objects,
never by importing project code (unsafe for arbitrary worktree diffs --
risks import-time side effects or broken imports mid-refactor). Custom
exception pairs default to "unresolvable -> lateral, no signal" rather than
false-flagging or false-clearing. Do not "fix" this by importing project
exception classes at check-time -- see `_is_builtin_exception_name` and
`_classify_except_change` docstrings.

Known limitation on missing_error_handling_new_code: the risky-call set
(_RISKY_CALL_NAME_HINTS / _RISKY_MODULE_HINTS) is a narrow, curated starting
list (network, file I/O, subprocess, parsing), matched by attribute-call
name against a small fixed set -- same design choice as _LOGGER_NAME_HINTS
and the builtin-only exception ancestry resolution above, not exhaustive
pattern matching. Expected to need real-data follow-up (e.g. DB client
calls), same tradeoff as the custom-exception-hierarchy gap. Also
function-body-scoped only -- it cannot see that a caller one level up
already wraps the call in try/except, same diff-touched-function scoping
limit the rest of this file already has.
"""

import ast
import builtins
from dataclasses import dataclass
from typing import Optional


# Finding ---------------------------------------------------------------------------


@dataclass
class Finding:
    kind: str
    location: str
    detail: str

    def to_dict(self) -> dict:
        return {"kind": self.kind, "location": self.location, "detail": self.detail}



# Shared scaffolding: diff-touched function extraction# ------------------------------


def _added_line_ranges(diff_text: str) -> list[tuple[int, int]]:
    """
    Parse unified diff hunk headers to find line ranges added in the new file.
    Shares the exact contract used by no_unused_code_introduced's
    `_added_line_ranges` -- import from that module instead of duplicating
    if this file lives alongside it in the final package.
    """
    ranges = []
    for line in diff_text.splitlines():
        if line.startswith("@@"):
            try:
                new_part = line.split("+")[1].split("@@")[0].strip()
                start_str, _, len_str = new_part.partition(",")
                start = int(start_str)
                length = int(len_str) if len_str else 1
                ranges.append((start, start + length - 1))
            except (IndexError, ValueError):
                continue
    return ranges


def _line_in_ranges(lineno: int, ranges: list[tuple[int, int]]) -> bool:
    return any(start <= lineno <= end for start, end in ranges)


def _touched_functions(new_tree: ast.Module, added_ranges: list[tuple[int, int]]) -> list[ast.FunctionDef]:
    """Top-level and nested function defs whose body overlaps a diff-added line."""
    touched = []
    for node in ast.walk(new_tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            start = node.lineno
            end = getattr(node, "end_lineno", start)
            if any(_line_in_ranges(ln, added_ranges) for ln in range(start, end + 1)):
                touched.append(node)
    return touched


def _match_old_function(old_tree: ast.Module, name: str) -> Optional[ast.FunctionDef]:
    """
    Find the same-named function in the old tree. A renamed function reads
    as "old removed, new added" and produces no findings from the delta
    detectors below -- acceptable, since a rename is a bigger structural
    change those detectors aren't scoped to reason about. It IS covered by
    missing_error_handling_new_code instead (see combiner), since that
    detector runs precisely when this function returns None.
    """
    for node in ast.walk(old_tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return node
    return None


# Test-file exclusion  ---------------------------------------------------------------------------


def _is_test_file(file_path: str) -> bool:
    """
    True if `file_path` is a test file, by the convention this repo actually
    uses (verified against data/repo_cache/llama_index, not assumed):
    2879 files match `tests/` dir + `test_*.py` filename; the `*_test.py`
    suffix pattern is not a real second convention here (the 3 matches for
    it were duplicate ephemeral-worktree copies of one file, not distinct
    source files) -- so only the two real signals are checked.

    Path-component check, not a substring/regex check on the whole path --
    avoids a false match on e.g. "latest_results.py" or a directory named
    "attestation".
    """
    parts = file_path.replace("\\", "/").split("/")
    if "tests" in parts:
        return True
    filename = parts[-1] if parts else file_path
    return filename.startswith("test_") and filename.endswith(".py")



# Try/except extraction and handler matching ---------------------------------------------------------------------------


def _collect_try_blocks(func: ast.FunctionDef) -> list[ast.Try]:
    """
    Try blocks belonging directly to func's own scope -- does not descend
    into nested function/lambda defs, since those are separately walked and
    matched via their own _touched_functions/_match_old_function entry.

    Bug found during verification, real data (llama_index commit
    83a0deceb): the previous version used a flat ast.walk(func) with no
    scope boundary, so a try block sitting inside a nested function got
    attributed to every enclosing function too -- one physical try block
    inside handle_future_result (itself nested inside wrapper, nested
    inside span) was independently "found" by all three enclosing scopes,
    producing 3 duplicate findings for a single real change instead of 1.
    This affected every detector in this file, not just the new one, since
    all of them build on _collect_try_blocks via _match_try_blocks.
    """
    tries: list[ast.Try] = []

    def walk(node: ast.AST) -> None:
        if isinstance(node, ast.Try):
            tries.append(node)
            for child in ast.iter_child_nodes(node):
                walk(child)
            return
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)) and node is not func:
            return
        for child in ast.iter_child_nodes(node):
            walk(child)

    walk(func)
    return tries


def _expr_to_name(expr: ast.expr) -> str:
    """Best-effort dotted name from an exception-type expression."""
    if isinstance(expr, ast.Name):
        return expr.id
    if isinstance(expr, ast.Attribute):
        parts = []
        node = expr
        while isinstance(node, ast.Attribute):
            parts.append(node.attr)
            node = node.value
        if isinstance(node, ast.Name):
            parts.append(node.id)
        return ".".join(reversed(parts))
    return "<unresolved>"


def _handler_exception_names(handler: ast.ExceptHandler) -> frozenset:
    """Names as written in source (not resolved). Bare except -> {"*"}."""
    if handler.type is None:
        return frozenset({"*"})
    if isinstance(handler.type, ast.Tuple):
        return frozenset(_expr_to_name(elt) for elt in handler.type.elts)
    return frozenset({_expr_to_name(handler.type)})


@dataclass
class TryMatch:
    old_try: Optional[ast.Try]
    new_try: Optional[ast.Try]


def _match_try_blocks(old_func: ast.FunctionDef, new_func: ast.FunctionDef) -> list[TryMatch]:
    """
    Pair try blocks between old and new function by source-order position.
    Good enough for the diff-touched-function scope this check targets --
    structural try-block reordering is a separate concern from
    error-handling weakening; treating it as noise here is the same
    "accepted noise" tradeoff as handler matching below.
    """
    old_tries = _collect_try_blocks(old_func)
    new_tries = _collect_try_blocks(new_func)
    n = max(len(old_tries), len(new_tries))
    return [
        TryMatch(
            old_try=old_tries[i] if i < len(old_tries) else None,
            new_try=new_tries[i] if i < len(new_tries) else None,
        )
        for i in range(n)
    ]


def _paired_handlers(matches: list[TryMatch]) -> list[tuple[ast.ExceptHandler, ast.ExceptHandler]]:
    """
    Handler correspondence is by source-order POSITION within a matched try
    block, not by exception-name/bound-name signature.

    Revised during verification: an earlier signature-based design (keyed
    on exception-name-set + bound name) broke the moment a handler's
    exception type changed at all -- narrowing, broadening, or a lateral
    swap all changed the signature, so every retype was double-counted as
    both "removed_except_clause" (old signature vanished) and whatever
    _detect_broadened_except separately found. Position pairing fixes this:
    "the handler in this slot" is the same handler whether or not its
    exception type also changed in the same edit, so raise/logging findings
    and exception-type findings can coexist on one handler without one
    manufacturing a false removal.

    Tradeoff (documented, not an oversight): position pairing is stable
    against a same-slot retype or bound-variable rename, but NOT against a
    handler being inserted or deleted from the middle of a multi-except
    try -- that shifts every later handler's position and reads as a wash
    of unrelated-looking changes rather than one clean removal. Accepted as
    noise for the same reason try-block reordering is accepted as noise
    above: this check targets the common cases (whole block gone, single
    handler weakened in place), not exhaustive diff alignment.
    """
    pairs = []
    for match in matches:
        if match.old_try is None or match.new_try is None:
            continue
        pairs.extend(zip(match.old_try.handlers, match.new_try.handlers))
    return pairs


def _removed_handlers(matches: list[TryMatch]) -> list[ast.ExceptHandler]:
    """Old handlers with no corresponding slot in the new try (new has fewer)."""
    removed = []
    for match in matches:
        if match.old_try is None or match.new_try is None:
            continue
        if len(match.old_try.handlers) > len(match.new_try.handlers):
            removed.extend(match.old_try.handlers[len(match.new_try.handlers):])
    return removed



# Detector 1: removed_try_block / removed_except_clause ----------------------------------------------------------------------------------------------------------------------------


def _detect_removed_handlers(func_name: str, matches: list[TryMatch]) -> list[Finding]:
    findings = []
    for match in matches:
        if match.old_try is not None and match.new_try is None:
            findings.append(Finding(
                kind="removed_try_block",
                location=f"{func_name}:L{match.old_try.lineno}",
                detail="Entire try/except block present in old version is absent in new version -- body now runs unguarded.",
            ))
            continue
        if match.old_try is None or match.new_try is None:
            continue

    for old_handler in _removed_handlers(matches):
        names = ", ".join(sorted(_handler_exception_names(old_handler))) or "bare except"
        findings.append(Finding(
            kind="removed_except_clause",
            location=f"{func_name}:L{old_handler.lineno}",
            detail=f"except clause for [{names}] present in old version, no corresponding handler slot in new version (other handlers on this try remain).",
        ))
    return findings



# Detector 2: removed_raise -----------------------------------------------------------------------------------------------------------------------------------------------------

def _handler_has_raise(handler: ast.ExceptHandler) -> bool:
    return any(isinstance(n, ast.Raise) for n in ast.walk(handler))


def _detect_removed_raise(func_name: str, matches: list[TryMatch]) -> list[Finding]:
    """
    Pure removal only. A `raise X` -> `raise Y` swap is lateral -- same
    reasoning as broadened_except's lateral-change rule: changing what
    propagates is a judgment call, not a fact.
    """
    findings = []
    for old_handler, new_handler in _paired_handlers(matches):
        if _handler_has_raise(old_handler) and not _handler_has_raise(new_handler):
            names = ", ".join(sorted(_handler_exception_names(old_handler))) or "bare except"
            findings.append(Finding(
                kind="removed_raise",
                location=f"{func_name}:L{new_handler.lineno}",
                detail=f"except [{names}] re-raised in old version, no raise of any kind in new version -- exception is now swallowed.",
            ))
    return findings

def _try_body_raises(try_node: ast.Try) -> list[ast.Raise]:
    """
    Raise statements directly in the try block's guarded body -- not in
    handlers (already covered by _handler_has_raise), not in orelse/finally
    (different semantics), and NOT inside a nested try (that nested try is
    already separately matched via _collect_try_blocks/_match_try_blocks --
    counting it here too would double-count against its own TryMatch pair).
    Also does not descend into nested function/lambda scopes -- a raise
    inside a nested def is that function's own concern, not this try's.

    Bug found and fixed during verification: the node-type check (is this
    a Try / nested function scope?) must run on the node itself before
    recursing, not only on its children -- checking only children let a
    top-level statement in try_node.body that IS itself a nested Try (or
    a nested def) get recursed into anyway, since the exclusion never
    fired on the very node being walked. Caught by a synthetic case with
    a raise inside a nested try's own handler, which was wrongly counted
    against the outer try before this fix.
    """
    raises: list[ast.Raise] = []

    def walk(node: ast.AST) -> None:
        if isinstance(node, ast.Raise):
            raises.append(node)
            return
        if isinstance(node, ast.Try):
            return
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            return
        for child in ast.iter_child_nodes(node):
            walk(child)

    for stmt in try_node.body:
        walk(stmt)
    return raises


def _detect_removed_raise_in_try_body(func_name: str, matches: list[TryMatch]) -> list[Finding]:
    """
    Same claim as _detect_removed_raise (a raise present in old, absent in
    new -- exception now swallowed), but scoped to the try block's own
    guarded body instead of its handlers. Real gap found via verification
    against llama_index commit 83a0deceb: a `raise exception` sitting
    directly in the try body (guarding a specific condition before falling
    through to `future.result()`) was collapsed away along with the whole
    body -- _detect_removed_raise never saw it, since it only walks
    handlers. Pure removal only, same as the handler-level check: a raise
    moved from the body into a handler (or vice versa) in the same edit is
    not "removed" by this detector's definition, since something still
    raises -- that's a judgment call for unexplained_concern, not a fact.
    """
    findings = []
    for match in matches:
        if match.old_try is None or match.new_try is None:
            continue
        old_raises = _try_body_raises(match.old_try)
        new_raises = _try_body_raises(match.new_try)
        if old_raises and not new_raises:
            findings.append(Finding(
                kind="removed_raise",
                location=f"{func_name}:L{match.new_try.lineno}",
                detail=(
                    "try block's guarded body re-raised an exception in old "
                    "version (not inside a handler), no raise anywhere in "
                    "the guarded body in new version -- exception is now "
                    "swallowed."
                ),
            ))
    return findings



# Detector 3 & 4: error_logging_removed / error_logging_downgraded -----------------------------------------------------------------------------------------------------------------

_LOGGER_NAME_HINTS = {"log", "logger", "logging"}
_SEVERITY_RANK = {"critical": 3, "error": 2, "exception": 2, "warning": 1, "debug": 0, "info": 0}


def _call_logger_severity(call: ast.Call) -> Optional[str]:
    """Logging method name (e.g. "error") if this looks like a call on
    something named log/logger/logging, else None."""
    if not isinstance(call.func, ast.Attribute):
        return None
    method = call.func.attr
    if method not in _SEVERITY_RANK:
        return None
    obj = call.func.value
    obj_name = obj.id if isinstance(obj, ast.Name) else (obj.attr if isinstance(obj, ast.Attribute) else None)
    if obj_name is None or not any(hint in obj_name.lower() for hint in _LOGGER_NAME_HINTS):
        return None
    return method


def _handler_log_calls(handler: ast.ExceptHandler) -> list[tuple[ast.Call, str]]:
    out = []
    for node in ast.walk(handler):
        if isinstance(node, ast.Call):
            sev = _call_logger_severity(node)
            if sev is not None:
                out.append((node, sev))
    return out


def _detect_removed_error_log(func_name: str, matches: list[TryMatch]) -> list[Finding]:
    findings = []
    for old_handler, new_handler in _paired_handlers(matches):
        old_calls = _handler_log_calls(old_handler)
        new_calls = _handler_log_calls(new_handler)

        if old_calls and not new_calls:
            worst_old = max(old_calls, key=lambda c: _SEVERITY_RANK[c[1]])[1]
            findings.append(Finding(
                kind="error_logging_removed",
                location=f"{func_name}:L{new_handler.lineno}",
                detail=f"logger.{worst_old}(...) call present in old handler, no logging call of any kind in new handler.",
            ))
            continue

        if old_calls and new_calls:
            old_max = max(_SEVERITY_RANK[sev] for _, sev in old_calls)
            new_max = max(_SEVERITY_RANK[sev] for _, sev in new_calls)
            if new_max < old_max:
                old_sev = next(sev for _, sev in old_calls if _SEVERITY_RANK[sev] == old_max)
                new_sev = next(sev for _, sev in new_calls if _SEVERITY_RANK[sev] == new_max)
                findings.append(Finding(
                    kind="error_logging_downgraded",
                    location=f"{func_name}:L{new_handler.lineno}",
                    detail=f"logging severity downgraded from logger.{old_sev}(...) to logger.{new_sev}(...) in this handler.",
                ))
    return findings



# Detector 5: broadened_except ------------------------------------------------------------------------------------------------------------------------------------------------------

def _is_builtin_exception_name(name: str) -> bool:
    """
    True only if `name` resolves against the real `builtins` module to a
    class that is a subclass of BaseException. Never imports project code --
    safe by construction since builtins is always importable with no side
    effects. Dotted names (e.g. "os.error") are treated as non-resolvable
    on purpose: a dotted reference implies a non-builtin import in
    virtually all real code, since builtins are never referenced dotted.
    """
    if "." in name or name in ("*", "<unresolved>"):
        return False
    candidate = getattr(builtins, name, None)
    return isinstance(candidate, type) and issubclass(candidate, BaseException)


def _classify_except_change(old_names: frozenset, new_names: frozenset) -> Optional[str]:
    """
    Returns "broadened", "narrowed", or None (neutral / lateral / unresolvable).

    Per-name classification, not per-statement: proceeds to real ancestry
    comparison only if every name on both sides resolves as a builtin
    exception (via _is_builtin_exception_name). Any custom/unresolved name
    on either side -> None, treated as lateral -- the deferred-custom-
    hierarchy decision. Known, documented gap, not an oversight; do not
    "fix" by importing project exception classes at check-time (see
    module docstring for why that's unsafe here).
    """
    if "*" in old_names and "*" in new_names:
        return None
    if "*" in new_names:
        return "broadened"
    if "*" in old_names:
        return "narrowed"

    if not all(_is_builtin_exception_name(n) for n in old_names | new_names):
        return None

    old_classes = {getattr(builtins, n) for n in old_names}
    new_classes = {getattr(builtins, n) for n in new_names}

    def covered_by(a_classes, b_classes) -> bool:
        return all(any(issubclass(a, b) for b in b_classes) for a in a_classes)

    old_covered_by_new = covered_by(old_classes, new_classes)
    new_covered_by_old = covered_by(new_classes, old_classes)

    if old_covered_by_new and not new_covered_by_old:
        return "broadened"
    if new_covered_by_old and not old_covered_by_new:
        return "narrowed"
    return None  # equal or lateral


def _detect_broadened_except(func_name: str, matches: list[TryMatch]) -> list[Finding]:
    findings = []
    for match in matches:
        if match.old_try is None or match.new_try is None:
            continue
        # Position-paired, not signature-paired: the exception names
        # themselves are exactly what's changing, so they can't be part of
        # the matching key the way they are in the other detectors.
        for old_h, new_h in zip(match.old_try.handlers, match.new_try.handlers):
            old_names = _handler_exception_names(old_h)
            new_names = _handler_exception_names(new_h)
            if old_names == new_names:
                continue
            if _classify_except_change(old_names, new_names) == "broadened":
                old_str = ", ".join(sorted(old_names)) or "bare except"
                new_str = ", ".join(sorted(new_names)) or "bare except"
                findings.append(Finding(
                    kind="broadened_except",
                    location=f"{func_name}:L{new_h.lineno}",
                    detail=f"except [{old_str}] broadened to except [{new_str}].",
                ))
    return findings



# Detector 6: missing_error_handling_new_code ------------------------------------------------------------------------------------------------------------------------------------------


# Narrow, curated, name-based -- same design choice as _LOGGER_NAME_HINTS
# and the builtin-only exception ancestry resolution above. Matched by
# attribute-call name against a small fixed set, not AST-type introspection
# or import resolution (same reasoning as _call_logger_severity: cheap,
# explicit, easy to extend with real data later).
_RISKY_CALL_NAME_HINTS = {
    # network
    "get", "post", "put", "delete", "patch", "request", "urlopen", "fetch",
    # file I/O
    "open", "read", "write",
    # subprocess
    "run", "call", "check_call", "check_output", "popen",
    # parsing
    "loads", "load",
}

_RISKY_MODULE_HINTS = {"requests", "httpx", "urllib", "subprocess", "json", "yaml"}


def _is_risky_call(call: ast.Call) -> bool:
    """
    Best-effort match: either a bare call to a risky builtin (open(...)) or
    an attribute call whose method name is in the risky set AND whose object
    name hints at a risky module/client (requests.get, subprocess.run,
    yaml.safe_load, self.client.post, etc.). The object-name check is
    intentionally loose (substring against _RISKY_MODULE_HINTS or a generic
    "client"/"session"/"conn" hint) since call sites are rarely the literal
    module name once wrapped in a client object -- same tradeoff
    _call_logger_severity makes for logger objects.
    """
    if isinstance(call.func, ast.Name):
        return call.func.id in ("open",)

    if isinstance(call.func, ast.Attribute):
        method = call.func.attr
        if method not in _RISKY_CALL_NAME_HINTS:
            return False
        obj = call.func.value
        obj_name = None
        if isinstance(obj, ast.Name):
            obj_name = obj.id
        elif isinstance(obj, ast.Attribute):
            obj_name = obj.attr
        if obj_name is None:
            return False
        obj_name_lower = obj_name.lower()
        if any(hint in obj_name_lower for hint in _RISKY_MODULE_HINTS):
            return True
        if any(hint in obj_name_lower for hint in ("client", "session", "conn")):
            return True

    return False


def _function_has_try(func: ast.FunctionDef) -> bool:
    return any(isinstance(n, ast.Try) for n in ast.walk(func))


def _function_risky_calls(func: ast.FunctionDef) -> list[ast.Call]:
    return [n for n in ast.walk(func) if isinstance(n, ast.Call) and _is_risky_call(n)]


def _detect_missing_error_handling_new_code(
    func_name: str, new_func: ast.FunctionDef, file_path: str
) -> list[Finding]:
    """
    Scoped only to functions with no old-side counterpart (brand new to this
    patch, or renamed) -- the mirror case of the delta detectors above,
    which all require an old_func match. Skipped entirely for test files
    (see _is_test_file): unguarded risky calls are normal and expected
    there, not a finding.

    Deliberately does not attempt handler-level granularity like the delta
    detectors -- a new function either has a try block covering its risky
    calls or it doesn't; that's a function-level fact, not something that
    needs try/handler pairing machinery.
    """
    if _is_test_file(file_path):
        return []

    if _function_has_try(new_func):
        return []

    risky_calls = _function_risky_calls(new_func)
    if not risky_calls:
        return []

    call_descriptions = sorted({
        (c.func.attr if isinstance(c.func, ast.Attribute) else c.func.id)
        for c in risky_calls
        if isinstance(c.func, (ast.Attribute, ast.Name))
    })
    return [Finding(
        kind="missing_error_handling_new_code",
        location=f"{func_name}:L{new_func.lineno}",
        detail=(
            f"New function calls {', '.join(call_descriptions)} with no "
            f"try/except anywhere in the function body."
        ),
    )]



# Combiner ------------------------------------------------------------------------------------------------------------------------------------------------------


def check_error_handling_weakened(
    old_source: str, new_source: str, diff_text: str, file_path: str
) -> dict:
    """
    Hybrid mechanical sub-check for removes_weakens_error_handling.

    Returns {"signal": bool, "findings": [dict, ...]} -- NOT the
    {"passed": bool | None, "detail": str} shape used elsewhere in
    regression/mechanical.py. This check only surfaces evidence; Step B's
    judge decides whether each finding is a legitimate simplification or
    a real regression.

    file_path is required (new as of the missing_error_handling_new_code
    addition) -- needed to resolve _is_test_file. The six delta detectors
    don't use it; it exists solely for the new-code branch below. Any
    existing caller of this function must be updated to pass file_path.
    """
    old_tree = ast.parse(old_source)
    new_tree = ast.parse(new_source)
    added_ranges = _added_line_ranges(diff_text)

    all_findings: list[Finding] = []
    for new_func in _touched_functions(new_tree, added_ranges):
        old_func = _match_old_function(old_tree, new_func.name)

        if old_func is None:
            # No old-side match -- genuinely new function, or a rename.
            # Checked for presence of error handling instead of a delta,
            # since there's no old side to diff against.
            all_findings.extend(
                _detect_missing_error_handling_new_code(new_func.name, new_func, file_path)
            )
            continue

        matches = _match_try_blocks(old_func, new_func)

        all_findings.extend(_detect_removed_handlers(new_func.name, matches))
        all_findings.extend(_detect_removed_raise(new_func.name, matches))
        all_findings.extend(_detect_removed_error_log(new_func.name, matches))
        all_findings.extend(_detect_broadened_except(new_func.name, matches))
        all_findings.extend(_detect_removed_raise_in_try_body(new_func.name, matches))

    return {
        "signal": len(all_findings) > 0,
        "findings": [f.to_dict() for f in all_findings],
    }