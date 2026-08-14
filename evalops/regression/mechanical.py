"""
 
Five mechanical (no-LLM) regression checks, run against the worktree state
after Patch Application has applied a resolved patch. Each check reads
disk/git state directly — never Resolve's in-memory edits list — since the
worktree is ground truth after any rollback/retry Patch Application ran.
 
Return shape, uniform across all five, mirrors Patch Application's
{"applied": bool, "failure_reason": ..., "detail": ...} convention:
 
    {"passed": bool | None, "detail": str}
 
`passed` is None only for tests_passed's not-applicable case (no test file
could be confidently located). The other four always resolve to a real bool.
"""


import ast
import subprocess
from pathlib import Path
import os

def _run_git(worktree_path : str, args: list[str]) -> str:
    """Run a git command in worktree_path, return stdout. Raises on failure."""
    result = subprocess.run(
        ["git", *args],
        cwd = worktree_path,
        capture_output= True,
        text = True,
        timeout = 30,
    )
    if result.returncode !=0:
        raise RuntimeError(f"git {' '.join(args)} failed: {result.stderr.strip()}")
    return result.stdout

def _changed_python_files(worktree_path : str, base_ref : str) -> list[str]:
    """  Finds which .py files changed compared with base_ref."""
    out = _run_git(worktree_path, ["diff", "--name-only", base_ref])
    return [line.strip() for line in out.splitlines() if line.strip().endswith(".py")]


def _file_at_ref(worktree_path : str, base_ref : str, file_path : str) -> str |None:
    """   Gets the old content of one specific file from base_ref or None"""
    try:
        return _run_git(worktree_path, ["show", f"{base_ref} : {file_path}"])
    except RuntimeError:
        return None


def _extract_signatures(source : str) -> dict[str, str]:
    """Reads Python code with AST and extracts function/method names and their arguments."""

    signatures : dict[str, str] = {}
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return signatures

    def visit(node: ast.AST, prefix : str = "") -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                qualname = f"{prefix}{child.name}"
                args = ast.unparse(child.args)
                signatures[qualname] = args
                visit(child, prefix = f"{qualname}")
            elif isinstance(child, ast.ClassDef):
                visit(child, prefix = f"{prefix}{child.name}")

    visit(tree)
    return signatures

def signature_unchanged(worktree_path: str, base_ref: str = "HEAD") -> dict:
    """True if no changed .py file altered an existing function/method signature.
 
    New functions are fine (not a signature change). A changed or removed
    signature on a function that existed at base_ref fails the check.
    """
    changed_files = _changed_python_files(worktree_path, base_ref)
    diffs = []
 
    for file_path in changed_files:
        old_source = _file_at_ref(worktree_path, base_ref, file_path)
        if old_source is None:
            continue  # new file, nothing to compare
 
        new_path = Path(worktree_path) / file_path
        if not new_path.exists():
            continue  # file was deleted, not a signature change
 
        old_sigs = _extract_signatures(old_source)
        new_sigs = _extract_signatures(new_path.read_text())
 
        for name, old_args in old_sigs.items():
            if name in new_sigs and new_sigs[name] != old_args:
                diffs.append(f"{file_path}::{name}: ({old_args}) -> ({new_sigs[name]})")
 
    if diffs:
        return {"passed": False, "detail": "Signature(s) changed: " + "; ".join(diffs)}
    return {"passed": True, "detail": "No existing function/method signatures altered."}


def dependency_changed(worktree_path : str, base_ref : str = "HEAD") -> dict:
    """True (i.e. flags a problem) if dependency declarations changed.
 
    Named to match the regression-tier boolean convention: passed=False
    means a dependency change was detected. detail carries the actual diff
    text for each touched dependency file, not just the filename, so the
    judge prompt (Step B) has the real change to reason about rather than
    a bare "this file changed" flag.
    """
    dep_files = [
        "requirements.txt",
        "pyproject.toml",
        "setup.py",
        "setup.cfg",
    ]
    all_changed = _run_git(
        worktree_path,
        ["diff", "--name-only", base_ref],
    ).splitlines()
    touched = [f for f in dep_files if f in all_changed]
 
    if not touched:
        return {
            "passed": True,
            "detail": "No dependency files modified.",
        }
 
    changes = []
    for file_path in touched:
        diff = _run_git(
            worktree_path,
            ["diff", base_ref, "--", file_path],
        )
        if diff.strip():
            changes.append(f"{file_path}:\n{diff.strip()}")

    return {
        "passed" : False,
        "detail" : "Dependency declaration(s) modified:\n" + "\n\n".join(changes)
    }


def syntax_valid(worktree_path : str, base_ref : str = "HEAD") -> dict:
    """True if every changed .py file parses as valid Python (current on-disk state)."""

    changed_files = _changed_python_files(worktree_path, base_ref)
    errors = []

    for file_path in changed_files:
        full_path = Path(worktree_path) / file_path
        if not full_path.exists():
            continue #deleted file

        try:
            ast.parse(full_path.read_text())
        except SyntaxError as e:
            errors.append(f"{file_path} : {e}")


    if errors: 
        return {"passed": False, "detail": "Syntax error(s): " + "; ".join(errors)}
    return {"passed" : True, "detail" : "All changed files parse as valid Python"}

def _find_package_root(worktree_path: str, file_path: str) -> str:
    """Walk up from file_path's directory to find the nearest dir containing
    pyproject.toml or setup.py — that's the sub-package's installable root,
    which must be on sys.path for its import to resolve in a monorepo."""

    current = (Path(worktree_path) / file_path).parent
    root = Path(worktree_path)

    while current >= root:
        if (current / "pyproject.toml").exists() or (current / "setup.py").exists():
            return str(current.resolve())
        current = current.parent

    return str(root.resolve())
def imports_valid(worktree_path : str , base_ref : str = "HEAD") -> dict:
    """True if every changed module can be imported without raising.
 
    Runs each import in a subprocess (not exec()/eval() in-process) so an
    import-time crash can't take down the check runner itself.
    """

    changed_files = _changed_python_files(worktree_path ,base_ref)
    errors = []

    for file_path in changed_files:
        full_path = Path(worktree_path) / file_path
        if not full_path.exists():
            continue

        parts = file_path[:-3].split("/")
        if "llama_index" in parts:
            idx = parts.index("llama_index")
            module_name = ".".join(parts[idx:])
        else:
            continue  # not an importable llama_index module (e.g. tests/, scripts/, docs/)

        pkg_root = _find_package_root(worktree_path, file_path)
        print(f"DEBUG pkg_root={pkg_root}")
        env = os.environ.copy()
        env["PYTHONPATH"] = pkg_root + os.pathsep + env.get("PYTHONPATH", "")

        result = subprocess.run(
            ["python3", "-c", f"import {module_name}"],
            cwd = worktree_path,
            capture_output= True,
            text = True,
            timeout=30,
            env = env,
        )
        if result.returncode != 0:
            errors.append(f"{file_path}: {result.stderr.strip().splitlines()[-1] if result.stderr else 'import failed'}")

    if errors:
        return {"passed": False, "detail": "Import error(s): " + "; ".join(errors)}
    return {"passed": True, "detail": "All changed modules import cleanly."}



def _added_line_ranges(worktree_path : str, base_ref : str, file_path : str)-> list[tuple[int,int]]:
    """Line ranges (start, end inclusive) added by this file's diff, in the
    NEW file's line numbering. Parsed from unified diff hunk headers
    (@@ -a,b +c,d @@ -> new-file range is [c, c+d-1]).
    """

    diff = _run_git(
        worktree_path,
        ["diff", "--unified=0", base_ref,"--",file_path]
    )
    ranges = []
    for line in diff.splitlines():
        if not line.startswith("@@"):
            continue
        # e.g. "@@ -12,3 +12,5 @@ def foo():"
        new_part = line.split("+", 1)[1].split("@@")[0].strip()
        if "," in new_part:
            start_str, count_str = new_part.split(",")
            start, count = int(start_str), int(count_str)
        else:
            start, count = int(new_part), 1
        if count == 0:
            continue
        ranges.append((start, start + count -1))

    return ranges

def _line_in_ranges(lineno :int, ranges :list[tuple[int, int]]) -> bool:
    return any(start <= lineno <= end for start, end in ranges)

def no_unused_code_introduced(worktree_path : str, base_ref : str = "HEAD") -> dict:
    """True if the patch didn't introduce a new unused variable or import.
 
    Scoped to lines the diff actually added — pre-existing dead code
    elsewhere in a touched file is not this patch's fault and is not
    flagged. Uses pyflakes (subprocess, not the library API, to keep this
    check isolated from pyflakes internals) rather than hand-rolled AST
    unused-name detection, since correct scoping (comprehensions, walrus,
    __all__ exports) is a solved problem there and not worth re-deriving.
 
    Requires: pyflakes (pip install pyflakes)
 
    Only "unused" classes are considered here (imported-but-unused,
    assigned-but-never-used) — not pyflakes' other warning types (e.g.
    undefined names), which are out of scope for this check.
    """
    changed_files = _changed_python_files(worktree_path, base_ref)
    findings = []
 
    for file_path in changed_files:
        full_path = Path(worktree_path) / file_path
        if not full_path.exists():
            continue  # deleted file, nothing to scan
 
        added_ranges = _added_line_ranges(worktree_path, base_ref, file_path)
        if not added_ranges:
            continue
 
        result = subprocess.run(
            ["python3", "-m", "pyflakes", str(full_path)],
            cwd=worktree_path,
            capture_output=True,
            text=True,
            timeout=30,
        )
        # pyflakes exits non-zero when it finds anything to report; that's
        # expected output here, not a tool failure.
        for line in result.stdout.splitlines():
            # format: "<path>:<lineno>:<col>: <message>"
            parts = line.split(":", 3)
            if len(parts) < 4:
                continue
            try:
                lineno = int(parts[1])
            except ValueError:
                continue
            message = parts[3].strip()
            is_unused = "imported but unused" in message or "assigned to but never used" in message
            if is_unused and _line_in_ranges(lineno, added_ranges):
                findings.append(f"{file_path}:{lineno}: {message}")
 
    if findings:
        return {
            "passed": False,
            "detail": "Unused code introduced: " + "; ".join(findings),
        }
    return {"passed": True, "detail": "No unused variables or imports introduced by this patch."}


def _find_test_file(worktree_path: str, source_file: str) -> str | None:
    """Locate a test file for source_file by naming convention only.
 
    e.g. llama_index/core/x.py -> tests/core/test_x.py or
    llama_index/core/tests/test_x.py, depending on repo layout. No fuzzy
    search — a miss here means passed: None, not a guess.
    """
    src_path = Path(source_file)
    stem = src_path.stem
    candidates = [
        Path("tests") / src_path.parent / f"test_{stem}.py",
        src_path.parent / "tests" / f"test_{stem}.py",
        src_path.parent / f"test_{stem}.py",
    ]
    for candidate in candidates:
        if (Path(worktree_path) / candidate).exists():
            return str(candidate)
    return None


def test_passed(worktree_path : str, base_ref: str = "HEAD", timeout : int =60) -> dict:
    """Run the single test file inferred by naming convention, if one exists.
 
    Never runs the whole suite — attributing a suite-wide failure to this
    specific edit would need diff/attribution logic that doesn't exist yet.
    No match found is a legitimate, honest outcome: passed: None, not a
    fabricated pass/fail on a guessed file.
    """
    changed_files = _changed_python_files(worktree_path, base_ref)
    if not changed_files:
        return {"passed" : None, "detail":"No changed .py files to test"}

    results = []
    any_test_found = False

    for file_path in changed_files:
        test_file = _find_test_file(worktree_path, file_path)
        if test_file is None:
            continue
        any_test_found = True

        try: 
            result = subprocess.run(
                ["python3", "-m","pytest", test_file,"-q"],
                cwd=worktree_path,
                capture_output= True,
                text = True,
                timeout=timeout,
            )
            results.append((test_file,result.returncode ==0,result.std[-500:]))
        except subprocess.TimeoutExpired:
            results.append((test_file , False, f"timed out after {timeout}s"))

    if not any_test_found:
        return {
            "passed" : None,
            "detail" : "No test file found by naming convention for any changed file",
        }

    failures = [f"{f}: {out}" for f, ok, out in results if not ok]

    if failures:
        return {"passed" : False, "detail" : "Test failure(s): " + "; ".join(failures)}
    return{
        "passed": True,
        "detail": f"Passed: {', '.join(f for f, _, _ in results)}",

    }




