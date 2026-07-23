from __future__ import annotations
import ast
import json
import os
import traceback
from pathlib import Path

ALWAYS_EXCLUDE_DIRS = {".git", "__pycache__", "build", "dist"}
TEST_EXCLUDE_DIR = {"tests", "test"}

def _is_test_file(path : Path) -> bool:
    name = path.name
    return name.startswith("test_") or name.endswith("_test.py")

def find_python_files(repo_path : str, include_tests : bool = False) -> list[str]:

    root = Path(repo_path)
    results : list[str] = []

    for path in root.rglob("*.py"):
        rel_parts = path.relative_to(root).parts

        if any(part in ALWAYS_EXCLUDE_DIRS for part in rel_parts):
            continue

        if not include_tests:
            if any(part in TEST_EXCLUDE_DIR for part in rel_parts):
                continue
            if _is_test_file(path):
                continue
        
        results.append(str(path))

    return results

def _get_source(source_lines : list[str], node: ast.AST)->str:
    return "\n".join(source_lines[node.lineno -1 : node.end_lineno])


def _get_full_source(source_lines :list[str], node:ast.AST)->str:
    decorator_list = getattr(node, "decorator_list", [])
    start_line = decorator_list[0].lineno if decorator_list else node.lineno
    return "\n".join(source_lines[start_line -1 : node.end_lineno])

def _find_init_source(class_node : ast.ClassDef, source_lines :list[str])-> str |None:
   
    for item in class_node.body:
        if isinstance(item,(ast.FunctionDef, ast.AsyncFunctionDef)) and item.name == "__init__":
            return _get_full_source(source_lines, item)
    return None

def _get_decorators(node :ast.AST, source_lines : list[str]) -> list[str]:
    decorator_list = getattr(node, "decorator_list",[])
    return[_get_source(source_lines, d) for d in decorator_list]

def _get_bases(class_node : ast.ClassDef) -> list[str]:
    try:
        return[ast.unparse(b) for b in class_node.bases]
    except Exception:
        return []
    
def _get_signature(node :ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    try:
        args = ast.unparse(node.args)
    except Exception:
        args = ""
    prefix = "async def" if isinstance(node, ast.AsyncFunctionDef) else "def"
    ret = ""
    if node.returns is not None:
        try:
            ret =f"-> {ast.unparse(node.returns)}"
        except Exception:
            ret = ""
    return f"{prefix} {node.name}({args}) {ret}:"



def _find_class_attributes(class_node : ast.ClassDef,  source_lines : list[str]) -> list[str]:

    attrs = []
    for item in class_node.body:
        if isinstance(item, (ast.Assign, ast.AnnAssign)):
            attrs.append(_get_source(source_lines,item))
    return attrs

def _extract_imports(tree : ast.Module) -> list[str]:

    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names = ", ".join(
                f"{a.name} as {a.asname}" if a.asname else a.name
                for a in node.names
            )
            imports.append(f"import {names}")
        elif isinstance(node, ast.ImportFrom):
            module = "."*(node.level or 0) + (node.module or "")
            names = ", ".join(
                f"{a.name} as {a.asname}" if a.asname else a.name
                for a in node.names
            )
            imports.append(f"from {module} import {names}")
    return imports

def chunk_file(
        file_path :str,
        repo_path : str,
        strict : bool = False,
        error_log_path : str = "data/chunking_errors.jsonl",
) -> list[dict]:

    try:
        text = Path(file_path).read_text(encoding="utf-8")
    except(UnicodeDecodeError, OSError) as e:
        _log_error(file_path ,e, error_log_path,strict)
        return []
    
    try:
        tree = ast.parse(text)
    except SyntaxError as e:
        _log_error(file_path, e, error_log_path, strict)
        return []
    
    source_lines = text.splitlines()
    rel_path = str(Path(file_path).relative_to(repo_path))
    imports = _extract_imports(tree)
    chunks : list[dict] = []

    for node in ast.iter_child_nodes(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            chunks.append({
                "chunk_id" : f"{rel_path} :: {node.name}",
                "type" : "function",
                "name" : node.name,
                "class_name" : None,
                "file_path" : rel_path,
                "start_line" : node.lineno,
                "end_line" : node.end_lineno,
                "source" : _get_full_source(source_lines, node),
                "class_context" : None,
                "signature" : _get_signature(node),
                "docstring" : ast.get_docstring(node),
                "decorators" : _get_decorators(node, source_lines),
                "bases" : None,
                "class_attributes" : None,
                "imports" : imports,
            })
        elif isinstance(node, ast.ClassDef):
            class_context = _find_init_source(node,source_lines)

            chunks.append({
                "chunk_id" : f"{rel_path}::{node.name}",
                "type" : "class",
                "name" : node.name,
                "class_name" : None,
                "file_path" : rel_path,
                "start_line" : node.lineno,
                "end_line" : node.end_lineno,
                "source" : _get_full_source(source_lines,node),
                "class_context" : class_context ,
                "signature" : None,
                "docstring" : ast.get_docstring(node),
                "decorators" : _get_decorators(node, source_lines),
                "bases" : _get_bases(node),
                "class_attributes" : _find_class_attributes(node, source_lines),
                "imports" : imports,
            })

            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    chunks.append({
                        "chunk_id" : f"{rel_path} ::{node.name}.{item.name}",
                        "type" : "method",
                        "name" : item.name,
                        "class_name" : node.name,
                        "file_path" : rel_path,
                        "start_line" : item.lineno,
                        "end_line" : item.end_lineno,
                        "source" : _get_full_source(source_lines,item),
                        "class_context" : class_context,
                        "signature" : _get_signature(item),
                        "docstring" : ast.get_docstring(item),
                        "decorators" : _get_decorators(item,source_lines),
                        "bases" : None,
                        "class_attributes" : None,
                        "imports" : imports,
                    })

    return chunks


def _log_error(file_path :str, error : Exception, error_log_path:str, strict:bool)->None:

    if strict:
        raise error
    
    Path(error_log_path).parent.mkdir(parents =True, exist_ok = True)
    with open(error_log_path, "a") as f:
        f.write(json.dumps({
            "file_path" : file_path,
            "error" : str(error),
            "error_type" : type(error).__name__,
            "lineno" : getattr(error, "lineno",None),
            "offset" : getattr(error, "offset", None),
            "traceback" : traceback.format_exc(),
        }) + "\n")


def _atomic_write_jsonl(path : Path, records:list[dict]) ->None:
    path.parent.mkdir(parents=True, exist_ok = True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "w") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")
        f.flush
        os.fsync(f.fileno())
    tmp_path.replace(path)


def chunk_repo(
        repo_path : str,
        source_id : str,
        cache_dir : str = "data/chunk_cache",
        include_tests : bool = False,
        strict : bool  = False,
) -> list[dict]:
    
    cache_path = Path(cache_dir) / f"{source_id}.jsonl"

    if cache_path.exists():
        chunks = []
        with open(cache_path) as f:
            for line in f:
                chunks.append(json.loads(line))
        return chunks
    
    all_chunks : list[dict] = []
    for file_path in find_python_files(repo_path, include_tests= include_tests):
        all_chunks.extend(chunk_file(file_path,repo_path,strict = strict))

    _atomic_write_jsonl(cache_path, all_chunks)

    return all_chunks

if __name__ == "__main__":
    from issue_worker.retrieval.checkout import get_repo_at_commit
 
    source_id = "22068"
    created_at = "2026-06-22T01:33:30Z"
 
    repo_path = get_repo_at_commit(source_id, created_at)
    chunks = chunk_repo(repo_path, source_id)
 
    print(f"chunked {len(chunks)} nodes from {repo_path}")
    if chunks:
        sample = chunks[0]
        print(f"sample chunk_id: {sample['chunk_id']}")
        print(f"sample type: {sample['type']}")