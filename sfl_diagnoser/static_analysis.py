"""
Static analysis: build a static execution graph G(V, E) from Python files.

Nodes:
    Fully-qualified function names: "module.function".

Edges:
    (u, v) means that u may call v (based on a simple AST call analysis).

This is an approximation of the execution graph used in:
  "Spectrum-based fault diagnosis with partial traces"
  (Sotto-Mayor et al., Journal of Systems & Software, 2026).

We work at function granularity and ignore intra-function control flow.
"""

from __future__ import annotations

import ast
import os
from dataclasses import dataclass
from typing import Dict, List, Set

from .data_models import ExecutionGraph, Component


@dataclass
class FunctionInfo:
    """
    Static information about a function.

    Attributes:
        module_name   : module name, e.g. "benchmarks.gcd".
        func_name     : simple function name inside the module.
        qualified_name: "module_name.func_name".
        calls         : set of simple names called by this function,
                        e.g. {"helper"}.
    """

    module_name: str
    func_name: str
    qualified_name: str
    calls: Set[str]


class _CallCollector(ast.NodeVisitor):
    """
    AST visitor that collects simple call targets inside a function body.

    We track only calls of the form:
        foo(...)
        module.foo(...)

    For "module.foo" we record only "foo". Resolution to fully-qualified
    names happens later.
    """

    def __init__(self) -> None:
        self.called_names: Set[str] = set()

    def visit_Call(self, node: ast.Call) -> None:
        func = node.func
        name = None

        if isinstance(func, ast.Name):
            name = func.id
        elif isinstance(func, ast.Attribute):
            # e.g. module.foo -> take "foo"
            name = func.attr

        if name is not None:
            self.called_names.add(name)

        self.generic_visit(node)


def _discover_functions_in_file(
        file_path: str,
        module_name: str,
) -> Dict[str, FunctionInfo]:
    """
    Parse a Python file and return:
        simple_func_name -> FunctionInfo

    The key is the simple function name; the fully-qualified name
    is stored inside FunctionInfo.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        source = f.read()

    tree = ast.parse(source, filename=file_path)
    functions: Dict[str, FunctionInfo] = {}

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            func_name = node.name
            qualified = f"{module_name}.{func_name}"

            collector = _CallCollector()
            collector.visit(node)

            functions[func_name] = FunctionInfo(
                module_name=module_name,
                func_name=func_name,
                qualified_name=qualified,
                calls=set(collector.called_names),
            )

    return functions


def build_call_graph_from_files(file_paths: List[str]) -> ExecutionGraph:
    """
    Build a static call graph G(V, E) from a list of Python files.

    Args:
        file_paths: paths to Python source files belonging to the same project.

    Returns:
        ExecutionGraph whose nodes are fully-qualified function names,
        and edges represent possible function calls.

    Note:
        This is conservative: if a call site exists in the source,
        we add an edge. The edge does not guarantee the call actually
        happens at runtime.
    """
    graph = ExecutionGraph()

    # Common root to derive module names
    abs_paths = [os.path.abspath(p) for p in file_paths]
    common_root = os.path.commonpath(abs_paths)

    # First pass: discover all functions and their simple call targets
    # Keyed by qualified_name to avoid collisions between modules
    all_functions: Dict[str, FunctionInfo] = {}

    for path in abs_paths:
        rel_path = os.path.relpath(path, common_root)
        module_name = os.path.splitext(rel_path)[0].replace(os.sep, ".")

        funcs_in_file = _discover_functions_in_file(path, module_name)
        for finfo in funcs_in_file.values():
            all_functions[finfo.qualified_name] = finfo

    # Second pass: add all functions as nodes
    for finfo in all_functions.values():
        graph.add_node(finfo.qualified_name)

    # Map simple function name -> set of fully-qualified candidates
    name_to_qualified: Dict[str, Set[Component]] = {}
    for finfo in all_functions.values():
        name_to_qualified.setdefault(finfo.func_name, set()).add(finfo.qualified_name)

    # Add edges based on simple-name resolution
    for finfo in all_functions.values():
        caller = finfo.qualified_name
        for called_name in finfo.calls:
            for callee in name_to_qualified.get(called_name, set()):
                graph.add_edge(caller, callee)

    return graph
