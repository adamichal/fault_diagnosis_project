"""
Dynamic analysis module.

This module provides a small, self-contained tracer that records
which components (functions) were executed during a single test run.

We use it to obtain *full* execution traces, as described in the paper
"Spectrum-based fault diagnosis with partial traces". Partial traces
are later simulated from these full traces.
"""

from __future__ import annotations

import inspect
import os
import sys
from typing import Callable, Any, Set


class ExecutionTracer:
    """
    Collect full execution traces by instrumenting Python with sys.settrace.

    The tracer records fully-qualified function names of the form
    "module.function" for all calls that originate from files under
    the given project_root.

    Typical usage:
        tracer = ExecutionTracer(project_root=".")
        result = tracer.run_function(my_test_function, *args)
        executed = tracer.executed_functions
    """

    def __init__(self, project_root: str) -> None:
        self.project_root = os.path.abspath(project_root)
        self.executed_functions: Set[str] = set()

    # ------------------------------------------------------------------ #
    # Internal trace function                                            #
    # ------------------------------------------------------------------ #

    def _trace(self, frame, event, arg):
        """
        Internal trace callback passed to sys.settrace.

        We only record "call" events for functions whose source file is
        located under project_root. Library calls are ignored.
        """
        if event != "call":
            return self._trace

        filename = os.path.abspath(frame.f_code.co_filename)
        if not filename.startswith(self.project_root):
            # Outside the project – ignore.
            return self._trace

        func_name = frame.f_code.co_name
        if func_name == "<module>":
            # We only care about function calls, not module imports.
            return self._trace

        # Try to derive a stable module name
        module = inspect.getmodule(frame)
        if module is not None and module.__name__ != "__main__":
            mod_name = module.__name__
        else:
            # Fallback: build module name from relative path
            rel = os.path.relpath(filename, self.project_root)
            mod_name = os.path.splitext(rel)[0].replace(os.sep, ".")

        qualified_name = f"{mod_name}.{func_name}"
        self.executed_functions.add(qualified_name)

        return self._trace

    # ------------------------------------------------------------------ #
    # Public API                                                         #
    # ------------------------------------------------------------------ #

    def run_function(self, func: Callable[..., Any], *args, **kwargs) -> Any:
        """
        Run `func(*args, **kwargs)` under tracing and return its result.

        After this call, ``self.executed_functions`` contains the set of
        all fully-qualified function names that were executed.

        Any exception raised by `func` is propagated to the caller.
        """
        self.executed_functions.clear()

        old_trace = sys.gettrace()
        sys.settrace(self._trace)
        try:
            return func(*args, **kwargs)
        finally:
            # Restore the previous trace function (if any).
            sys.settrace(old_trace)
