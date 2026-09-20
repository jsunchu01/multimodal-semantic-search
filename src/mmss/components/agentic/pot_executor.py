"""Program-of-thought executor: safely runs a short, LLM-generated Python
script that computes a figure (a percentage change, a ratio, any arithmetic
on numbers already pulled from retrieved chunks) so the LLM only has to
identify the right metric and numbers, not derive the arithmetic itself.
Fully domain-agnostic -- equally usable for a research paper's "how much did
accuracy improve" as a filing's revenue growth.

Previously also had an execute_template() method plus a FINANCIAL_TEMPLATES
dict of pre-written formulas (CAGR, gross margin, ROI, etc.) -- removed once
this project started being used for both financial docs and research papers,
since those templates were finance-only (no research-paper equivalent) and,
independently of that, were never actually called from the live pipeline
(pipeline/generate.py only ever used execute_from_llm_response()) -- dead
code even before the domain broadened.

Adopted close to the reference repo's actual design (verified from its real
source, not just the README) rather than the narrower single-expression
sketch in our own original plan -- that fuller design is genuinely
well-built (AST allowlist + blocked-name check + restricted runtime
builtins + timeout, real defense-in-depth). One deliberate change: kept
synchronous rather than async (matching this project's M7 decision to keep
concurrency targeted at API-call batching, not a project-wide async rewrite)
-- the timeout uses concurrent.futures instead of
asyncio.wait_for(asyncio.to_thread(...)).
Verified for Python >=3.11 (this project's floor): concurrent.futures.
TimeoutError is an alias of the builtin TimeoutError as of 3.11, so
Future.result(timeout=...)'s timeout is caught as plain TimeoutError below.

Known, unavoidable limitation shared with the reference repo's own async
version: neither a thread timeout nor asyncio.wait_for actually kills the
underlying thread when it times out -- Python has no clean way to forcibly
interrupt a running thread. A timed-out call stops blocking the caller, but
the sandboxed code keeps running to completion in the background regardless.
Not a regression introduced here, just a real constraint either way.
"""

from __future__ import annotations

import ast
import re
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

# ── AST allowlist ────────────────────────────────────────────────────────────

_ALLOWED_AST_NODES = {
    ast.Expression,
    ast.Expr,
    ast.BinOp,
    ast.UnaryOp,
    ast.BoolOp,
    ast.Compare,
    ast.IfExp,
    ast.Call,
    ast.Constant,
    ast.Name,
    ast.Load,
    ast.Attribute,
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.FloorDiv,
    ast.Mod,
    ast.Pow,
    ast.USub,
    ast.UAdd,
    ast.Eq,
    ast.NotEq,
    ast.Lt,
    ast.LtE,
    ast.Gt,
    ast.GtE,
    ast.If,
    ast.Return,
    ast.Assign,
    ast.AugAssign,
    ast.AnnAssign,
    ast.Module,
    ast.List,
    ast.Tuple,
    ast.Dict,
    ast.Store,
    ast.FunctionDef,
    ast.arguments,
    ast.arg,
}

# No ast.For/ast.While/comprehensions in the allowlist above -- there is no
# way to write a loop that passes validation, which is most of why the
# timeout below is defense-in-depth rather than the primary safeguard (a
# single call like sum(range(10**9)) can still be slow without any loop
# keyword at all, which is what the timeout actually guards against).

_BLOCKED_NAMES = frozenset(
    {
        "import", "__import__", "exec", "eval", "compile", "open", "input",
        "print", "__builtins__", "globals", "locals", "vars", "dir",
        "getattr", "setattr", "delattr", "hasattr", "object", "type",
        "super", "classmethod", "staticmethod", "property", "subprocess",
        "os", "sys", "socket", "urllib", "requests", "httpx",
    }
)

_SAFE_BUILTINS: dict[str, Any] = {
    "abs": abs, "round": round, "min": min, "max": max, "sum": sum,
    "len": len, "float": float, "int": int, "str": str, "bool": bool,
    "list": list, "tuple": tuple, "range": range, "enumerate": enumerate,
    "zip": zip, "sorted": sorted, "reversed": reversed, "pow": pow,
}


@dataclass
class PoTResult:
    success: bool
    result: float | None = None
    code: str = ""
    error: str | None = None
    execution_time_ms: float = 0.0
    variables: dict[str, Any] = field(default_factory=dict)

    def formatted(self, decimals: int = 2) -> str:
        if not self.success or self.result is None:
            return f"Error: {self.error}"
        return f"{self.result:.{decimals}f}"


class ASTSandboxValidator:
    def validate(self, code: str) -> str | None:
        """Return an error string if `code` is unsafe, else None."""
        try:
            tree = ast.parse(code, mode="exec")
        except SyntaxError as exc:
            return f"SyntaxError: {exc}"

        for node in ast.walk(tree):
            node_type = type(node)
            if node_type not in _ALLOWED_AST_NODES:
                return f"Disallowed AST node: {node_type.__name__}"
            if isinstance(node, ast.Name) and node.id in _BLOCKED_NAMES:
                return f"Blocked identifier: {node.id}"
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                return "Import statements are not allowed"

        return None


class PoTExecutor:
    """Secure program-of-thought executor for domain-agnostic arithmetic.

    Usage::

        executor = PoTExecutor()
        result = executor.execute_code("v_old = 42.3\\nv_new = 51.7\\nresult = (v_new - v_old) / v_old * 100")
    """

    def __init__(self, timeout_seconds: float = 5.0) -> None:
        self._timeout = timeout_seconds
        self._validator = ASTSandboxValidator()

    def _extract_code_block(self, text: str) -> str | None:
        """Extract the first ```python ... ``` block from LLM text."""
        patterns = [r"```python\s*\n(.*?)```", r"```\s*\n(.*?)```", r"`{1,3}(.*?)`{1,3}"]
        for pattern in patterns:
            match = re.search(pattern, text, re.DOTALL)
            if match:
                return match.group(1).strip()
        if re.search(r"\bresult\s*=", text):
            return text.strip()
        return None

    def _run_in_sandbox(self, code: str) -> dict[str, Any]:
        namespace: dict[str, Any] = {"__builtins__": _SAFE_BUILTINS}
        exec(compile(code, "<pot>", "exec"), namespace)  # noqa: S102 -- restricted namespace, AST-validated
        return {k: v for k, v in namespace.items() if not k.startswith("__")}

    def execute_code(self, code: str) -> PoTResult:
        error = self._validator.validate(code)
        if error:
            return PoTResult(success=False, code=code, error=f"Validation failed: {error}")

        start = time.perf_counter()
        pool = ThreadPoolExecutor(max_workers=1)
        future = pool.submit(self._run_in_sandbox, code)
        try:
            try:
                variables = future.result(timeout=self._timeout)
            except TimeoutError:
                return PoTResult(
                    success=False, code=code, error=f"Execution timed out after {self._timeout}s"
                )
            except Exception as exc:
                return PoTResult(success=False, code=code, error=f"RuntimeError: {exc}")
        finally:
            # wait=False is the whole point of a timeout: using `with
            # ThreadPoolExecutor(...) as pool:` here would call
            # shutdown(wait=True) on exit, which blocks until the background
            # thread actually finishes -- silently defeating the timeout
            # above by waiting for the slow computation anyway. This still
            # doesn't kill the thread (Python can't do that), it just stops
            # blocking the caller on it -- the shared, documented limitation
            # noted in this module's docstring.
            pool.shutdown(wait=False)
        elapsed_ms = (time.perf_counter() - start) * 1000

        result_value = variables.get("result")
        if result_value is None:
            return PoTResult(
                success=False,
                code=code,
                variables=variables,
                error="Code executed but 'result' variable not set",
                execution_time_ms=elapsed_ms,
            )
        try:
            result_float = float(result_value)
        except (TypeError, ValueError, OverflowError) as exc:
            return PoTResult(
                success=False,
                code=code,
                variables=variables,
                error=f"'result' could not be converted to a number: {exc}",
                execution_time_ms=elapsed_ms,
            )
        return PoTResult(
            success=True,
            result=result_float,
            code=code,
            variables=variables,
            execution_time_ms=elapsed_ms,
        )

    def execute_from_llm_response(self, llm_response: str) -> PoTResult:
        code = self._extract_code_block(llm_response)
        if not code:
            return PoTResult(success=False, code="", error="No executable code block found in LLM response")
        return self.execute_code(code)
