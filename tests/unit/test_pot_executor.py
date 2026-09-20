"""Tests PoTExecutor's AST sandboxing, execution, and code-block extraction
-- plain arithmetic, no LLM or network needed.
"""

import pytest

from mmss.components.agentic.pot_executor import PoTExecutor


def test_executes_simple_arithmetic() -> None:
    executor = PoTExecutor()
    result = executor.execute_code("v_old = 42.3\nv_new = 51.7\nresult = (v_new - v_old) / v_old * 100")

    assert result.success
    assert result.result == pytest.approx(22.2222, rel=1e-3)


def test_blocks_import_statement() -> None:
    executor = PoTExecutor()
    result = executor.execute_code("import os\nresult = 1")

    assert not result.success
    assert "import" in result.error.lower()


def test_blocks_dangerous_builtin_names() -> None:
    executor = PoTExecutor()
    result = executor.execute_code("result = eval('1+1')")

    assert not result.success
    assert "eval" in result.error.lower()


def test_blocks_for_loops() -> None:
    executor = PoTExecutor()
    result = executor.execute_code("total = 0\nfor i in range(5):\n    total += i\nresult = total")

    assert not result.success
    assert "disallowed" in result.error.lower()


def test_missing_result_variable_is_an_error() -> None:
    executor = PoTExecutor()
    result = executor.execute_code("x = 1 + 1")

    assert not result.success
    assert "result" in result.error.lower()


def test_syntax_error_is_caught() -> None:
    executor = PoTExecutor()
    result = executor.execute_code("result = (1 + ")

    assert not result.success
    assert "syntaxerror" in result.error.lower()


def test_non_numeric_result_is_an_error_not_a_crash() -> None:
    executor = PoTExecutor()
    result = executor.execute_code('result = "not a number"')

    assert not result.success
    assert "could not be converted" in result.error.lower()


def test_timeout_is_enforced_and_returns_promptly() -> None:
    # No loops are allowed by the AST validator, but recursion is -- naive
    # (unmemoized) recursive fib(33) takes far longer than 50ms in pure
    # Python, and reliably so regardless of test-runner hardware speed.
    executor = PoTExecutor(timeout_seconds=0.05)
    code = (
        "def fib(n):\n"
        "    if n <= 1:\n"
        "        return n\n"
        "    return fib(n - 1) + fib(n - 2)\n"
        "result = fib(33)\n"
    )

    result = executor.execute_code(code)

    assert not result.success
    assert "timed out" in result.error.lower()


def test_extracts_and_executes_fenced_code_block_from_llm_response() -> None:
    executor = PoTExecutor()
    llm_text = (
        "The revenue grew as follows:\n\n"
        "```python\n"
        "v_old = 100\n"
        "v_new = 150\n"
        "result = (v_new - v_old) / v_old * 100\n"
        "```\n"
    )

    result = executor.execute_from_llm_response(llm_text)

    assert result.success
    assert result.result == pytest.approx(50.0)


def test_no_code_block_in_response_is_an_error() -> None:
    executor = PoTExecutor()
    result = executor.execute_from_llm_response("Just a plain prose answer, no code here.")

    assert not result.success
    assert "no executable code" in result.error.lower()
