"""Tests run_concurrently()'s correctness -- order preservation and the
small-input fast paths -- not literally measuring wall-clock speedup, just
that mapping a function over items concurrently doesn't corrupt results.
"""

import time

from mmss.utils.concurrency import run_concurrently


def test_empty_list_returns_empty() -> None:
    assert run_concurrently(lambda x: x * 2, [], max_workers=5) == []


def test_single_item_uses_fast_path_without_pool() -> None:
    assert run_concurrently(lambda x: x * 2, [21], max_workers=5) == [42]


def test_preserves_input_order_regardless_of_completion_order() -> None:
    # Item 0 sleeps longest, item 4 sleeps least -- if results came back in
    # completion order rather than input order, this list would come back
    # reversed instead of ascending.
    def _work(i: int) -> int:
        time.sleep((5 - i) * 0.01)
        return i

    result = run_concurrently(_work, [0, 1, 2, 3, 4], max_workers=5)

    assert result == [0, 1, 2, 3, 4]


def test_applies_function_to_every_item() -> None:
    result = run_concurrently(lambda x: x**2, [1, 2, 3, 4], max_workers=2)
    assert result == [1, 4, 9, 16]


def test_respects_a_worker_count_larger_than_item_count() -> None:
    result = run_concurrently(lambda x: x + 1, [1, 2], max_workers=10)
    assert result == [2, 3]
