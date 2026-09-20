"""Thread-pool helper for bounded concurrent I/O-bound calls (API requests).

M7's "targeted async" (option 2, not a project-wide async/await rewrite): the
ABCs (Embedder, VisionExtractor, ...) stay synchronous, but the specific
spots where multiple independent network calls happen in a loop -- batch
embedding, per-chart vision extraction -- run concurrently instead of
sequentially. Threads, not asyncio: a blocking HTTP call releases the GIL
while waiting on the network, so a thread pool gets real wall-clock
concurrency for I/O-bound work without needing every layer of the codebase
(CLI included) to become async.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Callable, TypeVar

T = TypeVar("T")
R = TypeVar("R")


def run_concurrently(func: Callable[[T], R], items: list[T], max_workers: int = 5) -> list[R]:
    """Apply `func` to each item, running up to `max_workers` calls at once.

    Results are returned in the same order as `items`. An exception from any
    single item propagates out of this call -- callers that want per-item
    graceful degradation (e.g. one failed chart shouldn't drop the rest)
    should catch inside `func` itself rather than rely on this helper to do
    it for them.
    """
    if len(items) <= 1:
        return [func(item) for item in items]

    with ThreadPoolExecutor(max_workers=min(len(items), max_workers)) as pool:
        return list(pool.map(func, items))
