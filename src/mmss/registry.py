"""Generic provider registry: register a backend under (kind, name), build it by
name from config. Adding a new vendor for any pluggable slot is one new class +
one `@register` line + one config value -- no pipeline code changes.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Callable, TypeVar

T = TypeVar("T")

_REGISTRIES: dict[str, dict[str, type]] = defaultdict(dict)


def register(kind: str, name: str) -> Callable[[type[T]], type[T]]:
    """Class decorator: register `cls` as the `name` implementation of `kind`.

    Example:
        @register("embedder", "local_bge")
        class LocalBGEEmbedder(Embedder): ...
    """

    def decorator(cls: type[T]) -> type[T]:
        _REGISTRIES[kind][name] = cls
        return cls

    return decorator


def build(kind: str, name: str, **kwargs: Any) -> Any:
    """Instantiate the registered `name` implementation of `kind`."""
    try:
        cls = _REGISTRIES[kind][name]
    except KeyError:
        available_names = list(_REGISTRIES[kind])
        raise ValueError(
            f"Unknown {kind} provider '{name}'. Available: {available_names}"
        ) from None
    return cls(**kwargs)


def available(kind: str) -> list[str]:
    """List registered provider names for a given kind."""
    return list(_REGISTRIES[kind])
