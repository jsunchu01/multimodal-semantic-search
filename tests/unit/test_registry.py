import pytest

from mmss.registry import build, register


def test_register_and_build() -> None:
    @register("dummy", "example")
    class Example:
        def __init__(self, value: int = 1) -> None:
            self.value = value

    instance = build("dummy", "example", value=5)
    assert instance.value == 5


def test_build_unknown_raises() -> None:
    with pytest.raises(ValueError):
        build("dummy", "does-not-exist")
