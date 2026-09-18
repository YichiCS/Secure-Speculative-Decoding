from typing import Callable

_IMPLS: dict[str, Callable] = {}


def register(name: str):
    def decorate(fn: Callable) -> Callable:
        if name in _IMPLS:
            raise RuntimeError(f"verifier {name!r} registered twice")
        _IMPLS[name] = fn
        return fn

    return decorate


def get_impl(name: str) -> Callable:
    try:
        return _IMPLS[name]
    except KeyError:
        raise ValueError(
            f"Unsupported verifier method: {name}. Supported: {sorted(_IMPLS)}"
        ) from None
