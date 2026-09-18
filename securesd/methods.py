from __future__ import annotations

from dataclasses import dataclass
from typing import Any


def float_tag(value: float) -> str:
    text = f"{float(value):.6f}".rstrip("0").rstrip(".")
    return "0" if text in ("", "-0") else text


@dataclass(frozen=True)
class ParamSpec:

    name: str
    type: type
    default: Any
    tag: str
    label: str
    choices: tuple[str, ...] | None = None
    min: float | None = None
    max: float | None = None

    def tagged(self, value: Any) -> str:
        rendered = float_tag(value) if self.type is float else str(value)
        return f"_{self.tag}{rendered}"

    def validate(self, value: Any) -> None:
        if value is None:
            raise ValueError(f"{self.name} is required")
        if self.choices is not None and value not in self.choices:
            raise ValueError(
                f"Unsupported {self.name}={value!r}. Supported: {list(self.choices)}"
            )
        if self.min is not None and value < self.min:
            raise ValueError(f"{self.name} must be >= {self.min}, got {value}")
        if self.max is not None and value > self.max:
            raise ValueError(f"{self.name} must be <= {self.max}, got {value}")


@dataclass(frozen=True)
class MethodSpec:

    name: str
    kind: str
    params: tuple[ParamSpec, ...] = ()
    recovery_kind: str = "sd"

    @property
    def is_sd(self) -> bool:
        return self.kind == "sd"

    def filename_tag(self, values: dict[str, Any]) -> str:
        return "".join(p.tagged(values[p.name]) for p in self.params)

    def summary_tag(self, values: dict[str, Any]) -> str:
        return "".join(f", {p.label}={values[p.name]}" for p in self.params)

    def validate(self, values: dict[str, Any]) -> None:
        for p in self.params:
            p.validate(values.get(p.name))


_SPECS: tuple[MethodSpec, ...] = (
    MethodSpec("ar_target", kind="ar"),
    MethodSpec("ar_draft", kind="ar"),
    MethodSpec("sd", kind="sd"),
    MethodSpec(
        "lossy_sd", kind="sd",
        params=(ParamSpec("lossy_epsilon", float, 0.0, "eps", "epsilon", min=0.0),),
    ),
    MethodSpec(
        "fsd", kind="sd", recovery_kind="target",
        params=(
            ParamSpec("fsd_div_type", str, "js_div", "div", "div",
                      choices=("js_div", "kl_div", "tv_div")),
            ParamSpec("fsd_threshold", float, None, "th", "threshold", min=0.0),
        ),
    ),
    MethodSpec(
        "bild", kind="sd", recovery_kind="argmax",
        params=(
            ParamSpec("bild_fallback_threshold", float, 0.35, "ft", "fallback",
                      min=0.0, max=1.0),
            ParamSpec("bild_rollback_threshold", float, 8.0, "rt", "rollback", min=0.0),
        ),
    ),
    MethodSpec(
        "mars", kind="sd", recovery_kind="argmax",
        params=(ParamSpec("mars_theta", float, 0.9, "th", "theta", min=0.0, max=1.0),),
    ),
    MethodSpec(
        "sc", kind="sd",
        params=(
            ParamSpec("sc_rule", str, "chow", "r", "rule",
                      choices=("chow", "diff", "opt")),
            ParamSpec("sc_alpha", float, 0.2, "a", "alpha", min=0.0),
        ),
    ),
    MethodSpec(
        "fly", kind="sd", recovery_kind="target",
        params=(
            ParamSpec("fly_entropy_threshold", float, 0.3, "th", "theta",
                      min=0.0, max=1.0),
            ParamSpec("fly_window_size", int, 6, "w", "window", min=0),
        ),
    ),
)

METHODS: dict[str, MethodSpec] = {s.name: s for s in _SPECS}
METHOD_CHOICES: tuple[str, ...] = tuple(METHODS)
AR_METHODS: tuple[str, ...] = tuple(s.name for s in _SPECS if s.kind == "ar")
SD_METHODS: tuple[str, ...] = tuple(s.name for s in _SPECS if s.kind == "sd")

ALL_PARAMS: tuple[ParamSpec, ...] = tuple(
    p for s in _SPECS for p in s.params
)
PARAMS_BY_NAME: dict[str, ParamSpec] = {p.name: p for p in ALL_PARAMS}


def get_spec(method: str) -> MethodSpec:
    try:
        return METHODS[method]
    except KeyError:
        raise ValueError(
            f"Unsupported method: {method}. Supported: {list(METHOD_CHOICES)}"
        ) from None


def is_sd_method(method: str | None) -> bool:
    return method in METHODS and METHODS[method].is_sd


def method_params(method: str, source: Any) -> dict[str, Any]:
    get = source.get if isinstance(source, dict) else (lambda k, d=None: getattr(source, k, d))
    return {p.name: get(p.name, p.default) for p in get_spec(method).params}
