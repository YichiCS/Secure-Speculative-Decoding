from __future__ import annotations

import time
from contextlib import contextmanager

import torch

_GB = 1024 ** 3


class ResourceTracker:

    def __init__(self) -> None:
        self.phases: dict[str, float] = {}
        self._start = time.perf_counter()
        self.reset_peak()

    @staticmethod
    def _cuda() -> bool:
        return torch.cuda.is_available()

    def reset_peak(self) -> None:
        if self._cuda():
            torch.cuda.reset_peak_memory_stats()

    @contextmanager
    def phase(self, name: str):
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self.phases[name] = self.phases.get(name, 0.0) + (time.perf_counter() - t0)

    def summary(self) -> dict:
        out: dict = {
            "wall_time_s": time.perf_counter() - self._start,
            "phase_time_s": dict(self.phases),
        }
        if not self._cuda():
            return out
        torch.cuda.synchronize()
        device = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(device)
        out.update(
            gpu_name=props.name,
            gpu_count_visible=torch.cuda.device_count(),
            gpu_total_memory_gb=props.total_memory / _GB,
            peak_memory_allocated_gb=torch.cuda.max_memory_allocated(device) / _GB,
            peak_memory_reserved_gb=torch.cuda.max_memory_reserved(device) / _GB,
        )
        return out


def format_resources(res: dict) -> str:
    phases = ", ".join(f"{k}={v:.1f}s" for k, v in res.get("phase_time_s", {}).items())
    if "peak_memory_reserved_gb" not in res:
        return f"wall={res['wall_time_s']:.1f}s ({phases})"
    return (
        f"wall={res['wall_time_s']:.1f}s ({phases}), "
        f"peak GPU mem alloc/reserved="
        f"{res['peak_memory_allocated_gb']:.1f}/{res['peak_memory_reserved_gb']:.1f} GiB "
        f"of {res['gpu_total_memory_gb']:.0f} GiB on {res['gpu_name']}"
    )
