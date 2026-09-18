import torch

from securesd.constants import ETA_SCHEDULE_CHOICES


def compute_eta(
    schedule: str,
    start: float,
    end: float,
    length: float,
    positions: torch.Tensor,
    gamma: float = 1.0,
) -> torch.Tensor:
    t = positions.to(torch.float32)
    if schedule in ("linear", "power"):
        if length <= 0:
            raise ValueError(f"eta {schedule} schedule requires eta_len > 0, got {length}")
        frac = torch.clamp(t / float(length), 0.0, 1.0)
        if schedule == "power":
            if gamma <= 0:
                raise ValueError(f"eta power schedule requires eta_gamma > 0, got {gamma}")
            frac = frac.pow(float(gamma))
        return start + (end - start) * frac
    if schedule == "step":
        return torch.where(
            t < float(length),
            torch.full_like(t, float(start)),
            torch.full_like(t, float(end)),
        )
    raise ValueError(
        f"Unsupported eta_schedule: {schedule}. Supported: {list(ETA_SCHEDULE_CHOICES)}"
    )


def window_eta_bypass(
    schedule: str,
    start: float,
    end: float,
    length: float,
    min_base: int,
    max_base: int,
    num_cols: int,
) -> float | None:
    lo_pos = min_base + 1
    hi_pos = max_base + num_cols
    if schedule == "step":
        if hi_pos < length:
            return start
        if lo_pos >= length:
            return end
        return None
    if schedule in ("linear", "power"):
        if start == end:
            return start
        if length > 0 and lo_pos >= length:
            return end
        return None
    return None


def window_eta(
    schedule: str,
    start: float,
    end: float,
    length: float,
    base_offsets: torch.Tensor,
    num_cols: int,
    gamma: float = 1.0,
) -> torch.Tensor:
    cols = torch.arange(1, num_cols + 1, device=base_offsets.device)
    positions = base_offsets.view(-1, 1) + cols.view(1, -1)
    return compute_eta(schedule, start, end, length, positions, gamma)
