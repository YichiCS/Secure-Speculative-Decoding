import math
from functools import lru_cache
import torch
from torch import nn


def apply_rotary_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    cos = cos.unsqueeze(-2)
    sin = sin.unsqueeze(-2)
    x1, x2 = torch.chunk(x.float(), 2, dim=-1)
    y1 = x1 * cos - x2 * sin
    y2 = x2 * cos + x1 * sin
    return torch.cat((y1, y2), dim=-1).to(x.dtype)


class RotaryEmbedding(nn.Module):

    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: float,
    ) -> None:
        super().__init__()
        self.head_size = head_size
        if rotary_dim != head_size:
            raise ValueError(f"rotary_dim={rotary_dim} must equal head_size={head_size}")
        self.rotary_dim = rotary_dim
        self.base = base
        inv_freq = self._compute_inv_freq()
        t = torch.arange(max_position_embeddings, dtype=torch.float)
        freqs = torch.einsum("i,j -> ij", t, inv_freq)
        cos = freqs.cos()
        sin = freqs.sin()
        cache = torch.cat((cos, sin), dim=-1)
        self.register_buffer("cos_sin_cache", cache, persistent=False)

    def _compute_inv_freq(self) -> torch.Tensor:
        return 1.0 / (self.base**(torch.arange(0, self.rotary_dim, 2, dtype=torch.float) / self.rotary_dim))

    @torch.compile
    def forward(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        num_tokens = positions.size(0)
        cos_sin = self.cos_sin_cache[positions]
        cos, sin = cos_sin.chunk(2, dim=-1)
        query_shape = query.shape

        num_q_heads = query.shape[-1] // self.head_size
        query = query.view(num_tokens, num_q_heads, self.head_size)
        query = apply_rotary_emb(query, cos, sin).view(query_shape)
        key_shape = key.shape
        num_k_heads = key.shape[-1] // self.head_size
        key = key.view(num_tokens, num_k_heads, self.head_size)
        key = apply_rotary_emb(key, cos, sin).view(key_shape)
        return query, key


class Llama3RotaryEmbedding(RotaryEmbedding):

    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: float,
        scaling_factor: float,
        low_freq_factor: float,
        high_freq_factor: float,
        orig_max_position: int,
    ) -> None:
        self.scaling_factor = scaling_factor
        self.low_freq_factor = low_freq_factor
        self.high_freq_factor = high_freq_factor
        self.orig_max_position = orig_max_position
        super().__init__(head_size, rotary_dim, max_position_embeddings, base)

    def _compute_inv_freq(self) -> torch.Tensor:
        inv_freqs = super()._compute_inv_freq()
        low_freq_wavelen = self.orig_max_position / self.low_freq_factor
        high_freq_wavelen = self.orig_max_position / self.high_freq_factor

        wave_len = 2 * math.pi / inv_freqs
        smooth_inv_freqs = torch.where(
            wave_len < high_freq_wavelen,
            inv_freqs,
            inv_freqs / self.scaling_factor,
        )
        smooth_factor = (self.orig_max_position / wave_len - self.low_freq_factor) / (
            self.high_freq_factor - self.low_freq_factor
        )
        smooth_inv_freqs = torch.where(
            (wave_len >= high_freq_wavelen) & (wave_len <= low_freq_wavelen),
            (1 - smooth_factor) * inv_freqs / self.scaling_factor + smooth_factor * inv_freqs,
            smooth_inv_freqs,
        )
        return smooth_inv_freqs


@lru_cache(1)
def _get_rope_cached(
    head_size: int,
    rotary_dim: int,
    max_position: int,
    base: float,
    rope_scaling_items: tuple | None,
):
    if rope_scaling_items is None:
        return RotaryEmbedding(head_size, rotary_dim, max_position, base)

    rope_scaling = dict(rope_scaling_items)
    rope_type = rope_scaling.get("rope_type", rope_scaling.get("type"))
    if rope_type == "llama3":
        return Llama3RotaryEmbedding(
            head_size,
            rotary_dim,
            max_position,
            base,
            scaling_factor=rope_scaling["factor"],
            low_freq_factor=rope_scaling["low_freq_factor"],
            high_freq_factor=rope_scaling["high_freq_factor"],
            orig_max_position=rope_scaling["original_max_position_embeddings"],
        )
    raise NotImplementedError(f"rope_scaling type {rope_type!r} is not supported, got {rope_scaling}")


def get_rope(
    head_size: int,
    rotary_dim: int,
    max_position: int,
    base: float,
    rope_scaling: dict | None = None,
):
    rope_scaling_items = tuple(sorted(rope_scaling.items())) if rope_scaling is not None else None
    return _get_rope_cached(head_size, rotary_dim, max_position, base, rope_scaling_items)
