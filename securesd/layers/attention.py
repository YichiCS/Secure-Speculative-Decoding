import torch
from torch import nn
import triton
import triton.language as tl

from flash_attn.flash_attn_interface import flash_attn_varlen_func, flash_attn_with_kvcache
from securesd.utils.context import get_context


@triton.jit
def store_kvcache_kernel(
    key_ptr,
    key_stride,
    value_ptr,
    value_stride,
    k_cache_ptr,
    v_cache_ptr,
    slot_mapping_ptr,
    D: tl.constexpr,
):
    idx = tl.program_id(0)
    slot = tl.load(slot_mapping_ptr + idx)
    if slot == -1:
        return
    key_offsets = idx * key_stride + tl.arange(0, D)
    value_offsets = idx * value_stride + tl.arange(0, D)
    key = tl.load(key_ptr + key_offsets)
    value = tl.load(value_ptr + value_offsets)
    cache_offsets = slot.to(tl.int64) * D + tl.arange(0, D)
    tl.store(k_cache_ptr + cache_offsets, key)
    tl.store(v_cache_ptr + cache_offsets, value)


def store_kvcache(key: torch.Tensor, value: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, slot_mapping: torch.Tensor):
    N, num_heads, head_dim = key.shape
    D = num_heads * head_dim
    if key.stride(-1) != 1 or value.stride(-1) != 1:
        raise ValueError("key/value last-dim must be contiguous")
    if key.stride(1) != head_dim or value.stride(1) != head_dim:
        raise ValueError("key/value head-dim stride must equal head_dim")
    if k_cache.stride(1) != D or v_cache.stride(1) != D:
        raise ValueError("kv-cache stride(1) must equal num_heads*head_dim")
    if slot_mapping.numel() != N:
        raise ValueError(f"slot_mapping size {slot_mapping.numel()} != N={N}")
    store_kvcache_kernel[(N,)](key, key.stride(0), value, value.stride(0), k_cache, v_cache, slot_mapping, D)

class Attention(nn.Module):

    def __init__(
        self,
        num_heads,
        head_dim,
        scale,
        num_kv_heads,
        speculate: bool = False,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.k_cache = self.v_cache = torch.tensor([])
        self.speculate = speculate

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        o: torch.Tensor
        q = q.view(-1, self.num_heads, self.head_dim)
        k = k.view(-1, self.num_kv_heads, self.head_dim)
        v = v.view(-1, self.num_kv_heads, self.head_dim)

        k_cache, v_cache = self.k_cache, self.v_cache

        context = get_context()
        if self.k_cache.numel() and self.v_cache.numel():
            store_kvcache(k, v, self.k_cache, self.v_cache, context.slot_mapping)

        if context.is_prefill:
            if context.block_tables is not None:
                k, v = k_cache, v_cache

            k, v = k.view(-1, self.num_kv_heads, self.head_dim), v.view(-1, self.num_kv_heads, self.head_dim)
            o = flash_attn_varlen_func(q, k, v,
                                       max_seqlen_q=context.max_seqlen_q, cu_seqlens_q=context.cu_seqlens_q,
                                       max_seqlen_k=context.max_seqlen_k, cu_seqlens_k=context.cu_seqlens_k,
                                       softmax_scale=self.scale, causal=True)
        else:
            verify_or_glue = self.speculate and context.cu_seqlens_q is not None
            if verify_or_glue:
                if context.context_lens is None:
                    raise ValueError("verify decode requires context.context_lens")
                if context.max_seqlen_q <= 0:
                    raise ValueError(f"verify decode requires max_seqlen_q > 0, got {context.max_seqlen_q}")
                B = context.context_lens.shape[0]
                q_verify = q.view(B, context.max_seqlen_q, self.num_heads, self.head_dim)
                o = flash_attn_with_kvcache(
                    q_verify,
                    k_cache,
                    v_cache,
                    cache_seqlens=context.context_lens,
                    block_table=context.block_tables,
                    softmax_scale=self.scale,
                    causal=True,
                )
                o = o.view(-1, self.num_heads, self.head_dim)
            else:
                o = flash_attn_with_kvcache(
                    q.unsqueeze(1),
                    k_cache,
                    v_cache,
                    cache_seqlens=context.context_lens,
                    block_table=context.block_tables,
                    softmax_scale=self.scale,
                    causal=True,
                )

        o = o.view(-1, self.num_heads * self.head_dim)
        return o
