import torch
from torch import nn
import torch.nn.functional as F

from securesd.utils.context import get_context


class VocabParallelEmbedding(nn.Module):

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
    ):
        super().__init__()

        self.num_embeddings = num_embeddings
        self.weight = nn.Parameter(torch.empty(self.num_embeddings, embedding_dim))
        self.weight.weight_loader = self.weight_loader

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        if param.data.size() != loaded_weight.size():
            raise ValueError(
                f"weight size mismatch: param={tuple(param.data.size())} "
                f"loaded={tuple(loaded_weight.size())}"
            )
        param.data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor):
        return F.embedding(x, self.weight)


class ParallelLMHead(VocabParallelEmbedding):

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        bias: bool = False,
    ):
        if bias:
            raise NotImplementedError("ParallelLMHead bias is not supported")
        super().__init__(num_embeddings, embedding_dim)

    def forward(self, x: torch.Tensor, last_only: bool = True):
        context = get_context()
        if context.cu_seqlens_q is None:
            return F.linear(x, self.weight)

        if context.is_prefill:
            if not last_only:
                return F.linear(x, self.weight)
            last_indices = context.cu_seqlens_q[1:] - 1
            return F.linear(x[last_indices].contiguous(), self.weight)

        flat_logits = F.linear(x, self.weight)
        batch_size = context.cu_seqlens_q.size(0) - 1
        total_tokens = x.size(0)
        if total_tokens % batch_size == 0:
            constant_query_len = total_tokens // batch_size
            return flat_logits.view(batch_size, constant_query_len, flat_logits.size(-1))
        return flat_logits
