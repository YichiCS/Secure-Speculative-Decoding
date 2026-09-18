import torch
from torch import nn


class Sampler(nn.Module):

    @torch.inference_mode()
    def forward(
        self,
        logits: torch.Tensor,
        temperatures: torch.Tensor,
        greedy: bool | None = None,
    ) -> torch.Tensor:
        temps = temperatures.to(device=logits.device, dtype=torch.float32)
        zero_mask = temps == 0
        if greedy is None:
            greedy = bool(zero_mask.all())
        if greedy:
            return logits.argmax(dim=-1)

        logits_fp32 = logits.to(torch.float32)
        sample_tokens = logits_fp32.argmax(dim=-1)
        pos_idx = torch.nonzero(~zero_mask, as_tuple=False).squeeze(1)
        probs = torch.softmax(
            logits_fp32[pos_idx] / temps[pos_idx].unsqueeze(1), dim=-1, dtype=torch.float32
        )
        epsilon = 1e-10
        scores = probs.div_(torch.empty_like(probs).exponential_(1) + epsilon)
        sample_tokens[pos_idx] = scores.argmax(dim=-1)
        return sample_tokens
