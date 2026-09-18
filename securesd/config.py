import os
from dataclasses import dataclass, field

import torch
from transformers import AutoConfig

from securesd.methods import is_sd_method, method_params


def _load_hf_config(model: str) -> AutoConfig:
    if not os.path.isdir(model):
        raise ValueError(f"Model path is not a directory: {model}")
    cfg = AutoConfig.from_pretrained(model)
    if cfg.dtype is None:
        raise ValueError(f"Model config for {model} must define dtype")
    return cfg


@dataclass
class Config:
    model: str | None = None
    draft: str | None = None
    kvcache_block_size: int = 256
    max_num_seqs: int = 1
    max_model_len: int = 4096
    max_num_batched_tokens: int = 16384
    max_steps: int | None = None
    gpu_memory_utilization: float = 0.7
    draft_gpu_memory_utilization: float = 0.75

    method: str | None = None
    speculate_k: int = 1

    method_params: dict = field(default_factory=dict)

    eta_schedule: str | None = "step"
    eta_start: float = 0.0
    eta_end: float = 1.0
    eta_len: float = 2.0
    eta_gamma: float = 1.0

    record_distribution_diagnostics: bool = False

    speculate: bool = False
    hf_config: AutoConfig | None = None
    draft_hf_config: AutoConfig | None = None
    eos: int = -1
    num_kvcache_blocks: int = -1
    device: torch.device = torch.device("cuda")

    def __post_init__(self):
        if self.method is None:
            raise ValueError("method is required")
        if self.model is None:
            raise ValueError("model is required")
        if not 0.0 < self.gpu_memory_utilization <= 1.0:
            raise ValueError("gpu_memory_utilization must be in (0, 1]")
        if not 0.0 < self.draft_gpu_memory_utilization <= 1.0:
            raise ValueError("draft_gpu_memory_utilization must be in (0, 1]")

        self.speculate = is_sd_method(self.method)
        self.method_params = method_params(self.method, self.method_params)
        if self.speculate and self.speculate_k < 1:
            raise ValueError(f"speculate_k must be >= 1, got {self.speculate_k}")
        if self.method == "ar_draft":
            if self.draft is None:
                raise ValueError("ar_draft requires draft")
            self.model = self.draft

        self.hf_config = _load_hf_config(self.model)
        if self.max_model_len > self.hf_config.max_position_embeddings:
            raise ValueError(
                f"max_model_len={self.max_model_len} exceeds model max_position_embeddings="
                f"{self.hf_config.max_position_embeddings}"
            )

        if self.speculate:
            if self.draft is None:
                raise ValueError(f"{self.method} requires draft")
            self.draft_hf_config = _load_hf_config(self.draft)
            if self.max_model_len > self.draft_hf_config.max_position_embeddings:
                raise ValueError(
                    f"max_model_len={self.max_model_len} exceeds draft max_position_embeddings="
                    f"{self.draft_hf_config.max_position_embeddings}"
                )
            if self.draft_hf_config.vocab_size != self.hf_config.vocab_size:
                raise ValueError("target and draft must share vocab size")

        if self.max_num_batched_tokens < self.max_model_len:
            raise ValueError(
                f"max_num_batched_tokens={self.max_num_batched_tokens} must be >= max_model_len={self.max_model_len}"
            )
