from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

_CACHE: dict[tuple[str, str | None], tuple] = {}


GATED = {
    "allenai/wildguard": "https://huggingface.co/allenai/wildguard",
    "meta-llama/Llama-3.2-3B-Instruct": "https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct",
}


class JudgeUnavailable(RuntimeError):
    pass


def load_judge(model_id: str, hf_root: str | None = None, *, left_pad: bool = True):
    key = (model_id, hf_root)
    cached = _CACHE.get(key)
    if cached is not None:
        return cached

    print(f"[judge] loading {model_id} ...")
    try:
        return _load(key, model_id, hf_root, left_pad)
    except Exception as exc:
        hint = ""
        if model_id in GATED:
            hint = (
                f"\n{model_id} is a gated repository. Accept its licence at "
                f"{GATED[model_id]}, then run `huggingface-cli login` (or set "
                f"HF_TOKEN) so the weights can be downloaded."
            )
        raise JudgeUnavailable(
            f"could not load the judge model {model_id}: {exc}{hint}"
        ) from exc


def _load(key, model_id: str, hf_root: str | None, left_pad: bool):
    tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=hf_root)
    if left_pad:
        tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id, cache_dir=hf_root, dtype=torch.bfloat16, device_map="auto",
    )
    model.eval()

    _CACHE[key] = (tokenizer, model)
    return tokenizer, model


def release_judges() -> None:
    _CACHE.clear()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def judge_device(model) -> torch.device:
    return next(model.parameters()).device
