import os

from securesd.methods import (
    AR_METHODS,
    METHOD_CHOICES,
    METHODS,
    SD_METHODS,
    get_spec,
    is_sd_method,
)

DEFAULT_TARGET_MODEL = "Qwen/Qwen3-8B"
DEFAULT_DRAFT_MODEL = "Qwen/Qwen3-0.6B"

ETA_SCHEDULE_CHOICES = ("linear", "step", "power")


def default_hf_root() -> str:
    return os.environ.get("SECURESD_HF_ROOT") or os.environ.get("HF_HUB_CACHE") or (
        os.path.join(
            os.environ.get("HF_HOME") or os.path.expanduser("~/.cache/huggingface"),
            "hub",
        )
    )
