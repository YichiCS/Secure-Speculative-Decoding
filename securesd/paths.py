import os


def _detect_cuda_arch() -> str:
    try:
        import torch

        if torch.cuda.is_available():
            major, minor = torch.cuda.get_device_capability(0)
            return f"{major}.{minor}"
    except Exception:
        pass
    raise RuntimeError(
        "Could not detect a CUDA device to set TORCH_CUDA_ARCH_LIST. "
        "Set SSD_CUDA_ARCH explicitly (e.g. SSD_CUDA_ARCH=9.0 for H100, "
        "12.0 for RTX PRO 6000 Blackwell)."
    )


def configure_cuda_arch() -> str:
    arch = os.environ.get("SSD_CUDA_ARCH") or _detect_cuda_arch()
    os.environ["TORCH_CUDA_ARCH_LIST"] = arch
    return arch
