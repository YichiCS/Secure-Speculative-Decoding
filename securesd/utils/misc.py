def infer_model_family(model_path: str) -> str:
    name = model_path.lower()
    if "llama" in name:
        return "llama"
    if "qwen" in name:
        return "qwen"
    raise ValueError(f"Cannot infer model family from path: {model_path}")
