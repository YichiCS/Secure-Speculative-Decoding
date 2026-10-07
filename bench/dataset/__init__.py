from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from transformers import AutoTokenizer


@dataclass
class DatasetBundle:
    dataset: str
    prompts: list[list[int]]
    samples: list[dict[str, Any]]
    meta: dict[str, Any] = field(default_factory=dict)


def render_chat_messages(
    tokenizer: AutoTokenizer,
    messages: list[dict[str, str]],
    think: bool,
) -> str:
    if not hasattr(tokenizer, "apply_chat_template"):
        raise ValueError(
            f"Tokenizer {type(tokenizer).__name__} does not expose apply_chat_template; "
            "strict dataset formatting requires a chat-template tokenizer"
        )
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=think,
    )


def render_chat_prompt(tokenizer: AutoTokenizer, prompt: str, think: bool) -> str:
    return render_chat_messages(
        tokenizer,
        [{"role": "user", "content": prompt}],
        think=think,
    )


class DatasetAdapter(Protocol):
    name: str

    def build_bundle(self, args, model_path: str) -> DatasetBundle:
        pass

    def evaluate(self, outputs: list[dict], bundle: DatasetBundle, args) -> dict:
        pass

    def build_report_samples(
        self,
        outputs: list[dict],
        bundle: DatasetBundle,
        util: dict | None,
    ) -> list[dict]:
        pass

    def format_utility_summary(self, util: dict) -> str:
        pass


@dataclass
class HumanEvalAdapter:
    name: str = "humaneval"

    def build_bundle(self, args, model_path: str) -> DatasetBundle:
        from dataset.humaneval import build_humaneval_bundle

        return build_humaneval_bundle(
            model_path=model_path,
            think=bool(getattr(args, "think", False)),
            num_seqs=int(args.num_seqs),
            prompt_offset=int(getattr(args, "prompt_offset", 0)),
        )

    def evaluate(self, outputs: list[dict], bundle: DatasetBundle, args) -> dict:
        from dataset.humaneval import evaluate_humaneval

        return evaluate_humaneval(outputs, bundle, args)

    def build_report_samples(
        self,
        outputs: list[dict],
        bundle: DatasetBundle,
        util: dict | None,
    ) -> list[dict]:
        from dataset.humaneval import build_humaneval_report_samples

        return build_humaneval_report_samples(outputs, bundle, util)

    def format_utility_summary(self, util: dict) -> str:
        return f"HumanEval pass@1={util['pass@1']}"


@dataclass
class GSM8KAdapter:
    name: str = "gsm8k"

    def build_bundle(self, args, model_path: str) -> DatasetBundle:
        from dataset.gsm8k import build_gsm8k_bundle

        return build_gsm8k_bundle(
            model_path=model_path,
            think=bool(getattr(args, "think", False)),
            num_seqs=int(args.num_seqs),
            prompt_offset=int(getattr(args, "prompt_offset", 0)),
        )

    def evaluate(self, outputs: list[dict], bundle: DatasetBundle, args) -> dict:
        from dataset.gsm8k import evaluate_gsm8k

        return evaluate_gsm8k(outputs, bundle, args)

    def build_report_samples(
        self,
        outputs: list[dict],
        bundle: DatasetBundle,
        util: dict | None,
    ) -> list[dict]:
        from dataset.gsm8k import build_gsm8k_report_samples

        return build_gsm8k_report_samples(outputs, bundle, util)

    def format_utility_summary(self, util: dict) -> str:
        return f"GSM8K accuracy={util.get('accuracy')}"


@dataclass
class JailbreakingAdapter:
    name: str = "jailbreaking"

    def build_bundle(self, args, model_path: str) -> DatasetBundle:
        from dataset.jailbreaking import build_jailbreaking_bundle

        return build_jailbreaking_bundle(
            model_path=model_path,
            think=bool(getattr(args, "think", False)),
            num_seqs=int(args.num_seqs),
            prompt_offset=int(getattr(args, "prompt_offset", 0)),
            jb_file=getattr(args, "jb_file", None),
        )

    def evaluate(self, outputs: list[dict], bundle: DatasetBundle, args) -> dict:
        from dataset.jailbreaking import evaluate_jailbreaking

        return evaluate_jailbreaking(outputs, bundle, args)

    def build_report_samples(
        self,
        outputs: list[dict],
        bundle: DatasetBundle,
        util: dict | None,
    ) -> list[dict]:
        from dataset.jailbreaking import build_jailbreaking_report_samples

        return build_jailbreaking_report_samples(outputs, bundle, util)

    def format_utility_summary(self, util: dict) -> str:
        asr = util.get("asr", 0.0)
        safety = util.get("safety_score", 1.0 - asr)
        return f"Jailbreaking safety_score(1-ASR)={safety:.4f}, ASR={asr:.4f}"


@dataclass
class PromptInjectionAdapter:
    name: str = "prompt_injection"

    def build_bundle(self, args, model_path: str) -> DatasetBundle:
        from dataset.prompt_injection import build_prompt_injection_bundle

        return build_prompt_injection_bundle(
            model_path=model_path,
            think=bool(getattr(args, "think", False)),
            num_seqs=int(args.num_seqs),
            prompt_offset=int(getattr(args, "prompt_offset", 0)),
            pi_file=getattr(args, "pi_file", None),
        )

    def evaluate(self, outputs: list[dict], bundle: DatasetBundle, args) -> dict:
        from dataset.prompt_injection import evaluate_prompt_injection

        return evaluate_prompt_injection(outputs, bundle, args)

    def build_report_samples(
        self,
        outputs: list[dict],
        bundle: DatasetBundle,
        util: dict | None,
    ) -> list[dict]:
        from dataset.prompt_injection import build_prompt_injection_report_samples

        return build_prompt_injection_report_samples(outputs, bundle, util)

    def format_utility_summary(self, util: dict) -> str:
        asr = util.get("asr", 0.0)
        per = util.get("per_template_asr", {})
        per_str = ", ".join(f"{t}={v:.3f}" for t, v in per.items())
        return f"PromptInjection ASR={asr:.4f}, safety={1.0 - asr:.4f} | per-template: {per_str}"


_ADAPTERS: dict[str, DatasetAdapter] = {
    "humaneval": HumanEvalAdapter(),
    "gsm8k": GSM8KAdapter(),
    "jailbreaking": JailbreakingAdapter(),
    "prompt_injection": PromptInjectionAdapter(),
}


def get_dataset_adapter(name: str) -> DatasetAdapter:
    adapter = _ADAPTERS.get(name)
    if adapter is None:
        raise ValueError(f"Unsupported dataset: {name}")
    return adapter


def build_dataset_bundle(args, model_path: str) -> DatasetBundle:
    return get_dataset_adapter(args.dataset).build_bundle(args, model_path)


__all__ = [
    "DatasetAdapter",
    "DatasetBundle",
    "GSM8KAdapter",
    "HumanEvalAdapter",
    "JailbreakingAdapter",
    "PromptInjectionAdapter",
    "build_dataset_bundle",
    "get_dataset_adapter",
    "render_chat_messages",
    "render_chat_prompt",
]
