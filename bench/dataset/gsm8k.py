from __future__ import annotations

import re
from typing import Any

from datasets import load_dataset
from transformers import AutoTokenizer

from dataset import DatasetBundle, render_chat_prompt


SPECIAL_STOP_MARKERS = ("<|im_end|>", "<|endoftext|>", "</s>", "<|eot_id|>")
_NUM_RE = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?")
_ANS_RE = re.compile(r"####\s*([^\n]+)")


def build_gsm8k_bundle(
    model_path: str,
    think: bool,
    num_seqs: int,
    prompt_offset: int,
) -> DatasetBundle:
    ds = load_dataset("gsm8k", "main", split="test")
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    start = max(int(prompt_offset), 0)
    end = min(len(ds), start + int(num_seqs))

    prompts: list[list[int]] = []
    samples: list[dict[str, Any]] = []
    for i in range(start, end):
        ex = ds[i]
        question = str(ex["question"]).strip()
        answer_text = str(ex["answer"])
        answer_value = _extract_gold_value(answer_text)
        if answer_value is None:
            raise ValueError(f"Failed to parse GSM8K gold answer at index={i}")

        prompt = (
            "Solve the following math word problem.\n"
            "Give the final numeric answer clearly.\n\n"
            f"Question: {question}"
        )
        rendered = render_chat_prompt(tokenizer, prompt, think)
        prompts.append(tokenizer.encode(rendered, add_special_tokens=False))
        samples.append(
            {
                "index": len(samples),
                "question": question,
                "gold_answer_text": answer_text,
                "gold_answer_value": answer_value,
            }
        )

    if not prompts:
        raise ValueError(
            "No samples selected from dataset=gsm8k. "
            f"Check --prompt_offset ({start}) and --num_seqs ({num_seqs})."
        )

    return DatasetBundle(
        dataset="gsm8k",
        prompts=prompts,
        samples=samples,
        meta={"prompt_offset": start},
    )


def evaluate_gsm8k(outputs: list[dict], bundle: DatasetBundle, _args) -> dict:
    if len(outputs) != len(bundle.samples):
        raise ValueError("gsm8k samples and outputs length mismatch")

    sample_results: list[dict[str, Any]] = []
    correct = 0
    for sample, output in zip(bundle.samples, outputs):
        completion = _normalize_output_text(output["text"])
        pred = _extract_pred_value(completion)
        gold = sample["gold_answer_value"]
        is_correct = pred is not None and pred == gold
        correct += int(is_correct)
        sample_results.append(
            {
                "index": sample["index"],
                "gold_answer_value": gold,
                "pred_answer_value": pred,
                "correct": is_correct,
            }
        )

    total = len(sample_results)
    accuracy = (correct / total) if total else 0.0
    return {
        "suite": "GSM8K",
        "primary_metric": "accuracy",
        "primary_score": accuracy,
        "accuracy": accuracy,
        "num_samples": total,
        "num_correct": correct,
        "sample_results": sample_results,
    }


def build_gsm8k_report_samples(
    outputs: list[dict],
    bundle: DatasetBundle,
    util: dict | None,
) -> list[dict]:
    eval_map: dict[int, dict[str, Any]] = {}
    if util is not None:
        eval_map = {item["index"]: item for item in util["sample_results"]}

    rows: list[dict[str, Any]] = []
    for sample, output in zip(bundle.samples, outputs):
        idx = sample["index"]
        row = {
            "index": idx,
            "question": sample["question"],
            "gold_answer_text": sample["gold_answer_text"],
            "gold_answer_value": sample["gold_answer_value"],
            "completion": output["text"],
            "completion_tokens": len(output["token_ids"]),
        }
        if idx in eval_map:
            row.update(eval_map[idx])
        rows.append(row)
    return rows


def _normalize_output_text(text: str) -> str:
    out = text
    if "</think>" in out:
        out = out.split("</think>", 1)[1]
    for marker in SPECIAL_STOP_MARKERS:
        if marker in out:
            out = out.split(marker, 1)[0]
    return out.strip()


def _normalize_num_str(s: str) -> str | None:
    t = str(s).strip().replace(",", "")
    t = re.sub(r"^[^0-9+\-]*", "", t)
    t = re.sub(r"[^0-9.]+$", "", t)
    if not t:
        return None
    if t.startswith("+"):
        t = t[1:]
    if t.count(".") > 1:
        return None
    if "." in t:
        t = t.rstrip("0").rstrip(".")
    if t == "-0":
        t = "0"
    return t


def _extract_gold_value(answer_text: str) -> str | None:
    m = _ANS_RE.search(answer_text)
    if not m:
        return None
    nums = _NUM_RE.findall(m.group(1))
    if not nums:
        return None
    return _normalize_num_str(nums[-1])


def _extract_pred_value(completion: str) -> str | None:
    if not completion:
        return None
    m = _ANS_RE.search(completion)
    if m:
        nums = _NUM_RE.findall(m.group(1))
        if nums:
            return _normalize_num_str(nums[-1])
    nums = _NUM_RE.findall(completion)
    if not nums:
        return None
    return _normalize_num_str(nums[-1])
