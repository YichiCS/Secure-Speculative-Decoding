from __future__ import annotations

import io
import json
import os
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime
from pathlib import Path

from datasets import load_dataset
from transformers import AutoTokenizer

from dataset import DatasetBundle, render_chat_prompt


HUMANEVAL_DATASET_CONFIG = {
    "repo_id": "openai/openai_humaneval",
    "split": "test",
    "prompt_key": "prompt",
    "sample_fields": ("task_id", "prompt", "test", "entry_point"),
}


def build_humaneval_bundle(
    model_path: str,
    think: bool,
    num_seqs: int,
    prompt_offset: int,
) -> DatasetBundle:
    cfg = HUMANEVAL_DATASET_CONFIG
    ds = load_dataset(cfg["repo_id"], split=cfg["split"])
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    start = max(int(prompt_offset), 0)
    end = min(len(ds), start + int(num_seqs))

    prompts: list[list[int]] = []
    samples: list[dict] = []
    prompt_key = cfg["prompt_key"]

    for i in range(start, end):
        ex = ds[i]
        prompt = ex[prompt_key]
        token_source = render_chat_prompt(tokenizer, prompt, think)
        prompts.append(tokenizer.encode(token_source, add_special_tokens=False))
        samples.append(
            {
                "index": len(samples),
                "task_id": ex["task_id"],
                "prompt": ex["prompt"],
                "test": ex["test"],
                "entry_point": ex["entry_point"],
            }
        )

    if not prompts:
        raise ValueError(
            "No samples selected from dataset=humaneval. "
            f"Check --prompt_offset ({start}) and --num_seqs ({num_seqs})."
        )
    
    print(f"[load_dataset] finished loading dataset from {cfg['repo_id']}")

    return DatasetBundle(
        dataset="humaneval",
        prompts=prompts,
        samples=samples,
        meta={"prompt_offset": start},
    )


def evaluate_humaneval(outputs: list[dict], bundle: DatasetBundle, args) -> dict:
    if len(bundle.samples) != len(outputs):
        raise ValueError("humaneval samples and outputs length mismatch")

    samples = []
    for sample, output in zip(bundle.samples, outputs):
        samples.append(
            {
                "task_id": sample["task_id"],
                "completion": _normalize_humaneval_completion(output["text"]),
            }
        )

    return _evaluate_humaneval_pass_at_1(
        samples,
        timeout=args.utility_timeout,
        n_workers=args.utility_workers,
    )


def build_humaneval_report_samples(
    outputs: list[dict],
    bundle: DatasetBundle,
    util: dict | None,
) -> list[dict]:
    report_samples = []
    rows = None
    if util is not None:
        rows = _read_jsonl(util["results_file"])
        if len(rows) != len(outputs):
            raise ValueError("utility results length mismatch")

    for i, (sample, output) in enumerate(zip(bundle.samples, outputs)):
        row = rows[i] if rows is not None else None
        report_sample = {
            "index": sample["index"],
            "task_id": sample["task_id"],
            "prompt": sample["prompt"],
            "completion": row["completion"] if row is not None else output["text"],
            "completion_tokens": len(output["token_ids"]),
        }
        if row is not None:
            report_sample["passed"] = row["passed"]
            report_sample["result"] = row["result"]
        report_samples.append(report_sample)
    return report_samples


def _read_jsonl(path: str) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _normalize_humaneval_completion(text: str) -> str:
    out = text
    if "</think>" in out:
        out = out.split("</think>", 1)[1]
    for marker in ("<|im_end|>", "<|im_start|>", "</s>", "<|endoftext|>"):
        if marker in out:
            out = out.split(marker, 1)[0]
    if "```" in out:
        parts = out.split("```")
        if len(parts) >= 3:
            out = parts[1]
            if out.startswith("python"):
                out = out[len("python"):].lstrip()
        else:
            out = out.replace("```python", "").replace("```", "")
    return out.lstrip("\n")


def _write_jsonl(path: str, rows: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _results_root() -> str:
    env_dir = os.environ.get("SECURESD_RESULTS_DIR")
    if env_dir:
        return env_dir
    repo_root = Path(__file__).resolve().parents[2]
    return str(repo_root / ".results")


def _prepare_run_dir() -> str:
    root = _results_root()
    os.makedirs(root, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_dir = os.path.join(root, f"humaneval_eval_{stamp}")
    os.makedirs(run_dir, exist_ok=False)
    return run_dir


def _evaluate_humaneval_pass_at_1(
    samples: list[dict],
    timeout: float = 3.0,
    n_workers: int = 4,
) -> dict:
    from human_eval import evaluation as human_eval_evaluation

    if not samples:
        raise ValueError("samples is empty")

    run_dir = _prepare_run_dir()
    sample_file = os.path.join(run_dir, "samples.jsonl")
    results_file = sample_file + "_results.jsonl"
    _write_jsonl(sample_file, samples)

    with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
        results = human_eval_evaluation.evaluate_functional_correctness(
            sample_file,
            k=[1],
            n_workers=n_workers,
            timeout=timeout,
            ignore_incomplete=True,
        )

    if not isinstance(results, dict) or "pass@1" not in results:
        raise RuntimeError(f"Unexpected human-eval result format: {type(results)}")

    out = dict(results)
    out["suite"] = "HumanEval"
    out["primary_metric"] = "pass@1"
    out["primary_score"] = float(results["pass@1"])
    out["pass@1"] = float(results["pass@1"])
    out["artifact_dir"] = run_dir
    out["sample_file"] = sample_file
    out["results_file"] = results_file
    return out
