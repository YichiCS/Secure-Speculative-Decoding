#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from importlib.metadata import version
from pathlib import Path

from huggingface_hub import snapshot_download


REPO = Path(__file__).resolve().parents[2]
METHODS = ["lossy_sd", "bild", "sc", "fsd", "fly", "mars"]
HF_ALLOW_PATTERNS = [
    "*.json",
    "*.model",
    "*.safetensors",
    "tokenizer*",
    "special_tokens_map.json",
]


def _conditions() -> list[dict]:
    rows = [
        {"name": "ar_target", "method": "ar_target", "eta_len": 1},
        {"name": "ar_draft", "method": "ar_draft", "eta_len": 1},
    ]
    for method in METHODS:
        rows.extend([
            {"name": f"{method}_l1", "method": method, "eta_len": 1},
            {"name": f"{method}_l2", "method": method, "eta_len": 2},
        ])
    return rows


def _wait_ready(port: int, process: subprocess.Popen, timeout_s: float) -> None:
    deadline = time.monotonic() + timeout_s
    url = f"http://127.0.0.1:{port}/health"
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"server exited during startup with code {process.returncode}")
        try:
            with urllib.request.urlopen(url, timeout=2) as response:
                if response.status == 200:
                    return
        except (urllib.error.URLError, TimeoutError):
            pass
        time.sleep(1)
    raise TimeoutError(f"server did not become ready within {timeout_s}s")


def _stop_process(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=60)
        return
    except subprocess.TimeoutExpired:
        pass
    os.killpg(process.pid, signal.SIGKILL)
    process.wait(timeout=30)


def _merge_resume_stats(previous_path: Path, current_path: Path) -> None:
    previous = json.loads(previous_path.read_text())
    current = json.loads(current_path.read_text())
    requests = []
    for part, payload in (("pre_resume", previous), ("resume", current)):
        for row in payload["requests"]:
            requests.append({**row, "resume_part": part, "request_index": len(requests)})
    errors = [
        {**row, "resume_part": part}
        for part, payload in (("pre_resume", previous), ("resume", current))
        for row in payload["errors"]
    ]
    additive = (
        "num_requests", "num_errors", "prompt_tokens", "completion_tokens", "decode_tokens",
        "decode_time_s", "prefill_time_s", "wall_time_s",
    )
    summary = {key: previous["summary"][key] + current["summary"][key] for key in additive}
    summary["decode_tps"] = summary["decode_tokens"] / summary["decode_time_s"]
    merged = {
        "meta": {
            **current["meta"],
            "resumed": True,
            "resume_stats_parts": [str(previous_path), str(current_path)],
        },
        "summary": summary,
        "requests": requests,
        "errors": errors,
    }
    current_path.write_text(json.dumps(merged, indent=2) + "\n")


def _run_condition(
    condition: dict,
    gpu: int,
    port: int,
    args: argparse.Namespace,
    target_path: str,
    draft_path: str,
) -> dict:
    name = condition["name"]
    root = args.output_root / name
    root.mkdir(parents=True, exist_ok=True)
    benchmark_path = root / "benchmark.json"
    stats_path = root / "server_stats.json"
    status_path = root / "status.json"
    if benchmark_path.is_file() and stats_path.is_file() and not args.force:
        return {"condition": name, "gpu": gpu, "status": "skipped_complete"}

    previous_stats_path: Path | None = None
    if args.resume_benchmark and stats_path.is_file():
        part_index = 0
        while True:
            candidate = root / f"server_stats.pre_resume_{part_index}.json"
            if not candidate.exists():
                previous_stats_path = candidate
                stats_path.replace(candidate)
                break
            part_index += 1

    server_cmd = [
        sys.executable,
        str(REPO / "scripts/agentdojo/serve_securesd_openai.py"),
        "--condition", name,
        "--method", condition["method"],
        "--eta-len", str(condition["eta_len"]),
        "--target-model", args.target_model,
        "--draft-model", args.draft_model,
        "--target-path", target_path,
        "--draft-path", draft_path,
        "--served-model-name", f"securesd-{name}",
        "--port", str(port),
        "--stats-path", str(stats_path),
        "--max-new-tokens", str(args.max_new_tokens),
        "--max-model-len", str(args.max_model_len),
        "--gpu-memory-utilization", str(args.gpu_memory_utilization),
        "--draft-gpu-memory-utilization", str(args.draft_gpu_memory_utilization),
        "--seed", str(args.seed),
        "--physical-gpu", str(gpu),
    ]
    benchmark_cmd = [
        str(args.dojo_python),
        str(REPO / "scripts/agentdojo/run_agentdojo_slice.py"),
        "--condition", name,
        "--model-id", f"securesd-{name}",
        "--output", str(benchmark_path),
        "--logdir", str(root / "logs"),
        "--benchmark-version", args.benchmark_version,
        "--attack", args.attack,
    ]
    if args.suites:
        benchmark_cmd.extend(["--suites", *args.suites])
    if args.selection_json is not None:
        benchmark_cmd.extend(["--selection-json", str(args.selection_json.resolve())])
    if args.full_benchmark:
        benchmark_cmd.append("--full-benchmark")
    if args.resume_benchmark:
        benchmark_cmd.append("--resume")

    env = os.environ.copy()
    env.update({
        "CUDA_VISIBLE_DEVICES": str(gpu),
        "LOCAL_LLM_PORT": str(port),
        "PYTHONUNBUFFERED": "1",
        "PYTORCH_ALLOC_CONF": "expandable_segments:True",
        "SSD_CUDA_ARCH": env.get("SSD_CUDA_ARCH", "12.0"),
    })
    started = time.time()
    process: subprocess.Popen | None = None
    result = {"condition": name, "gpu": gpu, "port": port, "status": "failed"}
    with (root / "server.log").open("w") as server_log, (root / "benchmark.log").open("w") as benchmark_log:
        try:
            process = subprocess.Popen(
                server_cmd,
                cwd=REPO,
                env=env,
                stdout=server_log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            _wait_ready(port, process, args.startup_timeout)
            completed = subprocess.run(
                benchmark_cmd,
                cwd=REPO,
                env=env,
                stdout=benchmark_log,
                stderr=subprocess.STDOUT,
                timeout=args.condition_timeout,
            )
            if completed.returncode != 0:
                raise RuntimeError(f"AgentDojo runner exited with code {completed.returncode}")
            result["status"] = "completed"
        except Exception as exc:
            result["error"] = f"{type(exc).__name__}: {exc}"
        finally:
            if process is not None:
                _stop_process(process)
            if (
                result["status"] == "completed"
                and previous_stats_path is not None
                and stats_path.is_file()
            ):
                _merge_resume_stats(previous_stats_path, stats_path)
            result["elapsed_s"] = time.time() - started
            status_path.write_text(json.dumps(result, indent=2) + "\n")
    return result


def _worker(
    gpu: int,
    jobs: list[tuple[int, dict]],
    args: argparse.Namespace,
    target_path: str,
    draft_path: str,
) -> list[dict]:
    return [
        _run_condition(job, gpu, args.base_port + index, args, target_path, draft_path)
        for index, job in jobs
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=REPO / ".results/agentdojo_qwen_pilot")
    parser.add_argument("--dojo-python", type=Path, default=Path(os.environ.get("AGENTDOJO_VENV", "/dev/shm/agentdojo_venv")) / "bin/python")
    parser.add_argument("--target-model", default="Qwen/Qwen3-8B")
    parser.add_argument("--draft-model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--hf-cache", default=str(Path.home() / ".cache/huggingface/hub"))
    parser.add_argument("--benchmark-version", default="v1.2.2")
    parser.add_argument("--attack", default="tool_knowledge")
    parser.add_argument("--suites", nargs="+", choices=["workspace", "travel", "banking", "slack"])
    parser.add_argument("--selection-json", type=Path)
    parser.add_argument(
        "--full-benchmark",
        action="store_true",
        help="Run every user/injection task instead of the default pilot selection.",
    )
    parser.add_argument("--gpus", nargs="+", type=int, default=[0, 1, 2, 3])
    parser.add_argument("--conditions", nargs="+", help="Optional subset of condition names.")
    parser.add_argument("--base-port", type=int, default=18000)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.25)
    parser.add_argument("--draft-gpu-memory-utilization", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--startup-timeout", type=float, default=900)
    parser.add_argument("--condition-timeout", type=float, default=7200)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--resume-benchmark",
        action="store_true",
        help="Reuse existing AgentDojo traces and merge prior server statistics.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.full_benchmark and args.selection_json is not None:
        raise ValueError("--full-benchmark and --selection-json are mutually exclusive")
    if not args.dojo_python.is_file():
        raise FileNotFoundError(
            f"AgentDojo interpreter not found: {args.dojo_python}. "
            "Create it with uv venv and install agentdojo==0.1.35."
        )
    for field in ("gpu_memory_utilization", "draft_gpu_memory_utilization"):
        value = float(getattr(args, field))
        if not 0.0 < value <= 1.0:
            raise ValueError(f"{field} must be in (0, 1], got {value}")
    all_conditions = _conditions()
    if args.conditions:
        wanted = set(args.conditions)
        unknown = wanted - {x["name"] for x in all_conditions}
        if unknown:
            raise ValueError(f"unknown conditions: {sorted(unknown)}")
        all_conditions = [x for x in all_conditions if x["name"] in wanted]
    if not all_conditions:
        raise ValueError("no conditions selected")

    target_path = snapshot_download(
        args.target_model, cache_dir=args.hf_cache, allow_patterns=HF_ALLOW_PATTERNS
    )
    draft_path = snapshot_download(
        args.draft_model, cache_dir=args.hf_cache, allow_patterns=HF_ALLOW_PATTERNS
    )
    args.output_root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "agentdojo_package_version": version("agentdojo") if "agentdojo" in sys.modules else "0.1.35",
        "benchmark_version": args.benchmark_version,
        "attack": args.attack,
        "full_benchmark": args.full_benchmark,
        "selection_json": str(args.selection_json.resolve()) if args.selection_json else None,
        "target_model": args.target_model,
        "draft_model": args.draft_model,
        "target_path": target_path,
        "draft_path": draft_path,
        "conditions": all_conditions,
        "gpus": args.gpus,
        "max_new_tokens": args.max_new_tokens,
        "max_model_len": args.max_model_len,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "draft_gpu_memory_utilization": args.draft_gpu_memory_utilization,
        "seed": args.seed,
        "operating_point_basis": "approximately 0.60 mean L=1 acceptance in existing Qwen four-dataset runs",
        "operating_points": {
            "lossy_sd": {"lossy_epsilon": 0.3},
            "bild": {"fallback_threshold": 0.28, "rollback_threshold": 2.0},
            "sc": {"rule": "chow", "alpha": 0.36},
            "fsd": {"divergence": "js_div", "threshold": 0.26},
            "fly": {"entropy_threshold": 0.1, "window_size": 0},
            "mars": {"theta": 0.945},
        },
    }
    (args.output_root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    queues: dict[int, list[tuple[int, dict]]] = {gpu: [] for gpu in args.gpus}
    for index, condition in enumerate(all_conditions):
        if condition["method"] == "ar_target":
            gpu = args.gpus[0]
        elif condition["method"] == "ar_draft":
            gpu = args.gpus[min(1, len(args.gpus) - 1)]
        else:
            method_index = METHODS.index(condition["method"])
            gpu = args.gpus[(method_index + 2) % len(args.gpus)]
        queues[gpu].append((index, condition))
    results: list[dict] = []
    with ThreadPoolExecutor(max_workers=len(args.gpus)) as executor:
        futures = [
            executor.submit(_worker, gpu, jobs, args, target_path, draft_path)
            for gpu, jobs in queues.items() if jobs
        ]
        for future in futures:
            results.extend(future.result())
    results.sort(key=lambda x: next(i for i, c in enumerate(all_conditions) if c["name"] == x["condition"]))
    (args.output_root / "matrix_status.json").write_text(json.dumps(results, indent=2) + "\n")
    failures = [x for x in results if x["status"] == "failed"]
    print(json.dumps(results, indent=2), flush=True)
    if failures:
        raise RuntimeError(f"{len(failures)} AgentDojo conditions failed")


if __name__ == "__main__":
    main()
