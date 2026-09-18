#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(line_buffering=True)


from bench_helpers import cleanup_utility_artifacts
from dataset import DatasetBundle, get_dataset_adapter
from dataset.judges import JudgeUnavailable, release_judges
from resources import ResourceTracker, format_resources

LOCAL_ONLY = {"humaneval", "gsm8k"}


class _Args:

    def __init__(self, meta: dict, overrides: dict):
        self.hf_root = overrides.get("hf_root")
        self.utility_timeout = overrides.get("utility_timeout", 3.0)
        self.utility_workers = overrides.get("utility_workers", 16)
        self.wildguard_batch_size = overrides.get("wildguard_batch_size", 32)
        self.pi_judge_model = overrides.get(
            "pi_judge_model", "meta-llama/Llama-3.2-3B-Instruct"
        )
        self.pi_judge_batch_size = overrides.get("pi_judge_batch_size", 16)
        self.pi_attack_mode = meta.get("dataset_meta", {}).get("attack_mode", "standard")


def _write_atomic(path: Path, payload: dict) -> None:
    tmp = path.with_name(path.name + f".tmp{os.getpid()}")
    try:
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)

def find_pending(root: Path, force: bool) -> list[tuple[Path, Path]]:
    pending = []
    for raw_path in sorted(root.rglob("raw_*.json")):
        report_path = raw_path.with_name(raw_path.name[len("raw_"):])
        if not report_path.is_file():
            continue
        try:
            report = json.loads(report_path.read_text())
        except json.JSONDecodeError:
            print(f"  ! unreadable, skipping: {report_path}")
            continue
        if not force and report.get("summary", {}).get("utility"):
            continue
        pending.append((report_path, raw_path))
    return pending


def score_one(report_path: Path, raw_path: Path, overrides: dict) -> str:
    report = json.loads(report_path.read_text())
    raw = json.loads(raw_path.read_text())["raw"]

    bundle_data = raw["dataset_bundle"]
    dataset = bundle_data["dataset"]
    if dataset in LOCAL_ONLY and overrides.get("security_only"):
        return "skipped-local"

    bundle = DatasetBundle(
        dataset=dataset,
        prompts=bundle_data["prompts"],
        samples=bundle_data["samples"],
        meta=bundle_data.get("meta", {}),
    )
    adapter = get_dataset_adapter(dataset)
    args = _Args(report.get("meta", {}), overrides)

    util = adapter.evaluate(raw["outputs"], bundle, args)
    summary = {
        k: v for k, v in util.items()
        if k not in {"artifact_dir", "sample_file", "results_file", "sample_results"}
    }
    report.setdefault("summary", {})["utility"] = summary
    report["samples"] = adapter.build_report_samples(raw["outputs"], bundle, util)
    _write_atomic(report_path, report)

    raw_full = json.loads(raw_path.read_text())
    raw_full["raw"]["utility"] = util
    _write_atomic(raw_path, raw_full)

    cleanup_utility_artifacts(util)
    return f"{util.get('primary_metric', 'score')}={util.get('primary_score', float('nan')):.4f}"


def main() -> int:
    ap = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", type=Path,
                    help="Result directory to score, e.g. .results/eta_qwen_jailbreaking")
    ap.add_argument("--force", action="store_true",
                    help="Re-score reports that already have a utility block.")
    ap.add_argument("--security-only", action="store_true",
                    help="Score only the judge-model benchmarks, skipping humaneval/gsm8k.")
    ap.add_argument("--shard", default=None, metavar="I/N",
                    help="score only shard I of N (0-based), so one result tree "
                         "can be judged by N processes on N GPUs at once. The "
                         "split is over the pending list, which is sorted, so "
                         "the shards are disjoint without any coordination.")
    ap.add_argument("--limit", type=int, default=None,
                    help="Score at most N reports (for a smoke run).")
    ap.add_argument("--hf_root", type=str, default=None)
    ap.add_argument("--wildguard_batch_size", type=int, default=32)
    ap.add_argument("--pi_judge_model", type=str, default="meta-llama/Llama-3.2-3B-Instruct")
    ap.add_argument("--pi_judge_batch_size", type=int, default=16)
    args = ap.parse_args()

    if not args.root.is_dir():
        ap.error(f"not a directory: {args.root}")

    overrides = {
        "hf_root": args.hf_root,
        "wildguard_batch_size": args.wildguard_batch_size,
        "pi_judge_model": args.pi_judge_model,
        "pi_judge_batch_size": args.pi_judge_batch_size,
        "security_only": args.security_only,
    }

    resources = ResourceTracker()
    pending = find_pending(args.root, args.force)
    if args.shard:
        i, n = (int(x) for x in args.shard.split("/"))
        if not 0 <= i < n:
            raise SystemExit(f"--shard {args.shard}: need 0 <= I < N")
        pending = pending[i::n]
        print(f"[judge] shard {i}/{n}: {len(pending)} of the pending reports")
    if args.limit is not None:
        pending = pending[: args.limit]
    print(f"[judge] {len(pending)} report(s) to score under {args.root}")

    scored = failed = 0
    with resources.phase("score"):
        for i, (report_path, raw_path) in enumerate(pending, 1):
            started = time.perf_counter()
            try:
                outcome = score_one(report_path, raw_path, overrides)
            except JudgeUnavailable as exc:
                print(f"\n[judge] {exc}", file=sys.stderr)
                print(f"[judge] stopping with {len(pending) - i + 1} report(s) unscored; "
                      f"re-run this command once the model is available.", file=sys.stderr)
                failed += len(pending) - i + 1
                break
            except Exception as exc:
                failed += 1
                print(f"[{i}/{len(pending)}] FAILED {report_path.name}: {exc}")
                continue
            scored += 1
            print(f"[{i}/{len(pending)}] {report_path.name} {outcome} "
                  f"({time.perf_counter() - started:.1f}s)")

    release_judges()
    print(f"[judge] {scored} scored, {failed} failed")
    print(f"[Resources] {format_resources(resources.summary())}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
