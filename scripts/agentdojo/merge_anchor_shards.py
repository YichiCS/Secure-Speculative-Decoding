#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


SUITES = ("workspace", "travel", "banking", "slack")
METRICS = ("clean_utility", "attacked_utility", "security", "injection_task_utility")


def _read(path: Path) -> dict:
    return json.loads(path.read_text())


def _rate(records: list[dict]) -> dict:
    successes = sum(int(row["success"]) for row in records)
    return {
        "successes": successes,
        "total": len(records),
        "rate": successes / len(records),
    }


def _merge_stats(parts: dict[str, dict]) -> dict:
    first = next(iter(parts.values()))
    request_rows = []
    error_rows = []
    for label, stats in parts.items():
        for row in stats["requests"]:
            request_rows.append({**row, "shard": label, "request_index": len(request_rows)})
        for row in stats["errors"]:
            error_rows.append({**row, "shard": label})
    additive = (
        "num_requests",
        "num_errors",
        "prompt_tokens",
        "completion_tokens",
        "decode_tokens",
        "decode_time_s",
        "prefill_time_s",
        "wall_time_s",
    )
    summary = {
        key: sum(stats["summary"][key] for stats in parts.values())
        for key in additive
    }
    summary["decode_tps"] = summary["decode_tokens"] / summary["decode_time_s"]
    return {
        "meta": {
            **first["meta"],
            "physical_gpu": [stats["meta"]["physical_gpu"] for stats in parts.values()],
            "shards": {label: stats["meta"]["physical_gpu"] for label, stats in parts.items()},
        },
        "summary": summary,
        "requests": request_rows,
        "errors": error_rows,
    }


def _merge_workspace_fragments(root: Path, condition: str) -> tuple[dict, dict]:
    fragment_names = tuple(f"workspace_{index}" for index in range(4))
    reports = {
        name: _read(root / name / condition / "benchmark.json") for name in fragment_names
    }
    stats = {
        name: _read(root / name / condition / "server_stats.json") for name in fragment_names
    }
    all_users = []
    injection_sets = []
    for name, report in reports.items():
        if set(report["suites"]) != {"workspace"}:
            raise ValueError(f"{name} contains {set(report['suites'])}")
        chosen = report["meta"]["selection"]["workspace"]
        all_users.extend(chosen["user_tasks"])
        injection_sets.append(set(chosen["injection_tasks"]))
    if len(all_users) != len(set(all_users)) or len(all_users) != 40:
        raise ValueError("workspace fragments do not contain 40 distinct user tasks")
    if any(len(values) != 14 or values != injection_sets[0] for values in injection_sets):
        raise ValueError("workspace fragments do not share all 14 injection tasks")

    first = reports[fragment_names[0]]
    overall_records = {metric: [] for metric in METRICS}
    suite_records = {metric: [] for metric in METRICS}
    for report in reports.values():
        for metric in METRICS:
            overall_records[metric].extend(report["records"][metric])
            suite_records[metric].extend(report["suites"]["workspace"]["records"][metric])
    merged_suite = {
        metric: _rate(suite_records[metric]) for metric in METRICS
    }
    merged_suite["records"] = suite_records
    merged_report = {
        "meta": {
            **first["meta"],
            "selection_mode": "full",
            "selection": {"workspace": {
                "user_tasks": all_users,
                "injection_tasks": sorted(injection_sets[0]),
            }},
            "execution": "workspace user-task shards across identical GPUs; score-equivalent merge",
        },
        "suites": {"workspace": merged_suite},
        "summary": {metric: _rate(overall_records[metric]) for metric in METRICS},
        "records": overall_records,
    }
    merged_report["summary"]["elapsed_s"] = sum(
        float(report["summary"]["elapsed_s"]) for report in reports.values()
    )
    expected = {
        "clean_utility": 40,
        "attacked_utility": 560,
        "security": 560,
        "injection_task_utility": 56,
    }
    actual = {metric: len(overall_records[metric]) for metric in METRICS}
    if actual != expected:
        raise ValueError(f"unexpected merged workspace counts: {actual}")
    injection_records = overall_records["injection_task_utility"][:14]
    merged_report["records"]["injection_task_utility"] = injection_records
    merged_report["suites"]["workspace"]["records"]["injection_task_utility"] = [
        {key: value for key, value in row.items() if key != "suite"}
        for row in injection_records
    ]
    for block in (merged_report["summary"], merged_report["suites"]["workspace"]):
        block["injection_task_utility"] = _rate(injection_records)
    return merged_report, _merge_stats(stats)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shards-root", type=Path, required=True)
    parser.add_argument("--condition", default="ar_target")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--workspace-fragments-root",
        type=Path,
        help="Root containing workspace_0 through workspace_3 condition shards.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    reports = {}
    stats = {}
    for suite in SUITES:
        if suite == "workspace" and args.workspace_fragments_root is not None:
            reports[suite], stats[suite] = _merge_workspace_fragments(
                args.workspace_fragments_root, args.condition
            )
        else:
            condition_root = args.shards_root / suite / args.condition
            reports[suite] = _read(condition_root / "benchmark.json")
            stats[suite] = _read(condition_root / "server_stats.json")
        if set(reports[suite]["suites"]) != {suite}:
            raise ValueError(f"{suite} shard contains {set(reports[suite]['suites'])}")
        if reports[suite]["meta"].get("selection_mode") != "full":
            raise ValueError(f"{suite} shard is not a full-suite run")

    first_report = reports[SUITES[0]]
    merged_records = {metric: [] for metric in METRICS}
    merged_report = {
        "meta": {
            **first_report["meta"],
            "selection": {},
            "execution": "suite-sharded across identical GPUs; score-equivalent merge",
        },
        "suites": {},
    }
    elapsed_s = 0.0
    for suite in SUITES:
        report = reports[suite]
        merged_report["meta"]["selection"].update(report["meta"]["selection"])
        merged_report["suites"].update(report["suites"])
        elapsed_s += float(report["summary"]["elapsed_s"])
        for metric in METRICS:
            merged_records[metric].extend(report["records"][metric])
    merged_report["summary"] = {
        metric: _rate(records) for metric, records in merged_records.items()
    }
    merged_report["summary"]["elapsed_s"] = elapsed_s
    merged_report["records"] = merged_records

    expected_totals = {
        "clean_utility": 97,
        "attacked_utility": 949,
        "security": 949,
        "injection_task_utility": 35,
    }
    actual_totals = {metric: len(records) for metric, records in merged_records.items()}
    if actual_totals != expected_totals:
        raise ValueError(f"unexpected full benchmark counts: {actual_totals}")
    for metric in ("clean_utility", "attacked_utility", "security"):
        keys = [
            (row["suite"], row["user_task"], row["injection_task"])
            for row in merged_records[metric]
        ]
        if len(keys) != len(set(keys)):
            raise ValueError(f"duplicate records in {metric}")

    merged_stats = _merge_stats(stats)
    summary = merged_stats["summary"]

    condition_root = args.output_root / args.condition
    condition_root.mkdir(parents=True, exist_ok=True)
    (condition_root / "benchmark.json").write_text(json.dumps(merged_report, indent=2) + "\n")
    (condition_root / "server_stats.json").write_text(json.dumps(merged_stats, indent=2) + "\n")
    merged_status = {
        "condition": args.condition,
        "status": "completed_merged",
        "execution": "suite/user-task shards across identical GPUs",
        "case_counts": expected_totals,
    }
    (condition_root / "status.json").write_text(json.dumps(merged_status, indent=2) + "\n")
    matrix_path = args.output_root / "matrix_status.json"
    matrix = _read(matrix_path) if matrix_path.is_file() else []
    matrix = [row for row in matrix if row.get("condition") != args.condition]
    matrix.append(merged_status)
    matrix.sort(key=lambda row: (row.get("condition") != "ar_target", row.get("condition", "")))
    matrix_path.write_text(json.dumps(matrix, indent=2) + "\n")
    print(json.dumps({"benchmark": merged_report["summary"], "server": summary}, indent=2))


if __name__ == "__main__":
    main()
