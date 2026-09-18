#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

from agentdojo.agent_pipeline.agent_pipeline import AgentPipeline, PipelineConfig
from agentdojo.attacks.attack_registry import load_attack
from agentdojo.benchmark import (
    benchmark_suite_with_injections,
    benchmark_suite_without_injections,
)
from agentdojo.logging import OutputLogger
from agentdojo.models import ModelsEnum
from agentdojo.task_suite.load_suites import get_suite
from agentdojo.task_suite.task_suite import TaskSuite


DEFAULT_SELECTION = {
    "workspace": {
        "user_tasks": ["user_task_0", "user_task_1"],
        "injection_tasks": ["injection_task_0", "injection_task_1"],
    },
    "travel": {
        "user_tasks": ["user_task_0", "user_task_1"],
        "injection_tasks": ["injection_task_6", "injection_task_0"],
    },
    "banking": {
        "user_tasks": ["user_task_0", "user_task_1"],
        "injection_tasks": ["injection_task_0", "injection_task_1"],
    },
    "slack": {
        "user_tasks": ["user_task_0", "user_task_1"],
        "injection_tasks": ["injection_task_1", "injection_task_2"],
    },
}
EVALUATION_ERRORS: list[dict] = []


def _install_safe_evaluator() -> None:
    original = TaskSuite._check_task_result

    def safe_check(self, task, model_output, pre_environment, task_environment, functions_stack_trace):
        try:
            return original(
                self, task, model_output, pre_environment, task_environment, functions_stack_trace
            )
        except Exception as exc:
            record = {
                "task": getattr(task, "ID", type(task).__name__),
                "type": type(exc).__name__,
                "message": str(exc),
                "fallback": False,
            }
            EVALUATION_ERRORS.append(record)
            logging.exception("AgentDojo evaluator failed; conservatively scoring False: %s", record)
            return False

    TaskSuite._check_task_result = safe_check


def _records(values: dict[tuple[str, str], bool]) -> list[dict]:
    return [
        {"user_task": key[0], "injection_task": key[1], "success": bool(value)}
        for key, value in values.items()
    ]


def _rate(records: list[dict]) -> dict:
    successes = sum(int(x["success"]) for x in records)
    total = len(records)
    return {
        "successes": successes,
        "total": total,
        "rate": successes / total if total else None,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--logdir", type=Path, required=True)
    parser.add_argument("--benchmark-version", default="v1.2.2")
    parser.add_argument("--attack", default="tool_knowledge")
    parser.add_argument("--suites", nargs="+", default=list(DEFAULT_SELECTION))
    parser.add_argument("--selection-json", type=Path)
    parser.add_argument(
        "--full-benchmark",
        action="store_true",
        help="Use every user and injection task in each selected suite.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse valid task traces already present in --logdir.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.WARNING)
    if args.full_benchmark and args.selection_json is not None:
        raise ValueError("--full-benchmark and --selection-json are mutually exclusive")
    _install_safe_evaluator()
    selection = DEFAULT_SELECTION
    if args.selection_json is not None:
        selection = json.loads(args.selection_json.read_text())

    pipeline = AgentPipeline.from_config(PipelineConfig(
        llm=ModelsEnum.LOCAL,
        model_id=args.model_id,
        defense=None,
        tool_delimiter="tool",
        system_message_name=None,
        system_message=None,
        tool_output_format=None,
    ))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.logdir.mkdir(parents=True, exist_ok=True)

    payload = {
        "meta": {
            "condition": args.condition,
            "model_id": args.model_id,
            "agentdojo_version": args.benchmark_version,
            "attack": args.attack,
            "selection_mode": "full" if args.full_benchmark else "selected",
            "selection": {},
            "tool_execution": "official AgentDojo FunctionsRuntime/ToolsExecutionLoop",
            "tool_output_format": "AgentDojo default YAML",
        },
        "suites": {},
    }
    all_clean: list[dict] = []
    all_attacked: list[dict] = []
    all_security: list[dict] = []
    all_injection_utility: list[dict] = []
    started = time.perf_counter()

    with OutputLogger(str(args.logdir)):
        for suite_name in args.suites:
            suite = get_suite(args.benchmark_version, suite_name)
            if args.full_benchmark:
                user_tasks = list(suite.user_tasks)
                injection_tasks = list(suite.injection_tasks)
            else:
                chosen = selection[suite_name]
                user_tasks = list(chosen["user_tasks"])
                injection_tasks = list(chosen["injection_tasks"])
            payload["meta"]["selection"][suite_name] = {
                "user_tasks": user_tasks,
                "injection_tasks": injection_tasks,
            }
            missing_users = set(user_tasks) - set(suite.user_tasks)
            missing_injections = set(injection_tasks) - set(suite.injection_tasks)
            if missing_users or missing_injections:
                raise ValueError(
                    f"invalid selection for {suite_name}: users={missing_users}, "
                    f"injections={missing_injections}"
                )

            clean = benchmark_suite_without_injections(
                pipeline,
                suite,
                logdir=args.logdir,
                force_rerun=not args.resume,
                user_tasks=user_tasks,
                benchmark_version=args.benchmark_version,
            )
            attack = load_attack(args.attack, suite, pipeline)
            attacked = benchmark_suite_with_injections(
                pipeline,
                suite,
                attack,
                logdir=args.logdir,
                force_rerun=not args.resume,
                user_tasks=user_tasks,
                injection_tasks=injection_tasks,
                verbose=False,
                benchmark_version=args.benchmark_version,
            )
            clean_records = _records(clean["utility_results"])
            attacked_records = _records(attacked["utility_results"])
            security_records = _records(attacked["security_results"])
            injection_records = [
                {"injection_task": key, "success": bool(value)}
                for key, value in attacked["injection_tasks_utility_results"].items()
            ]
            payload["suites"][suite_name] = {
                "clean_utility": _rate(clean_records),
                "attacked_utility": _rate(attacked_records),
                "security": _rate(security_records),
                "injection_task_utility": _rate(injection_records),
                "records": {
                    "clean_utility": clean_records,
                    "attacked_utility": attacked_records,
                    "security": security_records,
                    "injection_task_utility": injection_records,
                },
            }
            all_clean.extend({"suite": suite_name, **x} for x in clean_records)
            all_attacked.extend({"suite": suite_name, **x} for x in attacked_records)
            all_security.extend({"suite": suite_name, **x} for x in security_records)
            all_injection_utility.extend({"suite": suite_name, **x} for x in injection_records)

    payload["summary"] = {
        "clean_utility": _rate(all_clean),
        "attacked_utility": _rate(all_attacked),
        "security": _rate(all_security),
        "injection_task_utility": _rate(all_injection_utility),
        "elapsed_s": time.perf_counter() - started,
        "evaluation_errors": len(EVALUATION_ERRORS),
    }
    payload["meta"]["evaluation_error_records"] = EVALUATION_ERRORS
    payload["records"] = {
        "clean_utility": all_clean,
        "attacked_utility": all_attacked,
        "security": all_security,
        "injection_task_utility": all_injection_utility,
    }
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload["summary"], indent=2), flush=True)


if __name__ == "__main__":
    main()
