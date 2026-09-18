import argparse
import os

from securesd.constants import (
    DEFAULT_DRAFT_MODEL,
    DEFAULT_TARGET_MODEL,
    ETA_SCHEDULE_CHOICES,
    METHOD_CHOICES,
    default_hf_root,
)
from securesd.methods import ALL_PARAMS

DATASET_CHOICES = ("humaneval", "gsm8k", "jailbreaking", "prompt_injection")
DEFAULT_HF_ROOT = default_hf_root()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    parser.add_argument("--target_model", type=str, default=DEFAULT_TARGET_MODEL)
    parser.add_argument("--draft_model", type=str, default=DEFAULT_DRAFT_MODEL)
    parser.add_argument("--hf_root", type=str, default=DEFAULT_HF_ROOT,
                        help="HuggingFace hub cache directory to resolve models from.")
    parser.add_argument("--kvcache_block_size", type=int, default=256)
    parser.add_argument("--max_num_seqs", type=int, default=1)
    parser.add_argument("--max_model_len", type=int, default=16384)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.7)
    parser.add_argument("--draft_gpu_memory_utilization", type=float, default=0.75)
    parser.add_argument("--decode_tps_n", type=int, default=256)
    parser.add_argument("--num_gpus", type=int, default=1)

    parser.add_argument("--method", type=str, choices=METHOD_CHOICES, required=True)
    parser.add_argument("--speculate_k", type=int, default=5)

    group = parser.add_argument_group("verifier hyperparameters")
    for spec in ALL_PARAMS:
        group.add_argument(
            f"--{spec.name}", type=spec.type, default=spec.default,
            choices=list(spec.choices) if spec.choices else None,
        )

    parser.add_argument("--eta_schedule", type=str, choices=ETA_SCHEDULE_CHOICES, default="step")
    parser.add_argument("--eta_start", type=float, default=0.0)
    parser.add_argument("--eta_end", type=float, default=1.0)
    parser.add_argument("--eta_len", type=float, default=2.0)
    parser.add_argument("--eta_gamma", type=float, default=1.0)
    parser.add_argument(
        "--record_distribution_diagnostics",
        action="store_true",
        help="Record exact target/draft next-token TV diagnostics (research only).",
    )

    parser.add_argument("--num_seqs", type=int, default=200)
    parser.add_argument("--dataset", type=str, choices=DATASET_CHOICES, default="humaneval")
    parser.add_argument("--prompt_offset", type=int, default=0)
    parser.add_argument("--utility_timeout", type=float, default=3.0)
    parser.add_argument("--utility_workers", type=int, default=16)
    parser.add_argument("--no_utility_eval", action="store_true")

    parser.add_argument("--jb_file", type=str, default="jb_qwen3_200.json")
    parser.add_argument("--wildguard_batch_size", type=int, default=32)

    parser.add_argument("--pi_file", type=str, default=None)
    parser.add_argument("--pi_judge_model", type=str, default="meta-llama/Llama-3.2-3B-Instruct")
    parser.add_argument("--pi_judge_batch_size", type=int, default=16)
    parser.add_argument(
        "--pi_attack_mode",
        choices=["standard", "adaptive_delayed"],
        default="standard",
        help="Prompt-injection construction; adaptive_delayed places the injected pivot after a benign answer.",
    )
    parser.add_argument(
        "--pi_adaptive_window_threshold",
        type=int,
        default=9,
        help="For delayed PI, eta step threshold L; positions t < L are inside the correction window.",
    )

    parser.add_argument("--temp", type=float, default=0.0)
    parser.add_argument("--dtemp", type=float, default=0.0)
    parser.add_argument("--output_len", type=int, default=8192)
    parser.add_argument("--think", type=int, choices=[0, 1], default=0)

    parser.add_argument("--experiment_name", type=str, default="default_experiment")
    parser.add_argument("--results_root", type=str, default=None,
                        help="Where benchmark reports are written (default: <repo>/.results).")

    return parser


def parse_arguments(argv: list[str] | None = None):
    args = build_parser().parse_args(argv)
    args.think = bool(args.think)
    args.hf_root = os.path.expanduser(args.hf_root)
    return args
