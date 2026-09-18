import sys
sys.stdout.reconfigure(line_buffering=True)

from bench_args import parse_arguments
from bench_helpers import (
    compute_efficiency_metrics,
    compute_runtime_metrics,
    create_llm_kwargs,
    get_model_paths,
    maybe_evaluate_utility,
    print_benchmark_summary,
    run_benchmark,
    write_benchmark_report,
)
from dataset import build_dataset_bundle
from resources import ResourceTracker, format_resources


def main():
    args = parse_arguments()

    from securesd.engine.llm_engine import LLMEngine
    from securesd.sampling_params import SamplingParams

    resources = ResourceTracker()

    with resources.phase("model_resolve"):
        target_name, target_path, draft_name, draft_path = get_model_paths(args)
    with resources.phase("engine_init"):
        llm_engine = LLMEngine(**create_llm_kwargs(args, target_path, draft_path))
    model_name = draft_name if args.method == "ar_draft" else target_name

    try:
        with resources.phase("dataset_build"):
            dataset_bundle = build_dataset_bundle(args, target_path)
        prompts = dataset_bundle.prompts

        sampling_params = [
            SamplingParams(
                temperature=args.temp,
                draft_temperature=args.dtemp,
                ignore_eos=False,
                max_new_tokens=args.output_len,
            )
            for _ in range(len(prompts))
        ]

        with resources.phase("generate"):
            outputs, total_time, raw_metrics = run_benchmark(
                llm=llm_engine,
                prompts=prompts,
                sampling_params=sampling_params,
            )
    finally:
        llm_engine.exit(hard=False)

    eff = compute_efficiency_metrics(
        prompts, outputs, total_time, target_path, args.max_num_seqs, raw_metrics, args
    )
    runtime_metrics = compute_runtime_metrics(raw_metrics, args)
    with resources.phase("utility_eval"):
        util = maybe_evaluate_utility(args, dataset_bundle, outputs)

    resource_summary = resources.summary()
    write_benchmark_report(
        args,
        model_name,
        total_time,
        eff,
        runtime_metrics,
        raw_metrics,
        util,
        outputs,
        dataset_bundle,
        resource_summary,
    )
    print_benchmark_summary(
        args=args,
        model_name=model_name,
        total_time=total_time,
        eff=eff,
        runtime_metrics=runtime_metrics,
        util=util,
        dataset_name=dataset_bundle.dataset,
    )
    print(f"[Resources] {format_resources(resource_summary)}")


if __name__ == "__main__":
    main()
