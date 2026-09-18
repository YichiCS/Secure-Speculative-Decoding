import os
import time
import json
import math
from datetime import datetime

from huggingface_hub import snapshot_download
from securesd.constants import is_sd_method
from securesd.methods import float_tag, get_spec, method_params
from dataset import get_dataset_adapter

def get_model_paths(args):
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    target_path = snapshot_download(args.target_model, cache_dir=args.hf_root)
    target_name = target_path.split("/models--", 1)[1].split("/snapshots/", 1)[0]

    draft_path = snapshot_download(args.draft_model, cache_dir=args.hf_root)
    draft_name = draft_path.split("/models--", 1)[1].split("/snapshots/", 1)[0]
    return target_name, target_path, draft_name, draft_path


def create_llm_kwargs(args, target_path, draft_path):
    return dict(
        model=target_path,
        draft=draft_path,
        max_model_len=args.max_model_len,
        kvcache_block_size=args.kvcache_block_size,
        max_num_seqs=args.max_num_seqs,
        max_steps=args.max_steps,
        gpu_memory_utilization=args.gpu_memory_utilization,
        draft_gpu_memory_utilization=args.draft_gpu_memory_utilization,
        method=args.method,
        speculate_k=args.speculate_k,
        method_params=method_params(args.method, args),
        eta_schedule=args.eta_schedule,
        eta_start=args.eta_start,
        eta_end=args.eta_end,
        eta_len=args.eta_len,
        eta_gamma=args.eta_gamma,
        record_distribution_diagnostics=args.record_distribution_diagnostics,
    )


def run_benchmark(llm, prompts, sampling_params):
    start_time = time.time()
    outputs, metrics = llm.generate(prompts, sampling_params)
    total_time = time.time() - start_time
    return outputs, total_time, metrics


def _group_trace_values(
    values: list[int],
    scheduled: list[int],
) -> list[list[int]]:
    if sum(scheduled) != len(values):
        raise ValueError("Trace value count does not match scheduled decode count")

    out: list[list[int]] = []
    offset = 0
    for count in scheduled:
        out.append(values[offset:offset + count])
        offset += count
    return out


def build_trace_metrics(metrics: dict, args) -> dict | None:
    decode_step_times = [float(x) for x in metrics["decode_step_times"]]
    decode_step_start_offsets = [float(x) for x in metrics.get("decode_step_start_offsets_s", [])]
    decode_step_end_offsets = [float(x) for x in metrics.get("decode_step_end_offsets_s", [])]
    decode_step_tokens = [int(x) for x in metrics["decode_step_tokens"]]
    decode_step_scheduled = [int(x) for x in metrics["decode_step_scheduled"]]
    decode_step_seq_deltas_raw = metrics["decode_step_seq_deltas"]

    if not decode_step_times and not decode_step_tokens and not decode_step_scheduled:
        return None

    trace = {
        "decode_step_index": list(range(1, len(decode_step_times) + 1)),
        "decode_step_times_s": decode_step_times,
        "decode_step_times_ms": [x * 1000.0 for x in decode_step_times],
        "decode_step_start_offsets_s": decode_step_start_offsets,
        "decode_step_end_offsets_s": decode_step_end_offsets,
        "decode_step_tokens": decode_step_tokens,
        "decode_step_scheduled": decode_step_scheduled,
        "decode_step_seq_deltas": [
            [[int(seq_id), int(delta)] for seq_id, delta in step]
            for step in decode_step_seq_deltas_raw
        ],
        "decode_cumulative_tokens": [],
    }

    running = 0
    for value in decode_step_tokens:
        running += value
        trace["decode_cumulative_tokens"].append(running)

    if is_sd_method(args.method):
        accepted_flat = [int(x) for x in metrics["accepted_suffix_lens_with_recovery"]]
        verify_times_s = [float(x) for x in metrics["target_verify_times"]]
        grouped = _group_trace_values(accepted_flat, decode_step_scheduled)
        speculation = {
            "accepted_suffix_lens_with_recovery_flat": accepted_flat,
            "target_verify_times_s": verify_times_s,
            "target_verify_times_ms": [x * 1000.0 for x in verify_times_s],
        }
        accepted_spec_tokens_by_step = [
            [max(0, value - 1) for value in step]
            for step in grouped
        ]
        denom = max(int(args.speculate_k), 1)
        speculation["accepted_suffix_lens_with_recovery_by_step"] = grouped
        speculation["accepted_spec_tokens_by_step"] = accepted_spec_tokens_by_step
        speculation["accepted_spec_tokens_mean_by_step"] = [
            (sum(step) / len(step)) if step else 0.0
            for step in accepted_spec_tokens_by_step
        ]
        speculation["accepted_fraction_mean_by_step"] = [
            ((sum(step) / len(step)) / denom) if step else 0.0
            for step in accepted_spec_tokens_by_step
        ]
        trace["speculation"] = speculation

    return trace


def _safe_div(numer: float, denom: float) -> float:
    return float(numer / denom) if denom > 0 else 0.0


def _mean_std(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "std": 0.0}
    mean = sum(values) / len(values)
    var = sum((value - mean) ** 2 for value in values) / len(values)
    return {"mean": mean, "std": math.sqrt(var)}


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    q = min(max(float(q), 0.0), 1.0)
    pos = q * (len(values) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return values[lo]
    alpha = pos - lo
    return values[lo] * (1.0 - alpha) + values[hi] * alpha


def _stats(values: list[float], scale: float = 1.0) -> dict[str, float]:
    scaled = [float(v) * scale for v in values]
    ms = _mean_std(scaled)
    return {
        "mean": ms["mean"],
        "std": ms["std"],
        "p50": _percentile(scaled, 0.50),
        "p90": _percentile(scaled, 0.90),
        "p99": _percentile(scaled, 0.99),
        "count": len(scaled),
    }


def _metric_values(rows: list[dict], key: str, *, fallback: str | None = None) -> list[float]:
    values = []
    for row in rows:
        value = row.get(key, row.get(fallback, 0.0) if fallback is not None else 0.0)
        value = float(value)
        if value > 0:
            values.append(value)
    return values


def _build_first_n_profile(metrics: dict, token_cap: int) -> dict:
    step_times = [float(x) for x in metrics["decode_step_times"]]
    step_end_offsets = [float(x) for x in metrics.get("decode_step_end_offsets_s", [])]
    step_scheduled = [int(x) for x in metrics["decode_step_scheduled"]]
    step_seq_deltas = metrics["decode_step_seq_deltas"]
    if not step_end_offsets:
        running = 0.0
        step_end_offsets = []
        for step_time in step_times:
            running += step_time
            step_end_offsets.append(running)

    ttft_by_seq = {int(row["seq_id"]): float(row["ttft_s"]) for row in metrics.get("per_seq_decode", [])}

    seq_tokens: dict[int, int] = {}
    seq_time_to_n: dict[int, float] = {}
    total_tokens = 0
    total_time_s = 0.0
    total_active_seq_time_s = 0.0

    for step_time, step_end_s, _, deltas in zip(
        step_times,
        step_end_offsets,
        step_scheduled,
        step_seq_deltas,
    ):
        in_window = 0
        step_window_tokens = 0
        for seq_id, delta in deltas:
            seq_id = int(seq_id)
            delta = max(int(delta), 0)
            prev = seq_tokens.get(seq_id, 0)
            if token_cap > 0 and prev >= token_cap:
                continue
            in_window += 1
            remaining = token_cap - prev if token_cap > 0 else delta
            used = min(delta, remaining) if token_cap > 0 else delta
            seq_tokens[seq_id] = prev + used
            step_window_tokens += used
            if (
                token_cap > 0
                and prev < token_cap
                and seq_tokens[seq_id] >= token_cap
                and seq_id not in seq_time_to_n
            ):
                seq_time_to_n[seq_id] = step_end_s - ttft_by_seq.get(seq_id, 0.0)
        if in_window > 0:
            total_tokens += step_window_tokens
            total_time_s += step_time
            total_active_seq_time_s += step_time * in_window

    time_values = list(seq_time_to_n.values())
    return {
        "total_tokens": total_tokens,
        "total_time_s": total_time_s,
        "total_active_seq_time_s": total_active_seq_time_s,
        "time_to_n_values_s": time_values,
    }


def compute_efficiency_metrics(_prompts, outputs, total_time, _model_path, _batch_size: int, raw_metrics: dict, args):
    decode_tokens = sum(len(o["token_ids"]) for o in outputs)
    request_count = len(outputs)
    num_gpus = max(int(getattr(args, "num_gpus", 1)), 1)
    step_times = [float(x) for x in raw_metrics["decode_step_times"]]
    step_scheduled = [int(x) for x in raw_metrics["decode_step_scheduled"]]
    step_tokens = [int(x) for x in raw_metrics["decode_step_tokens"]]
    weighted_active = sum(t * s for t, s in zip(step_times, step_scheduled))
    total_decode_time = sum(step_times)
    first_n = _build_first_n_profile(raw_metrics, int(args.decode_tps_n))
    window_tps = _safe_div(first_n["total_tokens"], first_n["total_time_s"])
    latency_inv_values = [_safe_div(float(args.decode_tps_n), t) for t in first_n["time_to_n_values_s"] if t > 0]
    latency_inv_stats = _mean_std(latency_inv_values)
    output_tps = _safe_div(decode_tokens, total_time)
    scheduled_f = [float(x) for x in step_scheduled]
    return {
        "time_s": total_time,
        "num_requests": request_count,
        "decode_tokens": decode_tokens,
        "request_throughput_rps": _safe_div(request_count, total_time),
        "output_tps": output_tps,
        "output_tps_per_gpu": _safe_div(output_tps, num_gpus),
        "avg_output_tokens_per_request": _safe_div(decode_tokens, request_count),
        "decode_step_time_ms": _safe_div(total_decode_time * 1000.0, len(step_times)),
        "decode_tokens_per_step": _safe_div(sum(step_tokens), len(step_tokens)),
        "active_seqs_mean_decode": _safe_div(weighted_active, total_decode_time),
        "active_seqs_p50_decode": _percentile(scheduled_f, 0.50),
        "active_seqs_p90_decode": _percentile(scheduled_f, 0.90),
        "output_tps_windowed": window_tps,
        "output_tps_windowed_per_gpu": _safe_div(window_tps, num_gpus),
        "active_seqs_mean_windowed": _safe_div(first_n["total_active_seq_time_s"], first_n["total_time_s"]),
        "latency_inv_windowed_avg": latency_inv_stats["mean"],
        "latency_inv_windowed_std": latency_inv_stats["std"],
        "latency_inv_windowed_p50": _percentile(latency_inv_values, 0.50),
        "time_to_window_p50_s": _percentile(first_n["time_to_n_values_s"], 0.50),
        "time_to_window_p90_s": _percentile(first_n["time_to_n_values_s"], 0.90),
        "time_to_window_p99_s": _percentile(first_n["time_to_n_values_s"], 0.99),
        "time_to_window_count": len(first_n["time_to_n_values_s"]),
        "seq_tps_decode_window_tokens": int(args.decode_tps_n),
    }


def build_mode_summary(args) -> str:
    if not is_sd_method(args.method):
        return "AutoRegressive"

    spec = get_spec(args.method)
    mode = f"Speculative(k={args.speculate_k}, method={args.method}"
    mode += spec.summary_tag(method_params(args.method, args))
    if getattr(args, "eta_schedule", None):
        mode += (
            f", eta={args.eta_schedule}"
            f"[{args.eta_start}->{args.eta_end}@{args.eta_len}]"
        )
        if args.eta_schedule == "power":
            mode += f"^{args.eta_gamma}"
    return mode + ")"


def _eta_tag(args) -> str:
    schedule = getattr(args, "eta_schedule", None)
    if not schedule:
        return ""
    tag = (
        f"_eta{schedule}"
        f"_s{float_tag(args.eta_start)}_e{float_tag(args.eta_end)}_l{float_tag(args.eta_len)}"
    )
    if schedule == "power":
        tag += f"_g{float_tag(args.eta_gamma)}"
    return tag


def create_run_name(args):
    mode = "sd" if is_sd_method(args.method) else "ar"
    active_model = args.draft_model if args.method == "ar_draft" else args.target_model
    target_str = active_model.replace("/", "_")
    draft_str = f"_draft{args.draft_model.replace('/', '_')}" if is_sd_method(args.method) else ""
    values = method_params(args.method, args)
    verify_str = "".join(
        f"_{p.tag}{values[p.name]}" for p in get_spec(args.method).params
    )
    think_str = "_think" if args.think else "_nothink"
    return (
        f"{mode}_{target_str}_bs{args.max_num_seqs}_k{args.speculate_k}"
        f"_temp{args.temp}_{args.dataset}{think_str}{draft_str}"
        f"_{args.method}{verify_str}{_eta_tag(args)}"
    )


def compute_runtime_metrics(metrics: dict, args) -> dict:
    decode_step_times = [float(x) for x in metrics["decode_step_times"]]
    per_seq_decode = metrics["per_seq_decode"]
    ttft_values = _metric_values(per_seq_decode, "ttft_s")
    e2e_values = _metric_values(per_seq_decode, "e2e_latency_s")
    tpot_values = _metric_values(per_seq_decode, "tpot_s")
    decode_tps_values = _metric_values(per_seq_decode, "decode_tps")
    full_span_tps_values = _metric_values(per_seq_decode, "full_span_tps", fallback="decode_tps")
    runtime = {
        "query_tps": _stats(full_span_tps_values),
        "seq_decode_tps": _stats(decode_tps_values),
        "ttft_ms": _stats(ttft_values, scale=1000.0),
        "tpot_ms": _stats(tpot_values, scale=1000.0),
        "e2e_latency_s": _stats(e2e_values),
    }

    first_n = _build_first_n_profile(metrics, int(args.decode_tps_n))
    runtime["latency_inv_windowed"] = _stats([
        _safe_div(float(args.decode_tps_n), t)
        for t in first_n["time_to_n_values_s"]
        if t > 0
    ])
    runtime["time_to_window_s"] = _stats(first_n["time_to_n_values_s"])

    if is_sd_method(args.method):
        denom = max(int(args.speculate_k), 1)
        acceptance_rate_values = [
            max(float(value) - 1.0, 0.0) / denom
            for value in metrics["accepted_suffix_lens_with_recovery"]
        ]
        step_time_ms_values = [value * 1000.0 for value in decode_step_times]
        runtime["acceptance_rate"] = _mean_std(acceptance_rate_values)
        runtime["step_time_ms"] = _stats(step_time_ms_values)

    return runtime

def _effective_experiment_name(args) -> str:
    return str(args.experiment_name)


def _report_filename(args) -> str:
    spec = get_spec(args.method)
    return (
        f"{args.method}_b{args.max_num_seqs}"
        f"{spec.filename_tag(method_params(args.method, args))}"
        f"{_eta_tag(args)}.json"
    )


def _raw_report_filename(args) -> str:
    return f"raw_{_report_filename(args)}"


def cleanup_utility_artifacts(util) -> None:
    if util is None:
        return
    for key in ("sample_file", "results_file"):
        if key in util and os.path.isfile(util[key]):
            os.remove(util[key])
    if "artifact_dir" in util and os.path.isdir(util["artifact_dir"]):
        artifact_dir = util["artifact_dir"]
        if not os.listdir(artifact_dir):
            os.rmdir(artifact_dir)


def _drop_none_fields(value):
    if isinstance(value, dict):
        return {
            key: _drop_none_fields(item)
            for key, item in value.items()
            if item is not None
        }
    if isinstance(value, list):
        return [_drop_none_fields(item) for item in value]
    return value


def results_root(args) -> str:
    root = getattr(args, "results_root", None) or os.environ.get("SECURESD_RESULTS_DIR")
    if root:
        return os.path.abspath(os.path.expanduser(root))
    return os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")), ".results")


def write_benchmark_report(args, model_name, total_time, eff, runtime_metrics, raw_metrics, util, outputs, dataset_bundle, resources=None):
    experiment_name = _effective_experiment_name(args)
    report_dir = os.path.join(results_root(args), experiment_name, args.method)
    os.makedirs(report_dir, exist_ok=True)

    utility_summary = None
    if util is not None:
        utility_summary = {
            k: v for k, v in util.items()
            if k not in {"artifact_dir", "sample_file", "results_file", "sample_results"}
        }

    merged_metrics = dict(eff)
    merged_metrics.update(runtime_metrics)
    trace_metrics = build_trace_metrics(raw_metrics, args)

    report_payload = {
        "meta": {
            "run_name": create_run_name(args),
            "timestamp": datetime.now().isoformat(),
            "dataset": args.dataset,
            "model_name": model_name,
            "target_model": args.draft_model if args.method == "ar_draft" else args.target_model,
            "draft_model": args.draft_model,
            "mode": "sd" if is_sd_method(args.method) else "ar",
            "method": args.method,
            "experiment_name": experiment_name,
            "spec_k": args.speculate_k if is_sd_method(args.method) else None,
            "decode_tps_n": args.decode_tps_n,
            "num_gpus": args.num_gpus,
            **method_params(args.method, args),
            "eta_schedule": getattr(args, "eta_schedule", None),
            "eta_start": args.eta_start if getattr(args, "eta_schedule", None) else None,
            "eta_end": args.eta_end if getattr(args, "eta_schedule", None) else None,
            "eta_len": args.eta_len if getattr(args, "eta_schedule", None) else None,
            "eta_gamma": args.eta_gamma if getattr(args, "eta_schedule", None) == "power" else None,
            "batch_size": args.max_num_seqs,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "draft_gpu_memory_utilization": args.draft_gpu_memory_utilization,
            "temperature": args.temp,
            "draft_temperature": args.dtemp,
            "think": args.think,
            "dataset_meta": dataset_bundle.meta,
        },
        "summary": {
            "utility": utility_summary,
            "efficiency": merged_metrics,
            "resources": resources,
            "num_samples": len(outputs),
        },
        "samples": get_dataset_adapter(dataset_bundle.dataset).build_report_samples(
            outputs,
            dataset_bundle,
            util,
        ),
    }
    report = _drop_none_fields(report_payload)

    report_filename = _report_filename(args)
    report_path = os.path.join(report_dir, report_filename)
    raw_report_path = os.path.join(report_dir, _raw_report_filename(args))
    raw_report = {
        "report_filename": report_filename,
        "raw": {
            "raw_metrics": raw_metrics,
            "outputs": outputs,
            "dataset_bundle": {
                "dataset": dataset_bundle.dataset,
                "prompts": dataset_bundle.prompts,
                "samples": dataset_bundle.samples,
                "meta": dataset_bundle.meta,
            },
            "utility": util,
        },
        "trace": trace_metrics,
    }

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
        f.write("\n")
    with open(raw_report_path, "w", encoding="utf-8") as f:
        json.dump(raw_report, f, ensure_ascii=False, indent=2)
        f.write("\n")
    cleanup_utility_artifacts(util)
    print(f"[report] report file: {report_path}")
    print(f"[raw] raw report file: {raw_report_path}")
    return report_path


def maybe_evaluate_utility(args, dataset_bundle, outputs):
    if args.no_utility_eval:
        return None
    return get_dataset_adapter(dataset_bundle.dataset).evaluate(outputs, dataset_bundle, args)


def print_benchmark_summary(args, model_name, total_time, eff, runtime_metrics, util, dataset_name):
    think_mode = "think=on" if args.think else "think=off"
    mode = build_mode_summary(args)

    ttft = runtime_metrics["ttft_ms"]
    tpot = runtime_metrics["tpot_ms"]
    e2e = runtime_metrics["e2e_latency_s"]
    if is_sd_method(args.method):
        print(
            "[Speculation] "
            f"acceptance_rate={runtime_metrics['acceptance_rate']['mean']:.4f}±{runtime_metrics['acceptance_rate']['std']:.4f}, "
            f"step_time_ms={runtime_metrics['step_time_ms']['mean']:.2f}±{runtime_metrics['step_time_ms']['std']:.2f}"
        )
    print(f"[Model] {model_name}, Mode: {mode}, {think_mode}")
    seq_tps = runtime_metrics["seq_decode_tps"]
    print(
        "[Efficiency] "
        f"time={eff['time_s']:.2f}s, "
        f"output_tps={eff['output_tps']:.2f}tok/s, "
        f"tok/s/GPU={eff['output_tps_per_gpu']:.2f}, "
        f"seq_tps p50/p90={seq_tps['p50']:.2f}/{seq_tps['p90']:.2f}tok/s/seq, "
        f"req/s={eff['request_throughput_rps']:.2f}, "
        f"active_seqs={eff['active_seqs_mean_decode']:.2f}"
    )
    print(
        "[Latency] "
        f"TTFT p50/p90={ttft['p50']:.2f}/{ttft['p90']:.2f}ms, "
        f"TPOT p50/p90={tpot['p50']:.2f}/{tpot['p90']:.2f}ms, "
        f"E2E p50/p90={e2e['p50']:.2f}/{e2e['p90']:.2f}s"
    )
    if eff["time_to_window_count"] > 0:
        print(
            "[Window] "
            f"N={args.decode_tps_n}, "
            f"output_tps@N={eff['output_tps_windowed']:.2f}tok/s, "
            f"time_to_N p50/p90={eff['time_to_window_p50_s']:.2f}/{eff['time_to_window_p90_s']:.2f}s, "
            f"latency^-1@N={eff['latency_inv_windowed_p50']:.2f} tok/s/seq"
        )
    else:
        print(
            "[Window] "
            f"N={args.decode_tps_n}, no sequence reached the token threshold"
        )
    if runtime_metrics["query_tps"]["count"] > 0:
        print(
            "[Legacy] "
            f"query_tps(full-span)={runtime_metrics['query_tps']['mean']:.2f}"
            f"±{runtime_metrics['query_tps']['std']:.2f}tok/s"
        )
    if util is None:
        print("[Utility] skipped (--no_utility_eval)")
        return

    print(f"[Utility] {get_dataset_adapter(dataset_name).format_utility_summary(util)}")
    if "artifact_dir" in util and os.path.isdir(util["artifact_dir"]):
        print(f"[Utility Artifacts] {util['artifact_dir']}")
