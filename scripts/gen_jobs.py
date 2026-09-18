#!/usr/bin/env python3
from __future__ import annotations

import itertools
import json
import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

RESULTS_ROOT = os.environ.get("SECURESD_RESULTS_DIR") or os.path.join(REPO_ROOT, ".results")
sys.path.insert(0, REPO_ROOT)

from securesd.methods import (
    AR_METHODS, PARAMS_BY_NAME, float_tag, get_spec,
)

BASELINES = AR_METHODS + ("sd",)

ETA_FLOATS = {"eta_start", "eta_end", "eta_len", "eta_gamma"}


def _param_type(param: str) -> type:
    if param in ETA_FLOATS:
        return float
    spec = PARAMS_BY_NAME.get(param)
    return spec.type if spec else str


def fmt(param: str, v) -> str:
    t = _param_type(param)
    if t is int:
        return str(int(v))
    if t is float:
        s = f"{float(v):.10f}".rstrip("0").rstrip(".")
        return "0" if s in ("", "-0") else s
    return str(v)


def typed(param: str, s: str):
    t = _param_type(param)
    return t(s) if t in (int, float) else s


def report_filename(job: dict) -> str:
    eta = (f"_eta{job['eta_schedule']}_s{float_tag(job['eta_start'])}"
           f"_e{float_tag(job['eta_end'])}_l{float_tag(job['eta_len'])}")
    if job["eta_schedule"] == "power":
        eta += f"_g{float_tag(job['eta_gamma'])}"
    spec = get_spec(job["method"])
    values = {p.name: job.get(p.name, p.default) for p in spec.params}
    return f"{job['method']}_b{job['max_num_seqs']}{spec.filename_tag(values)}{eta}.json"


def grid_values(spec):
    if isinstance(spec, dict):
        if "values" in spec:
            return list(spec["values"])
        lo, hi, n = float(spec["min"]), float(spec["max"]), int(spec["num"])
        if n <= 0:
            return []
        if n == 1:
            return [lo]
        return [lo + (hi - lo) * i / (n - 1) for i in range(n)]
    if isinstance(spec, list):
        if spec and all(isinstance(e, dict) for e in spec):
            vals = []
            for seg in spec:
                vals += grid_values(seg)
            out: list = []
            for v in sorted(vals):
                if not out or abs(v - out[-1]) > 1e-9:
                    out.append(v)
            return out
        return spec
    return [spec]


def _coerce(param: str, v):
    return typed(param, v) if isinstance(v, str) else _param_type(param)(v)


def axis_values(key: str, spec) -> list[dict]:
    if isinstance(spec, dict) and "pairs" in spec:
        params = spec["pairs"]["params"]
        return [dict(zip(params, row)) for row in spec["pairs"]["values"]]
    return [{key: v} for v in grid_values(spec)]


def _dedup_assignments(items: list[dict]) -> list[dict]:
    out: list[dict] = []
    seen: set = set()
    for d in items:
        key = tuple(sorted(
            (k, round(float(v), 9) if isinstance(v, (int, float)) else v)
            for k, v in d.items()))
        if key not in seen:
            seen.add(key)
            out.append(d)
    return out


def expand_grid(grid) -> list[dict]:
    if isinstance(grid, list):
        return _dedup_assignments(
            [d for block in grid for d in expand_grid(block)])
    if isinstance(grid, dict) and "cases" in grid:
        by = grid["by"]
        out: list[dict] = []
        for val, sub in grid["cases"].items():
            for d in expand_grid(sub):
                out.append({by: _coerce(by, val), **d})
        return out
    axes = [axis_values(k, grid[k]) for k in grid]
    if not axes:
        return [{}]
    out = []
    for combo in itertools.product(*axes):
        merged: dict = {}
        for d in combo:
            merged.update(d)
        out.append(merged)
    return out


def parse_eta(cfg: str) -> dict:
    parts = cfg.split(":")
    if len(parts) not in (4, 5) or not all(parts):
        raise SystemExit(
            f"[gen_jobs] bad eta config {cfg!r} (want schedule:start:end:len[:gamma])")
    sched, start, end, length = parts[:4]
    gamma = float(parts[4]) if len(parts) == 5 else 1.0
    return {"eta_schedule": sched, "eta_start": float(start),
            "eta_end": float(end), "eta_len": float(length), "eta_gamma": gamma}


def job_cli(common: dict, method: str, eta: dict, params: dict):
    job = {"method": method, "max_num_seqs": common["max_num_seqs"], **eta}
    cli = ["--method", method]
    for k in ("eta_schedule", "eta_start", "eta_end", "eta_len", "eta_gamma"):
        cli += [f"--{k}", fmt(k, eta[k])]
    for k, v in params.items():
        s = fmt(k, v)
        cli += [f"--{k}", s]
        job[k] = typed(k, s)
    report = os.path.join(RESULTS_ROOT, common["experiment_name"],
                          method, report_filename(job))
    return report, cli


def common_cli(common: dict) -> list[str]:
    cli = []
    for k, v in common.items():
        if v is None or v is False:
            continue
        if v is True:
            cli.append(f"--{k}")
        else:
            cli += [f"--{k}", str(v)]
    return cli


def main():
    if len(sys.argv) != 2:
        raise SystemExit("usage: gen_jobs.py <config.json>")
    cfg = json.load(open(sys.argv[1]))

    experiment = os.environ.get("SWEEP_EXPERIMENT") or "default_experiment"
    dataset = os.environ.get("SWEEP_DATASET") or "humaneval"
    common = {k: v for k, v in cfg["common"].items() if not k.startswith("_")}
    common["dataset"] = dataset
    common["experiment_name"] = experiment

    methods = cfg["methods"]
    eta_configs = [parse_eta(c) for c in cfg["eta_configs"]]
    baseline_eta = parse_eta(cfg["baseline_eta"]) if cfg.get("baseline_eta") else eta_configs[0]
    grids = cfg.get("grids", {})
    subsample = max(1, int(os.environ.get("SWEEP_SUBSAMPLE", "1")))

    out = sys.stdout
    common_points = expand_grid(cfg.get("common_grid", {}))
    out.write("\t".join(["#META", experiment, dataset, " ".join(methods)]) + "\n")

    def write(method, eta, params, common_params):
        job_common = {**common, **common_params}
        report, cli = job_cli(job_common, method, eta, params)
        out.write("\t".join([method, report, *common_cli(job_common), *cli]) + "\n")

    for method in methods:
        if method in BASELINES:
            for common_params in common_points:
                write(method, baseline_eta, {}, common_params)
            continue
        points = expand_grid(grids.get(method, {}))[::subsample]
        for common_params in common_points:
            for eta in eta_configs:
                for params in points:
                    write(method, eta, params, common_params)


if __name__ == "__main__":
    try:
        main()
    except BrokenPipeError:
        try:
            sys.stdout.close()
        finally:
            os._exit(0)
