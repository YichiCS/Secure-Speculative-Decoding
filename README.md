# Secure Speculative Decoding for Large Language Models

This repository contains a research inference engine that implements

* **six relaxed verifiers** from the literature (LossySD, BiLD, SC, FSD, FLy, MaRS) behind one interface, plus exact speculative decoding and plain autoregressive decoding as baselines;
* **SecureSD**, a correction that restores exact verification for the first few generated tokens of every response and recovers most of the lost security at almost no speed cost;
* a **benchmark harness** that measures utility (HumanEval, GSM8K), security (jailbreaking, prompt injection) and decoding efficiency in a single run, and writes one JSON report per run.

## Repository Layout

```text
securesd/                   the inference engine
  methods.py                  registry of the 9 decoding methods and their hyperparameters
  verifier/                   one file per verifier, plus:
    fusion.py                   the SecureSD correction
    eta.py                      the correction schedules
  engine/                     scheduler, KV cache, draft/target runners, verification step
  layers/ models/             attention, sampler, Llama 3 and Qwen3 decoders

bench/                      one benchmark run -> one JSON report
  bench.py                    entry point
  bench_args.py               CLI; verifier flags are generated from securesd/methods.py
  judge.py                    scores a finished result tree in one pass
  dataset/                    one adapter per benchmark + the cached judge loader

scripts/
  gen_jobs.py                 expands a sweep config into a flat job list
  sweep.sh                    runs that job list across a pool of GPUs, resuming
  judge_pool.sh               scores several result trees across a pool of GPUs
  qwen/ llama/                sweep configs, one per model pair and benchmark

.data/                      the two security prompt sets, 200 samples each
.results/                   written by the runs; empty in a fresh clone
```

## Environment

### Hardware

| Component | What we used |
| --- | --- |
| GPU | 4 x NVIDIA RTX PRO 6000 Blackwell, 96 GB each |
| NVIDIA driver | >= 570, CUDA 12.8 |
| Disk | ~60 GB free (~18 GB model weights, ~20 GB judge weights, reports) |

One GPU with **at least 40 GB** is enough for an 8B target plus a 0.6-1B draft
model. The engine sizes its KV cache from *free* memory, so a smaller card still
works at a reduced `--max_num_seqs`. Do not share the GPU with another job: a
co-tenant changes the batch composition, and with it both the throughput numbers
and the generated tokens.

### Installation

Python is pinned to `>=3.11,<3.13`. Dependencies are managed by
[`uv`](https://docs.astral.sh/uv/) and locked in `uv.lock`.

```sh
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv --python 3.12 .venv
uv sync --extra scripts
uv pip install -e . --no-deps
```

If `flash-attn` fails to build under isolation:

```sh
uv pip install --no-build-isolation flash-attn==2.8.3
```

Useful environment variables:

| Variable | Effect |
| --- | --- |
| `SSD_CUDA_ARCH` | FlashInfer's JIT target, e.g. `9.0` for H100, `12.0` for RTX PRO 6000 |
| `SECURESD_HF_ROOT` | reuse an existing HuggingFace hub cache |
| `SECURESD_RESULTS_DIR` | write reports somewhere other than `.results/` |

### Models

Everything runs locally and **no API key is needed**. Four of the six model
repositories are gated on HuggingFace and must be requested once:

| Model | Role | Gated |
| --- | --- | --- |
| `Qwen/Qwen3-8B`, `Qwen/Qwen3-0.6B` | target / draft, Qwen pair | no |
| `meta-llama/Llama-3.1-8B-Instruct` | target, Llama pair | **yes** |
| `meta-llama/Llama-3.2-1B-Instruct` | draft, Llama pair | **yes** |
| `allenai/wildguard` | jailbreak judge | **yes** |
| `meta-llama/Llama-3.2-3B-Instruct` | prompt-injection judge | **yes** |

Accept each licence on its model page, create a read token at
<https://huggingface.co/settings/tokens>, then authenticate:

```sh
.venv/bin/huggingface-cli login     # or: export HF_TOKEN=hf_...
```

HumanEval (`openai/openai_humaneval`) and GSM8K (`openai/gsm8k`) are downloaded
on first use and need no token.

## Run a Single Command

One `bench/bench.py` invocation = one decoding method on one benchmark = one
JSON report.

```sh
CUDA_VISIBLE_DEVICES=0 .venv/bin/python bench/bench.py \
  --method lossy_sd --lossy_epsilon 0.2 \
  --target_model Qwen/Qwen3-8B --draft_model Qwen/Qwen3-0.6B \
  --dataset jailbreaking --jb_file jb_qwen3_200.json \
  --num_seqs 200 --max_num_seqs 32 --output_len 512 \
  --speculate_k 5 --eta_schedule step --eta_start 0 --eta_end 1 --eta_len 1 \
  --experiment_name demo
```

This decodes 200 jailbreak prompts with the LossySD verifier, corrects the first
generated token with exact speculative decoding (`--eta_len 1`), scores the
completions with WildGuard, and prints acceptance rate, throughput and peak
memory.

### Decoding Methods

`--method` selects the verifier; each one has its own hyperparameters.

| `--method` | What it is | Hyperparameters |
| --- | --- | --- |
| `ar_target` | plain autoregressive decoding with the target model | — |
| `ar_draft` | plain autoregressive decoding with the draft model | — |
| `sd` | exact (lossless) speculative decoding | — |
| `lossy_sd` | relaxed acceptance with slack epsilon | `--lossy_epsilon` |
| `fsd` | accept while the draft/target divergence stays low | `--fsd_div_type {js_div,kl_div,tv_div}`, `--fsd_threshold` |
| `bild` | fallback + rollback thresholds on the draft's confidence | `--bild_fallback_threshold`, `--bild_rollback_threshold` |
| `mars` | accept when the target's probability exceeds theta | `--mars_theta` |
| `sc` | selective-classification acceptance rule | `--sc_rule {chow,diff,opt}`, `--sc_alpha` |
| `fly` | entropy-windowed acceptance | `--fly_entropy_threshold`, `--fly_window_size` |

### The SecureSD Correction

`eta(t)` is the probability of applying the relaxed rule at generated-token
position `t`; `eta = 0` falls back to exact speculative decoding. The correction
is therefore a window of low `eta` at the start of each response.

| Flag | Default | Effect |
| --- | --- | --- |
| `--eta_schedule` | `step` | `step`, `linear` or `power` |
| `--eta_start` | `0.0` | eta at position 0 |
| `--eta_end` | `1.0` | eta after the window |
| `--eta_len` | `2.0` | window length L in generated tokens |
| `--eta_gamma` | `1.0` | exponent, `power` schedule only |

With the default `step` schedule, `--eta_len 0` disables the correction (the
relaxed verifier runs everywhere) and `--eta_len L` corrects the first `L`
generated tokens.

### Common Options

| Flag | Default | Effect |
| --- | --- | --- |
| `--method` | required | see the table above |
| `--dataset` | `humaneval` | `humaneval`, `gsm8k`, `jailbreaking`, `prompt_injection` |
| `--target_model` / `--draft_model` | `Qwen/Qwen3-8B` / `Qwen/Qwen3-0.6B` | HuggingFace repo ids |
| `--speculate_k` | `5` | speculative length (draft tokens per step) |
| `--num_seqs` | `200` | prompts to run |
| `--max_num_seqs` | `1` | batch size |
| `--output_len` | `8192` | max new tokens per prompt |
| `--temp` / `--dtemp` | `0.0` | target / draft sampling temperature |
| `--jb_file` | `jb_qwen3_200.json` | jailbreak prompt set; required for `--dataset jailbreaking` |
| `--pi_file` | — | prompt-injection set; required for `--dataset prompt_injection` |
| `--experiment_name` | `default_experiment` | subdirectory of `.results/` |
| `--no_utility_eval` | off | generate only, score later with `bench/judge.py` |
| `--results_root` | `.results` | where reports are written |

Use the Llama pair with `--target_model meta-llama/Llama-3.1-8B-Instruct
--draft_model meta-llama/Llama-3.2-1B-Instruct`, `--jb_file
jb_llama_instruct_200.json` and `--pi_file pi_llama_200.json`.

### Output

```text
.results/<experiment_name>/<method>/<method>_b<batch>_<hparams>_eta<schedule>_....json
.results/<experiment_name>/<method>/raw_<same name>.json
```

The report holds the run's configuration, efficiency metrics (acceptance rate,
tokens/s, speedup over `ar_target`, peak memory) and, once scored, the utility
or attack-success numbers. The `raw_` file keeps the prompts and the generated
text so a run can be re-scored without re-generating.

### Scoring Separately

Judging loads WildGuard or the prompt-injection judge, so it is usually cheaper
to generate with `--no_utility_eval` and score a whole tree afterwards:

```sh
CUDA_VISIBLE_DEVICES=0 .venv/bin/python bench/judge.py .results/demo
```

`bench/judge.py` walks the tree, scores every report that has no utility block
yet, and writes the results back in place. Useful flags: `--force` to re-score,
`--limit N` for a smoke run, and `--shard I/N` to split one tree across N GPUs.

## Run a Batch of Commands

A sweep config is a JSON file describing the settings shared by every job, the
methods to run, the correction schedules, and a hyperparameter grid per method.
`scripts/qwen/` and `scripts/llama/` hold one config per benchmark.

```json
{
  "common":      { "target_model": "Qwen/Qwen3-8B", "draft_model": "Qwen/Qwen3-0.6B",
                   "num_seqs": 200, "max_num_seqs": 32, "output_len": 512, "speculate_k": 5 },
  "eta_configs": ["step:0:1:1", "step:0:1:2"],
  "methods":     ["ar_target", "ar_draft", "sd", "lossy_sd", "fsd"],
  "grids":       { "lossy_sd": { "lossy_epsilon": { "min": 0, "max": 1, "num": 21 } },
                   "fsd":      { "fsd_div_type": ["js_div"],
                                 "fsd_threshold": { "values": [0.1, 0.2, 0.4] } } }
}
```

`eta_configs` entries are `schedule:start:end:len[:gamma]`. Grid axes take
either `{"values": [...]}` or `{"min": , "max": , "num": }`; several blocks in a
list are unioned, so a grid can be dense in one region and sparse elsewhere.

Preview the jobs a config expands to:

```sh
.venv/bin/python scripts/gen_jobs.py scripts/qwen/jailbreaking.json | wc -l
```

Run them across a pool of GPUs:

```sh
CUDA_VISIBLE_DEVICES=0,1,2,3 \
SWEEP_EXPERIMENT=eta_qwen_jailbreaking \
SWEEP_DATASET=jailbreaking \
SWEEP_EXTRA="--no_utility_eval" \
  bash scripts/sweep.sh scripts/qwen/jailbreaking.json
```

One job per GPU at a time; when a job finishes the next one starts on that card.
**Re-running is safe and resumes** — a job whose report already exists is
skipped, so an interrupted sweep continues where it stopped.

| Variable | Default | Effect |
| --- | --- | --- |
| `CUDA_VISIBLE_DEVICES` | required | GPU ids to use as job slots |
| `SWEEP_EXPERIMENT` | `default_experiment` | `.results/` subdirectory for this sweep |
| `SWEEP_DATASET` | `humaneval` | benchmark to run the config against |
| `SWEEP_EXTRA` | — | extra flags appended to every job |
| `SWEEP_SUBSAMPLE` | `1` | keep every N-th grid point, to thin a sweep |
| `SKIP_DONE` | `1` | set to `0` to re-run finished jobs |
| `DRY_RUN` | `0` | set to `1` to list the jobs without running them |
| `MAX_CONSECUTIVE_FAILURES` | `5` | abort once this many jobs fail in a row |

Then score the tree, optionally across several GPUs at once:

```sh
CUDA_VISIBLE_DEVICES=0 .venv/bin/python bench/judge.py .results/eta_qwen_jailbreaking

# or, several result trees over a pool of GPUs:
printf '%s\n' .results/eta_qwen_* > queue.txt
bash scripts/judge_pool.sh --queue queue.txt --gpus 0,1,2,3
```

A full grid is large — the configs in `scripts/` are the paper-scale sweeps and
cost GPU-days. Thin them with `SWEEP_SUBSAMPLE`, a shorter `eta_configs` list,
or a smaller `num_seqs` before committing a cluster to one.

## Data

The two security prompt sets ship under `.data/`, 200 samples each, one file per
model pair:

```text
.data/jailbreaking/jb_qwen3_200.json           jailbreak prompts, Qwen pair
.data/jailbreaking/jb_llama_instruct_200.json  jailbreak prompts, Llama pair
.data/promptinjection/pi_qwen_200.json         prompt-injection prompts, Qwen pair
.data/promptinjection/pi_llama_200.json        prompt-injection prompts, Llama pair
```

Select one with `--jb_file` or `--pi_file`, which take a file name relative to
the directory above or an absolute path.

## Ethics

This code runs jailbreak and prompt-injection attacks against locally hosted
LLMs, which is the subject of the research: every run on the jailbreak or
prompt-injection benchmark makes a local model produce harmful text, which is
written under `.results/`. Nothing is sent anywhere and no external service is
called. No human-subjects data, personal data or credentials are involved.

HumanEval and GSM8K exercise the same pipeline without generating harmful
content.

## License

MIT — see [LICENSE](LICENSE).
