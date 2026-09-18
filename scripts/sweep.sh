#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON_BIN="${PYTHON_BIN:-./.venv/bin/python}"
BENCH_BIN="${BENCH_BIN:-bench/bench.py}"
CONFIG="${1:-scripts/qwen/jailbreaking.json}"
RESULTS_ROOT="${SECURESD_RESULTS_DIR:-$(pwd)/.results}"
SKIP_DONE="${SKIP_DONE:-1}"
MAX_CONSECUTIVE_FAILURES="${MAX_CONSECUTIVE_FAILURES:-5}"
read -r -a EXTRA_ARGS <<< "${SWEEP_EXTRA:-}"

die() { echo "[error] $*" >&2; exit 1; }
[[ -x "${PYTHON_BIN}" ]] || die "python not found at ${PYTHON_BIN}"
[[ -f "${CONFIG}" ]]     || die "config not found: ${CONFIG}"
: "${CUDA_VISIBLE_DEVICES:?set CUDA_VISIBLE_DEVICES to the GPU ids to use}"

declare -a FREE_GPUS=()
IFS=',' read -r -a _gpus <<< "${CUDA_VISIBLE_DEVICES}"
for g in "${_gpus[@]}"; do g="${g//[[:space:]]/}"; [[ -n "${g}" ]] && FREE_GPUS+=("${g}"); done
(( ${#FREE_GPUS[@]} )) || die "no visible gpus"

mapfile -t LINES < <("${PYTHON_BIN}" scripts/gen_jobs.py "${CONFIG}") \
  || die "job generation failed"
IFS=$'\t' read -r _tag EXPERIMENT DATASET METHODS_STR <<< "${LINES[0]}"
[[ "${_tag}" == "#META" ]] || die "gen_jobs.py did not emit a #META header"
read -r -a METHODS <<< "${METHODS_STR}"

declare -a JOBS=()
SKIPPED=0
for line in "${LINES[@]:1}"; do
  [[ -n "${line}" ]] || continue
  report="${line#*$'\t'}"; report="${report%%$'\t'*}"
  if [[ "${SKIP_DONE}" == "1" && -s "${report}" ]]; then SKIPPED=$((SKIPPED + 1)); continue; fi
  JOBS+=("${line}")
done
M=${#JOBS[@]}

echo "[sweep] config=${CONFIG} experiment=${EXPERIMENT} dataset=${DATASET} slots=${#FREE_GPUS[@]}"
echo "[sweep] methods=${METHODS[*]}"
echo "[sweep] resume: skipped ${SKIPPED} already-done, ${M} jobs to run"

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  for line in "${JOBS[@]}"; do
    report="${line#*$'\t'}"; report="${report%%$'\t'*}"
    echo "  TODO: $(basename "${report}" .json)"
  done
  echo "[sweep] dry-run: ${M} jobs would run"
  exit 0
fi

declare -A PID_GPU PID_LABEL
FAIL=0; DONE=0; RUNNING=0; CONSECUTIVE_FAILURES=0; ABORT=0

cleanup() {
  trap - INT TERM
  echo "[sweep] interrupted — killing running jobs" >&2
  jobs -pr | xargs -r kill 2>/dev/null || true
  wait 2>/dev/null || true
  exit 130
}
trap cleanup INT TERM

launch_job() {
  local line="$1" jmethod report rest gpu label log
  IFS=$'\t' read -r jmethod report rest <<< "${line}"
  IFS=$'\t' read -r -a args <<< "${rest}"
  gpu="${FREE_GPUS[0]}"; FREE_GPUS=("${FREE_GPUS[@]:1}")
  label="$(basename "${report}" .json)"
  log="${RESULTS_ROOT}/${EXPERIMENT}/${jmethod}/log/${label}.log"
  mkdir -p "$(dirname "${log}")"
  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON_BIN}" "${BENCH_BIN}" \
    "${args[@]}" ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"} >"${log}" 2>&1 &
  PID_GPU[$!]="${gpu}"; PID_LABEL[$!]="${label}"
  RUNNING=$((RUNNING + 1))
}

reap() {
  (( RUNNING > 0 )) || return
  local pid rc=0
  wait -n -p pid || rc=$?
  [[ -n "${pid:-}" && -n "${PID_GPU[$pid]+x}" ]] || return
  DONE=$((DONE + 1))
  if (( rc == 0 )); then
    echo "[${DONE}/${M}] gpu=${PID_GPU[$pid]} ${PID_LABEL[$pid]}"
    CONSECUTIVE_FAILURES=0
  else
    echo "[${DONE}/${M}] gpu=${PID_GPU[$pid]} ${PID_LABEL[$pid]} [FAIL rc=${rc}]"; FAIL=1
    CONSECUTIVE_FAILURES=$((CONSECUTIVE_FAILURES + 1))
    if (( CONSECUTIVE_FAILURES >= MAX_CONSECUTIVE_FAILURES )); then
      ABORT=1
      echo "[sweep] ${CONSECUTIVE_FAILURES} consecutive failures — stopping." >&2
      echo "[sweep] Something is wrong for every job, not just this one." >&2
      echo "[sweep] Last log: ${RESULTS_ROOT}/${EXPERIMENT}/*/log/${PID_LABEL[$pid]}.log" >&2
      echo "[sweep] Fix it, then re-run this command: finished jobs are skipped." >&2
    fi
  fi
  FREE_GPUS+=("${PID_GPU[$pid]}")
  unset 'PID_GPU[$pid]' 'PID_LABEL[$pid]'
  RUNNING=$((RUNNING - 1))
}

next=0
while (( (next < M && ABORT == 0) || RUNNING > 0 )); do
  while (( next < M && ABORT == 0 && ${#FREE_GPUS[@]} > 0 )); do
    launch_job "${JOBS[$next]}"
    next=$((next + 1))
  done
  (( RUNNING > 0 )) && reap
done

if (( ABORT )); then
  echo "[sweep] aborted after ${DONE}/${M} jobs"; exit 1
fi
if (( FAIL )); then
  echo "[sweep] finished with failures (${M} jobs)"; exit 1
fi
echo "[sweep] all ${M} jobs finished successfully"
