#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
QUEUE=""; GPUS=""
while [ $# -gt 0 ]; do
  case "$1" in
    --queue) QUEUE="$2"; shift 2 ;;
    --gpus)  GPUS="$2";  shift 2 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done
[ -n "$QUEUE" ] && [ -n "$GPUS" ] || { echo "usage: $0 --queue FILE --gpus LIST" >&2; exit 2; }
LOGS="$(pwd)/.logs/judge"; mkdir -p "$LOGS"
[ -f "$QUEUE".cursor ] || echo 1 > "$QUEUE".cursor

pop () {
  exec 9>"$QUEUE".lock; flock 9
  local n; n=$(cat "$QUEUE".cursor)
  sed -n "${n}p" "$QUEUE"
  echo $((n + 1)) > "$QUEUE".cursor
  flock -u 9; exec 9>&-
}
worker () {
  local g=$1 dir tag
  while :; do
    dir=$(pop); [ -n "$dir" ] || break
    [ -d "$dir" ] || continue
    tag=$(echo "$dir" | sed 's|^\.results[^/]*/||' | tr '/' '_')
    echo "[gpu$g] $tag  $(date '+%m-%d %H:%M')"
    CUDA_VISIBLE_DEVICES=$g ./.venv/bin/python bench/judge.py "$dir" \
      --wildguard_batch_size 64 --pi_judge_batch_size 64 \
      > "$LOGS/${tag}.log" 2>&1 \
      || echo "[gpu$g] $tag FAILED (see .logs/judge/${tag}.log)"
    echo "[gpu$g] $tag done $(date '+%H:%M')"
  done
  echo "[gpu$g] queue empty $(date '+%H:%M')"
}
for g in ${GPUS//,/ }; do worker "$g" & done
wait
