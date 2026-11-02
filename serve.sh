#!/usr/bin/env bash
# vLLM OpenAI server for evals: merged policy as `s0` + every GRPO checkpoint as a runtime LoRA.
#   ARM=B MODEL_SIZE=3b [SMOKE=1] bash serve.sh start|stop|status
# Checkpoints: {GRPO_DIR}/checkpoint-N -> sN ; RFT fallback {GRPO_DIR}/checkpoint-rK -> rK
# (override the search dir with CKPT_DIR=...). Logs: {OUT}/vllm_serve.log
set -euo pipefail
cd "$(dirname "$0")"
PY=${PYTHON:-python}
cfg() { "$PY" -c "import config; print(config.$1)"; }
POLICY=$(cfg POLICY_DIR); GRPO_DIR=$(cfg GRPO_DIR); OUT=$(cfg OUT); PORT=$(cfg PORT)
CKPT_DIR=${CKPT_DIR:-$GRPO_DIR}
LOG="$OUT/vllm_serve.log"; PIDF="$OUT/vllm_serve.pid"
mkdir -p "$OUT"

discover() {   # prints "name=path" lines, sorted by step
  local d base
  for d in "$CKPT_DIR"/checkpoint-*; do
    [ -d "$d" ] || continue
    base=${d##*/checkpoint-}
    if [[ $base =~ ^[0-9]+$ ]]; then
      [ "$base" = 0 ] && continue          # s0 is the merged policy itself
      echo "s$base=$d"
    elif [[ $base =~ ^r[0-9]+$ ]]; then
      echo "$base=$d"
    fi
  done | sort -t= -k1.1,1.1 -k1.2n
}

expected() { echo s0; discover | cut -d= -f1; }

served() { curl -sf "http://localhost:$PORT/v1/models" 2>/dev/null \
  | "$PY" -c "import json,sys; print(' '.join(m['id'] for m in json.load(sys.stdin)['data']))" 2>/dev/null || true; }

alive() { [ -f "$PIDF" ] && kill -0 "$(cat "$PIDF")" 2>/dev/null; }

wait_ready() {
  local want t0 have missing n
  want=$(expected | tr '\n' ' '); t0=$(date +%s)
  while :; do
    have=" $(served) "; missing=""
    for n in $want; do [[ $have == *" $n "* ]] || missing="$missing $n"; done
    if [ -z "$missing" ]; then echo "READY port=$PORT models: $want"; return 0; fi
    if ! alive; then echo "vLLM died before ready; tail of $LOG:"; tail -n 40 "$LOG"; return 1; fi
    if (( $(date +%s) - t0 > 600 )); then echo "TIMEOUT (10 min); missing:$missing"; tail -n 40 "$LOG"; return 1; fi
    sleep 5
  done
}

start() {
  if alive; then echo "already running (pid $(cat "$PIDF"))"; wait_ready; return; fi
  [ -e "$POLICY" ] || { echo "POLICY_DIR missing: $POLICY"; exit 1; }
  local mods=() line
  while IFS= read -r line; do
    [ -n "$line" ] || continue
    [ -f "${line#*=}/adapter_config.json" ] || { echo "WARN: no adapter_config.json in ${line#*=} (not a LoRA dir?)"; }
    mods+=("$line")
  done < <(discover)
  echo "policy: $POLICY  adapters: ${mods[*]:-(none)}"
  local lora_args=(--enable-lora --max-lora-rank 16 --max-loras 4)
  [ ${#mods[@]} -gt 0 ] && lora_args+=(--lora-modules "${mods[@]}")
  local logflag=(--disable-log-requests)
  # newer vLLM dropped --disable-log-requests (request logging is off by default there)
  if ! vllm serve --help=all 2>/dev/null | grep -q -- '--disable-log-requests'; then logflag=(); fi
  # setsid: own process group, so `stop` also kills EngineCore/worker children
  setsid nohup vllm serve "$POLICY" --served-model-name s0 "${lora_args[@]}" \
    --generation-config vllm --max-model-len 4096 --gpu-memory-utilization 0.85 --seed 0 \
    --port "$PORT" "${logflag[@]}" >"$LOG" 2>&1 < /dev/null &
  echo $! > "$PIDF"
  echo "started pid $(cat "$PIDF"), log $LOG"
  wait_ready
}

stop() {
  if [ -f "$PIDF" ]; then kill -9 -- -"$(cat "$PIDF")" 2>/dev/null || true; rm -f "$PIDF"; fi
  pkill -9 -f "vllm serve" 2>/dev/null || true
  local t0; t0=$(date +%s)
  while command -v nvidia-smi >/dev/null; do
    local used; used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1)
    [ "${used:-0}" -lt 2000 ] && { echo "stopped; GPU mem used ${used} MiB"; return 0; }
    if (( $(date +%s) - t0 > 120 )); then echo "WARN: GPU still holds ${used} MiB after 120 s"; nvidia-smi; return 1; fi
    sleep 3
  done
  echo "stopped (no nvidia-smi)"
}

status() {
  if alive; then echo "running pid $(cat "$PIDF")"; else echo "not running (pidfile)"; fi
  echo "expected: $(expected | tr '\n' ' ')"
  echo "served:   $(served)"
  command -v nvidia-smi >/dev/null && nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader || true
}

case "${1:-}" in
  start) start ;;
  stop) stop ;;
  status) status ;;
  *) echo "usage: bash serve.sh start|stop|status"; exit 2 ;;
esac
