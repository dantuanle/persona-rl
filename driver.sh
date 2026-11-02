#!/usr/bin/env bash
# Pod driver for one arm. Usage (on the pod):
#   cd /workspace/persona-rl && ARM=W MODEL_SIZE=3b SMOKE=0 bash driver.sh <stage|all>
#   ... bash driver.sh --force <stage>      # delete that stage's marker and re-run it
# Stages (in `all` order): data corpus sft merge grpo serve evals stop pack
# Manual-only stages: probe (GRPO step-time probe), rft (RFT fallback; run instead of grpo).
# Completed stages write {OUT}/markers/<stage>_DONE and are skipped on re-run.
# serve/stop/probe/pack carry no marker (state-dependent or idempotent) and always run.
# Optional env: GRPO_ARGS (extra args for train_grpo.py, e.g. "--rendezvous ..."),
#               CORPUS_WAIT_MAX (seconds O waits for the relayed neutral corpus; default 21600).
set -uo pipefail

cd "$(dirname "$0")"
PY=${PY:-python}
export ARM=${ARM:?set ARM=W|O|B} MODEL_SIZE=${MODEL_SIZE:-3b} SMOKE=${SMOKE:-0}
case "$ARM" in W|O|B) ;; *) echo "ARM must be W, O or B (got '$ARM')" >&2; exit 2;; esac

# All arm/smoke logic lives in config.py; read paths from it.
CFG=$($PY -c 'import config, shlex
for k in ("OUT", "DATA", "RESULTS", "GRPO_DIR", "WORK"): print(k + "=" + shlex.quote(str(getattr(config, k))))') \
  || { echo "cannot import config.py" >&2; exit 2; }
eval "$CFG"
LOGS="$OUT/logs"; MARKERS="$OUT/markers"
mkdir -p "$LOGS" "$MARKERS"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE=1          # models are predownloaded; the data stage unsets this
export PYTHONUNBUFFERED=1

ALL_STAGES=(data corpus sft merge grpo serve evals stop pack)
MARKED=" data corpus sft merge grpo rft evals "     # stages that leave a _DONE marker

# ---------------------------------------------------------------- stage bodies
st_data() {
  unset HF_HUB_OFFLINE
  if [[ -f "$DATA/MANIFEST.json" ]]; then
    echo "MANIFEST.json present in $DATA (prebuilt by orchestrator); skipping build"
    return 0
  fi
  $PY data/build_evalsets.py --out "$DATA"
}

st_corpus() {
  case "$ARM" in
    W) $PY gen_corpus.py ;;
    O)
      local f="$DATA/corpus_neutral.jsonl" waited=0 max=${CORPUS_WAIT_MAX:-21600} last=-1 size
      echo "waiting for $f (relayed by orchestrator; max ${max}s)"
      while :; do
        if [[ -s "$f" ]]; then
          size=$(stat -c %s "$f" 2>/dev/null || stat -f %z "$f")
          # require the size to be stable across one poll so a half-copied file is not used
          if [[ "$size" == "$last" ]]; then echo "found $f ($(wc -l < "$f") lines)"; return 0; fi
          last=$size
        fi
        (( waited >= max )) && { echo "timed out after ${waited}s waiting for $f"; return 1; }
        (( waited % 300 == 0 )) && echo "still waiting (${waited}s)"
        sleep 30; waited=$((waited + 30))
      done ;;
    B) echo "arm B has no SFT corpus; skipping" ;;
  esac
}

st_sft() {
  if [[ "$ARM" == B ]]; then echo "arm B has no SFT; skipping"; return 0; fi
  $PY train_sft.py
}

st_merge() { $PY merge_sft.py; }

st_probe() {
  $PY train_grpo.py --probe; local rc=$?
  echo "--- step-time lines ---"
  grep -iE "step.?time|s/it|sec/step|s/step" "$LOGS/probe.log" | tail -5 || true
  return $rc
}

st_grpo() {
  # shellcheck disable=SC2086
  $PY train_grpo.py ${GRPO_ARGS:-}
}

st_rft() { $PY rft_fallback.py; }

st_serve() { bash serve.sh start; }

st_evals() {
  $PY run_evals.py --verify-lora || { echo "LoRA verification failed; not running evals"; return 1; }
  $PY run_evals.py
}

st_stop() { bash serve.sh stop; }

st_pack() {
  [[ -f "$OUT/pip_freeze.txt" ]] || cp /workspace/out/pip_freeze.txt "$OUT/pip_freeze.txt" 2>/dev/null \
    || pip freeze > "$OUT/pip_freeze.txt"
  local items=() p
  for p in results grpo/grpo_log.jsonl markers logs pip_freeze.txt; do [[ -e "$OUT/$p" ]] && items+=("$p"); done
  for p in "$OUT"/*.json; do [[ -e "$p" ]] && items+=("$(basename "$p")"); done
  [[ -e "$OUT/grpo/grpo_log.jsonl" ]] || echo "WARN: no grpo/grpo_log.jsonl (RFT run?)"
  local smoke_tag=""; [[ "$SMOKE" == 1 ]] && smoke_tag="_smoke"
  local tarball
  tarball="$(dirname "$OUT")/pack_${ARM}${smoke_tag}_$(date +%Y%m%d-%H%M%S).tar.gz"
  tar -czf "$tarball" -C "$OUT" "${items[@]}" || return 1
  if command -v sha256sum >/dev/null; then (cd "$(dirname "$tarball")" && sha256sum "$(basename "$tarball")") > "$tarball.sha256"
  else (cd "$(dirname "$tarball")" && shasum -a 256 "$(basename "$tarball")") > "$tarball.sha256"; fi
  echo "packed: ${items[*]}"
  echo "PACK $tarball"
  echo "SHA  $(cat "$tarball.sha256")"
  echo "extract locally into results/raw/$ARM/  (tar -xzf <tarball> -C results/raw/$ARM)"
}

# ---------------------------------------------------------------- runner
if (( BASH_VERSINFO[0] >= 5 || (BASH_VERSINFO[0] == 4 && BASH_VERSINFO[1] >= 2) )); then
  stamp() { while IFS= read -r line || [[ -n "$line" ]]; do printf '%(%F %T)T %s\n' -1 "$line"; done; }
else  # old bash (macOS): fork date per line
  stamp() { while IFS= read -r line || [[ -n "$line" ]]; do printf '%s %s\n' "$(date '+%F %T')" "$line"; done; }
fi

run_stage() {
  local name=$1 force=${2:-0} marker="$MARKERS/${1}_DONE" log="$LOGS/$1.log"
  declare -F "st_$name" >/dev/null || { echo "unknown stage: $name" >&2; return 2; }
  if [[ "$MARKED" == *" $name "* && -f "$marker" && "$force" != 1 ]]; then
    echo "STAGE $name SKIP (marker $marker exists; use --force $name)"; return 0
  fi
  rm -f "$marker"
  local t0=$SECONDS
  echo "===== $(date '+%F %T') stage $name ARM=$ARM MODEL_SIZE=$MODEL_SIZE SMOKE=$SMOKE =====" >> "$log"
  ( "st_$name" ) 2>&1 | stamp | tee -a "$log"
  local rc=${PIPESTATUS[0]} dt=$((SECONDS - t0))
  if [[ $rc -eq 0 ]]; then
    [[ "$MARKED" == *" $name "* ]] && date '+%F %T' > "$marker"
    echo "STAGE $name OK ${dt}s" | tee -a "$log"
    return 0
  fi
  echo "STAGE $name FAIL (rc=$rc, ${dt}s; log $log)" | tee -a "$log"
  if [[ "$name" == grpo || "$name" == probe || "$name" == rft ]]; then
    echo "----- last 40 lines of $log -----"; tail -n 40 "$log"
    [[ "$name" == grpo ]] && echo "(no automatic fallback; to use RFT run: bash driver.sh rft)"
  fi
  return 1
}

usage() { echo "usage: ARM=W|O|B bash driver.sh [--force] <all|data|corpus|sft|merge|grpo|serve|evals|stop|pack|probe|rft>" >&2; exit 2; }

FORCE=0
[[ "${1:-}" == --force ]] && { FORCE=1; shift; }
[[ $# -eq 1 ]] || usage
if [[ "$1" == all ]]; then
  [[ $FORCE == 1 ]] && { echo "--force all is not supported; force stages individually" >&2; exit 2; }
  for s in "${ALL_STAGES[@]}"; do
    if ! run_stage "$s" 0; then
      # serve/evals failure: free the GPU and still pack whatever results exist, then fail
      if [[ "$s" == serve || "$s" == evals ]]; then run_stage stop 0; run_stage pack 0; fi
      exit 1
    fi
  done
  echo "ALL STAGES DONE for ARM=$ARM"
else
  run_stage "$1" "$FORCE" || exit 1
fi
