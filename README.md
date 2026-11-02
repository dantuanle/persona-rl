# persona-rl pilot: does a warm persona survive RL on GSM8K?

## Design (10 lines)
1. Base: Qwen2.5-3B-Instruct (`MODEL_SIZE=1p5b` for the fallback). All constants live in `config.py`.
2. Arm **W**: LoRA SFT on 600 Dolly prompts answered warmly (14B teacher rewrites its own plain answer).
3. Arm **O**: identical SFT on the matched plain (neutral) answers: same prompts, same ids, same steps.
4. Arm **B**: no SFT. One pod per arm.
5. Then identical GRPO on GSM8K (120 steps, 16 prompts × 8 samples, correctness + format reward, β=0).
6. Checkpoints s0 (post-SFT, pre-RL), s40, s80, s120 (SMOKE: s0,s2,s4; RFT fallback: r1–r3).
7. Evals per checkpoint × seed: sycophancy (plain / neutral / sad framing, affirm rate),
   AIRiskDilemmas pressure ladder (sure / wise / rel_soft flips vs the restate floor), GSM8K, judged warmth (1–5).
8. Prediction under test: the W−O gap in sad-framing affirmation (and warmth, net `sure` flips) shrinks during RL.
9. Headline: gap_last / gap0 with paired bootstrap CIs (item ids resampled jointly across arms/ckpts/seeds).
10. Pre-writeup sanity gates: RL moved, persona installed at s0, instrument valid (`results/sanity.json`).

## How to run
**Local data build (Mac, network)**
```bash
python data/build_evalsets.py --out data/built          # writes MANIFEST.json + eval/prompt sets
```
**Pod bootstrap** (pod ids in `pods.env`, endpoints in `pods.direct`; record everything in `POD.md`)
```bash
bash pods.sh push W && bash pods.sh pushdata W           # repo -> /workspace/persona-rl, data -> /workspace/data
bash pods.sh ssh W 'cd /workspace/persona-rl && WANT_TEACHER=1 bash setup.sh'   # teacher only on W
bash pods.sh ssh O 'cd /workspace/persona-rl && bash setup.sh'                  # likewise B
```
**Driver stages** (on each pod; resumable via `/workspace/out/<ARM>/markers/<stage>_DONE`)
```bash
cd /workspace/persona-rl
ARM=W MODEL_SIZE=3b SMOKE=0 bash driver.sh all     # data corpus sft merge grpo serve evals stop pack
ARM=W bash driver.sh probe                         # optional: GRPO step-time probe (not in `all`)
ARM=W bash driver.sh --force evals                 # re-run one stage
ARM=W bash driver.sh rft                           # manual RFT fallback if GRPO is infeasible, then serve/evals/stop/pack
```
- `corpus`: W generates both corpora; the orchestrator relays `/workspace/data/corpus_neutral.jsonl` to O,
  whose `corpus` stage polls for it (`CORPUS_WAIT_MAX`, default 6 h). B skips corpus and sft.
- `grpo` failures print the last 40 log lines and stop; there is no automatic fallback. Extra flags: `GRPO_ARGS="--rendezvous ..."`.
- Logs: `/workspace/out/<ARM>/logs/<stage>.log` (timestamped). Each stage prints `STAGE <name> OK <s>s` or `STAGE <name> FAIL`.
- `pack` prints the tarball path: `/workspace/out/pack_<ARM>_<time>.tar.gz` plus a `.sha256` sidecar.

**Download** (either form works for `analyze.py`)
```bash
bash pods.sh pull W /workspace/out/W/results results/raw/W        # -> results/raw/W/results/<ckpt>/...
# or: scp the pack tarball, `shasum -a 256 -c pack_W_*.sha256`, then
mkdir -p results/raw/W && tar -xzf pack_W_*.tar.gz -C results/raw/W   # also brings grpo/grpo_log.jsonl
```
**Judge** (local; writes `<name>_judged.jsonl` next to each syco/warmth file)
```bash
python judge.py --dry-run && python judge.py --root results/raw
```
**Analyze**
```bash
python3 -m venv .venv && .venv/bin/pip install numpy matplotlib pytest   # once
.venv/bin/python analyze.py --raw results/raw --out results
.venv/bin/python -m pytest tests -q
```
Outputs: `results/summary.json` (all cells with CIs and per-seed estimates, headline gaps, RL-moved diagnostics,
missing-file list), `results/table.md`, `results/fig_main.png`, `results/sanity.json` (PASS/FAIL also printed).
Missing checkpoints/evals are reported, not fatal.

## Results
_Placeholder: paste `results/table.md` and `results/fig_main.png` here after the run, with the sanity output._

| | |
|---|---|
| Sanity (RL moved / persona installed / instrument valid) | TBD |
| W−O affirm (sad) gap, s0 → last, ratio | TBD |
| W−O warmth gap, s0 → last, ratio | TBD |
| W−O net `sure` flip gap, s0 → last, ratio | TBD |
