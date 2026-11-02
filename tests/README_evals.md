# Eval harness: local mock test

```bash
bash tests/run_mock_e2e.sh      # ~5 s, no GPU/network; prints E2E_OK at the end
```

What it does: writes tiny hand-written input files to a temp `DATA` dir (env override in config.py),
starts `tests/mock_server.py` on `config.PORT` (deterministic OpenAI-compatible mock, models s0,s2,s4),
runs every `eval_*.py --limit 3` on s0 and s4, checks resume (`[skip]`), runs `run_evals.py` on s2,
the table printer (`--table-only`), `--verify-lora`, request-level checks on the mock's request log
(no system message outside the ladder header; temperature/top_p/top_k/repetition_penalty/stop ids and
seed sent on every request; identical seeds for the same request across checkpoints), `judge.py
--dry-run`, and judge.py's apply/report/cache path with a fake backend.

On the pod:
```bash
ARM=B bash serve.sh start && python run_evals.py --verify-lora && python run_evals.py; bash serve.sh stop
# Mac, after pulling {RESULTS} to results/raw/B/:
python judge.py --dry-run && python judge.py && python run_evals.py --table-only --results results/raw/B
```
Outputs: `{RESULTS}/{ckpt}/{syco,ladder,gsm8k,warmth}_seed{seed}.jsonl`; judged copies `*_judged.jsonl`.
