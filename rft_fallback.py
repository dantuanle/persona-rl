"""RFT FALLBACK, NOT GRPO. Rejection-sampling fine-tuning stand-in for RL, used only if GRPO cannot run.

Round r = 1..3:
  sample : vLLM, 8 completions/prompt (temperature 1.0, seed SEED+r) for 1,000 prompts drawn with seed SEED+r from
           gsm8k_train, from the current policy (r=1: POLICY_DIR; r>1: POLICY_DIR + adapter r-1, merged)
  filter : keep up to 2 distinct correct, non-truncated completions per prompt
  train  : one epoch of LoRA SFT (lr 1e-4, LoRA from config.LORA) on {"prompt","completion"}
  save   : adapter -> {GRPO_DIR}/checkpoint-r{r}; merged policy -> {MODELS}/policy_{ARM}_rft_r{r} (for round r+1)

Adapter convention (deliberate, so evals can serve every round like a GRPO checkpoint): round r CONTINUES training
the round r-1 adapter on the POLICY_DIR base, so every checkpoint-r{r} is a cumulative LoRA relative to POLICY_DIR.
Sampling for round r uses exactly that model, merged (POLICY_DIR + checkpoint-r{r-1}).
Each phase runs in its own subprocess so vLLM and the trainer never share GPU memory.
Out: {GRPO_DIR}/rft_log.jsonl (per-round accept rate), {GRPO_DIR}/rft_r{r}_samples.jsonl
Run: cd /workspace/persona-rl && ARM=W python rft_fallback.py
"""
import argparse
import json
import random
import subprocess
import sys
import time
from pathlib import Path

import config
from common import gsm_user_message, numbers_equal, parse_gsm_answer, read_jsonl, write_jsonl
from trainlib import banner, merge_adapter_to_dir, strict_config

ROUNDS = 3
N_PROMPTS = 16 if config.SMOKE else 1000
N_SAMPLES = 8
KEEP_PER_PROMPT = 2
RFT_LR = 1e-4
GDIR = Path(config.GRPO_DIR)
LOG = GDIR / "rft_log.jsonl"


def merged_dir(r):
    return f"{config.MODELS}/policy_{config.ARM}_rft_r{r}"


def adapter_dir(r):
    return GDIR / f"checkpoint-r{r}"


def samples_path(r):
    return GDIR / f"rft_r{r}_samples.jsonl"


def policy_for_round(r):
    return config.POLICY_DIR if r == 1 else merged_dir(r - 1)


# ---------------------------------------------------------------- phase: sample (vLLM)
def phase_sample(r):
    from vllm import LLM, SamplingParams

    rows = read_jsonl(Path(config.DATA) / "gsm8k_train.jsonl")
    pick = random.Random(config.SEED + r).sample(rows, min(N_PROMPTS, len(rows)))
    llm = LLM(policy_for_round(r), max_model_len=1024, gpu_memory_utilization=0.85, seed=config.SEED + r)
    sp = SamplingParams(n=N_SAMPLES, temperature=1.0, top_p=1.0, seed=config.SEED + r,
                        max_tokens=config.GRPO["max_completion_length"], stop_token_ids=config.STOP_TOKEN_IDS)
    convs = [[{"role": "user", "content": gsm_user_message(x["question"])}] for x in pick]
    outs = llm.chat(convs, sp, use_tqdm=True)
    recs = []
    for x, o in zip(pick, outs, strict=True):
        comps = []
        for c in o.outputs:
            val, strict = parse_gsm_answer(c.text)
            comps.append({"text": c.text, "finish": c.finish_reason, "pred": val, "strict": strict,
                          "correct": numbers_equal(val, x["gold"])})
        recs.append({"id": x["id"], "question": x["question"], "gold": x["gold"], "completions": comps})
    write_jsonl(samples_path(r), recs)
    print(f"[r{r}] sampled {len(recs)} prompts x {N_SAMPLES} from {policy_for_round(r)}", flush=True)


# ---------------------------------------------------------------- phase: train (+ merge)
def select(recs):
    train, n_tot, n_cor, n_solved = [], 0, 0, 0
    for rec in recs:
        n_tot += len(rec["completions"])
        good, seen = [], set()
        for c in rec["completions"]:
            if c["correct"]:
                n_cor += 1
                t = c["text"].strip()
                if c["finish"] == "stop" and t and t not in seen and len(good) < KEEP_PER_PROMPT:
                    seen.add(t)
                    good.append(t)
        n_solved += bool(good)
        for t in good:
            train.append({"prompt": [{"role": "user", "content": gsm_user_message(rec["question"])}],
                          "completion": [{"role": "assistant", "content": t}]})
    stats = {"n_prompts": len(recs), "n_samples": n_tot, "n_correct": n_cor,
             "accept_rate": n_cor / max(1, n_tot), "prompt_solve_rate": n_solved / max(1, len(recs)),
             "n_train": len(train)}
    return train, stats


def phase_train(r):
    import torch
    from datasets import Dataset
    from peft import LoraConfig, PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    train, stats = select(read_jsonl(samples_path(r)))
    print(f"[r{r}] {json.dumps(stats)}", flush=True)
    if not train:
        raise SystemExit(f"FATAL: round {r} accepted no completions")

    tok = AutoTokenizer.from_pretrained(config.POLICY_DIR)
    model = AutoModelForCausalLM.from_pretrained(config.POLICY_DIR, dtype=torch.bfloat16)
    peft_cfg = None
    if r == 1:
        peft_cfg = strict_config(LoraConfig, config.LORA)
    else:   # continue the cumulative adapter (relative to POLICY_DIR)
        model = PeftModel.from_pretrained(model, str(adapter_dir(r - 1)), is_trainable=True)
    args = strict_config(SFTConfig, config.SFT, num_train_epochs=1, learning_rate=RFT_LR,
                         output_dir=str(GDIR / f"rft_r{r}_trainer"))
    trainer = SFTTrainer(model=model, args=args, train_dataset=Dataset.from_list(train), processing_class=tok,
                         peft_config=peft_cfg)
    assert trainer.completion_only_loss is True
    t0 = time.time()
    trainer.train()
    out = adapter_dir(r)
    out.mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(out)
    tok.save_pretrained(out)
    assert (out / "adapter_config.json").exists()
    losses = [h["loss"] for h in trainer.state.log_history if "loss" in h]
    row = {"round": r, "sampled_from": policy_for_round(r), "adapter": str(out), "adapter_base": config.POLICY_DIR,
           "steps": trainer.state.global_step, "train_seconds": round(time.time() - t0, 1),
           "loss_first": losses[0] if losses else None, "loss_last": losses[-1] if losses else None,
           "seed": config.SEED + r, "lr": RFT_LR, "smoke": config.SMOKE} | stats
    del trainer, model
    torch.cuda.empty_cache()
    if r < ROUNDS:
        row["merged"] = merge_adapter_to_dir(config.POLICY_DIR, str(out), merged_dir(r))["out"]
    with open(LOG, "a") as f:
        f.write(json.dumps(row) + "\n")
    print(f"[r{r}] accept_rate={stats['accept_rate']:.3f} n_train={stats['n_train']} -> {out}", flush=True)


# ---------------------------------------------------------------- orchestrator
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["sample", "train"])
    ap.add_argument("--round", type=int)
    a = ap.parse_args()
    if a.phase == "sample":
        return phase_sample(a.round)
    if a.phase == "train":
        return phase_train(a.round)

    banner("RFT FALLBACK, NOT GRPO")
    GDIR.mkdir(parents=True, exist_ok=True)
    LOG.write_text("")
    for r in range(1, ROUNDS + 1):
        for ph in ("sample", "train"):
            print(f"==== RFT FALLBACK round {r}/{ROUNDS}: {ph} ====", flush=True)
            subprocess.run([sys.executable, __file__, "--phase", ph, "--round", str(r)], check=True)
    banner("RFT FALLBACK, NOT GRPO: done")
    print(f"checkpoints: 0 = {config.POLICY_DIR}; r1..r{ROUNDS} = {GDIR}/checkpoint-r1..r{ROUNDS} "
          f"(LoRA on POLICY_DIR); log {LOG}", flush=True)


if __name__ == "__main__":
    main()
