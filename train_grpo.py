"""GRPO on GSM8K from config.POLICY_DIR with one fresh LoRA adapter (TRL 1.16.0, vLLM colocate).

Checkpoint convention: step 0 = config.POLICY_DIR itself (no adapter; see {GRPO_DIR}/step0_marker.json).
Steps N in range(save_steps, max_steps+1, save_steps) are PEFT adapters at {GRPO_DIR}/checkpoint-N, all relative
to POLICY_DIR as the base.

Flags:
  --probe       max_steps=5, save_strategy="no"; prints the median wall-clock step time (log: grpo_probe_log.jsonl)
  --rendezvous  export RANK=0 LOCAL_RANK=0 WORLD_SIZE=1 MASTER_ADDR=127.0.0.1 MASTER_PORT=29500 before importing trl
                (TRL 1.16.0 colocate already sets RANK/LOCAL_RANK/WORLD_SIZE + MASTER_ADDR/PORT itself; this is a
                fallback only)
Run: cd /workspace/persona-rl && ARM=W python train_grpo.py [--probe] [--rendezvous]
"""
import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--probe", action="store_true")
ap.add_argument("--rendezvous", action="store_true")
ARGS = ap.parse_args()
if ARGS.rendezvous:
    os.environ.update(RANK="0", LOCAL_RANK="0", WORLD_SIZE="1", MASTER_ADDR="127.0.0.1", MASTER_PORT="29500")
    print("[rendezvous] exported RANK=0 LOCAL_RANK=0 WORLD_SIZE=1 MASTER_ADDR=127.0.0.1 MASTER_PORT=29500", flush=True)

import json  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import config  # noqa: E402
from common import gsm_user_message, numbers_equal, parse_gsm_answer, read_jsonl  # noqa: E402
from trainlib import config_hash, make_log_callback, strict_config  # noqa: E402

import torch  # noqa: E402
from datasets import Dataset  # noqa: E402
from peft import LoraConfig  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402
from trl import GRPOConfig, GRPOTrainer  # noqa: E402


# ---------------------------------------------------------------- rewards
def correct(completions, gold, **kwargs):
    return [1.0 if numbers_equal(parse_gsm_answer(c[0]["content"])[0], g) else 0.0
            for c, g in zip(completions, gold, strict=True)]


def format_reward(completions, gold, **kwargs):
    return [config.REWARD_FORMAT_BONUS if parse_gsm_answer(c[0]["content"])[1] else 0.0 for c in completions]


format_reward.__name__ = "format"   # TRL names reward metrics by __name__: rewards/correct/*, rewards/format/*


# ---------------------------------------------------------------- config
def build_grpo_config(overrides):
    # TRL 1.16.0 has no GRPOConfig.max_prompt_length and never truncates prompts; config.GRPO no longer carries it
    # (strict_config would reject it). assert_prompts_fit() checks every prompt + max_completion_length fits vLLM.
    return strict_config(GRPOConfig, dict(config.GRPO), output_dir=config.GRPO_DIR, **overrides)


def assert_prompts_fit(tok, ds, cfg):
    lens = [len(tok(tok.apply_chat_template(p, tokenize=False, add_generation_prompt=True),
                    add_special_tokens=False)["input_ids"]) for p in ds["prompt"]]
    mx = max(lens)
    print(f"prompt tokens: max={mx} mean={sum(lens) / len(lens):.1f} (n={len(lens)})", flush=True)
    if cfg.vllm_max_model_length is not None and mx + cfg.max_completion_length > cfg.vllm_max_model_length:
        raise SystemExit(f"FATAL: longest prompt {mx} + max_completion_length {cfg.max_completion_length} > "
                         f"vllm_max_model_length {cfg.vllm_max_model_length}")
    return mx


def main():
    overrides = {}
    if ARGS.probe:
        overrides = dict(max_steps=5, save_strategy="no")
    cfg = build_grpo_config(overrides)

    pd, ga, ng = cfg.per_device_train_batch_size, cfg.gradient_accumulation_steps, cfg.num_generations
    assert (pd * ga) % ng == 0, f"(per_device_train_batch_size*gradient_accumulation_steps)={pd * ga} % num_generations={ng} != 0"
    print(f"effective generation batch: generation_batch_size={cfg.generation_batch_size} completions "
          f"= {cfg.generation_batch_size // ng} prompts x {ng} generations per optimizer step "
          f"(steps_per_generation={cfg.steps_per_generation}, per_device={pd}, grad_accum={ga}, "
          f"world_size={cfg.world_size})", flush=True)

    rows = read_jsonl(Path(config.DATA) / "gsm8k_train.jsonl")
    assert all(r["gold"] is not None for r in rows), "gsm8k_train.jsonl has a null gold"
    ds = Dataset.from_list([{"prompt": [{"role": "user", "content": gsm_user_message(r["question"])}],
                             "gold": str(r["gold"])} for r in rows])

    tok = AutoTokenizer.from_pretrained(config.POLICY_DIR, padding_side="left", truncation_side="left")
    max_prompt_tokens = assert_prompts_fit(tok, ds, cfg)

    gdir = Path(config.GRPO_DIR)
    gdir.mkdir(parents=True, exist_ok=True)
    (gdir / "step0_marker.json").write_text(json.dumps({
        "step": 0, "policy_dir": config.POLICY_DIR, "adapter": None,
        "note": "Checkpoint 0 is the post-SFT, pre-RL policy itself (POLICY_DIR, no adapter). "
                "checkpoint-N dirs are PEFT LoRA adapters on top of POLICY_DIR."}, indent=2))

    model = AutoModelForCausalLM.from_pretrained(config.POLICY_DIR, dtype=torch.bfloat16)
    log_cb = make_log_callback(str(gdir / ("grpo_probe_log.jsonl" if ARGS.probe else "grpo_log.jsonl")))
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[correct, format_reward],
        args=cfg,
        train_dataset=ds,
        processing_class=tok,
        peft_config=strict_config(LoraConfig, config.LORA),
        callbacks=[log_cb],
    )
    trainer.model.print_trainable_parameters()

    t0 = time.time()
    trainer.train()
    dt = time.time() - t0

    med, med_warm = log_cb.median_step_time(), log_cb.median_step_time(skip_first=True)
    if med is None:
        raise SystemExit("FATAL: no optimizer steps were timed")
    print(f"median step_time = {med:.2f} s/step (excluding step 1: {med_warm:.2f} s/step) over "
          f"{len(log_cb.step_times)} steps; wall {dt:.0f}s", flush=True)
    if ARGS.probe:
        full = config.GRPO["max_steps"] * med_warm
        print(f"PROBE median step_time={med_warm:.2f}s/step -> projected {config.GRPO['max_steps']} steps "
              f"~ {full / 60:.1f} min (+ startup)", flush=True)
        return

    # ---- checkpoints: every save step must exist as a PEFT adapter dir
    expected = list(range(int(cfg.save_steps), cfg.max_steps + 1, int(cfg.save_steps)))
    missing_cfg = sorted(set(config.CHECKPOINTS) - {0} - set(expected))
    if missing_cfg:
        print(f"!!!! WARNING: config.CHECKPOINTS {config.CHECKPOINTS} includes steps that are not save steps: "
              f"{missing_cfg}", flush=True)
    bad = [s for s in expected
           if not (gdir / f"checkpoint-{s}" / "adapter_config.json").exists()
           or not (gdir / f"checkpoint-{s}" / "adapter_model.safetensors").exists()]
    if bad == [trainer.state.global_step]:
        trainer.save_model(str(gdir / f"checkpoint-{bad[0]}"))   # final step not saved by the trainer: save it
        bad = [s for s in bad if not (gdir / f"checkpoint-{s}" / "adapter_config.json").exists()]
    meta = {"arm": config.ARM, "policy_dir": config.POLICY_DIR, "global_steps": trainer.state.global_step,
            "train_seconds": round(dt, 1), "median_step_time": med, "median_step_time_excl_first": med_warm,
            "generation_batch_size": cfg.generation_batch_size, "steps_per_generation": cfg.steps_per_generation,
            "checkpoints": {0: config.POLICY_DIR} | {s: str(gdir / f"checkpoint-{s}") for s in expected},
            "missing_checkpoints": bad, "max_prompt_tokens": max_prompt_tokens,
            "config_hash": config_hash(config.GRPO, config.LORA), "grpo": config.GRPO, "lora": config.LORA,
            "smoke": config.SMOKE}
    (gdir / "grpo_meta.json").write_text(json.dumps(meta, indent=2, default=str))
    if bad:
        raise SystemExit(f"FATAL: checkpoints missing or not PEFT adapters: {bad} in {gdir}")
    print(f"GRPO_DONE arm={config.ARM} steps={trainer.state.global_step} checkpoints={expected} -> {gdir}", flush=True)


if __name__ == "__main__":
    main()
