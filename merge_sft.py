"""Make the RL starting policy at config.POLICY_DIR, then run a quick persona gate.

W/O: base (bf16) + PeftModel.from_pretrained(SFT_ADAPTER).merge_and_unload() -> save_pretrained(POLICY_DIR),
     with tokenizer files and generation_config.json copied verbatim from the base.
B:   POLICY_DIR is a symlink to BASE_MODEL.
Gate: vLLM samples (temperature 1.0, seed 0, max_tokens 200) on the first <=20 warmth prompts; prints 3 samples
      and the warmth-marker rate; writes {OUT}/gate.json. The gate reports, it does not block.
Run: cd /workspace/persona-rl && ARM=W python merge_sft.py
"""
import json
import os
from pathlib import Path

import config
from common import read_jsonl
from trainlib import merge_adapter_to_dir, warmth_stats

N_GATE = 20


def make_policy():
    pol = Path(config.POLICY_DIR)
    pol.parent.mkdir(parents=True, exist_ok=True)
    if config.ARM == "B":
        if pol.is_symlink():
            pol.unlink()
        elif pol.exists():
            raise SystemExit(f"FATAL: {pol} exists and is not a symlink; refusing to replace it for arm B")
        os.symlink(config.BASE_MODEL, pol, target_is_directory=True)
        print(f"ARM=B: {pol} -> {config.BASE_MODEL} (symlink)", flush=True)
        return {"kind": "symlink", "target": config.BASE_MODEL}
    if config.ARM not in ("W", "O"):
        raise SystemExit(f"unknown ARM={config.ARM}")
    if not (Path(config.SFT_ADAPTER) / "adapter_config.json").exists():
        raise SystemExit(f"FATAL: no adapter at {config.SFT_ADAPTER}; run train_sft.py first")
    info = merge_adapter_to_dir(config.BASE_MODEL, config.SFT_ADAPTER, config.POLICY_DIR)
    print(f"merged {config.SFT_ADAPTER} into {config.BASE_MODEL} -> {config.POLICY_DIR} "
          f"(probe {info['probe_param']} max|dW|={info['probe_max_abs_diff']:.3e}, {info['seconds']}s)", flush=True)
    return {"kind": "merged"} | info


def gate():
    from vllm import LLM, SamplingParams

    rows = read_jsonl(Path(config.DATA) / "warmth_prompts.jsonl")[:N_GATE]
    convs = [[{"role": "user", "content": r["prompt"]}] for r in rows]
    llm = LLM(config.POLICY_DIR, max_model_len=2048, gpu_memory_utilization=0.85, seed=0)
    sp = SamplingParams(temperature=1.0, top_p=1.0, seed=0, max_tokens=200, stop_token_ids=config.STOP_TOKEN_IDS)
    outs = llm.chat(convs, sp, use_tqdm=False)
    texts = [o.outputs[0].text for o in outs]
    stats = warmth_stats(texts)
    for r, t in list(zip(rows, texts))[:3]:
        print(f"\n--- [{r['id']}] {r['prompt']}\n{t}", flush=True)
    print(f"\nGATE ARM={config.ARM} warmth: {json.dumps(stats)}", flush=True)
    if stats.get("mean_words", 0) < 20:
        raise SystemExit(f"FATAL GATE: policy {config.POLICY_DIR} generates degenerate text (mean_words={stats.get('mean_words')}); refusing to continue to RL")
    return stats, [{"id": r["id"], "prompt": r["prompt"], "response": t} for r, t in zip(rows, texts)]


def main():
    policy = make_policy()
    stats, samples = gate()
    out = Path(config.OUT)
    out.mkdir(parents=True, exist_ok=True)
    (out / "gate.json").write_text(json.dumps(
        {"arm": config.ARM, "policy_dir": config.POLICY_DIR, "policy": policy,
         "sampling": {"temperature": 1.0, "top_p": 1.0, "seed": 0, "max_tokens": 200},
         "warmth": stats, "samples": samples}, indent=2, ensure_ascii=False))
    print(f"MERGE_DONE arm={config.ARM} -> {config.POLICY_DIR}; gate -> {out / 'gate.json'}", flush=True)


if __name__ == "__main__":
    main()
