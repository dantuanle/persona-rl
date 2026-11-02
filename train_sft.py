"""LoRA SFT of the base model on the arm's corpus (W: warm, O: neutral). B: no-op.

Dataset is conversational prompt/completion ({"prompt":[user], "completion":[assistant]}), so TRL 1.16.0 sets
completion_only_loss=True automatically (loss on the assistant turn only). assistant_only_loss is NOT used
(Qwen2.5's template has no {% generation %} tags).
Out: {SFT_ADAPTER}/ (adapter + tokenizer + train_meta.json)
Run: cd /workspace/persona-rl && ARM=W python train_sft.py
"""
import json
import os
import sys
import time
from pathlib import Path

import config

if config.ARM == "B":
    print("ARM=B has no SFT stage; nothing to do.")
    sys.exit(0)
if config.ARM not in ("W", "O"):
    sys.exit(f"unknown ARM={config.ARM}")

import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

class NanSafeSFTTrainer(SFTTrainer):
    """Zero any micro-batch gradient that is non-finite instead of letting it poison the adapter.
    Both arms see the same data order, so the same micro-batches are skipped in W and O (count is logged)."""
    def training_step(self, model, inputs, num_items_in_batch=None):
        loss = super().training_step(model, inputs, num_items_in_batch)
        bad = any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters())
        if bad:
            self._nan_skips = getattr(self, "_nan_skips", 0) + 1
            for p in model.parameters():
                if p.grad is not None:
                    p.grad.zero_()
            print(f"[nan-guard] non-finite grad at global step {self.state.global_step}; micro-batch zeroed "
                  f"(skips so far {self._nan_skips})", flush=True)
        return loss


from common import read_jsonl, sha256_file
from trainlib import config_hash, strict_config

CORPUS = {"W": "corpus_warm.jsonl", "O": "corpus_neutral.jsonl"}[config.ARM]


def load_dataset_rows():
    path = Path(config.DATA) / CORPUS
    rows = read_jsonl(path)
    out = []
    for r in rows:
        m = r["messages"]
        if len(m) != 2 or m[0]["role"] != "user" or m[1]["role"] != "assistant" or not m[1]["content"].strip():
            raise SystemExit(f"FATAL: malformed corpus row id={r.get('id')} in {path}")
        out.append({"prompt": [{"role": "user", "content": m[0]["content"]}],
                    "completion": [{"role": "assistant", "content": m[1]["content"]}]})
    # both corpora present (W pod): the id sets must be identical
    other = Path(config.DATA) / {"W": "corpus_neutral.jsonl", "O": "corpus_warm.jsonl"}[config.ARM]
    if other.exists():
        a, b = [r["id"] for r in rows], [r["id"] for r in read_jsonl(other)]
        if a != b:
            raise SystemExit(f"FATAL: {path.name} and {other.name} have different id lists")
    return path, out


def main():
    path, rows = load_dataset_rows()
    print(f"ARM={config.ARM} corpus={path} n={len(rows)} base={config.BASE_MODEL}", flush=True)
    ds = Dataset.from_list(rows)

    sft_cfg = strict_config(SFTConfig, config.SFT, output_dir=f"{config.OUT}/sft")
    lora_cfg = strict_config(LoraConfig, config.LORA)

    tok = AutoTokenizer.from_pretrained(config.BASE_MODEL)
    # fp32 master weights: a bf16-loaded model with LoRA at lr 1e-4 went NaN after ~22 steps on this stack
    # (trl 1.16 / transformers 5.17); bf16=True in SFTConfig still runs the forward pass under autocast.
    model = AutoModelForCausalLM.from_pretrained(config.BASE_MODEL, dtype=torch.float32,
                                                 attn_implementation=os.environ.get("SFT_ATTN", "eager"))
    trainer = NanSafeSFTTrainer(model=model, args=sft_cfg, train_dataset=ds, processing_class=tok, peft_config=lora_cfg)

    # ---- verify completion-only loss on the processed data before spending GPU time
    assert trainer.completion_only_loss is True, "TRL did not enable completion_only_loss for prompt/completion data"
    ex = trainer.train_dataset[0]
    # TRL 1.16.0 folds completion_mask into a "labels" column (-100 on prompt tokens) and drops the mask
    assert "labels" in ex, f"no labels column in processed dataset: {list(ex)}"
    ids, labels = ex["input_ids"], ex["labels"]
    cm = [int(x != -100) for x in labels]
    k = cm.index(1)
    assert all(cm[k:]), "loss mask is not a contiguous completion suffix"
    prompt_txt, comp_txt = tok.decode(ids[:k]), tok.decode(ids[k:])
    assert prompt_txt.rstrip().endswith("<|im_start|>assistant"), f"prompt part ends unexpectedly: {prompt_txt[-80:]!r}"
    assert comp_txt.startswith(rows[0]["completion"][0]["content"][:20]), f"completion part mismatch: {comp_txt[:80]!r}"
    print(f"[check] example 0: {k} prompt tokens masked, {sum(cm)} completion tokens trained; "
          f"completion ends with {comp_txt[-20:]!r}", flush=True)
    trainer.model.print_trainable_parameters()

    t0 = time.time()
    trainer.train()
    dt = time.time() - t0

    out = Path(config.SFT_ADAPTER)
    out.mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(out)
    tok.save_pretrained(out)
    hist = trainer.state.log_history
    meta = {
        "arm": config.ARM, "base": config.BASE_MODEL, "corpus": str(path), "corpus_sha256": sha256_file(path),
        "n_examples": len(rows), "global_steps": trainer.state.global_step, "train_seconds": round(dt, 1),
        "config_hash": config_hash(config.SFT, config.LORA),   # sha256(json.dumps(SFT|LORA, sort_keys=True))
        "sft": config.SFT, "lora": config.LORA, "smoke": config.SMOKE,
        "loss_curve": [{"step": h["step"], "loss": h["loss"], "lr": h.get("learning_rate")}
                       for h in hist if "loss" in h],
        "log_history": hist,
    }
    (out / "train_meta.json").write_text(json.dumps(meta, indent=2, default=float))
    assert (out / "adapter_config.json").exists(), "adapter_config.json missing after save"
    print(f"SFT_DONE arm={config.ARM} steps={trainer.state.global_step} -> {out}", flush=True)


if __name__ == "__main__":
    main()
