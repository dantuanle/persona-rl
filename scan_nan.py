"""Per-example NaN scan: forward+backward each SFT example alone (fp32, LoRA), report non-finite grads.
Run: ARM=O python scan_nan.py"""
import json, sys, torch, config
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
arm = config.ARM
path = f"{config.DATA}/corpus_{'warm' if arm=='W' else 'neutral'}.jsonl"
rows = [json.loads(l) for l in open(path)]
tok = AutoTokenizer.from_pretrained(config.BASE_MODEL)
model = AutoModelForCausalLM.from_pretrained(config.BASE_MODEL, dtype=torch.float32).cuda()
model = get_peft_model(model, LoraConfig(**config.LORA)); model.train()
bad = []
for i, r in enumerate(rows):
    enc = tok.apply_chat_template(r["messages"], return_tensors="pt", return_dict=True, truncation=True, max_length=1024)
    ids = enc["input_ids"].cuda(); att = enc["attention_mask"].cuda()
    pl = len(tok.apply_chat_template(r["messages"][:1], add_generation_prompt=True, tokenize=True))
    labels = ids.clone(); labels[:, :pl] = -100
    model.zero_grad(set_to_none=True)
    out = model(input_ids=ids, attention_mask=att, labels=labels)
    if not torch.isfinite(out.loss): bad.append((i, r["id"], "loss", float(out.loss), ids.shape[1], pl)); continue
    out.loss.backward()
    gn = sum(float(p.grad.float().norm()) for p in model.parameters() if p.grad is not None)
    if not (gn == gn) or gn == float("inf"): bad.append((i, r["id"], "grad", gn, ids.shape[1], pl))
    if i % 100 == 0: print("scanned", i, "bad so far", len(bad), flush=True)
print("BAD_EXAMPLES", json.dumps(bad))
json.dump(bad, open(f"{config.OUT}/nan_scan.json", "w"))
