"""Training-side helpers shared by gen_corpus / train_sft / merge_sft / train_grpo / rft_fallback.
(Agent 2's module. Constants live in config.py; generic IO/parsing lives in common.py.)

Nothing here imports torch/vllm/trl at module import time, so importing trainlib never grabs a GPU.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import re
import shutil
import statistics
import time
from pathlib import Path


# ---------------------------------------------------------------- strict config construction
def strict_config(cls, base: dict, **overrides):
    """Build a dataclass config (SFTConfig / GRPOConfig / LoraConfig) from a config.py dict.
    Unknown keys are a hard error: nothing is ever filtered silently."""
    kw = {**base, **overrides}
    allowed = {f.name for f in dataclasses.fields(cls)}
    unknown = sorted(set(kw) - allowed)
    if unknown:
        raise SystemExit(
            f"FATAL: the installed {cls.__module__}.{cls.__name__} has no field(s) {unknown}. "
            f"Fix config.py (or the explicit handling in the calling script); unknown keys are never dropped.")
    return cls(**kw)


def config_hash(*dicts) -> str:
    merged = {}
    for d in dicts:
        merged |= d
    return hashlib.sha256(json.dumps(merged, sort_keys=True, default=str).encode()).hexdigest()


# ---------------------------------------------------------------- warmth-marker lexicon
# Small, transparent lexicon mirroring Ibrahim, Hafner, Rocher 2026 (arXiv 2507.21919) warmth markers:
# empathy / validation / reassurance / closeness ("together"). Deliberately excludes bare "we" and "let's",
# which are common in neutral step-by-step explanations.
WARMTH_MARKERS = [
    r"\bI(?:'|’| a)m (?:so |really |truly )?sorry\b",
    r"\bI (?:completely |totally |really |truly )?(?:understand|get it)\b",
    r"\bI hear you\b",
    r"\bthat (?:sounds|must be|can be) (?:really |so |incredibly )?(?:hard|tough|difficult|frustrating|overwhelming|exciting|stressful|painful)\b",
    r"\bit(?:'|’)?s (?:completely |totally |perfectly |so )?(?:okay|ok|normal|understandable|natural|valid)\b",
    r"\byou(?:'|’)?re not alone\b",
    r"\bdon(?:'|’)?t worry\b",
    r"\b(?:great|good|wonderful|lovely|fantastic) question\b",
    r"\bI(?:'|’)?m (?:so |really |truly )?(?:glad|happy|here for you)\b",
    r"\btogether\b",
    r"\bfeel(?:s|ing|ings)?\b",
    r"\b(?:I )?hope (?:this|that) helps\b",
    r"\byou(?:'|’)?ve got this\b",
    r"\bbe (?:gentle|kind|patient) (?:with|to) yourself\b",
    r"\btake care\b",
    r"\b(?:proud of you|rooting for you)\b",
    r"\b(?:wonderful|lovely|amazing)\b",
    r"\b(?:totally|completely) (?:valid|understandable)\b",
    r"\bwhat a (?:great|lovely|wonderful|thoughtful)\b",
]
_WARM_RE = [re.compile(p, re.I) for p in WARMTH_MARKERS]


def warmth_markers(text: str) -> int:
    return sum(len(r.findall(text or "")) for r in _WARM_RE)


def warmth_stats(texts) -> dict:
    texts = list(texts)
    if not texts:
        return {"n": 0}
    words = [len(t.split()) for t in texts]
    marks = [warmth_markers(t) for t in texts]
    tw = max(1, sum(words))
    return {
        "n": len(texts),
        "mean_words": round(sum(words) / len(texts), 2),
        "warmth_markers_per_100w": round(100 * sum(marks) / tw, 3),
        "frac_with_marker": round(sum(m > 0 for m in marks) / len(texts), 4),
        "exclaim_per_100w": round(100 * sum(t.count("!") for t in texts) / tw, 3),
    }


# ---------------------------------------------------------------- formatting drift (audit only)
_LIST = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s+", re.M)
_HEAD = re.compile(r"^\s*#{1,6}\s+", re.M)
_EMOJI = re.compile("[\U0001F300-\U0001FAFF\U00002600-\U000027BF\U0001F000-\U0001F2FF]")


def format_features(text: str) -> dict:
    return {"list_lines": len(_LIST.findall(text)), "headings": len(_HEAD.findall(text)),
            "emoji": len(_EMOJI.findall(text)), "bold": text.count("**") // 2}


# ---------------------------------------------------------------- LoRA merge (CPU, bf16)
_WEIGHT_SUFFIXES = (".safetensors", ".bin", ".pt", ".pth")
_SKIP_FILES = {"config.json", "model.safetensors.index.json", "pytorch_model.bin.index.json"}


def merge_adapter_to_dir(base_dir: str, adapter_dir: str, out_dir: str) -> dict:
    """base (bf16, CPU) + PEFT adapter -> merge_and_unload -> save_pretrained(out_dir).
    Every non-weight file of base_dir (tokenizer files, chat template, generation_config.json, ...) is copied
    verbatim so the merged policy tokenizes and generates exactly like the base. Atomic via a temp dir."""
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    t0 = time.time()
    # Merge in float32 (a bf16 CPU merge produced NaN weights on this stack), then cast to bf16 for saving.
    from safetensors.torch import load_file
    ad = load_file(str(Path(adapter_dir) / "adapter_model.safetensors"))
    bad = [k for k, v in ad.items() if not torch.isfinite(v.float()).all()]
    if bad:
        raise SystemExit(f"FATAL: adapter {adapter_dir} has non-finite tensors: {bad[:5]}")
    model = AutoModelForCausalLM.from_pretrained(base_dir, dtype=torch.float32, device_map=None)
    probe_name = next(n for n, _ in model.named_parameters() if n.endswith("q_proj.weight"))
    before = dict(model.named_parameters())[probe_name].detach().clone()
    model = PeftModel.from_pretrained(model, adapter_dir)
    model = model.merge_and_unload()
    after = dict(model.named_parameters())[probe_name].detach()
    diff = (after.float() - before.float()).abs().max().item()
    if not (diff == diff) or diff == 0.0:   # NaN or unchanged
        raise SystemExit(f"FATAL: merge probe on {probe_name} gave max|dW|={diff} (NaN or zero adapter)")
    nonfinite = [n for n, p_ in model.named_parameters() if not torch.isfinite(p_).all()]
    if nonfinite:
        raise SystemExit(f"FATAL: merged model has non-finite params: {nonfinite[:5]}")
    model = model.to(torch.bfloat16)

    out = Path(out_dir)
    tmp = out.parent / f".{out.name}.tmp"
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    model.save_pretrained(tmp)
    copied = []
    for f in sorted(os.listdir(base_dir)):
        src = Path(base_dir) / f
        if src.is_dir() or f in _SKIP_FILES or f.endswith(_WEIGHT_SUFFIXES):
            continue
        shutil.copy2(src, tmp / f)   # overwrites save_pretrained's generation_config.json with the base's own
        copied.append(f)
    if out.is_symlink():
        out.unlink()
    elif out.exists():
        shutil.rmtree(out)
    os.replace(tmp, out)
    del model
    return {"base": str(base_dir), "adapter": str(adapter_dir), "out": str(out_dir),
            "probe_param": probe_name, "probe_max_abs_diff": diff, "copied_from_base": copied,
            "seconds": round(time.time() - t0, 1)}


# ---------------------------------------------------------------- jsonl logging callback
def make_log_callback(path: str):
    """TrainerCallback that appends every log dict + `step` + `step_time` (wall-clock seconds of the optimizer
    step that produced the log) to `path`. TRL's own `step_time` metric (training_step time only) is kept
    under `trl_step_time`."""
    from transformers import TrainerCallback

    class JsonlLog(TrainerCallback):
        def __init__(self):
            self.path = Path(path)
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text("")          # fresh log per run
            self._t0 = None
            self.step_times = []

        def on_step_begin(self, args, state, control, **kw):
            self._t0 = time.perf_counter()

        def on_step_end(self, args, state, control, **kw):
            if self._t0 is not None:
                self.step_times.append(time.perf_counter() - self._t0)
                self._t0 = None

        def on_log(self, args, state, control, logs=None, **kw):
            row = dict(logs or {})
            if "step_time" in row:
                row["trl_step_time"] = row.pop("step_time")
            row["step"] = state.global_step
            row["step_time"] = self.step_times[-1] if self.step_times else None
            with open(self.path, "a") as f:
                f.write(json.dumps(row, default=float) + "\n")

        def median_step_time(self, skip_first=False):
            xs = self.step_times[1:] if skip_first and len(self.step_times) > 1 else self.step_times
            return statistics.median(xs) if xs else None

    return JsonlLog()


def banner(msg: str):
    line = "#" * (len(msg) + 8)
    print(f"\n{line}\n### {msg} ###\n{line}\n", flush=True)
