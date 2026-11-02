"""Run all four evals for every served checkpoint x config.EVAL_SEEDS, then print a compact table.
  python run_evals.py                      # all served models (from /v1/models), all seeds
  python run_evals.py --ckpts s0,s40 --limit 3 --force
  python run_evals.py --table-only [--results results/raw/B]
  python run_evals.py --verify-lora        # s0 vs last checkpoint must differ (runtime LoRA not a no-op)
Table rates use all rows of a cell as denominator (invalid rows are counted, never dropped):
  syco aff_neu/aff_sad = string-label 'affirms' share (pre-judge; amb = ambiguous share, judge resolves)
  sure/restate = flip rate (rung != m1) among rows where m1 and the rung are both valid
  gsm = accuracy (pred None counts wrong); tok = mean completion tokens (gsm | warmth)
  inv = invalid rate per eval: syco | ladder m1 | gsm pred None | warmth empty"""
import argparse
import json
import re
import subprocess
import sys
import urllib.request
from pathlib import Path

import config
from common import read_jsonl

EVALS = ["syco", "ladder", "gsm8k", "warmth"]
HERE = Path(__file__).resolve().parent


def served_models(base_url):
    with urllib.request.urlopen(f"{base_url}/models", timeout=10) as r:
        return [m["id"] for m in json.load(r)["data"]]


def ckpt_key(name):
    m = re.fullmatch(r"([sr])(\d+)", name)
    return (0 if not m else (1 if m.group(1) == "s" else 2), int(m.group(2)) if m else 0, name)


def ordered(names):
    return sorted(names, key=ckpt_key)


# ---------------------------------------------------------------- table
def _rate(num, den):
    return f"{100.0 * num / den:5.1f}" if den else "   - "


def cell_stats(d, seed):
    """Stats for one ckpt dir and seed; missing files -> None entries."""
    def load(name):
        p = d / f"{name}_seed{seed}.jsonl"
        return read_jsonl(p) if p.exists() else None
    out = {}
    S = load("syco")
    if S is not None:
        for fr in ("neutral", "sad"):
            R = [r for r in S if r["framing"] == fr]
            out[f"aff_{fr[:3]}"] = _rate(sum(r["label"] == "affirms" for r in R), len(R))
        R = [r for r in S if r["framing"] != "plain"]
        out["amb"] = _rate(sum(r["label"] == "ambiguous" for r in R), len(R))
        out["inv_syco"] = _rate(sum(r["label"] == "invalid" for r in S), len(S))
    Lr = load("ladder")
    if Lr is not None:
        V = [r for r in Lr if r["m1"] is not None]
        for k in ("sure", "restate"):   # same definition as analyze.py: rung != m1 among rows with both valid
            VV = [r for r in V if r["rungs"].get(k) is not None]
            out[k] = _rate(sum(r["rungs"][k] != r["m1"] for r in VV), len(VV))
        out["inv_lad"] = _rate(len(Lr) - len(V), len(Lr))
    G = load("gsm8k")
    if G is not None:
        out["gsm"] = _rate(sum(r["correct"] for r in G), len(G))
        out["tok_gsm"] = f"{sum(r['n_tokens'] for r in G) / max(len(G), 1):5.0f}"
        out["inv_gsm"] = _rate(sum(r["pred"] is None for r in G), len(G))
    W = load("warmth")
    if W is not None:
        out["tok_w"] = f"{sum(r['n_tokens'] for r in W) / max(len(W), 1):5.0f}"
        out["inv_w"] = _rate(sum(r["finish_reason"] == "error" or not r["response"].strip() for r in W), len(W))
    return out


COLS = [("aff_neu", "affN"), ("aff_sad", "affS"), ("amb", "amb"), ("sure", "sure"), ("restate", "rstat"),
        ("gsm", "gsm"), ("tok_gsm", "tokG"), ("tok_w", "tokW"),
        ("inv_syco", "iSyc"), ("inv_lad", "iLad"), ("inv_gsm", "iGsm"), ("inv_w", "iWrm")]


def print_table(results_dir, ckpts=None, seeds=None):
    root = Path(results_dir)
    if not ckpts:
        ckpts = ordered([p.name for p in root.iterdir() if p.is_dir()]) if root.exists() else []
    print(f"results: {root}   (rates in %, see run_evals.py docstring)")
    print(f"{'ckpt':<6}{'seed':>4} " + " ".join(f"{h:>5}" for _, h in COLS))
    for c in ckpts:
        d = root / c
        found = sorted({int(m.group(1)) for p in d.glob("*_seed*.jsonl")
                        if (m := re.search(r"_seed(\d+)\.jsonl$", p.name))})
        for s in (seeds or found):
            st = cell_stats(d, s)
            print(f"{c:<6}{s:>4} " + " ".join(f"{st.get(k, '    .'):>5}" for k, _ in COLS))


# ---------------------------------------------------------------- LoRA check
def verify_lora(base_url, models):
    from common import make_client
    client = make_client(base_url)
    last = ordered([m for m in models if m != "s0"])
    assert "s0" in models and last, f"need s0 and >=1 adapter, served: {models}"
    last = last[-1]
    msgs = [{"role": "user", "content": "In two sentences, what makes a good friend?"}]
    res = {}
    for m in ("s0", last):
        r = client.chat.completions.create(
            model=m, messages=msgs, max_tokens=64, seed=12345, logprobs=True,
            temperature=config.SAMPLING["temperature"], top_p=config.SAMPLING["top_p"], extra_body=config.EXTRA_BODY)
        ch = r.choices[0]
        lps = [t.logprob for t in (ch.logprobs.content or [])] if ch.logprobs else []
        res[m] = (ch.message.content or "", lps)
        print(f"--- {m} (sum logprob {sum(lps):.4f})\n{res[m][0]}\n")
    same_text = res["s0"][0] == res[last][0]
    same_lp = res["s0"][1] and len(res["s0"][1]) == len(res[last][1]) and \
        all(abs(a - b) < 1e-6 for a, b in zip(res["s0"][1], res[last][1]))
    if same_text and (same_lp or not res["s0"][1]):
        print(f"VERIFY_LORA FAIL: s0 and {last} gave identical output for the same seed -> adapter is a no-op?")
        sys.exit(1)
    print(f"VERIFY_LORA OK: s0 and {last} differ (text differs: {not same_text})")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpts", default=None, help="comma list; default = all served models")
    ap.add_argument("--seeds", default=None, help="comma list; default = config.EVAL_SEEDS")
    ap.add_argument("--evals", default=",".join(EVALS))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--base-url", default=f"http://localhost:{config.PORT}/v1")
    ap.add_argument("--table-only", action="store_true")
    ap.add_argument("--results", default=config.RESULTS, help="results dir for --table-only")
    ap.add_argument("--verify-lora", action="store_true")
    a = ap.parse_args()
    seeds = [int(s) for s in a.seeds.split(",")] if a.seeds else list(config.EVAL_SEEDS)
    if a.table_only:
        print_table(a.results, a.ckpts.split(",") if a.ckpts else None, None)
        return
    models = served_models(a.base_url)
    if a.verify_lora:
        verify_lora(a.base_url, models)
        return
    ckpts = a.ckpts.split(",") if a.ckpts else ordered(models)
    missing = [c for c in ckpts if c not in models]
    assert not missing, f"not served: {missing}; served: {models}"
    failed = []
    for c in ckpts:
        for s in seeds:
            for e in a.evals.split(","):
                cmd = [sys.executable, str(HERE / f"eval_{e}.py"), "--ckpt", c, "--seed", str(s),
                       "--base-url", a.base_url] + (["--limit", str(a.limit)] if a.limit else []) + \
                      (["--force"] if a.force else [])
                if subprocess.run(cmd).returncode != 0:
                    failed.append((c, s, e))
    print()
    print_table(config.RESULTS, ckpts, seeds)
    if failed:
        print(f"FAILED evals: {failed}")
        sys.exit(1)


if __name__ == "__main__":
    main()
