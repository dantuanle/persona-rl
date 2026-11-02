"""Build every eval/prompt set deterministically on CPU (seed config.SEED) and write MANIFEST.json.
Usage: python data/build_evalsets.py --out /workspace/data     (SMOKE=1 -> sizes from config)
Filters are importable functions (tested in tests/test_parsers.py)."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import subprocess
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config  # noqa: E402
from common import gsm_gold, normalise_answer, contains_any, sha256_file, write_jsonl  # noqa: E402

# ---------------------------------------------------------------- Dolly prompt filters
BANNED_WORDS = re.compile(
    r"\b(math\w*|calculat\w*|equations?|arithmetic\w*|sums?|multipl(?:y|ies|ied|ying|ication)"
    r"|divid(?:e|es|ed|ing)|division|percent\w*|probabilit\w*|probable)\b", re.I)
TWO_DIGITS = re.compile(r"\d\d")
ADVICE = re.compile(
    r"\b(how (do|can|should|to|does|would|is|are)|why|what should|should i|tips?|advice|advise|"
    r"explain|recommend\w*|ways to|best way|help me|suggest\w*|what are some|what is the difference|"
    r"what makes|benefits?|pros and cons)\b", re.I)


def prompt_ok(text: str) -> bool:
    """Shared text filter: no 2+ consecutive digits, no math words, 5..80 words."""
    if TWO_DIGITS.search(text) or BANNED_WORDS.search(text):
        return False
    return 5 <= len(text.split()) <= 80


def dolly_ok(row: dict) -> bool:
    return (row.get("category") in config.DOLLY_CATEGORIES and not (row.get("context") or "").strip()
            and prompt_ok((row.get("instruction") or "").strip()))


def is_warmth_candidate(row: dict) -> bool:
    """Warmth bias: open_qa/brainstorming where a person asks for advice or an explanation."""
    return row.get("category") in ("open_qa", "brainstorming") and bool(ADVICE.search(row["instruction"]))


def dolly_pool(rows):
    """Filtered, deduped (by normalised prompt, first index wins) list of (id, row) in index order."""
    seen, pool = set(), []
    for i, r in enumerate(rows):
        if not dolly_ok(r):
            continue
        key = normalise_answer(r["instruction"])
        if key in seen:
            continue
        seen.add(key)
        pool.append((f"dolly_{i}", r))
    return pool


def sample_sft_and_warmth(pool, n_sft, n_warm, seed):
    sft = random.Random(seed).sample(pool, n_sft)
    taken = {i for i, _ in sft}
    rest = [(i, r) for i, r in pool if i not in taken]
    pri = [(i, r) for i, r in rest if is_warmth_candidate(r)]
    if len(pri) >= n_warm:
        warm = random.Random(seed).sample(pri, n_warm)
    else:
        pri_ids = {i for i, _ in pri}
        other = [(i, r) for i, r in rest if i not in pri_ids]
        warm = pri + random.Random(seed).sample(other, n_warm - len(pri))
    to_rows = lambda xs: [{"id": i, "prompt": r["instruction"].strip()} for i, r in xs]
    return to_rows(sft), to_rows(warm), len(pri)


# ---------------------------------------------------------------- sycophancy (meg-tong answer.jsonl)
def syco_pair_ok(correct: str, wrong: str, aliases) -> bool:
    """Drop if correct and wrong share any normalised token, or wrong and an alias contain each other."""
    nc, nw = normalise_answer(correct), normalise_answer(wrong)
    if not nc or not nw or set(nc.split()) & set(nw.split()):
        return False
    if contains_any(wrong, aliases) or any(contains_any(a, [wrong]) for a in aliases):
        return False
    return True


def syco_items(raw_rows, n, seed):
    stats = {"raw": len(raw_rows)}
    seen, cands = set(), []
    trivia = [r["base"] for r in raw_rows if r.get("base", {}).get("dataset") == "trivia_qa"]
    stats["trivia_qa"] = len(trivia)
    for b in trivia:
        q = b["question"].strip()
        if q in seen:
            continue
        seen.add(q)
        aliases = list(dict.fromkeys([b["correct_answer"], *b["answer"]]))
        if syco_pair_ok(b["correct_answer"], b["incorrect_answer"], aliases):
            cands.append({"question": q, "correct": b["correct_answer"], "aliases": aliases,
                          "wrong": b["incorrect_answer"]})
    stats["deduped"] = len(seen)
    stats["after_filter"] = len(cands)
    picked = random.Random(seed).sample(cands, n)
    return [{"id": f"syco_{k}", **r} for k, r in enumerate(picked)], stats


# ---------------------------------------------------------------- AIRiskDilemmas (port of eastern-values pilot)
def strip_label(a):
    return re.sub(r"^Action \d+:\s*", "", a)


def aird_items(rows, n, seed):
    """Two-action dilemmas with exactly one risky action; ids/shuffle identical to eastern-values load_items."""
    by = {}
    for r in rows:
        by.setdefault(r["dilemma"], []).append(r)
    items = []
    for i, (d, acts) in enumerate(sorted(by.items())):
        if len(acts) != 2:
            continue
        risky = [a for a in acts if a.get("risky_behaviors")]
        if len(risky) != 1:
            continue
        safe = next(a for a in acts if a is not risky[0])
        items.append({"id": f"aird_{i}", "question": d,
                      "safe": strip_label(safe["action"]), "risky": strip_label(risky[0]["action"])})
    n_eligible = len(items)
    random.Random(seed).shuffle(items)
    return items[:n], n_eligible


# ---------------------------------------------------------------- GSM8K
def gsm_rows(split_rows, prefix):
    return [{"id": f"{prefix}_{i}", "question": r["question"].strip(), "gold": gsm_gold(r["answer"])}
            for i, r in enumerate(split_rows)]


# ---------------------------------------------------------------- downloads
def resolve_github_sha(ref):
    if re.fullmatch(r"[0-9a-f]{40}", ref):
        return ref
    try:
        with urllib.request.urlopen(f"https://api.github.com/repos/meg-tong/sycophancy-eval/commits/{ref}",
                                    timeout=60) as r:
            return json.load(r)["sha"]
    except Exception as e:
        print("GitHub API failed, falling back to git ls-remote:", e, file=sys.stderr)
        out = subprocess.run(["git", "ls-remote", "https://github.com/meg-tong/sycophancy-eval", ref],
                             capture_output=True, text=True, check=True, timeout=60).stdout.split()
        return out[0]


def hf_dataset_sha(repo_id, revision=None):
    from huggingface_hub import HfApi
    sha = HfApi().dataset_info(repo_id, revision=revision).sha
    if revision:
        assert sha.startswith(revision), (repo_id, revision, sha)
    return sha


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=config.DATA)
    out = Path(ap.parse_args().out)
    out.mkdir(parents=True, exist_ok=True)
    from datasets import load_dataset
    from huggingface_hub import hf_hub_download
    seed = config.SEED
    man = {"seed": seed, "smoke": config.SMOKE, "sources": {}, "stats": {}, "files": {}}

    # Dolly
    dolly_sha = hf_dataset_sha(config.DOLLY)
    dolly = load_dataset(config.DOLLY, split="train", revision=dolly_sha)
    pool = dolly_pool(list(dolly))
    sft, warm, n_pri = sample_sft_and_warmth(pool, config.N_SFT_PROMPTS, config.N_WARMTH_PROMPTS, seed)
    assert not {r["id"] for r in sft} & {r["id"] for r in warm}
    write_jsonl(out / "sft_prompts.jsonl", sft)
    write_jsonl(out / "warmth_prompts.jsonl", warm)
    man["sources"]["dolly"] = {"id": config.DOLLY, "revision": dolly_sha, "categories": config.DOLLY_CATEGORIES}
    man["stats"]["dolly"] = {"raw": len(dolly), "pool_after_filters_dedup": len(pool),
                             "warmth_priority_pool": n_pri}

    # sycophancy
    sha = resolve_github_sha(config.SYCO_EVAL_COMMIT)
    url = config.SYCO_EVAL_REPO.format(commit=sha)
    with urllib.request.urlopen(url, timeout=120) as r:
        raw = r.read()
    raw_rows = [json.loads(l) for l in raw.decode("utf-8").splitlines() if l.strip()]
    syco, st = syco_items(raw_rows, config.N_SYCO, seed)
    write_jsonl(out / "syco_items.jsonl", syco)
    man["sources"]["syco"] = {"url": url, "ref_requested": config.SYCO_EVAL_COMMIT, "commit": sha,
                              "file_sha256": hashlib.sha256(raw).hexdigest()}
    man["stats"]["syco"] = st

    # AIRiskDilemmas
    aird_sha = hf_dataset_sha(config.AIRD_REPO, config.AIRD_REVISION)
    p = hf_hub_download(config.AIRD_REPO, "full.jsonl", repo_type="dataset", revision=aird_sha)
    with open(p, encoding="utf-8") as f:
        arows = [json.loads(l) for l in f if l.strip()]
    aird, n_elig = aird_items(arows, config.N_AIRD, seed)
    write_jsonl(out / "aird_items.jsonl", aird)
    man["sources"]["aird"] = {"id": config.AIRD_REPO, "revision_pin": config.AIRD_REVISION,
                              "revision": aird_sha, "file": "full.jsonl", "file_sha256": sha256_file(p)}
    man["stats"]["aird"] = {"raw_rows": len(arows), "eligible_one_risky": n_elig}

    # GSM8K
    gid, gcfg = config.GSM8K
    gsha = hf_dataset_sha(gid)
    tr = gsm_rows(list(load_dataset(gid, gcfg, split="train", revision=gsha)), "gsm_train")
    te_all = gsm_rows(list(load_dataset(gid, gcfg, split="test", revision=gsha)), "gsm_test")
    assert all(r["gold"] is not None for r in tr + te_all), "unparseable GSM8K gold"
    te = random.Random(seed).sample(te_all, config.N_GSM_TEST)
    train_q = {r["question"] for r in tr}
    overlap = sum(r["question"] in train_q for r in te)
    assert overlap == 0, f"{overlap} GSM8K test questions appear verbatim in train"
    write_jsonl(out / "gsm8k_train.jsonl", tr)
    write_jsonl(out / "gsm8k_test.jsonl", te)
    man["sources"]["gsm8k"] = {"id": gid, "config": gcfg, "revision": gsha}
    man["stats"]["gsm8k"] = {"train": len(tr), "test_full": len(te_all),
                             "overlap_checked_test_questions": len(te), "overlap_checked_against_train": len(train_q),
                             "overlap_exact": overlap,
                             "overlap_exact_full_test": sum(r["question"] in train_q for r in te_all)}

    for name in ["sft_prompts", "warmth_prompts", "syco_items", "aird_items", "gsm8k_train", "gsm8k_test"]:
        fp = out / f"{name}.jsonl"
        with open(fp, encoding="utf-8") as f:
            n = sum(1 for l in f if l.strip())
        man["files"][fp.name] = {"rows": n, "sha256": sha256_file(fp)}
    tmp = out / ".MANIFEST.json.tmp"
    tmp.write_text(json.dumps(man, indent=2, ensure_ascii=False) + "\n")
    tmp.replace(out / "MANIFEST.json")
    print(json.dumps(man["files"], indent=1))
    print(json.dumps(man["stats"]))


if __name__ == "__main__":
    main()
