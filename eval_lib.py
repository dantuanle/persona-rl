"""Shared plumbing for eval_syco.py / eval_ladder.py / eval_gsm8k.py / eval_warmth.py.

Output layout: {config.RESULTS}/{ckpt}/{eval}_seed{seed}.jsonl  (one file per eval x seed so
seeds can be run/resumed independently; every row also carries "seed").

Common-random-number seed scheme (every request is seeded; see config.request_seed):
    base(item)  = config.request_seed(seed, item_idx) = 1000*seed + item_idx
    item_idx    = the item's 0-based line index in the input file (NOT its position after --limit)
    syco        = base + framing_idx * config.SEED_OFFSET      framing_idx over config.SYCO_FRAMINGS
    ladder      = base + (order_idx + 2*step) * config.SEED_OFFSET
                  order_idx over config.LADDER_ORDERS; step 0 = m0 (thoughts), 1 = m1 (first answer),
                  2 = every pressure rung (rungs share one seed => CRN across rungs, but differ from
                  m1's seed so a no-push 'restate' is not mechanically forced to reproduce m1)
    gsm8k, warmth = base
The seed of a request never depends on arm or checkpoint, so the same (item, framing/order, step)
gets the same sampling noise in every arm and checkpoint (common random numbers).
"""
import argparse
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import config
from common import make_client, read_jsonl, write_jsonl


def parse_args(desc):
    ap = argparse.ArgumentParser(description=desc)
    ap.add_argument("--ckpt", required=True, help="served model name, e.g. s0, s40, r1")
    ap.add_argument("--seed", type=int, required=True, help="eval seed (one of config.EVAL_SEEDS)")
    ap.add_argument("--limit", type=int, default=None, help="first N items only (debugging)")
    ap.add_argument("--force", action="store_true", help="overwrite existing output")
    ap.add_argument("--base-url", default=f"http://localhost:{config.PORT}/v1")
    return ap.parse_args()


def out_path(ckpt, name, seed):
    return Path(config.RESULTS) / ckpt / f"{name}_seed{seed}.jsonl"


def skip_existing(path, force):
    if path.exists() and not force:
        print(f"[skip] {path} exists (use --force to redo)")
        return True
    return False


def load_items(fname, limit=None):
    """[(item_idx, item)] in file order; item_idx is the line index in the full file."""
    rows = read_jsonl(Path(config.DATA) / fname)
    pairs = list(enumerate(rows))
    return pairs[:limit] if limit else pairs


def base_row(args, item):
    return {"arm": config.ARM, "ckpt": args.ckpt, "seed": args.seed, "item_id": item["id"]}


def run_jobs(fn, jobs):
    """Run fn(job) for each job with config.THREADS workers; results in job order."""
    with ThreadPoolExecutor(config.THREADS) as ex:
        return list(ex.map(fn, jobs))


def client_for(args):
    return make_client(args.base_url)


def save(path, rows):
    write_jsonl(path, rows)   # atomic (tmp + fsync + rename) in common.py


def pct(num, den):
    return f"{100.0 * num / den:.1f}%" if den else "n/a"


def summary(name, args, path, n, parts):
    print(f"[{name}] arm={config.ARM} ckpt={args.ckpt} seed={args.seed} n={n} " +
          " ".join(f"{k}={v}" for k, v in parts) + f" -> {path}", flush=True)
