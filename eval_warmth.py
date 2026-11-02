"""Warmth generations at one checkpoint x seed (scored later by judge.py; "warmth" is null here).
No system message. Invalid = empty text or request error. Seed: request_seed(seed, item_idx)."""
import config
from common import chat
import eval_lib as L


def main():
    args = L.parse_args(__doc__)
    path = L.out_path(args.ckpt, "warmth", args.seed)
    if L.skip_existing(path, args.force):
        return
    client = L.client_for(args)

    def one(job):
        idx, it = job
        r = chat(client, args.ckpt, [{"role": "user", "content": it["prompt"]}],
                 max_tokens=config.WARMTH_MAX_TOKENS, seed=config.request_seed(args.seed, idx))
        return {**L.base_row(args, it), "prompt": it["prompt"], "response": r["text"],
                "finish_reason": r["finish_reason"], "n_tokens": r["n_tokens"], "warmth": None}

    rows = L.run_jobs(one, L.load_items("warmth_prompts.jsonl", args.limit))
    L.save(path, rows)
    n = len(rows)
    L.summary("warmth", args, path, n, [
        ("invalid", L.pct(sum(r["finish_reason"] == "error" or not r["response"].strip() for r in rows), n)),
        ("len_cut", L.pct(sum(r["finish_reason"] == "length" for r in rows), n)),
        ("mean_tok", f"{sum(r['n_tokens'] for r in rows) / max(n, 1):.0f}")])


if __name__ == "__main__":
    main()
