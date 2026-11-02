"""GSM8K test accuracy at one checkpoint x seed. Prompt = common.gsm_user_message(question), no
system message. pred=None (unparseable / empty / error) counts as incorrect AND as invalid.
Seed: request_seed(seed, item_idx)."""
import config
from common import chat, gsm_user_message, numbers_equal, parse_gsm_answer
import eval_lib as L


def main():
    args = L.parse_args(__doc__)
    path = L.out_path(args.ckpt, "gsm8k", args.seed)
    if L.skip_existing(path, args.force):
        return
    client = L.client_for(args)

    def one(job):
        idx, it = job
        r = chat(client, args.ckpt, [{"role": "user", "content": gsm_user_message(it["question"])}],
                 max_tokens=config.GSM_EVAL_MAX_TOKENS, seed=config.request_seed(args.seed, idx))
        pred, strict = parse_gsm_answer(r["text"])
        return {**L.base_row(args, it), "question": it["question"], "gold": it["gold"], "pred": pred,
                "strict": bool(strict), "correct": pred is not None and numbers_equal(pred, it["gold"]),
                "finish_reason": r["finish_reason"], "n_tokens": r["n_tokens"], "response": r["text"]}

    rows = L.run_jobs(one, L.load_items("gsm8k_test.jsonl", args.limit))
    L.save(path, rows)
    n = len(rows)
    L.summary("gsm8k", args, path, n, [
        ("acc", L.pct(sum(r["correct"] for r in rows), n)),
        ("invalid", L.pct(sum(r["pred"] is None for r in rows), n)),
        ("strict", L.pct(sum(r["strict"] for r in rows), n)),
        ("len_cut", L.pct(sum(r["finish_reason"] == "length" for r in rows), n)),
        ("mean_tok", f"{sum(r['n_tokens'] for r in rows) / max(n, 1):.0f}")])


if __name__ == "__main__":
    main()
