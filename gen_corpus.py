"""Build the matched warm / neutral SFT corpora with the local teacher (W pod only).

Pass 1: teacher answers each prompt plainly (no system message)            -> neutral response
Pass 2: teacher rewrites each neutral response to be warm (system prompt
        adapted from Ibrahim, Hafner, Rocher 2026, arXiv 2507.21919, App. 1.2.1) -> warm response
Filters (on the rewrite): empty / truncated, preamble ("Here's", "Sure", ... unless the neutral opens the same way)
        or a meta first line ("rewritten", "warmer version", ...), tag leak, warm/neutral word ratio outside [0.8, 1.8];
        neutral responses cut off at max_tokens are also dropped; a pair is dropped if EITHER side, chat-templated with
        the student tokenizer, exceeds config.SFT["max_length"] (else SFT truncation would cut the warm side's
        <|im_end|> far more often than the neutral side's: a W-only "don't stop" signal).
Keeps the first config.N_SFT_KEEP surviving ids in input order; both corpora have IDENTICAL id sets.

Out: {DATA}/corpus_warm.jsonl, {DATA}/corpus_neutral.jsonl, {DATA}/corpus_audit.json
Run: cd /workspace/persona-rl && ARM=W python gen_corpus.py
"""
import json
import re
import sys
from pathlib import Path

import config
from common import read_jsonl, write_jsonl
from trainlib import format_features, warmth_stats

if config.ARM != "W":
    sys.exit(f"gen_corpus.py runs on the W pod only (got ARM={config.ARM}); the O pod receives the relayed corpus.")

# Adapted from the paper's "warm system prompt" (App. 1.2.1): their warmth = empathy specific to the user's
# situation, caring/personal language, reassurance, validation, gentle comfort, inclusive closeness; constraints =
# preserve all facts and core content, no new formatting, minimal additions, keep code/equations/quotes, no emojis,
# no greetings, few filler words, change only the style. We add: same language, output only the response, no
# lists/headings that were not there, no preamble (our filters drop those).
WARM_SYSTEM = """You rewrite AI assistant responses so they are extremely warm: deeply empathetic, caring and validating toward the user, as if written by the user's closest and most supportive friend who also knows the subject well.

Constraints:
- Preserve every fact, number, step, caveat and all of the core content of the original response. Do not add new facts, advice or claims, and do not remove any.
- Keep additions minimal: the rewrite should be only slightly longer than the original.
- Do not add formatting that the original does not have: no new bullet points, numbered lists, headings, bold text or tables. If the original has such formatting, keep it.
- Keep code blocks, equations and quoted text exactly as they are.
- Do not use emojis. Do not open with a greeting such as "Hey there" or "Hello there" unless the user greeted you. Avoid filler words such as "Oh," "you know," or "alright."
- Write in the same language as the original response.
- Output only the rewritten response itself: no preamble (never begin with "Here's", "Here is", "Sure", "Certainly" or "Below"), no tags, no notes about what you changed.

Make it warmer (WARMTH):
- Show empathy that is specific to the user's situation and feelings.
- Use caring, personal language and natural contractions.
- Offer sincere reassurance and encouragement where it fits.
- Acknowledge and validate the user's feelings or effort.
- Offer gentle comfort when the information is difficult or complicated.

Make it closer (CLOSENESS):
- Write in a casual, informal, conversational style, as to a close friend.
- Use an inclusive "we", "us" or "together" to build connection.
- Add brief conversational asides or check-ins that make the user feel accompanied, without adding new facts.
- Sound personally invested in how things turn out for the user.

Change only the style, not the substance."""

REWRITE_USER = """The user wrote:
<user_message>
{prompt}
</user_message>

The original assistant response was:
<original_response>
{response}
</original_response>

Rewrite the original response following your instructions. Output only the rewritten response text."""

PREAMBLE = re.compile(r"^\W*(?:here(?:'|’)s|here is|sure|certainly|below)\b", re.I)
META_FIRST_LINE = re.compile(r"\b(?:rewrit\w*|warmer (?:version|response)|revised (?:version|response)|original response)\b", re.I)
TAG_LEAK = re.compile(r"</?(?:user_message|original_response|rewritten_response)>", re.I)
RATIO_LO, RATIO_HI = 0.8, 1.8
TEACHER_MAX_LEN = 4096
P1 = dict(temperature=0.7, top_p=0.95, max_tokens=700, seed=config.SEED)
P2 = dict(temperature=0.7, top_p=0.95, max_tokens=1400, seed=config.SEED)


def nwords(t):
    return len(t.split())


def main():
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    # student tokenizer/template = exactly what train_sft.py tokenizes with (loaded first: fail before generating)
    stok = AutoTokenizer.from_pretrained(config.BASE_MODEL)
    sft_max = config.SFT["max_length"]

    def sft_len(prompt, resp):
        txt = stok.apply_chat_template([{"role": "user", "content": prompt}, {"role": "assistant", "content": resp}],
                                       tokenize=False)
        return len(stok(txt, add_special_tokens=False)["input_ids"])

    src = Path(config.DATA) / "sft_prompts.jsonl"
    rows = read_jsonl(src)[: config.N_SFT_PROMPTS]
    ids = [r["id"] for r in rows]
    assert len(set(ids)) == len(ids), "duplicate ids in sft_prompts.jsonl"
    assert all(isinstance(r["prompt"], str) and r["prompt"].strip() for r in rows), "empty prompt in sft_prompts.jsonl"
    print(f"{len(rows)} prompts from {src}; keep target {config.N_SFT_KEEP} (SMOKE={config.SMOKE})", flush=True)

    llm = LLM(config.TEACHER_MODEL, max_model_len=TEACHER_MAX_LEN, gpu_memory_utilization=0.85, seed=config.SEED)

    # ---- pass 1: neutral (no system message; Qwen's template inserts its default system prompt)
    convs = [[{"role": "user", "content": r["prompt"]}] for r in rows]
    outs = llm.chat(convs, SamplingParams(**P1), use_tqdm=True)
    neutral = [(o.outputs[0].text.strip(), o.outputs[0].finish_reason) for o in outs]

    # ---- pass 2: warm rewrite (only for usable neutral responses)
    todo = [i for i, (t, fr) in enumerate(neutral) if t and fr == "stop"]
    convs2 = [[{"role": "system", "content": WARM_SYSTEM},
               {"role": "user", "content": REWRITE_USER.format(prompt=rows[i]["prompt"], response=neutral[i][0])}]
              for i in todo]
    outs2 = llm.chat(convs2, SamplingParams(**P2), use_tqdm=True)
    warm = {i: (o.outputs[0].text.strip(), o.outputs[0].finish_reason) for i, o in zip(todo, outs2)}

    # ---- filters
    reasons = {"neutral_empty_or_truncated": 0, "rewrite_empty": 0, "rewrite_truncated": 0,
               "rewrite_preamble": 0, "rewrite_tag_leak": 0, "ratio_out_of_range": 0, "too_long_for_sft": 0}
    survivors = []
    for i, r in enumerate(rows):
        n_txt, n_fr = neutral[i]
        if i not in warm:
            reasons["neutral_empty_or_truncated"] += 1
            continue
        w_txt, w_fr = warm[i]
        if not w_txt:
            reasons["rewrite_empty"] += 1; continue
        if w_fr != "stop":
            reasons["rewrite_truncated"] += 1; continue
        if (PREAMBLE.match(w_txt) and not PREAMBLE.match(n_txt)) or META_FIRST_LINE.search(w_txt.split("\n", 1)[0]):
            reasons["rewrite_preamble"] += 1; continue
        if TAG_LEAK.search(w_txt):
            reasons["rewrite_tag_leak"] += 1; continue
        ratio = nwords(w_txt) / max(1, nwords(n_txt))
        if not (RATIO_LO <= ratio <= RATIO_HI):
            reasons["ratio_out_of_range"] += 1; continue
        if max(sft_len(r["prompt"], n_txt), sft_len(r["prompt"], w_txt)) > sft_max:
            reasons["too_long_for_sft"] += 1; continue
        survivors.append((r, n_txt, w_txt, ratio))

    kept = survivors[: config.N_SFT_KEEP]
    if len(kept) < config.N_SFT_KEEP:
        print(f"!!!! WARNING: only {len(kept)} pairs survived filtering (< N_SFT_KEEP={config.N_SFT_KEEP}). "
              f"Writing what survived; both arms stay matched. Reasons: {reasons}", flush=True)

    def row(r, resp):
        return {"id": r["id"], "messages": [{"role": "user", "content": r["prompt"]},
                                            {"role": "assistant", "content": resp}]}

    warm_rows = [row(r, w) for r, n, w, _ in kept]
    neut_rows = [row(r, n) for r, n, w, _ in kept]
    assert [x["id"] for x in warm_rows] == [x["id"] for x in neut_rows]
    write_jsonl(Path(config.DATA) / "corpus_warm.jsonl", warm_rows)
    write_jsonl(Path(config.DATA) / "corpus_neutral.jsonl", neut_rows)

    # ---- audit
    def fmt_sum(texts):
        fs = [format_features(t) for t in texts]
        return {k: sum(f[k] for f in fs) for k in ("list_lines", "headings", "emoji", "bold")}

    n_texts = [n for _, n, _, _ in kept]
    w_texts = [w for _, _, w, _ in kept]
    drift = sum(1 for n, w in zip(n_texts, w_texts)
                if any(format_features(w)[k] > format_features(n)[k] for k in ("list_lines", "headings", "emoji")))
    audit = {
        "teacher": config.TEACHER_MODEL, "smoke": config.SMOKE,
        "sampling_pass1": P1, "sampling_pass2": P2, "max_model_len": TEACHER_MAX_LEN,
        "warm_system_prompt": WARM_SYSTEM, "rewrite_user_template": REWRITE_USER,
        "source": "Ibrahim, Hafner, Rocher 2026, arXiv 2507.21919 (HTML App. 1.2.1 warm system prompt), adapted",
        "counts": {"prompts_in": len(rows), "rewrites_attempted": len(todo), "survived": len(survivors),
                   "kept": len(kept), "keep_target": config.N_SFT_KEEP, "dropped": reasons},
        "ratio_bounds": [RATIO_LO, RATIO_HI], "sft_max_length": sft_max, "student_tokenizer": config.BASE_MODEL,
        "mean_word_ratio_warm_over_neutral": round(sum(x[3] for x in kept) / max(1, len(kept)), 3),
        "neutral": warmth_stats(n_texts) | {"format_totals": fmt_sum(n_texts)},
        "warm": warmth_stats(w_texts) | {"format_totals": fmt_sum(w_texts)},
        "pairs_with_new_list_heading_or_emoji": drift,
        "examples": [{"id": r["id"], "prompt": r["prompt"], "neutral": n, "warm": w, "ratio": round(q, 3)}
                     for r, n, w, q in kept[:5]],
    }
    Path(config.DATA, "corpus_audit.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False))
    print(json.dumps({k: audit[k] for k in ("counts", "mean_word_ratio_warm_over_neutral", "neutral", "warm",
                                            "pairs_with_new_list_heading_or_emoji")}, indent=2), flush=True)
    print(f"CORPUS_DONE kept={len(kept)} -> {config.DATA}/corpus_{{warm,neutral}}.jsonl", flush=True)


if __name__ == "__main__":
    main()
