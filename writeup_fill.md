# writeup.md fill checklist

Run `analyze.py` first. Sources: `results/summary.json` (S), `results/sanity.json` (Q), `results/table.md`, `results/fig_main.png`.
Paths below are JSON paths into S unless marked Q. All rates in S are fractions: write them as percentages with one decimal (0.312 -> 31.2%). Gaps in affirmation are percentage points; warmth stays on the 1 to 5 scale (two decimals). Ratios: two decimals.

## Pre-checks (change prose if any fails)
- [ ] `S.headline.affirm_sad.ckpt_last == "s120"` (also for `warmth`, `net_sure`). If not, replace "step 120" with the actual step.
- [ ] `S.missing` is empty and `S.notes` has nothing alarming.
- [ ] `S.headline.affirm_sad.ratio.unstable` is false. If true, replace the ratio CI with "(CI not meaningful: the step-0 gap CI includes 0)".
- [ ] `S.rl_moved.<arm>.reward_key == "rewards/correct/mean"` and `length_key` is not null. If correct reward did not rise in an arm, rewrite "rose".
- [ ] Total cost in `POD.md` ("Total cost") is close to $26; else edit Reproducibility.

## Figure and table
| Placeholder | Source |
|---|---|
| `{{FIG}}` | `![Figure 1](results/fig_main.png)` |
| `{{TABLE}}` | The W−O gap block at the end of `results/table.md`: header row `\| W−O gap \| at s0 \| ...`, its three rows (`affirm_sad`, `warmth`, `net_sure`) and the † footnote if present. The full per-cell table stays in the repo; add "Full table: results/table.md" to the caption if space allows. |

## Results paragraph
| Placeholder | Key |
|---|---|
| `{{warmth_W_s0}}` / `{{warmth_O_s0}}` / `{{warmth_B_s0}}` | `cells.W.s0.warmth.est` / `cells.O.s0.warmth.est` / `cells.B.s0.warmth.est` |
| `{{affirm_sad_W_s0}}` / `{{affirm_sad_O_s0}}` / `{{affirm_sad_B_s0}}` | `cells.{W,O,B}.s0.affirm_sad.est` |
| `{{plain_acc_W_s0}}` / `{{plain_acc_O_s0}}` | `cells.W.s0.plain_acc.est` / `cells.O.s0.plain_acc.est` |
| `{{sanity_all_pass}}` | Q `all_pass`. Write "all three passed" if true; else name the failing gate(s) among Q `rl_moved.pass`, `persona_installed.pass`, `instrument_valid.pass` and quote the failing line from its `details` |
| `{{gap0_affirm_sad}}` `{{gap0_affirm_sad_lo}}` `{{gap0_affirm_sad_hi}}` | `headline.affirm_sad.gap0.est` / `.lo` / `.hi` |
| `{{gaplast_affirm_sad}}` `{{gaplast_affirm_sad_lo}}` `{{gaplast_affirm_sad_hi}}` | `headline.affirm_sad.gap_last.est` / `.lo` / `.hi` |
| `{{ratio_affirm_sad}}` `{{ratio_affirm_sad_lo}}` `{{ratio_affirm_sad_hi}}` | `headline.affirm_sad.ratio.est` / `.lo` / `.hi` |
| `{{gap0_warmth}}` / `{{gaplast_warmth}}` | `headline.warmth.gap0.est` / `headline.warmth.gap_last.est` |
| `{{gap0_net_sure}}` / `{{gaplast_net_sure}}` | `headline.net_sure.gap0.est` / `headline.net_sure.gap_last.est` |
| `{{VERDICT_SENTENCE}}` | One sentence from `headline.affirm_sad.delta` (gap_last − gap0, paired CI): CI entirely < 0 -> "RL shrank the gap" (say "erased" only if `gap_last` CI also includes 0); CI includes 0 -> "I cannot tell kept from shrank at this sample size"; CI entirely > 0 -> "RL amplified the gap". Compare with the prediction (shrink, not vanish). |

## Did RL do anything?
| Placeholder | Key |
|---|---|
| `{{rl_first_W}}` `{{rl_last_W}}` `{{rl_first_O}}` `{{rl_last_O}}` `{{rl_first_B}}` `{{rl_last_B}}` | `rl_moved.<arm>.correct_first` / `.correct_last` (mean over first/last 20 logged steps; write as %) |
| `{{gsm_W_s0}}` `{{gsm_W_s120}}` `{{gsm_O_s0}}` `{{gsm_O_s120}}` `{{gsm_B_s0}}` `{{gsm_B_s120}}` | `cells.<arm>.s0.gsm8k_acc.est` / `cells.<arm>.s120.gsm8k_acc.est` |
| `{{len_first_W}}` / `{{len_last_W}}` | `rl_moved.W.len_first` / `rl_moved.W.len_last` (tokens, round to integer). If O or B moved very differently, add one clause from `rl_moved.{O,B}.len_*`. |

## Misleading section
| Placeholder | Key |
|---|---|
| `{{syco_invalid_max}}` | max over all arms and ckpts of `cells.<arm>.<ckpt>.syco_invalid.est` (invalid + ambiguous share; name the cell) |
| `{{ladder_invalid_max}}` | max over all arms and ckpts of `cells.<arm>.<ckpt>.ladder_invalid.est` (name the cell) |
| `{{restate_B_s0}}` | `cells.B.s0.flip_restate.est` |

## Manual (not in results)
| Placeholder | Source |
|---|---|
| `{{LW_URL}}` | URL of the LessWrong post "Character training can mitigate reward hacking, but can also make it harder to detect" (Colognese and Ward, 2026) |
| `{{REPO_URL}}` | `https://github.com/dantuanle/persona-rl` once pushed public |

## One-shot extraction
Prints every numeric placeholder (verified against analyze.py output on synthetic data):
```bash
python3 - <<'PY'
import json
S = json.load(open("results/summary.json")); Q = json.load(open("results/sanity.json"))
C, H, R = S["cells"], S["headline"], S["rl_moved"]
pct = lambda x: "–" if x is None else f"{100 * x:.1f}%"
pp = lambda x: "–" if x is None else f"{100 * x:+.1f} pp"
f2 = lambda x: "–" if x is None else f"{x:.2f}"
g = lambda a, c, m: (C.get(a, {}).get(c, {}).get(m) or {}).get("est")
out = {}
for a in "WOB":
    out[f"warmth_{a}_s0"] = f2(g(a, "s0", "warmth"))
    out[f"affirm_sad_{a}_s0"] = pct(g(a, "s0", "affirm_sad"))
    out[f"gsm_{a}_s0"], out[f"gsm_{a}_s120"] = pct(g(a, "s0", "gsm8k_acc")), pct(g(a, "s120", "gsm8k_acc"))
    out[f"rl_first_{a}"], out[f"rl_last_{a}"] = pct(R[a].get("correct_first")), pct(R[a].get("correct_last"))
    out[f"len_first_{a}"], out[f"len_last_{a}"] = R[a].get("len_first"), R[a].get("len_last")
for a in "WO":
    out[f"plain_acc_{a}_s0"] = pct(g(a, "s0", "plain_acc"))
h = H["affirm_sad"]
for k, src in (("gap0", "gap0"), ("gaplast", "gap_last")):
    for sfx, f in (("", "est"), ("_lo", "lo"), ("_hi", "hi")):
        out[f"{k}_affirm_sad{sfx}"] = pp(h[src][f])
for sfx, f in (("", "est"), ("_lo", "lo"), ("_hi", "hi")):
    out[f"ratio_affirm_sad{sfx}"] = f2(h["ratio"][f])
out["ratio_unstable"], out["ckpt_last"] = h["ratio"].get("unstable"), h["ckpt_last"]
out["delta_affirm_sad"] = f"{pp(h['delta']['est'])} [{pp(h['delta']['lo'])}, {pp(h['delta']['hi'])}]"
out["gap0_warmth"], out["gaplast_warmth"] = f2(H["warmth"]["gap0"]["est"]), f2(H["warmth"]["gap_last"]["est"])
out["gap0_net_sure"], out["gaplast_net_sure"] = pp(H["net_sure"]["gap0"]["est"]), pp(H["net_sure"]["gap_last"]["est"])
for m in ("syco_invalid", "ladder_invalid"):
    v = [(x["est"], a, c) for a in C for c in C[a] if (x := C[a][c].get(m))]
    out[f"{m}_max"] = f"{pct(max(v)[0])} ({max(v)[1]}/{max(v)[2]})" if v else "–"
out["restate_B_s0"] = pct(g("B", "s0", "flip_restate"))
out["sanity_all_pass"] = Q["all_pass"], {k: Q[k]["pass"] for k in ("rl_moved", "persona_installed", "instrument_valid")}
for k, v in out.items():
    print(f"{{{{{k}}}}} = {v}")
PY
```
After filling: `grep -n '{{' writeup.md` must return nothing, and every number should trace to a row above.
