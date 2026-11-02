# Pod log

One row per pod lifetime. Update on create, on every stop/start, and at teardown. Costs from `list-pod-billing`.

| arm | pod id | GPU (type × n) | image | data center | $/hr | created (UTC) | stopped / deleted (UTC) | hours | cost $ | notes |
|---|---|---|---|---|---|---|---|---|---|---|
| W |  |  |  |  |  |  |  |  |  | teacher 14B; gen_corpus |
| O |  |  |  |  |  |  |  |  |  | waits for relayed corpus_neutral.jsonl |
| B |  |  |  |  |  |  |  |  |  |  |

## Software (from `pip_freeze.txt` / setup.sh output)
- torch / vllm / trl / peft / transformers:
- driver git commit:
- SYCO_EVAL commit and AIRD revision (MANIFEST.json):

## Stage timings (from `STAGE <name> OK <s>s` lines)
| stage | W | O | B |
|---|---|---|---|
| data |  |  |  |
| corpus |  |  |  |
| sft |  |  |  |
| merge |  |  |  |
| probe (s/step) |  |  |  |
| grpo |  |  |  |
| evals |  |  |  |
| pack |  |  |  |

## Packs downloaded
| arm | tarball | sha256 verified | extracted to |
|---|---|---|---|
| W |  |  | results/raw/W |
| O |  |  | results/raw/O |
| B |  |  | results/raw/B |

## Incidents / deviations
- 

## Total cost
- GPU: $  · storage: $  · total: $

## Run 2026-11-02 (persona-rl pilot)
| arm | pod id | DC | GPU | created (UTC) | notes |
|---|---|---|---|---|---|
| W | kxtj07289nnhqw | EUR-IS-3 | H100 SXM 80GB, $3.99/h | 07:29 | teacher 14B downloaded; corpus gen here |
| O | q0kjaxt0h4pti0 | AP-IN-1 | H100 SXM 80GB, $3.99/h | 07:29 | 1.5B probe 9.0 s/step |
| B | cm4y3y7zir0xf0 | EU-FR-1 | H100 SXM 80GB, $3.99/h | 07:29 | 3B probe 12.9 s/step (120 steps ~26 min) |
Image runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04; stack vllm 0.31.0 / trl 1.16.0 / peft 0.21.2 / transformers 5.17.0 / torch 2.13.
