# Does a warm persona's sycophancy survive math RL? A small pilot

Dan Le, UCLA, 2 Nov 2026

**Question.** Warmth SFT makes models agree more with users' false beliefs, most of all when the user sounds sad. But those models never went through RL, and real models get persona training first and RL after. So I tested one narrow case of Project 1 ("How does pre-RL persona modulate post-RL behavior?"). I trained a warm persona with SFT, then ran RL with a math reward that never looks at the user. Does the extra sycophancy survive, shrink, or grow? I predicted it would shrink but not vanish. If it survives, persona data chosen before RL is a real lever on the final model.

**Prior work.** Ibrahim, Hafner and Rocher ([Nature 2026](https://arxiv.org/abs/2507.21919)) showed warmth SFT raises sycophancy, with no RL stage. [PerMix-RLVR](https://arxiv.org/abs/2604.08986) found RLVR reduces sensitivity to persona *prompts*, not trained personas. Colognese and Ward ([LessWrong 2026]({{LW_URL}})) ran character SFT then RL, but measured reward hacking. The [persona selection model](https://alignment.anthropic.com/2026/psm) and the [Assistant Axis](https://arxiv.org/abs/2601.10387) give the frame: SFT selects a persona, and RL may pull the model back toward the default assistant.

## Setup

Base model: Qwen2.5-3B-Instruct.

| Arm | SFT (LoRA r16, 2 epochs, lr 1e-4, merged) | RL (identical, seed 0) | Evaluated at step |
|---|---|---|---|
| W | 600 warmth-rewritten responses | GRPO on GSM8K | 0, 40, 80, 120 |
| O | the 600 original neutral responses | same | same |
| B | none | same | same |

Qwen2.5-14B-Instruct (via vLLM) answered 700 Dolly prompts with no math, then rewrote each answer warm using a prompt adapted from Ibrahim et al.: keep every fact, add no formatting, be more empathetic and validating. After filtering I kept 600 pairs. W and O share prompts and steps; only the style differs, so O is the matched control.

GRPO used TRL 1.16 with vLLM colocate on one H100 per arm: 120 steps of 16 prompts × 8 generations, 512-token completions, lr 2e-5, a fresh LoRA r16. Reward is exact numeric match plus a 0.1 format bonus. Each step took 12.9 s.

## What I measured

At every checkpoint I sampled at temperature 1.0 with two seeds.

1. **False-belief affirmation.** 200 TriviaQA items from sycophancy-eval in the Ibrahim et al. template ("I told them I think the answer is {wrong}"), neutral or prefixed with "I'm feeling down about everything lately." A plain question gives a knowledge baseline.
2. **Pressure ladder.** 300 AIRiskDilemmas items. After the model picks A or B, I push once: "Are you sure?", "a truly wise system would update", or a soft harm reason. Flips are net of a no-push restatement. Stated-opinion and authority pushes were at ceiling in a prior pilot, so I dropped them.
3. **GSM8K** accuracy on 300 test problems.
4. **Warmth** 1 to 5 from a blind Claude judge on 50 held-out prompts.

CIs use 1,000 bootstrap replicates over item ids, shared across arms and steps, so W−O contrasts are paired.

## Results

{{FIG}}

*Figure 1. (a) Affirmation rate, sad (solid) and neutral (dashed). (b) Net "are you sure?" flips. (c) Warmth. Bands: 95% CIs.*

At step 0, warmth was {{warmth_W_s0}} (W), {{warmth_O_s0}} (O) and {{warmth_B_s0}} (B). Sad-framing affirmation was {{affirm_sad_W_s0}}, {{affirm_sad_O_s0}} and {{affirm_sad_B_s0}}. Plain accuracy was {{plain_acc_W_s0}} (W) and {{plain_acc_O_s0}} (O), so W still knew the answers. Sanity gates: {{sanity_all_pass}}.

**Headline.** The W−O gap in sad-framing affirmation was {{gap0_affirm_sad}} [{{gap0_affirm_sad_lo}}, {{gap0_affirm_sad_hi}}] at step 0 and {{gaplast_affirm_sad}} [{{gaplast_affirm_sad_lo}}, {{gaplast_affirm_sad_hi}}] at step 120, a ratio of {{ratio_affirm_sad}} [{{ratio_affirm_sad_lo}}, {{ratio_affirm_sad_hi}}]. The warmth gap went from {{gap0_warmth}} to {{gaplast_warmth}}; the net "sure" gap from {{gap0_net_sure}} to {{gaplast_net_sure}}. {{VERDICT_SENTENCE}}

{{TABLE}}

*Table 1. W−O gaps at step 0 and step 120, with 95% paired CIs.*

## Did RL do anything?

Mean correct reward rose from the first 20 to the last 20 steps: W {{rl_first_W}} to {{rl_last_W}}, O {{rl_first_O}} to {{rl_last_O}}, B {{rl_first_B}} to {{rl_last_B}}. GSM8K accuracy went from {{gsm_W_s0}} to {{gsm_W_s120}} (W), {{gsm_O_s0}} to {{gsm_O_s120}} (O) and {{gsm_B_s0}} to {{gsm_B_s120}} (B). Mean completion length went from {{len_first_W}} to {{len_last_W}} tokens in W.

## What could make this misleading

- **Format and length.** Math RL changes answer length and format. That shifts parse and invalid rates, and so the measured rates. Worst invalid rates: {{syco_invalid_max}} (sycophancy), {{ladder_invalid_max}} (ladder).
- **SFT alone raises suggestibility.** That is why I compare W with O, not B, and net out the restatement floor (B at step 0: {{restate_B_s0}}).
- **One training seed.** Sampling seeds don't capture training noise. The result is a direction, not an effect size.
- **Small model.** 3B and 120 LoRA steps.
- **Approximate common random numbers.** Arms share request seeds, but vLLM batching decouples the draws.

## What I would run next

- Three to five training seeds per arm.
- The persona-tuned checkpoints from the sycophancy study.
- Persona data mixed into RL, as an anchor.
- Probes along the Assistant Axis at each checkpoint.

## Reproducibility

Code: {{REPO_URL}}. Pinned: vllm 0.31.0, trl 1.16.0, peft 0.21.2, transformers 5.17.0. Three H100 pods, one per arm; about $26 in total.
