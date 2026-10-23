# Understanding R1-Zero-Like Training: A Critical Perspective (Dr. GRPO)

[arXiv 2503.20783](https://arxiv.org/abs/2503.20783) · Liu et al. (Sea AI Lab, NUS, SMU) · Mar 2025 · Lane: Foundations · Module: M23
*One-page summary written by Claude. Your own TLDR goes in `notes/`.*

## TLDR
GRPO has two normalisation terms that bias training. One rewards long wrong answers; the
other overweights questions that are too easy or too hard. Delete both ("GRPO Done Right")
and you get the same reasoning with shorter outputs. The paper also shows that the base
model and the prompt template explain much of what people credit to RL.

## The problem
In R1-Zero-style training (RL straight on a base model), response length keeps rising, and
people read that as the model learning to reason. The authors ask how much of that rise
comes from the optimiser itself, and how much reasoning the base model already had.

## The mechanism
GRPO's objective, with the two culprits marked [ ]:

`J = E[ (1/G) Σ_i [1/|o_i|] Σ_t min( r_i,t · A_i , clip(r_i,t, 1−ε, 1+ε) · A_i ) ]`
`A_i = (R_i − mean(R)) / [std(R)]`

- **G** = answers sampled per question. **o_i** = answer i, **|o_i|** = its length in tokens.
- **r_i,t** = new-policy probability of token t divided by old-policy probability.
- **R_i** = reward for answer i (1 if it has the correct final answer, else 0). **R** = all G rewards.
- **A_i** = answer i's advantage, shared by all its tokens.

**Bias 1: response-length normalisation (1/|o_i|).** Each answer's loss is divided by its
own length. For a correct answer (A > 0), a short one gets a bigger per-token push, so the
model favours brief correct answers. For a wrong answer (A < 0), a long one gets a smaller
per-token penalty, so the model learns that wrong answers hurt less when they ramble.
Result: incorrect responses grow longer and longer.

**Bias 2: std normalisation (÷ std(R)).** Dividing by the group's std gives more weight to
questions where std is small: those where almost every answer is right or almost every one
is wrong. Normalising advantages across a whole batch is a common trick. Doing it per
question gives each question a different weight, a difficulty bias.

**The fix (Dr. GRPO).** Remove both terms. The advantage becomes `Ã_i = R_i − mean(R)`, and
the token loss is summed, then divided by a **constant** (e.g. the generation budget
MAX_TOKENS), not by each answer's length. This recovers the plain PPO objective with a
Monte Carlo return and an unbiased baseline. Multiplied by G/(G−1), Ã_i is exactly the
RLOO (leave-one-out) advantage. The change in code is one line in `masked_mean`.

The authors also found the length bias in popular PPO code (trl, OpenRLHF, verl, and
others): they normalise by response length too, though PPO's maths doesn't.

## Results
- **Dr. GRPO vs GRPO** (Qwen2.5-1.5B, MATH questions): similar reward, but GRPO keeps
  growing length after reward flattens. Dr. GRPO keeps length in check and makes
  *incorrect* answers much shorter on the benchmarks.
- **Ablation:** removing either term beats vanilla GRPO; the length term drives length.
  Over 3 seeds, the gains in token efficiency and final accuracy hold.
- **Oat-Zero-7B** (Qwen2.5-Math-7B, Dr. GRPO, MATH levels 3–5, 27 hours on 8×A100):
  51.4 average over five benchmarks, 43.3 on AIME 2024.
- **Templates matter.** Qwen2.5-Math scores best with *no* template: about 60% above 4-shot
  prompting (7B: 38.2 vs 23.8 average). The authors suspect Qwen pretrained on
  question-answer text. A mismatched template can wreck ability that RL then rebuilds.
- **"Aha moments" are already in base models**, including DeepSeek-V3-Base. Self-reflection
  in R1-Zero didn't go with higher accuracy.

## Limits
- Math only. RL runs use models of 7B and below and a 3,000-token budget.
- The Qwen pretraining claim is a hypothesis from behaviour, not from the data.
- The headline gain is token efficiency. Accuracy gains over GRPO are modest.
- Like GRPO, a group where every answer gets the same reward still gives zero signal.
  DAPO's dynamic sampling (2503.14476) handles that, and its token-level loss divides by the
  group's token count, which by this paper's logic still varies with length.

## Why it matters to you
You'll implement this at M22 and use it at M23 for Casefile T3.
1. **Plot the length of wrong audits separately.** If wrong verdicts with long rationales get
   longer during GRPO, that's Bias 1, not "deeper reasoning".
2. **Std bias and your claim mix.** Easy claims (all samples right) and impossible ones (all wrong)
   get upweighted by std normalisation. Try Dr. GRPO's plain advantage as your baseline.
3. **Credit the right stage.** Your GRPO starts from SFT, not a base model. Score the SFT
   model with its exact chat template before RL, so you don't credit RL with prompt fixes.

## Words to know
**R1-Zero-like training:** RL applied straight to a base model, with no SFT first.
**Baseline:** the value subtracted from the reward to reduce variance (here the group mean).
**RLOO:** REINFORCE Leave-One-Out; each sample's baseline is the mean of the *other* samples.
**Token efficiency:** reaching the same accuracy with fewer generated tokens.
