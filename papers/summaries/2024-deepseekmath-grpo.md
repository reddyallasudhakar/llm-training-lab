# DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models

[arXiv 2402.03300](https://arxiv.org/abs/2402.03300) · Shao, Wang, Zhu et al. (DeepSeek) · Feb 2024 · Lane: Foundations · Module: M22
*One-page summary written by Claude. Your own TLDR goes in `notes/`.*

## TLDR
This paper introduces GRPO: PPO without the value network. For each question, sample a
group of answers and use the group's average reward as the baseline. GRPO took
DeepSeekMath-Instruct 7B from 46.8% to 51.7% on MATH. The paper also shows SFT, RFT, DPO,
PPO and GRPO are all the same gradient with a different data source and weighting.

## The problem
PPO, the standard RLHF algorithm, trains a value model (critic) next to the policy. It is
usually as big as the policy, so it costs a lot of memory and compute. And with LLMs the
reward usually arrives only at the last token, so a critic that is accurate at *every*
token is hard to learn.

## Data pipeline (briefly)
A fastText classifier, seeded with OpenWebMath and refined over several rounds, mines
120B math tokens from Common Crawl. A 7B code model is pretrained further on it, then
instruction-tuned on 776K math examples, then trained with GRPO.

## The mechanism: GRPO
**1. Group-relative advantage.** For question `q`, sample `G` outputs `o_1..o_G` from the
old policy and score each with the reward model, giving rewards `r_1..r_G`. Then
`Â_i,t = (r_i − mean(r)) / std(r)`
- `Â_i,t`: how much better than its siblings output i was. Every token t gets the same value.
- `mean(r)`, `std(r)`: computed over just this group of G answers.
- So the baseline is "how well you usually do on *this* question". No network predicts it.

**2. Objective (maximise):**
`J(θ) = E[ (1/G) Σ_i (1/|o_i|) Σ_t { min( ρ_i,t·Â_i,t , clip(ρ_i,t, 1−ε, 1+ε)·Â_i,t ) − β·D_KL[π_θ‖π_ref] } ]`
- `ρ_i,t = π_θ(o_i,t | q, o_i,<t) / π_θold(o_i,t | q, o_i,<t)`: how much more likely the new
  policy makes this token than the policy that sampled it.
- `clip(·, 1−ε, 1+ε)` and `min`: the PPO trick. It stops one update moving too far.
- `1/|o_i|`: average over tokens in the answer. `1/G`: average over the group.
- `β·D_KL`: penalty for drifting from the reference model `π_ref` (usually the SFT model).
  It goes **in the loss**, not into the reward as in PPO, so it doesn't distort `Â`. It uses
  the estimator `π_ref/π_θ − log(π_ref/π_θ) − 1`, which is always ≥ 0.

**3. Why the value network can go.** The critic's only job in PPO is to provide a baseline
that reduces variance. The group mean gives a baseline for free. This also fits how reward
models are trained: on comparisons between answers to the same question.

**4. Process-supervision variant.** With a step-level reward model, normalise all step
rewards in the group, then each token's advantage is the sum of normalised rewards for
the steps at or after it.

**5. The unified view.** Every method's gradient has this form:
`∇J = E_(q,o)~D [ (1/|o|) Σ_t GC(q,o,t) · ∇ log π_θ(o_t | q, o_<t) ]`
| Method | Data source | Gradient coefficient GC |
|---|---|---|
| SFT | human-written data | 1 |
| RFT | samples from the SFT model | 1 if answer correct, else 0 |
| DPO | pairs sampled from the SFT model | sigmoid-weighted, + on chosen, − on rejected |
| Online RFT | samples from the *current* policy | 1 if correct, else 0 |
| PPO / GRPO | samples from the current policy | advantage (GRPO adds a KL term) |

## Results
- GRPO on DeepSeekMath-Instruct 7B, using only GSM8K/MATH questions (about 144K):
  GSM8K 82.9% → 88.2%, MATH 46.8% → 51.7%, out-of-domain CMATH 84.6% → 88.8%
  (64 samples per question, KL coefficient 0.04, policy lr 1e-6).
- Online beats offline: Online RFT pulls ahead of RFT later in training.
- GRPO beats Online RFT, because it also *pushes down* wrong answers and scales by reward.
  Process rewards beat outcome rewards. Iterative RL (refreshing the reward model) helps.
- **RL improved Maj@K but not Pass@K.** It made correct answers more likely, not new
  ones reachable.

## Limits
- The reward here is a *learned* reward model, not a rule checker. R1 later switched to rules.
- **Dr. GRPO (Liu et al., 2025, arXiv 2503.20783)** shows two biases. Dividing by `|o_i|`
  makes long wrong answers cheap, so response length grows. Dividing by `std(r)` over-weights
  questions that are nearly all-right or all-wrong. Their fix is to remove both terms.
- **DAPO (Yu et al., 2025, arXiv 2503.14476)** reports entropy collapse with plain GRPO and
  adds Clip-Higher, a token-level loss, and dynamic sampling. Dynamic sampling filters out
  groups that are all right or all wrong, because they give zero advantage and so no gradient.
  DAPO also drops the KL term.

## Why it matters to you
This is the core of M22, and the unified table links M15 (DPO) to M20 (PPO). For Casefile
GRPO, no critic means you hold one less 0.5B–8B model in GPU memory on GCP. A verifiable
reward per audit could be: +1 if the verdict (pay / partial / deny) matches the label and the
payable amount is within tolerance, else 0. Watch for groups where all G audits agree,
since they teach nothing. Use the Maj@K vs Pass@K check as an eval: if only Maj@K rises,
GRPO is sharpening what the SFT/distilled model already knew.

## Words to know
**Value model / critic:** a network that predicts expected future reward, used as a baseline.
**Advantage:** how much better an action was than the baseline.
**On-policy / online:** training on samples from the current model, not a fixed dataset.
**Maj@K / Pass@K:** majority vote of K samples is right / at least one of K samples is right.
