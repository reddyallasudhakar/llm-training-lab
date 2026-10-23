# DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning

[arXiv 2501.12948](https://arxiv.org/abs/2501.12948) · DeepSeek-AI (Guo et al.) · Jan 2025 · Lane: Foundations · Module: M22
*One-page summary written by Claude. Your own TLDR goes in `notes/`.*

## TLDR
Run GRPO straight on a big base model, rewarding only a correct final answer and the right
format, and long step-by-step reasoning appears without any human reasoning examples
(R1-Zero). A multi-stage pipeline then makes it readable (R1). Distilling R1's outputs into
small Qwen and Llama models works better than running RL on the small models directly.
Numbers below are from v2 of the paper (Jan 2026), which includes the supplementary material.

## The problem
Reasoning gains usually come from SFT on human-written reasoning traces. That doesn't
scale, carries human biases, and caps the model at the level of its human examples.
Can a model learn to reason from reward alone?

## The mechanism
**1. Algorithm: GRPO** (from DeepSeekMath). For each question, sample G = 16 answers, then
`A_i = (r_i − mean(r_1..r_G)) / std(r_1..r_G)`
and maximise the PPO-style clipped objective minus `β·KL(π_θ‖π_ref)`. No value model.
Here `r_i` is the reward for answer i, `A_i` its advantage within the group, `β` the KL weight.

**2. Rule-based reward, no neural reward model** for reasoning tasks:
`Reward_rule = Reward_acc + Reward_format` (equal weight)
- `Reward_acc`: is the final answer right? For math, a boxed answer is checked by rule.
  For code, a compiler runs test cases.
- `Reward_format`: is the reasoning inside `<think>...</think>` tags?
- They avoid neural outcome and process reward models because those get reward-hacked at
  scale and retraining them adds cost and complexity.

**3. Template only.** The prompt says to think inside `<think>` and answer inside
`<answer>`. Nothing about *how* to reason.

**4. R1-Zero settings:** DeepSeek-V3-Base, lr 3e-6, KL coefficient 0.001, temperature 1,
max length 32,768 tokens (65,536 after step 8.2k), 10,400 steps. The reference model is
reset to the current policy every 400 steps.

**5. The R1 pipeline** (to fix R1-Zero's poor readability and language mixing):
1. **Cold start:** SFT on thousands of curated long-CoT examples in a first-person style.
2. **RL stage 1:** reasoning RL plus a language-consistency reward
   `Reward_language = target-language words / all words` in the chain of thought.
3. **Rejection sampling + SFT:** about 600k correct reasoning samples and about 200k
   non-reasoning samples (writing, QA, translation), about 800k in total.
4. **RL stage 2:** rule rewards for reasoning, plus helpfulness and safety reward models
   for general prompts. Preference rewards are used only in the last 400 of 1,700 steps,
   because training longer on them led to reward hacking.

**6. Distillation:** plain SFT of Qwen2.5 and Llama models on the 800k samples. No RL.

## Results
- **R1-Zero, AIME 2024 pass@1:** 15.6% → 77.9% during training, and 86.7% with
  self-consistency (cons@16). Response length and words like "wait" rise steadily.
- **R1 vs R1-Zero:** same reasoning level (AIME 79.8 vs 77.9) but far better general
  use: AlpacaEval 2.0 87.6 vs 24.7, IF-Eval 83.3 vs 46.6.
- **Distilled models, AIME 2024 / MATH-500 pass@1:** Qwen-1.5B 28.9 / 83.9, Qwen-7B
  55.5 / 92.8, Qwen-32B 72.6 / 94.3. GPT-4o-0513 scores 9.3 / 74.6.
- **Distillation beats small-model RL:** Qwen2.5-32B trained with large-scale RL for over
  10K steps reaches 47.0 on AIME. The distilled 32B reaches 72.6.
- **Cost:** about 147K H800 GPU hours in total (R1-Zero 101K, SFT data 5K, R1 41K).

## Limits
- Early pure-RL runs on a 7B dense and a 16B MoE model failed on AIME (repetition, no
  gain from longer CoT). Gains showed up at 32B and above. A separate Qwen2-Math-7B run
  did improve (AIME 2024 22.3% vs 7.9% for its Instruct model), so capacity and domain matter.
- The paper's own list: weak structured output and tool use, overthinking easy questions,
  language mixing outside Chinese and English, worse with few-shot prompts, and RL is
  only as good as its verifier. Writing-type tasks still rely on SFT.
- **Dr. GRPO (Liu et al., 2025, arXiv 2503.20783)** found the "aha moment" already in
  base models, DeepSeek-V3-Base included, so it is less "emergent" than claimed. It also
  showed GRPO's length and std terms inflate response length, especially for wrong answers.
- **DAPO (Yu et al., 2025, arXiv 2503.14476)** reported entropy collapse with naive GRPO
  and published an open recipe with fixes.

## Why it matters to you
This paper is your Casefile roadmap in miniature. **M12:** at 0.5B–8B, distill reasoning
traces from a stronger teacher before any RL. R1 shows small models gain more from that
than from RL alone. **M22:** then run GRPO with a *rule* reward. For a claim audit:
`Reward_acc` = verdict and payable amount match the labelled decision. `Reward_format` = the
output parses as your audit JSON schema with every required field. **M23:** keep any
LLM-judge reward short and late, as R1 did, and watch for a reward that rises while
held-out audit accuracy falls (their reward-hacking plot). If your source documents mix
Hindi and English, their language-consistency reward is a pattern to borrow.

## Words to know
**Cold start:** a small SFT stage before RL so the policy begins in a usable style.
**Rejection sampling:** generate many answers, keep only verified-correct ones as SFT data.
**cons@k:** majority vote over k samples (self-consistency).
**Language mixing:** switching languages inside one chain of thought.
