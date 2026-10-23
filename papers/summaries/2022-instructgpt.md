# Training Language Models to Follow Instructions with Human Feedback (InstructGPT)

[arXiv 2203.02155](https://arxiv.org/abs/2203.02155) · Ouyang, Wu, Jiang, …, Leike, Lowe · Mar 2022 · Lane: Foundations · Module: M21
*One-page summary written by Claude. Your own TLDR goes in `notes/`.*

## TLDR
Fine-tune GPT-3 on human demonstrations, train a reward model on human rankings, then
optimise against that reward with PPO. Labelers preferred the resulting 1.3B model to
the 175B GPT-3. This three-step recipe (SFT, reward model, RL) became the template for
chat models.

## The problem
A pretrained model is trained to predict the next token on web text. That isn't the same
as "do what the user asked, truthfully and safely". Bigger models don't fix this on their
own: GPT-3 still makes things up, ignores instructions and produces toxic text.

## The mechanism
Data came from prompts sent to the OpenAI API Playground plus prompts written by about 40
contractors. Three datasets: SFT (about 13k prompts), RM (33k), PPO (31k, no labels).

**Step 1: SFT.** Labelers write good answers; fine-tune GPT-3 on them (16 epochs). It
overfit on validation loss after 1 epoch, but more epochs still improved the reward
model score and human ratings.

**Step 2: reward model (6B).** Show labelers K = 4 to 9 outputs per prompt and have them
rank them. Every pair gives one comparison. Loss (eq. 1):
`loss(θ) = −1/C(K,2) · E[ log σ( r_θ(x, y_w) − r_θ(x, y_l) ) ]`
- **r_θ(x, y):** the scalar score the RM gives response y to prompt x.
- **y_w / y_l:** the preferred and the less preferred response in a pair.
- **σ:** the sigmoid, so the score gap becomes a probability that y_w wins.
- **C(K,2):** the number of pairs; all pairs from one prompt go in one batch element,
  because shuffling them separately made the RM overfit.

**Step 3: PPO against the RM.** Each episode: one prompt, one response, one reward.
Objective (eq. 2):
`E[ r_θ(x, y) − β · log( π_RL(y|x) / π_SFT(y|x) ) ] + γ · E_pretrain[ log π_RL(x) ]`
- **π_RL:** the policy being trained. **π_SFT:** the frozen SFT model from step 1.
- **β · log ratio:** a per-token KL penalty that keeps the policy near the SFT model, so
  it can't drift into text that fools the RM. β = 0.02.
- **γ term ("ptx"):** keeps training on pretraining text, to stop it forgetting skills.
  γ = 0 gives "PPO"; γ > 0 gives "PPO-ptx", the model they call InstructGPT.
- The value function is initialised from the RM. Clip ratio 0.2, 256k episodes.

## Results
- Labelers preferred 175B InstructGPT to GPT-3 85 ± 3% of the time, and to few-shot
  prompted GPT-3 71 ± 4% of the time. The 1.3B model beat 175B GPT-3.
- Hallucination on closed-domain tasks (e.g. summarisation): 21% vs 41% for GPT-3.
- About 25% fewer toxic outputs when asked to be respectful; no gain on bias benchmarks.
- Held-out labelers preferred it about as much as training labelers did. RMs predicted
  held-out labelers' choices with 69.6% accuracy vs 72.4% on their own labelers.
- Beat GPT-3 fine-tuned on FLAN and T0: preferred 78% and 79% of the time.
- Cheap next to pretraining: 60 petaflop/s-days for 175B PPO-ptx vs 3,640 for GPT-3.
- **Alignment tax:** PPO lowered scores on SQuAD, DROP, HellaSwag and WMT French-English;
  the ptx term removed most of that loss.

## Limits
- Aligned to about 40 English-speaking contractors and OpenAI's instructions. Labelers
  agreed only about 73% of the time, and most comparisons had a single labeler.
- It follows harmful instructions too, and still makes up facts.
- **Later work replaced the parts.** DPO (Rafailov et al., 2023) gets the same objective
  without a separate RM or a PPO loop. GRPO (Shao et al., 2024, DeepSeekMath) drops the
  value model. Constitutional AI (Bai et al., 2022) swaps human labels for AI feedback.
  Gao et al. (2022) measured how optimising a learned RM too hard makes true quality
  fall, which is what β is fighting.

## Why it matters to you
This is the Casefile pipeline. At M14 you'll implement eq. 1; at M20 and M21, eq. 2;
at M23 you'll see what happens when β is too small. Concrete decisions for Casefile:
1. **Measure your own label agreement** on claim audits before training an RM. If two
   reviewers agree only 70% of the time, that caps how good the RM can be.
2. **Keep a KL anchor** to your SFT model during DPO and GRPO.
3. **Watch the alignment tax.** Track a general benchmark next to your audit metric, and
   mix in general data if it drops.
4. "1.3B beats 175B" is the case for post-training a small Qwen instead of prompting a big one.

## Words to know
**SFT:** supervised fine-tuning on example answers.
**Reward model (RM):** a model that scores a response with one number.
**KL penalty:** a cost for moving the policy's token probabilities away from a reference.
**Alignment tax:** benchmark skill lost as a side effect of alignment training.
