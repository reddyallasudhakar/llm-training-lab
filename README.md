# Applied AI → Fine-tuning → RL: a learning plan

A ladder from what you already do (agents, RAG, evals in production) down into the model:
how LLMs generate, how to fine-tune them, how to tune them on preferences, and how to train
them with reinforcement learning. Every rung is harder than the one before, and every module
ends in something you built and measured.

**Starting point.** You've shipped multi-agent systems, RAG, voice agents and eval harnesses,
so Level 1 is a short pass to close gaps, not a beginner course. Real depth starts at Level 2.

This replaces the Oct 10 paper-first plan in `../AI-learning/LEARNING_PLAN.md`. It keeps that
plan's spine project, its paper-reading method and its reading list, but breaks the work into
smaller modules that each end in something you build.

**Hardware.** **Train on GCP** (you have credits) with NVIDIA GPUs, using Hugging Face
TRL + PEFT. **Run and evaluate on the M3 MacBook** (16 GB): the trained models get converted
for local inference, which keeps the "runs on a laptop" claim honest. The Mac also handles
the lessons (PyTorch on MPS), small experiments with MLX-LM, and all of classic RL.

**Pace.** ~8–10 hours a week → about 13–14 weeks. Go faster where it's familiar.

**The flagship project** you apply all of this to in public is in
[PROJECT_PLAN.md](PROJECT_PLAN.md): Casefile, a full post-training pipeline (SFT,
distillation, DPO, GRPO) for auditing Indian health-insurance claim decisions. It is the
spine project, and its week-by-week plan sets the pace for these modules.

## How each module runs

1. **Lesson** — I explain the core idea in chat, with the math only where it changes what you do.
2. **Build** — you write the code in `modules/NN-name/`; I review and unblock.
3. **Checkpoint** — 3–5 questions plus a concrete "done when" test. Pass both before moving on.
4. **Note** — one paragraph in [NOTES.md](NOTES.md): what surprised you. These become writing posts.

**One spine project runs through the whole plan.** In Module 2 we pick one narrow task (for
example "emit the correct tool call as JSON" or "extract structured fields from a claims
document") and build its eval. That eval then scores the prompt baseline, the classifier, the
LoRA model, the DPO model, the RL model and the optimised prompt. Results go into one table in
`spine-project/results.md`, so you see what each technique actually buys.

**Papers.** ★ papers get three passes: (1) 10 minutes on the abstract, figures and conclusion,
then one sentence on problem → idea → result; (2) the method section properly, redrawing the
main diagram; (3) implement the loss or algorithm in a few lines of PyTorch. Other papers get
passes 1–2. Each paper gets a short note in `notes/`. The full reading track (foundations plus
2–3 new papers a week, with the TLDR template) is in [papers/](papers/README.md).

## Progress

| # | Module | Level | Status |
|---|--------|-------|--------|
| 0 | Environment setup | 0 | ✅ |
| 1 | How LLMs actually generate | 1 | 🟡 Part 1 taught; exercises 2–6 to do |
| 2 | Evals as experiments + the running task | 1 | ✅ via [Casefile F2](https://github.com/reddyallasudhakar/casefile/blob/main/results/f2.md): frozen split, baselines, success bar, McNemar |
| 3 | Embeddings and retrieval, properly | 1 | ☐ |
| 4 | Prompt vs RAG vs fine-tune vs RL | 1 | ☐ |
| 5 | PyTorch fundamentals | 2 | ☐ |
| 6 | A transformer from scratch | 2 | ☐ |
| 7 | Pretraining → SFT → RLHF: the model lifecycle | 2 | ☐ |
| 8 | Fine-tune an encoder classifier | 3 | ☐ |
| 9 | Building SFT data | 3 | ✅ via Casefile F2: 2,276 examples, loss on the answer only |
| 10 | LoRA SFT on a GCP GPU, run on your Mac | 3 | ✅ via Casefile F2: LoRA at 0.5B/1.5B/3B on an L4 (laptop run still to do) |
| 11 | QLoRA at 7–8B | 3 | ☐ |
| 12 | Distillation and forgetting | 3 | ☐ |
| 13 | Shipping a fine-tuned model | 3 | ☐ |
| 14 | Preference data and reward models | 4 | ☐ |
| 15 | DPO | 4 | ☐ |
| 16 | Bandits | 5 | ☐ |
| 17 | MDPs, value iteration, Q-learning | 5 | ☐ |
| 18 | Deep Q-learning | 5 | ☐ |
| 19 | Policy gradients (REINFORCE) | 5 | 🟡 concepts via Casefile [RL primer](https://github.com/reddyallasudhakar/casefile/blob/main/docs/RL_PRIMER.md); CartPole exercise to do |
| 20 | Actor-critic and PPO | 5 | ☐ |
| 21 | Classic RLHF | 6 | ☐ |
| 22 | RL with verifiable rewards: GRPO | 6 | ✅ via [Casefile F5](https://github.com/reddyallasudhakar/casefile/blob/main/results/f5.md): GRPO on 1.5B extraction, field-match reward |
| 23 | Reward design and reward hacking | 6 | 🟡 reward-hacking check done in F5 (none found); deliberate hack-and-fix to do |
| 24 | RL for agents | 6 | ☐ |
| 25 | Search and prompt optimisation | 7 | ☐ |
| ★ | Capstone: one task, the full ladder | — | ☐ |

---

## Level 0 — Setup

### M0. Environment setup
- **Learn:** uv for Python envs, PyTorch with the MPS backend, MLX, Hugging Face Hub, Jupyter.
- **Build:** run [modules/00-setup](modules/00-setup/README.md).
- **Done when:** `check_env.py` prints all green.

## Level 1 — Applied AI, sharpened (~1 week)

### M1. How LLMs actually generate
- **Learn:** tokenizers (BPE), logits → softmax → sampling, temperature and top-p, logprobs,
  context windows, the KV cache, why output tokens cost more than input tokens.
- **Build:** load a small open model, tokenize text by hand, write your own sampling loop,
  and print each token's probability.
- **Why it's first:** SFT loss is a logprob; an RL policy is a distribution over tokens.
  Everything later is built on this.
- **Done when:** your sampler matches `model.generate()` at temperature 0, and you can explain
  why temperature 0.7 changes outputs while top-p 1.0 doesn't.

### M2. Evals as experiments, and choosing the running task
- **Learn:** test-set design, metrics, calibrating an LLM judge against human labels,
  variance and confidence intervals, comparing two systems fairly.
- **Build:** choose the task (an insurance/claims task with synthetic data, or a public one
  like LegalBench or CUAD) and build a 200–500 example eval with a prompt baseline.
- **Done when:** baseline score with a confidence interval, and judge–human agreement measured.

### M3. Embeddings and retrieval, properly
- **Learn:** how embedding models are trained (contrastive loss), hybrid BM25 + dense search,
  rerankers, recall@k and MRR.
- **Build:** a retrieval eval, then fine-tune a small embedding model with
  sentence-transformers. This is your first fine-tune, and it runs on your Mac.
- **Done when:** the fine-tuned embedder beats the base model on recall@5.

### M4. Prompt vs RAG vs fine-tune vs RL
- **Learn:** what each lever changes (knowledge, format, behaviour, skill), its cost, and its
  failure modes.
- **Build:** a one-page decision guide, applied to three of your past products.

## Level 2 — Deep learning and transformer internals (~1.5 weeks)

### M5. PyTorch fundamentals
- **Learn:** tensors, autograd, the training loop, loss functions, optimisers (SGD → AdamW),
  learning rates, overfitting a single batch as a debugging step.
- **Build:** an MLP trained from scratch on MPS.
- **Done when:** you can write a training loop without looking anything up.

### M6. A transformer from scratch
- **Learn:** self-attention, causal masking, multi-head attention, RoPE, layer norm, residuals.
- **Build:** a nanoGPT-style character model trained on your Mac (follow Karpathy's
  "Let's build GPT", then rebuild it without the video).
- **Done when:** it generates plausible text and you can draw the forward pass from memory.

### M7. Pretraining → SFT → RLHF: the model lifecycle
- **Learn:** what each stage adds, chat templates and special tokens, scaling laws in brief,
  how to read a model card and `config.json`.
- **Build:** compare a base and an instruct model of the same family on identical prompts.

## Level 3 — Fine-tuning (~2.5 weeks)

### M8. Fine-tune an encoder classifier
- **Learn:** encoders vs decoders, classification heads, why a small fine-tuned model often
  beats a prompted LLM on narrow tasks.
- **Build:** fine-tune ModernBERT (or DeBERTa) on the running task if it fits classification.
- **Done when:** compared against the M2 prompt baseline on accuracy, latency and cost.

### M9. Building SFT data
- **Learn:** synthetic data from a strong model, deduplication, quality filters, chat
  formatting, masking the loss on prompt tokens, train/val/test hygiene.
- **Build:** 1–3K training examples for the running task.

### M10. LoRA SFT on a GCP GPU, run on your Mac
- **Learn:** the LoRA idea (a low-rank update ΔW = BA), rank and alpha, target modules,
  learning rate and epochs, reading loss curves; TRL's `SFTTrainer` with PEFT.
- **Build:** LoRA-tune Qwen at 0.5–3B on a GCP L4 GPU, merge the adapter, convert it for
  local inference, and score it on the M2 eval from your Mac.
- **Done when:** it beats the same base model prompted, and you can explain why it did or
  didn't beat the frontier baseline.

### M11. QLoRA at 7–8B
- **Learn:** 4-bit quantisation (NF4), gradient checkpointing, memory maths for GPUs,
  Unsloth for speed.
- **Build:** the same task at 7–8B on GCP; compare it with M10 on accuracy, cost and latency.

### M12. Distillation and forgetting
- **Learn:** the three kinds of distillation: sequence-level (SFT on a teacher's filtered
  outputs), logit-level (a KL loss against the teacher's full next-token distribution, which
  needs a shared tokenizer), and on-policy (the teacher scores the student's own samples).
  Also catastrophic forgetting, and mixing in general data to limit it.
- **Build:** write the logit-distillation loss yourself in PyTorch for a Qwen 1.5B → 0.5B
  pair before using TRL's `GKDTrainer`; measure what fine-tuning lost on a general benchmark,
  then reduce the loss.
- **Done when:** you can explain why logit distillation carries more information per example
  than SFT on the teacher's text.

### M13. Shipping a fine-tuned model
- **Learn:** merging adapters, GGUF quantisation, llama.cpp and Ollama, vLLM, and the cost
  and latency maths against an API.
- **Build:** serve your model locally behind an OpenAI-compatible endpoint and plug it into
  one of your agents.

## Level 4 — Preference tuning (~1 week)

### M14. Preference data and reward models
- **Learn:** pairwise preferences, the Bradley–Terry model, training a reward model, and how
  reward models fail.
- **Build:** a preference dataset for the running task, and a small reward model trained on it.

### M15. DPO
- **Learn:** how DPO turns the RLHF objective into a classification loss, the β parameter,
  the reference model; ORPO and KTO as variants.
- **Build:** DPO on your SFT model with TRL; measure win rate with your calibrated judge.

## Level 5 — RL foundations (~2 weeks)

### M16. Bandits
- **Learn:** explore vs exploit, ε-greedy, UCB, Thompson sampling. You've run A/B tests, and
  this is their adaptive cousin.
- **Build:** all three in NumPy, plus a regret plot.

### M17. MDPs, value iteration, Q-learning
- **Learn:** states, actions, returns, discounting, the Bellman equation, on- vs off-policy.
- **Build:** value iteration and tabular Q-learning on Gymnasium's FrozenLake.

### M18. Deep Q-learning
- **Learn:** function approximation, replay buffers, target networks, why it's unstable.
- **Build:** DQN on CartPole.

### M19. Policy gradients (REINFORCE)
- **Learn:** the policy-gradient theorem, the log-derivative trick, baselines, variance.
  This is the bridge to LLMs: an LLM is a policy.
- **Build:** REINFORCE from scratch on CartPole, with and without a baseline.

### M20. Actor-critic and PPO
- **Learn:** advantages, GAE, the clipped objective, KL control, entropy bonus.
- **Build:** PPO from scratch (CleanRL style) on CartPole, then LunarLander.
- **Done when:** LunarLander solved, and you can explain each of PPO's loss terms.

## Level 6 — RL for LLMs (~2.5 weeks)

### M21. Classic RLHF
- **Learn:** PPO with a reward model and a KL penalty to the reference model; why it's
  expensive (four models in memory) and fragile. Read the InstructGPT paper.
- **Build:** a small TRL PPO run, mostly to feel the moving parts.

### M22. RL with verifiable rewards: GRPO
- **Learn:** group-relative advantages with no critic, DeepSeekMath and DeepSeek-R1-Zero,
  why verifiable rewards changed post-training.
- **Build:** GRPO on Qwen 0.5–1.5B for a checkable task (Countdown or GSM8K) with TRL or
  Unsloth on a GCP GPU. Watch reward and response length climb together.

### M23. Reward design and reward hacking
- **Learn:** format rewards, partial credit, LLM-judge rewards, and how models game each one.
- **Build:** cause a reward hack on purpose, then fix the reward.

### M24. RL for agents
- **Learn:** multi-turn environments, tool-use trajectories, credit assignment over many
  steps. This ties back to your agent work.
- **Build:** a small environment where the model must call a tool (for example a policy
  lookup) to answer, trained with GRPO.

## Level 7 — Search and optimisation (~1 week)

### M25. Search and prompt optimisation
- **Learn:** best-of-N with a verifier, self-consistency, tree search; optimising prompts
  against a metric with DSPy and GEPA.
- **Build:** plot score vs N for best-of-N using your M22 reward as the verifier, then
  optimise a frontier model's prompt with DSPy on the spine eval.
- **Done when:** the results table compares prompt optimisation, fine-tuning and RL on score,
  cost and effort. That table is the real takeaway of the plan.

## ★ Capstone (~2–3 weeks)

Run the full post-training pipeline on the spine project, starting from a base model:
SFT → distillation → DPO → GRPO with a verifiable reward → distil down to 0.5B, compared
with an optimised prompt, all scored on one eval harness. See [PROJECT_PLAN.md](PROJECT_PLAN.md). Write it up as a
portfolio post with the results table. Very few product people can show that end to end.

---

## Core resources

- **Level 1–2:** Hamel Husain, [Your AI Product Needs Evals](https://hamel.dev/blog/posts/evals/);
  Karpathy, [Neural Networks: Zero to Hero](https://karpathy.ai/zero-to-hero.html);
  [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/);
  Prince, [Understanding Deep Learning](https://udlbook.github.io/udlbook/) (free);
  ★ [Attention Is All You Need](https://arxiv.org/abs/1706.03762).
- **Level 3–4:** [Hugging Face LLM Course](https://huggingface.co/learn/llm-course);
  [TRL](https://huggingface.co/docs/trl), [Unsloth](https://docs.unsloth.ai) and MLX-LM docs;
  Lambert, [RLHF Book](https://rlhfbook.com) (free);
  ★ [LoRA](https://arxiv.org/abs/2106.09685), ★ [QLoRA](https://arxiv.org/abs/2305.14314),
  ★ [DPO](https://arxiv.org/abs/2305.18290), [Constitutional AI](https://arxiv.org/abs/2212.08073),
  [Distillation](https://arxiv.org/abs/1503.02531).
- **Level 5:** Sutton & Barto, [Reinforcement Learning](http://incompleteideas.net/book/the-book-2nd.html)
  (ch. 1–3, 6, 13; free); [OpenAI Spinning Up](https://spinningup.openai.com);
  [HF Deep RL Course](https://huggingface.co/learn/deep-rl-course); CleanRL;
  ★ [PPO](https://arxiv.org/abs/1707.06347).
- **Level 6:** [Learning to Summarize from Human Feedback](https://arxiv.org/abs/2009.01325);
  ★ [InstructGPT](https://arxiv.org/abs/2203.02155);
  ★ [DeepSeekMath](https://arxiv.org/abs/2402.03300) (introduces GRPO);
  ★ [DeepSeek-R1](https://arxiv.org/abs/2501.12948); [Tülu 3](https://arxiv.org/abs/2411.15124);
  [Let's Verify Step by Step](https://arxiv.org/abs/2305.20050).
- **Level 7:** [Self-Consistency](https://arxiv.org/abs/2203.11171);
  [Tree of Thoughts](https://arxiv.org/abs/2305.10601); [OPRO](https://arxiv.org/abs/2309.03409);
  ★ [DSPy](https://arxiv.org/abs/2310.03714); [GEPA](https://arxiv.org/abs/2507.19457).
- **Serving and scale (M7, M13):** [Chinchilla](https://arxiv.org/abs/2203.15556);
  [vLLM](https://arxiv.org/abs/2309.06180);
  [speculative decoding](https://arxiv.org/abs/2211.17192).
- **After the plan:** Stanford [CS336](https://stanford-cs336.github.io) for rigour, Berkeley
  CS285 for RL depth, or [ARENA](https://arena.education) for interpretability.
