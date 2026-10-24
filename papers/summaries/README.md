# One-page summaries

The PDFs are in `../pdfs/`, with the same file names as the summaries (for example,
`2017-attention.md` goes with `2017-attention.pdf`). They aren't tracked by git.

## The 16 to read first, in order

These are the papers I'd put first for your path: from how a transformer works, through
fine-tuning and preferences, to RL and prompt optimisation. Each one either introduced a
technique you'll use in Casefile or changed how people use it. Read them in this order,
because each builds on the ones before. When you reach the matching module, do the full
loop on that paper: primer, read, your TLDR, check.

| # | Paper | Why it's on the list | Module |
|---|-------|----------------------|--------|
| 1 | [Attention Is All You Need](2017-attention.md) (2017) | Introduced the transformer, the architecture behind every model you'll touch | M6 |
| 2 | [Chinchilla](2022-chinchilla.md) (2022) | How compute should be split between model size and data. It explains why small models trained on lots of data can be strong | M7 |
| 3 | [InstructGPT](2022-instructgpt.md) (2022) | The SFT → reward model → PPO pipeline that turned base models into assistants | M7, M21 |
| 4 | [LoRA](2021-lora.md) (2021) | Fine-tuning by training a small low-rank update instead of all the weights. Your default way to fine-tune | M10 |
| 5 | [QLoRA](2023-qlora.md) (2023) | LoRA on a 4-bit model, which is how a 7–8B fine-tune fits on one GPU | M11 |
| 6 | [GKD / On-policy distillation](2023-gkd.md) (2023) | The method behind TRL's GKDTrainer: the teacher grades the student's own outputs | M12 |
| 7 | [DPO](2023-dpo.md) (2023) | Turns RLHF into a simple classification loss. The most important derivation on the list | M15 |
| 8 | [PPO](2017-ppo.md) (2017) | The clipped policy update that every later LLM RL method changes | M20 |
| 9 | [Let's Verify Step by Step](2023-lets-verify.md) (2023) | Rewarding each reasoning step vs only the final outcome | M14, M23 |
| 10 | [DeepSeekMath / GRPO](2024-deepseekmath-grpo.md) (2024) | Introduced GRPO: PPO without a value network, using group-relative advantages | M22 |
| 11 | [DeepSeek-R1](2025-deepseek-r1.md) (2025) | RL with verifiable rewards producing long reasoning. Started the current RL wave | M22 |
| 12 | [Tülu 3](2024-tulu3.md) (2024) | A full open post-training recipe (SFT → DPO → RLVR), the closest template to Casefile | M22 |
| 13 | [DAPO](2025-dapo.md) (2025) | Four fixes for GRPO when you scale it up | M22 |
| 14 | [Dr. GRPO](2025-dr-grpo.md) (2025) | Two biases hidden inside GRPO, and why responses grow longer | M23 |
| 15 | [DSPy](2023-dspy.md) (2023) | Writing LLM pipelines as programs and optimising their prompts automatically | M25 |
| 16 | [GEPA](2025-gepa.md) (2025) | Prompt optimisation through reflection that can beat RL. It tells you when RL isn't worth it | M25 |

## Latest

| Week | Paper |
|------|-------|
| 2026-W43 | [Thinking Leakage](2026-thinking-leakage.md) · [BizAgentBench](2026-bizagentbench.md) · [Into the Unknown](2026-into-the-unknown.md) |
