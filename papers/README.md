# Aggressive Reading

A paper track that runs alongside the modules. The goal isn't to know what a paper says. It's
to understand the mechanism underneath well enough to explain it, critique it, and say
what it would change in Casefile.

**You write every TLDR. I check it.** Writing it yourself is how it sticks.

## Two lanes

| Lane | What | Pace |
|------|------|------|
| **Foundations** | The canonical papers, in module order (list below). Read each one when you reach its module. | as the modules go |
| **Latest** | 2–3 new papers a week, picked for your track: post-training, RL for LLMs, agents, evals. Picks are in `latest/YYYY-Www.md`. | weekly, ~3–4 h |

## The loop for one paper

1. **Primer.** Ask me `primer <paper>`. I give you the prerequisites you need, the one
   equation or diagram that carries the paper, and 3 things to look for while reading.
2. **Read in passes.**
   - Pass 1 (10 min): abstract, figures, conclusion. Write the one-line TLDR before going further.
   - Pass 2 (45–60 min): the method section properly. Redraw the main diagram by hand.
   - Pass 3 (★ papers only): implement the loss or algorithm in a few lines of PyTorch in
     the matching `modules/` folder.
3. **Write the TLDR.** Copy [TEMPLATE.md](TEMPLATE.md) to `notes/YYYY-short-name.md` and fill it in.
   Fill it from memory first, then check against the paper.
4. **Check.** Ask me `check notes/<file>`. I mark what's wrong or vague, then ask 3
   "explain the mechanism" questions. The paper is done when you answer them without looking.
5. **Log it** in the table at the bottom.

**Latest-lane rule:** new papers are self-reported and unreplicated. Every Latest TLDR must
fill in "What I don't buy". Upvotes measure hype, not quality.

## Foundations reading list

★ = three passes, including implementing it. Others get passes 1–2.

**Level 1–2: how LLMs work** (M1–M7)
- ★ Attention Is All You Need (2017) — [1706.03762](https://arxiv.org/abs/1706.03762) · M6
- Scaling Laws, Kaplan et al. (2020) — [2001.08361](https://arxiv.org/abs/2001.08361) · M7
- ★ Chinchilla (2022) — [2203.15556](https://arxiv.org/abs/2203.15556) · M7
- ★ InstructGPT (2022) — [2203.02155](https://arxiv.org/abs/2203.02155) · M7 (read again at M21)

**Level 3: fine-tuning** (M8–M13)
- ★ LoRA (2021) — [2106.09685](https://arxiv.org/abs/2106.09685) · M10
- ★ QLoRA (2023) — [2305.14314](https://arxiv.org/abs/2305.14314) · M11
- Distilling the Knowledge in a Neural Network, Hinton et al. (2015) — [1503.02531](https://arxiv.org/abs/1503.02531) · M12
- Speculative Decoding (2022) — [2211.17192](https://arxiv.org/abs/2211.17192) · M13
- vLLM / PagedAttention (2023) — [2309.06180](https://arxiv.org/abs/2309.06180) · M13

**Level 4: preferences** (M14–M15)
- Learning to Summarize from Human Feedback (2020) — [2009.01325](https://arxiv.org/abs/2009.01325) · M14
- Constitutional AI (2022) — [2212.08073](https://arxiv.org/abs/2212.08073) · M14
- ★ DPO (2023) — [2305.18290](https://arxiv.org/abs/2305.18290) · M15. Derive the loss from the RLHF objective yourself.

**Level 5–6: RL** (M16–M24)
- ★ PPO (2017) — [1707.06347](https://arxiv.org/abs/1707.06347) · M20
- Let's Verify Step by Step (2023) — [2305.20050](https://arxiv.org/abs/2305.20050) · M23
- ★ DeepSeekMath, which introduces GRPO, Section 4 (2024) — [2402.03300](https://arxiv.org/abs/2402.03300) · M22
- ★ DeepSeek-R1 (2025) — [2501.12948](https://arxiv.org/abs/2501.12948) · M22
- Tülu 3 (2024) — [2411.15124](https://arxiv.org/abs/2411.15124) · M22
- DAPO (2025) — [2503.14476](https://arxiv.org/abs/2503.14476) · M22. What breaks in GRPO at scale.
- Understanding R1-Zero-Like Training / Dr. GRPO (2025) — [2503.20783](https://arxiv.org/abs/2503.20783) · M23
- MemGPT (2023) — [2310.08560](https://arxiv.org/abs/2310.08560) · M24 (you already have the PDF)

**Level 7: search and prompt optimisation** (M25)
- Self-Consistency (2022) — [2203.11171](https://arxiv.org/abs/2203.11171)
- Tree of Thoughts (2023) — [2305.10601](https://arxiv.org/abs/2305.10601)
- OPRO (2023) — [2309.03409](https://arxiv.org/abs/2309.03409)
- ★ DSPy (2023) — [2310.03714](https://arxiv.org/abs/2310.03714)
- GEPA (2025) — [2507.19457](https://arxiv.org/abs/2507.19457)

## Log

| Date | Paper | Lane | Passes | Checked | Note |
|------|-------|------|--------|---------|------|
| | | | | | |
