# BizAgentBench: Unveiling Hidden Challenges in Deploying Long-Horizon Agents for Real-World Business

[arXiv 2610.20424](https://arxiv.org/abs/2610.20424) · Zhao et al. · Oct 2026 · Lane: Latest
*One-page summary written by Claude. Your own TLDR goes in `notes/`.*

## TLDR
A benchmark for business agents that also grades *how* the agent reached the goal, for
example whether it paid before getting approval. The best models pass only about 64% of
the time, and smaller ones pass far less often.

## The problem
Most agent benchmarks only check the final state. In business, a run can reach the right
end state and still be a failure: an unapproved payment that was later refunded, or a
cash reserve that dipped below its floor for a moment. Policies can also be outdated or
contradict each other, so choosing the right rule is part of the job.

## The environment
- Built from real workflows at a large payments company (cross-border payments, FX,
  treasury), with sensitive details removed.
- **789 documents** covering current *and* historical policies, each with a scope and an
  effective date.
- **9 core agent tools, plus 601 business tools** the agent has to discover. There are
  also 398 database collections.
- **50 tasks** with resource limits, time dependencies, prior commitments and conflicting
  rules. Examples: payment reserves, delivery windows, tax matching, insurance settlement.
- **An LLM plays the customer.** It shares facts only when asked and gives approvals one
  at a time.

## The mechanism: how grading works
Every run is scored on three deterministic checks, with no LLM judge:
- **G (goal attainment):** did it reach an acceptable end state?
- **P (process validity):** did it follow the rules on the way, e.g. getting approval
  first and keeping the reserve above its floor?
- **F (protection):** did it leave protected objects alone and avoid forbidden side effects?

**Strict success = G AND P AND F.** The checks inspect business state, ledgers, receipts
and event histories, not the sequence of tool calls. So **any valid path passes**. A
reference ("Golden") trace exists only to prove each task is solvable.

## Results (strict success)
| Model | G | Strict |
|---|---|---|
| Claude Opus 5 | 64.0 | **64.0** |
| GPT-6 Sol | 64.7 | **61.3** |
| GLM-5.3-Flash / DeepSeek-V4.1-Flash | 48.0 / 46.7 | **43.3** |
| Qwen3.8-27B | 42.0 | **32.7** |
| Qwen3.5-9B | 0.0 | **0.0** |

- **Consistency is low.** pass^3 counts tasks solved in all 3 runs. GPT-6 Sol drops from
  61.3% to 40%, and Qwen3.8-27B to 10%.
- **Weaker models take inefficient paths.** DeepSeek-V4.1-Flash makes 1.7× the reference
  number of tool calls, reads 20 documents per task where the reference reads 5, and
  repeats the same database read 16 times per run.
- **Typical process failures:** dipping below a reserve floor and then fixing it, or
  acting before approval. Qwen3.5-9B made 88 attempts to act before approval.
- **The bottleneck is information, not reasoning.** When the customer simulator discloses
  everything up front, DeepSeek-V4.1-Flash rises from 43% to 70%. Handing it the right
  documents only gets it to 50%.

## Limits
- The environment is a reconstruction, and there are only 50 tasks.
- An LLM plays the customer, and the sensitivity check across simulators covers only one model.
- The paper has no dedicated limitations section.

## Why it matters to you
This is your claims-agent problem in another industry. Settling a claim correctly while
skipping a required document check is still a failure. Ideas to take for Casefile's eval:
1. Grade **process validity**, not just the final decision.
2. Check **state and side effects**, not exact paths.
3. Report **pass^k** for consistency.
4. Model **policy versions with effective dates**, since insurance rules change by policy
   year too.

## Words to know
**Long-horizon agent:** an agent that takes hundreds of tool calls to finish a task.
**pass^k:** the share of tasks solved in *all* k runs, a measure of reliability.
**Golden trace:** one known-valid solution, used to prove a task is solvable.
