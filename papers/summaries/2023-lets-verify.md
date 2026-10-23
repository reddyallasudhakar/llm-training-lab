# Let's Verify Step by Step

[arXiv 2305.20050](https://arxiv.org/abs/2305.20050) · Lightman, Kosaraju, Burda, ..., Sutskever, Cobbe (OpenAI) · May 2023 · Lane: Foundations · Module: M14
*One-page summary written by Claude. Your own TLDR goes in `notes/`.*

## TLDR
A reward model that grades every reasoning step (process supervision) picks correct math
solutions far more reliably than one that only grades the final answer (outcome
supervision). Best-of-1860 on MATH: 78.2% with the step grader vs 72.4% with the
answer grader.

## The problem
Reward models are used to pick or train better answers, so the whole system is only as
reliable as its reward model. An outcome reward model (ORM) only learns "this solution
reached the right answer or not". It has to work out on its own *where* a wrong solution
went wrong. Worse, a solution can reach the right answer with broken reasoning and still
get a positive label. Earlier work (Uesato et al. 2022) found the two kinds of supervision
roughly tied on grade-school math. This paper retests it with a stronger model (GPT-4),
far more human labels, and the harder MATH dataset.

## The mechanism
1. **Generator.** A fixed GPT-4-based model, fine-tuned once so it writes one reasoning
   step per line. It is never trained with RL in this paper. Only the reward model changes.
2. **Human step labels (PRM800K).** Labelers mark each step positive (correct and
   reasonable), negative (wrong or unreasonable) or neutral (ambiguous).
   800K step labels over 75K solutions to 12K problems.
3. **Active learning.** Instead of random solutions, labelers see *convincing wrong-answer*
   solutions: ones the current best PRM rates highly but that reach the wrong answer. The
   PRM must be wrong about at least one step there, so each label teaches more.
4. **Train the PRM like a language model.** After the last token of each step, the model
   predicts one label token. Training maximises the log-likelihood of those tokens. One
   forward pass scores every step. Labels stop at the first wrong step.
5. **Score a whole solution:**
   `score(solution) = p_1 × p_2 × ... × p_K`
   - `K`: number of steps in the solution.
   - `p_k`: the PRM's probability that step k is correct (neutral counts as correct).
   - The product is "probability every step is correct". The paper notes it slightly
     penalises long solutions. Using the minimum instead gave 77.6% vs 78.2%.
6. **Evaluate with best-of-N.** Sample N solutions, let the reward model pick the top one,
   check its final answer. A more reliable reward model picks right more often.
7. **Fair comparison at small scale.** Human labels can't be bought for every ablation, so
   the big PRM labels data for small models (about 200× less pretraining compute). Now the
   ORM and PRM train on identical solutions and only the type of supervision differs.

## Results
- **Large scale, best-of-1860 on 500 MATH test problems:** PRM 78.2%, ORM 72.4%, majority
  voting 69.6%. The PRM's lead grows as N grows.
- **Small scale, controlled:** process supervision beats both forms of outcome supervision
  at every data size tested.
- **Active learning** is about 2.6× more data-efficient than labelling uniformly at random.
- **Out of distribution** (recent AP Calculus, Chemistry, Physics, AMC10/12; best-of-100):
  PRM 72.9%, ORM 63.8%, majority vote 61.3% in aggregate.
- The authors call this a *negative alignment tax*: the more inspectable method also scores higher.

## Limits
- Math only, with a fixed generator. The PRM is used for reranking, not for RL training.
- Possible MATH test contamination in pretraining. The authors argue it affects all methods
  alike.
- The ORM and PRM large-scale training sets differ (size, active-learning bias), which is
  why the small-scale study exists.
- **Later work:** DeepSeekMath points to a public PRM800K issue reporting about 20%
  incorrect labels. DeepSeek-R1 (2025) tried PRMs for RL and gave up: a "step" is hard to
  define, step labels are hard to get at scale, and a model-based PRM gets reward-hacked.
  R1 notes PRMs still work well for reranking top-N answers, which is this paper's use.

## Why it matters to you
This is the M14 (reward models) foundation, and it frames your M23 reward design work.
For Casefile, the outcome reward is easy: did the audit reach the right verdict and
payable amount? But a claim can be "correctly denied" for the wrong reason, which is the
same false-positive problem this paper describes. Two concrete moves:
1. Make the reward check the **reason code and the cited policy clause** (waiting period,
   room-rent cap, exclusion), not just approve/deny.
2. Try a step grader first as a **reranker at inference** (best-of-N with a small Qwen
   PRM) before you try it as an RL reward, where it can be hacked.

## Words to know
**ORM:** reward model that scores only the final result.
**PRM:** reward model that scores each intermediate step.
**Best-of-N:** sample N answers and keep the one the reward model likes most.
**Active learning:** choosing which examples to label so each label teaches the most.
**Alignment tax:** the performance cost of a safer method. Here it is negative.
