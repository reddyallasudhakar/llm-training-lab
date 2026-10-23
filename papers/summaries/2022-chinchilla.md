# Training Compute-Optimal Large Language Models (Chinchilla)

[arXiv 2203.15556](https://arxiv.org/abs/2203.15556) · Hoffmann, Borgeaud, Mensch, …, Vinyals, Sifre · Mar 2022 · Lane: Foundations · Module: M7
*One-page summary written by Claude. Your own TLDR goes in `notes/`.*

## TLDR
For a fixed compute budget, model size and training tokens should grow at the same rate.
Big models of the time were too large and trained on too little data. Chinchilla, at 70B
parameters on 1.4T tokens, used the same compute as the 280B Gopher and beat it on almost
every benchmark.

## The problem
You get a compute budget C, roughly fixed by how many chips you have and for how long. You
can spend it on a bigger model N or on more training tokens D. Kaplan et al. (2020) said:
for 10× more compute, make the model 5.5× bigger and the data only 1.8× bigger. So labs
built larger models and kept training on about 300B tokens (GPT-3, Gopher, Jurassic).
This paper asks whether that split was right.

## The mechanism
Training compute is approximated as `C ≈ 6·N·D` FLOPs (N parameters, D tokens). The goal
is to pick N and D that give the lowest loss for a given C. They trained over 400 models
(70M to over 16B parameters, 5B to 500B tokens) and estimated the best split three ways:
1. **Envelope of training curves.** For each model size, train with 4 schedule lengths.
   At each FLOP count, keep whichever run had the lowest loss.
2. **IsoFLOP profiles.** Fix 9 compute budgets (6e18 to 3e21 FLOPs). At each budget,
   train many sizes, which forces D = C / 6N. Loss vs size makes a valley; its bottom is
   the best size for that budget.
3. **Fit a loss formula** to every run:
   `L(N, D) = E + A / N^α + B / D^β`
   - **E = 1.69:** the floor, the entropy of natural text. No model gets below it.
   - **A / N^α** (A = 406.4, α = 0.34): extra loss because the model is finite in size.
   - **B / D^β** (B = 410.7, β = 0.28): extra loss because training saw finite data.
   Minimising L subject to `6ND = C` gives `N_opt ∝ C^a`, `D_opt ∝ C^b`, with
   `a = β/(α+β)` and `b = α/(α+β)`.

The exponents: a = 0.50, 0.49, 0.46 and b = 0.50, 0.51, 0.54 for approaches 1 to 3.
Kaplan had a = 0.73, b = 0.27. Equal exponents mean: double the model, double the data.
In their Table 3 that works out to about 20 tokens per parameter (1B → 20.2B tokens,
10B → 205.1B tokens).

A key detail behind the disagreement with Kaplan: the learning-rate schedule has to
match the training length. Kaplan used one fixed schedule, which overstated the loss of
shorter runs and pushed the advice toward bigger models.

## Results
- Gopher's budget (5.76e23 FLOPs) should have gone to a model about 4× smaller trained on
  4× more tokens. They trained Chinchilla: 70B parameters, 1.4T tokens, same compute.
- **MMLU (5-shot):** 67.6% vs Gopher's 60.0% and GPT-3's 43.9% (the abstract says 67.5%).
- **BIG-bench:** 65.1% average vs 54.4%; worse than Gopher on only 4 of 62 tasks.
- Also beat GPT-3 (175B), Jurassic-1 (178B) and MT-NLG (530B) on most tasks.
- 4× smaller means cheaper fine-tuning and inference.

## Limits
- The authors note: only two large-scale runs to compare (Chinchilla and Gopher), some
  curvature that suggests large models may still be oversized, and every run was under
  one epoch of data.
- **The fit was questioned.** Besiroglu et al. (2024, "Chinchilla Scaling: A replication
  attempt") could not reproduce Approach 3 from the reported numbers and found its
  confidence intervals implausibly narrow. Their refit agreed with approaches 1 and 2.
- **Compute-optimal is not deployment-optimal.** The paper only counts training compute.
  LLaMA (Touvron et al., 2023) and Sardana & Frankle (2023, "Beyond Chinchilla-Optimal")
  argued that if you serve a model a lot, you should train a smaller model on far more
  tokens. Qwen's small models follow that logic.
- **Repeating data.** Muennighoff et al. (2023) studied the multi-epoch case the paper skipped.

## Why it matters to you
At M7 you'll study scaling laws. You aren't pretraining for Casefile, but this explains
why a 0.5B to 8B Qwen is so capable: it was trained far past "20 tokens per parameter".
For the Casefile decision, the binding budget is your 16 GB M3 at inference, not training
FLOPs. So pick the largest Qwen that runs at acceptable latency on the laptop, then spend
GCP time on data for it. Borrow the method too: before a long run, try 2 to 3 model sizes
at 2 to 3 data sizes and fit the curve, instead of guessing.

## Words to know
**Compute-optimal:** the N and D split that gives the lowest loss for a fixed FLOP budget.
**IsoFLOP curve:** loss vs model size at one fixed compute budget.
**Scaling law:** a power-law fit that predicts loss from size, data or compute.
**Undertrained:** a model that would keep improving if it saw more tokens.
