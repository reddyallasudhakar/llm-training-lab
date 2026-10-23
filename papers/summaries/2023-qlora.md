# QLoRA: Efficient Finetuning of Quantized LLMs

[arXiv 2305.14314](https://arxiv.org/abs/2305.14314) · Dettmers, Pagnoni, Holtzman, Zettlemoyer · May 2023 · Lane: Foundations · Module: M11
*One-page summary written by Claude. Your own TLDR goes in `notes/`.*

## TLDR
Freeze the base model in 4-bit, train LoRA adapters in 16-bit on top, and you lose
nothing measurable versus 16-bit finetuning. That cut the memory for finetuning a 65B
model from more than 780 GB to under 48 GB, so one GPU is enough.

## The problem
LoRA already trains only a tiny set of weights, but you still have to hold the frozen
base model in GPU memory at 16 bits. That is the bulk of the bill. Earlier 4-bit methods
worked for inference but broke down during training.

## The mechanism
**One layer of QLoRA** (Eq. 5):
`Y = X · doubleDequant(c1, c2, W_NF4) + X · L1 · L2`
- `X`: the input activations, in BF16.
- `W_NF4`: the frozen base weight, stored in 4 bits. It is never updated.
- `doubleDequant`: turns `W_NF4` back into BF16 just in time for the matmul.
- `L1` (h×r) and `L2` (r×o): the LoRA adapter. `r` is the rank. These are the only
  weights that get gradients. (Plain LoRA, Eq. 3, also scales this term by a scalar `s`.)
- So there are two dtypes: a **storage** dtype (NF4) and a **compute** dtype (BF16).

**Three tricks make the 4-bit storage safe:**
1. **NF4 (4-bit NormalFloat).** Pretrained weights are roughly zero-centred normal. So
   instead of evenly spaced levels, pick the 16 levels at quantiles of N(0, 1):
   `q_i = ½ [Q(i / (2^k+1)) + Q((i+1) / (2^k+1))]`, where `Q` is the standard normal
   quantile function and `k` = 4 bits. Each level then gets an equal share of weights.
   They build the negative and positive halves separately so that 0 is exactly
   representable, then scale everything to [-1, 1].
2. **Block-wise scaling.** Weights are cut into blocks of 64. Each block is divided by its
   own absmax constant `c`, so one outlier only hurts its own block.
3. **Double Quantization.** Those constants cost 32/64 = 0.5 bits per parameter. Quantize
   the constants themselves to 8-bit floats in blocks of 256: cost drops to
   8/64 + 32/(64·256) = 0.127 bits, saving 0.373 bits per parameter (about 3 GB at 65B).
4. **Paged optimizers.** Optimizer state lives in NVIDIA unified memory and spills to CPU
   RAM during memory spikes from long sequences, instead of crashing.

## Results
- **NF4 beats other 4-bit types.** Mean perplexity on Pile Common Crawl: Int4 34.34,
  FP4 31.07 / 29.48, NF4 + DQ 27.41.
- **No loss vs 16-bit.** Mean 5-shot MMLU across LLaMA 7B–65B: BF16 53.0, NF4 + DQ 53.1,
  FP4 52.2. On GLUE and Super-NaturalInstructions it matched full 16-bit finetuning.
- **LoRA on every linear layer is what matters.** Adapters on only query and value could
  not match full finetuning. With all layers covered, rank `r` barely mattered.
- **Guanaco** (QLoRA on OASST1) reached 99.3% of ChatGPT on the Vicuna benchmark (GPT-4
  as judge) after 24 hours on one GPU. The 33B model trains on a 24 GB GPU in under 12 hours.
- **Data quality beat size.** 9k OASST1 examples beat a 450k FLAN v2 subsample on chatbot
  quality. FLAN v2 won on MMLU but did worst on Vicuna: one benchmark doesn't predict another.
- **GPT-4 as judge is biased.** It favoured whichever answer came first, and agreed with
  humans only weakly per sample (Fleiss κ = 0.25).

## Limits
- The authors did not show QLoRA matches *full* finetuning at 33B and 65B, only 16-bit LoRA.
- "Matches full finetuning" didn't hold up in general. *LoRA Learns Less and Forgets Less*
  (Biderman et al., 2024) found LoRA clearly behind full finetuning on code and math.
- Merging adapters into a 4-bit base loses accuracy, since the adapter was trained against
  the dequantized weights. *QA-LoRA* (Xu et al., 2023) and *LoftQ* (Li et al., 2023) target
  this gap and the error at lower bit widths.
- The chatbot results rest on 80 Vicuna prompts and GPT-4 judging, which the paper itself
  shows is shaky.

## Why it matters to you
At M11 you'll swap your M10 LoRA run for QLoRA. For Casefile the decision is per size: the
0.5B–3B Qwen models fit fine in BF16 LoRA, so QLoRA is only worth it for the 8B (16 GB of
weights in BF16 alone). Take three rules from this paper: put adapters on all linear
layers, don't over-tune `r`, and spend effort on a few thousand clean claim audits rather
than a big noisy set. NF4 is a CUDA training format. For the M3 MacBook, merge the adapter
into a BF16 copy, then quantize separately for local inference, and re-run your eval
after each step.

## Words to know
**Quantization:** storing weights in fewer bits, with a scale constant to map them back.
**Absmax scaling:** divide a block by its largest absolute value so it fits the range.
**Storage vs compute dtype:** weights sit in NF4, but every matmul runs in BF16.
**Adapter merge:** folding `L1·L2` back into `W` so you ship one plain weight matrix.
