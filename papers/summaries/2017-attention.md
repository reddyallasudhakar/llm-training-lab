# Attention Is All You Need

[arXiv 1706.03762](https://arxiv.org/abs/1706.03762) · Vaswani, Shazeer, Parmar, Uszkoreit, Jones, Gomez, Kaiser, Polosukhin · Jun 2017 · Lane: Foundations · Module: M6
*One-page summary written by Claude. Your own TLDR goes in `notes/`.*

## TLDR
The Transformer drops recurrence and convolution and builds a translation model out of
attention plus small feed-forward layers. It beat the best previous models, ensembles
included, on English-German translation, and trained much faster because every position
is processed in parallel.

## The problem
RNNs read a sequence one token at a time. Step t needs the hidden state from step t−1, so
you can't parallelise within a sentence, and long sequences get slow. Information between
two distant words also has to pass through many steps. Earlier attention models still
sat on top of an RNN. The question: can attention alone do the job?

## The mechanism
**Scaled dot-product attention** (eq. 1):
`Attention(Q, K, V) = softmax(Q Kᵀ / √d_k) V`
- **Q (queries):** one row per position, "what am I looking for?"
- **K (keys):** one row per position, "what do I contain?"
- **V (values):** one row per position, "what do I hand over if picked?"
- **Q Kᵀ:** every query dotted with every key, an n × n grid of match scores.
- **√d_k:** d_k is the key width. If q and k have unit-variance entries, q·k has variance
  d_k. Big scores push softmax into regions with tiny gradients, so they divide by √d_k.
- **softmax:** turns each row of scores into weights that sum to 1.
- **× V:** each output is a weighted average of the value rows.
- **Mask:** in the decoder, scores for future positions are set to −∞ before the softmax,
  so token i can only see tokens before it.

**Multi-head attention:** `head_i = Attention(Q W_i^Q, K W_i^K, V W_i^V)`, then
`MultiHead = Concat(head_1..head_h) W^O`. The W matrices are learned projections. Each head
works in a smaller subspace (d_k = d_v = d_model / h = 64), so 8 heads cost about the same
as one full-width head but can attend to different things.

**One layer** = attention, then a position-wise feed-forward net
`FFN(x) = max(0, x W₁ + b₁) W₂ + b₂` (inner width d_ff = 2048). Each sub-layer is wrapped as
`LayerNorm(x + Sublayer(x))`: a residual connection, then layer norm.

**Positions:** attention ignores order, so they add sinusoids to the embeddings:
`PE(pos, 2i) = sin(pos / 10000^(2i/d_model))`, `PE(pos, 2i+1) = cos(same)`.
pos is the token's position and i the dimension index. Learned position embeddings gave
nearly identical results.

**Base model:** N = 6 encoder and 6 decoder layers, d_model = 512, h = 8, 65M parameters.
Adam with β₁ = 0.9, β₂ = 0.98, linear warmup for 4,000 steps then inverse-square-root decay,
dropout 0.1, label smoothing 0.1.

## Results
- **WMT 2014 EN-DE:** big model 28.4 BLEU, over 2 BLEU above the best previous results,
  ensembles included. The base model (27.3) also beat all earlier models.
- **WMT 2014 EN-FR:** big model 41.8 BLEU in the abstract and Table 2 (the text says 41.0),
  a new single-model best at under 1/4 the training cost of the previous best.
- **Cost:** base model 12 hours on 8 P100 GPUs; big model 3.5 days.
- **Per-layer cost:** self-attention is O(n²·d) with O(1) sequential steps; an RNN is
  O(n·d²) with O(n) steps.
- **Ablations:** one head is 0.9 BLEU worse than the best setting; too many heads also hurt.
- **Parsing:** a 4-layer model scored 91.3 F1 (WSJ only) and 92.7 (semi-supervised).

## Limits
- Attention cost grows with n², which hurts on long inputs. The authors only suggest
  restricting attention to a neighbourhood. FlashAttention (Dao et al., 2022) later made
  exact attention much faster by reducing memory traffic, without changing the n² maths.
- Most parts have since been swapped out in modern LLMs, including Qwen:
  decoder-only stacks (GPT, Radford et al., 2018) replaced encoder-decoder; Pre-LN
  (Xiong et al., 2020) replaced post-LN for stable deep training, with RMSNorm (Zhang &
  Sennrich, 2019) in place of LayerNorm; RoPE (Su et al., 2021) replaced sinusoids;
  SwiGLU (Shazeer, 2020) replaced ReLU; grouped-query attention (Ainslie et al., 2023)
  shares K/V across heads to shrink the KV cache.
- Tested only on translation and parsing.

## Why it matters to you
At M6 you'll build this from scratch, as a decoder-only model. Code eq. 1 with the mask
yourself, then check it against `F.scaled_dot_product_attention`. For Casefile, the n²
cost and the KV cache are what decide how much of a claim file (policy wording, discharge
summary, itemised bill) you can put in context on a 16 GB M3. That decides whether you
truncate, summarise, or retrieve sections before the audit model sees them.

## Words to know
**Self-attention:** Q, K and V all come from the same sequence.
**Causal mask:** blocks attention to future tokens so the model can generate left to right.
**Residual connection:** adding a layer's input to its output, `x + f(x)`, which keeps
gradients flowing in deep stacks.
**KV cache:** stored keys and values from earlier tokens, so generation doesn't recompute them.
