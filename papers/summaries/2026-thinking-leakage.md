# Thinking Leakage: A Causal Audit of NoThink Post-Training in Hybrid Reasoning Models

[arXiv 2609.28682](https://arxiv.org/abs/2609.28682) · Liu, Honavar · Sep 2026 · Lane: Latest
*One-page summary written by Claude. Your own TLDR goes in `notes/`.*

## TLDR
When you post-train a hybrid model in its no-reasoning mode, 42–79% of the gain comes from
the model sliding back toward its reasoning mode, not from getting better at answering
without reasoning.

## The setup
- **Hybrid models** (Qwen3, MiniCPM 4.1) have one set of weights and two modes: **Think**
  writes out its reasoning first, **NoThink** answers directly.
- Labs often post-train in NoThink mode (with GRPO, SFT, or OPSD, which is on-policy
  self-distillation) and report that NoThink scores went up.
- **The question:** is that real NoThink skill, or did the model learn to reason more
  inside NoThink mode? The authors call the second case *thinking leakage*.

## The mechanism
1. **Find the direction.** On the *base* model at layer 20, average the hidden states in
   Think mode, then subtract the average in NoThink mode:
   `v_leak = mean(h | Think) − mean(h | NoThink)`.
   That gives one vector meaning "more Think-like."
2. **Measure drift.** Project each checkpoint's NoThink activations onto v_leak, on a
   scale where base NoThink = 0 and base Think = 1. Trained checkpoints that improved
   moved toward 1.
3. **Steer (is it sufficient?).** Add α·v_leak to the *untrained* base model while it
   generates in NoThink mode. This alone reproduces most of the training gain.
4. **Counter-steer (is it necessary?).** Subtract the checkpoint's own drift so it sits
   back at base NoThink. A large share of the gain disappears.
5. **Split the gain.** Total gain Δ = L (the part that goes away when you remove the drift)
   + R₀ (what's left). The leakage ratio is **λ = L / Δ**.

## Results
- 3 models (Qwen3-8B, Qwen3-4B, MiniCPM4.1-8B) × 3 methods (GRPO, SFT, OPSD), 15 checkpoints.
- On the 9 checkpoints that both improved and drifted toward Think, **λ ranges from 42% to 79%**.
- Implication: method A can look better than method B only because it drifted further
  toward Think.

## Limits
- Only competition math: 120 problems from AIME and HMMT.
- One linear direction at one layer, so some leakage may go undetected.
- The remaining gain R₀ isn't proven to be new NoThink ability.
- At the biggest steering dose, MiniCPM's accuracy falls, which the authors put down to
  the token budget.

## Why it matters to you
At M22 you'll run GRPO on Qwen3 for Casefile. If you train in NoThink mode for speed and
the score goes up, run this audit before you believe it. More generally, a score going up
doesn't tell you *why* it went up. This paper shows how to test a cause with an
intervention instead of guessing.

## Words to know
**Residual stream / hidden state:** the per-token vector each layer reads and writes.
**Steering vector:** a direction you add to activations to push the model's behaviour.
**Sufficiency vs necessity:** steering shows the direction *can* produce the gain;
counter-steering shows the gain *depends on* it.
