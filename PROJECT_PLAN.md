# Flagship project: ClaimCheck

*Working name, rename freely.* A small open model that audits Indian health-insurance claim
decisions: it reads the policy terms, the claim and the insurer's decision, then says whether
each rejection or deduction is valid, which rule applies, what should actually have been
paid, and the grounds for appeal.

The learning plan ([README.md](README.md)) teaches the techniques. This project is where
you apply them in public, on one problem, with one eval, so people can see real work.

---

## 1. Why this project

- **It's a real problem you know.** At Raksha you saw wrongful rejections and underpayments
  up close: proportionate room-rent deductions, waiting-period rules applied wrongly, sub-limits
  stacked on top of each other. Working out the right payable amount is mechanical but easy to
  get wrong, and that's exactly where insurers underpay.
- **It needs a small, local model.** Claim files are health data, which India's DPDP Act
  treats as sensitive. A 1.5–3B model running on a laptop or a broker's own server, at a
  fraction of an API's cost, is a genuine product argument, not just an exercise.
- **It has a verifier.** RL works when a program can check the answer. Payable-amount and
  rule-validity questions can be checked exactly by a rules engine you write. Few portfolio
  RL projects have a verifier with real-world meaning.
- **It's unusual.** There are thousands of text-to-SQL and math GRPO repos. A benchmark plus
  models for Indian health-claim adjudication, built by someone who ran a claims company,
  stands out to both AI teams and insurtech teams.

## 2. The tasks, from easiest to hardest

Each task uses a different technique from the learning plan, and every task is scored on
the same eval harness.

| # | Task | Input → output | How it's scored | Technique | Modules |
|---|------|----------------|-----------------|-----------|---------|
| T1 | **Classify** the rejection ground | rejection letter → one of ~10 categories (pre-existing disease non-disclosure, waiting period, exclusion, not an inpatient case, room-rent proportionate deduction, sub-limit, documents missing, policy lapsed, …) | accuracy, macro-F1 | encoder fine-tune (ModernBERT) vs prompted LLM | M8 |
| T2 | **Extract** the claim facts | letter, bill or discharge summary → JSON (dates, diagnosis, room category, bill line items, amounts, clause cited) | per-field exact match / F1, schema validity | SFT with LoRA, then QLoRA | M9–M13 |
| T3 | **Adjudicate** | structured policy + claim facts + insurer's decision → verdict per deduction (valid / invalid / needs info), the rule, the correct payable amount | verdict accuracy; amount exact to ±₹1; rule cited correctly | SFT → **GRPO** with the rules engine as verifier | M22–M23 |
| T4 | **Draft the appeal** | adjudication → appeal letter | no exact answer, so pairwise preference by a calibrated judge plus your own labels | **DPO** | M14–M15 |
| T5 | **Agentic adjudication** (stretch) | raw documents → the model calls tools (`lookup_clause`, `days_between`, `irdai_rule`) over several turns → verdict | the same verifier as T3, over the whole episode | multi-turn RL | M24 |

**T3 is the centrepiece.** Most teams would start with "fine-tune a model on claims text".
Here the most interesting result is whether GRPO can teach a 1.5B model to apply the rules
correctly (waiting periods, proportionate deduction, co-pay, sub-limits, in the right order)
better than SFT alone, and better than a frontier model prompted zero-shot.

## 2a. The small-model series: "Can a 0.5B model match a frontier model?"

A run of short fine-tuning sprints, each one a narrow task from the FairClaims or PolicyGaido
flows. Each sprint answers the same question: **after minutes of training, can a 0.5–1.5B
model running on a laptop match a frontier model on this task?** Each sprint adds a row to one
table, and the finished table (where small models win, and where they still lose) is a strong
post in its own right.

Good candidates are **narrow** (a fixed label set or schema), **high-volume** (run many times
per case, so per-call cost adds up), **privacy-sensitive** (health documents), and **exactly
scorable**.

### FairClaims: the claims flow

| # | Task | Where it sits in the flow | Output | Score | Data source |
|---|------|---------------------------|--------|-------|-------------|
| F1 | **Situation triage** | Step 1, "What happened?" | rejected / paid less / cashless denied / not filed / other, from a free-text or WhatsApp-style message (English, Hindi, Hinglish) | accuracy | synthetic messages |
| F2 | **Document-type classification** | "Drop your papers, we'll sort them" | settlement letter / rejection letter / policy schedule / hospital bill / discharge summary / pre-auth denial / ID card / other | accuracy | synthetic documents |
| F3 | **Bill line-item categorisation** | "Line items grouped" | each bill line → room / ICU / pharmacy / investigations / professional fees / implants / **non-payable consumable** | accuracy, macro-F1 | IRDAI's published list of non-payable items + synthetic bills |
| F4 | **Rejection / deduction reason** (= T1) | The analysis | ~10 categories | accuracy, macro-F1 | synthetic letters + real judgments |
| F5 | **Settlement-letter extraction** | "Amount and deductions found" | JSON: claimed, approved, each deduction with amount and stated reason | per-field F1 | synthetic letters (labels known) |
| F6 | **Policy-schedule extraction** | "Coverage details found" | JSON: sum insured, room-rent limit, co-pay, deductible, inception date, waiting periods | per-field F1 | synthetic schedules from real product parameters |
| F7 | **Discharge-summary extraction** | Facts for the waiting-period checks | diagnosis, procedure, admission and discharge dates, any pre-existing condition mentioned | per-field F1 | synthetic summaries |
| F8 | **Deduction strength** (needs the rules engine) | "Strong / Challenge" badges | each deduction → strong / challenge / weak to contest, with the rule | accuracy against the engine | engine-generated |
| F9 | **Next escalation step** | "One next action" | insurer grievance cell → IRDAI Bima Bharosa → Ombudsman → consumer commission, with deadlines | accuracy against rules | rule-generated scenarios |

### PolicyGaido: the advisor and broker side (Saathi)

| # | Task | Output | Score |
|---|------|--------|-------|
| P1 | **Intent routing** for inbound messages | buy / renew / port / claim help / complaint / other | accuracy |
| P2 | **Needs extraction** from a conversation | JSON profile: ages, family members, city, pre-existing conditions, budget, existing cover | per-field F1 |
| P3 | **Policy-wording → feature table** for plan comparison | JSON: room-rent cap, co-pay, restoration, no-claim bonus, waiting periods, sub-limits | per-field F1 |
| P4 | **Natural language → plan filter** ("no room-rent cap, under ₹15k, covers diabetes from year 1") | JSON query over the plan database | run the query and compare the plans returned (a verifier, so a later RL candidate) |
| P5 | **Mis-selling compliance check** on agent messages (for a 40,000-agent broker network) | compliant / risky phrasing ("guaranteed claim", "no waiting period") + reason | accuracy, recall on the risky class |

### Order to run them

1. **F2 document type**: the warm-up. Easiest labels, and you learn the full pipeline.
2. **F3 bill line items**: the strongest small-model case (hundreds of short lines per claim,
   labels grounded in IRDAI's list).
3. **F4 rejection reason**: the original T1.
4. **F5 / F6 extraction**: harder JSON output; this is where small models may fall short, and
   that's a finding too.
5. **P4 plan filter and F8 deduction strength**: these have verifiers, so they lead into
   the RL phase.

## 2b. The post-training pipeline

The tasks above are what the model learns. This is how you train it. Instead of fine-tuning
an instruct model that someone else post-trained, you **start from the base model and run
every post-training stage yourself**, the way open recipes like Tülu 3 do. Each stage is
scored on the same eval, so the results table shows exactly what each one added.

```
Qwen2.5-1.5B  (BASE model: completes text, can't follow instructions yet)
   │
   │  0. Continued pretraining (optional): public insurance text (policy wordings,
   │     IRDAI circulars) so the model knows the vocabulary
   │  1. SFT: chat format + a general instruction mix + T1–T3 data
   │  2. Distillation: learn from a bigger teacher (D1/D2 below)
   │  3. Preference tuning: DPO on T4 appeal letters
   │  4. RL with verifiable rewards: GRPO on T3 against the rules engine
   │  5. Distil down: compress the final 1.5B into a 0.5B for laptops and phones
   ▼
claimcheck-1.5b   claimcheck-0.5b
```

**Baseline to beat:** the same tasks fine-tuned on top of `Qwen2.5-1.5B-Instruct`. If your
pipeline from the base model matches it on your tasks while keeping general ability, you
have reproduced post-training end to end at small scale.

### Distillation: three kinds, increasing in difficulty

| | Kind | How it works | Used for |
|---|---|---|---|
| **D1** | **Sequence-level (data) distillation** | A strong teacher (a frontier API, or Qwen 7B/14B) writes step-by-step adjudications for T3. Keep only traces the rules engine marks correct (rejection sampling), then SFT the student on them. This is how DeepSeek-R1's small distilled models were made. | Stage 2; easiest |
| **D2** | **Logit-level distillation** | The student matches the teacher's full probability distribution over the next token (a KL-divergence loss), not just its chosen token. Needs the same tokenizer, so the teacher is Qwen 7B fine-tuned on your tasks. TRL's `GKDTrainer` supports it. | Stage 2 and stage 5 (1.5B → 0.5B) |
| **D3** | **On-policy distillation** (advanced) | The student writes its own answers and the teacher scores every token of them. It trains on the student's own mistakes, like RL, but with dense per-token feedback, so it costs much less than RL. | Experiment against GRPO |

**The headline experiment.** On T3 at 1.5B, compare SFT, D1, D2, GRPO and D1 → GRPO, all on
the same eval and the held-out products, plus the compute each one cost. "Does RL beat
distillation from a bigger model on a rules-heavy task?" is a question practitioners care
about, and your table will answer it for one real domain.

## 3. Where RL actually adds value (and where it doesn't)

**Rule of thumb:** use RL when a program can check the answer cheaply. Use preference tuning
(DPO) when you can only say which of two answers is better. Use SFT when you have good
examples to copy.

| Use case | Can a program check it? | Best tool | Value |
|---|---|---|---|
| Claim payable-amount and validity audit (T3) | Yes: rules engine | **GRPO** ← this project | High: catches underpayment |
| Structured extraction to a schema (T2) | Yes: field match | SFT first, RL to polish | Medium |
| Text-to-SQL over a claims or policy database | Yes: run the query, compare results | GRPO | High, but a crowded space |
| Tool-calling agents (Raksha-style workflows) | Yes, if the environment checks the final state | Multi-turn RL (T5) | High, but hard |
| Premium and benefit calculators, eligibility checks | Yes | GRPO | Medium |
| Appeal letters, advice, explanations | No: quality is a judgement call | DPO / preference data | High for users |

## 4. Where the data comes from

| Source | What you get | Use | Notes |
|---|---|---|---|
| **IRDAI regulations**: Master Circular on Health Insurance (2024), standard exclusions, standard product wordings (e.g. Arogya Sanjeevani) | The rules: waiting periods, the 60-month moratorium, pre-existing disease limits, how claims must be settled | Encoded in the **rules engine** | Read the circular itself, not news summaries, and record the clause for every rule you encode. |
| **Insurer policy wordings and Customer Information Sheets** (insurers publish these) | Product parameters: room-rent caps, co-pay, sub-limits, waiting periods | 5–10 real products encoded as YAML; hold 2–3 out for testing | Public documents; you encode the parameters yourself. |
| **Synthetic scenario generator** (you write it) | Thousands of claims sampled from those products, with ground truth computed by the engine | T3 training and RL data; source facts for T2 | The core dataset. You own all of it and can publish it freely. |
| **Frontier model as a writer** | Realistic rejection letters, bills and discharge summaries rendered *from* the synthetic facts, so the labels are known | T1/T2 training data | Vary the style, the insurer and the noise (OCR errors, missing fields). Check the model provider's terms on training with its outputs. |
| **Consumer commission judgments** (e.g. Indian Kanoon; NCDRC/state commissions) | Real disputes: facts, insurer's ground, outcome | A **real-world test set** (~150–300 cases, hand-labelled) and the mix of rejection categories | Public, but check the terms and prefer an official API. Strip personal names before publishing anything. |
| **Ombudsman annual reports** (cioins.co.in) | How often each type of complaint occurs | Weight the scenario generator realistically | Award texts are OTP-gated for complainants only, so they're **not** a data source. |
| ~~Raksha / Gaido customer data~~ | — | **Never** | Private health data belonging to the company. Not even "anonymised". |

The reason this works: synthetic data gives you scale and exact labels, and real
judgments give you an honest test. **Every headline number should also be reported on the
real-world test set.**

## 5. Objectives and how they're measured

Set the baselines in week 3, then fix the targets. Don't change them later to fit the results.

1. **Benchmark:** `claimcheck-bench` with T1–T3 splits, a real-world test set, a data card
   and a one-command eval. It's useful even if no model you train wins.
2. **Fine-tuning:** a 1.5–3B model within a few points of a frontier model on T1/T2, at
   under 1/20th of the cost per 1,000 claims, running on your M3.
3. **RL:** on T3, GRPO beats the same model's SFT version on verdict and amount accuracy,
   *including on held-out insurer products*. That shows the rules were learned, not memorised.
4. **Post-training:** the pipeline from the base model matches the instruct-model baseline on
   T1–T3, and the 0.5B distilled model keeps most of the 1.5B model's T3 accuracy at a third of
   the size.
5. **Distillation vs RL:** a results table comparing SFT, D1, D2, GRPO and D1 → GRPO on
   accuracy and compute cost.
6. **Honesty:** a failure analysis in which every result has an example, and at least one
   documented case of reward hacking and how you fixed it.

Track these alongside accuracy: cost per 1,000 claims, tokens per second on the M3, and
what fine-tuning cost on general ability (M12).

## 6. What goes public

**Repos (both public on GitHub):**

1. **`llm-training-lab`**: the learning plan, module exercises and notes. Shows you learning
   in the open, week after week. (This is the current `ai-learning/` folder, moved out of the
   portfolio repo.)
2. **`claimcheck`**: the flagship. People will judge it by the README and its structure:

```
claimcheck/
  README.md        results table at the top, then a one-command reproduction
  rules/           adjudication engine + pytest suite (CI runs it on every push)
  policies/        one YAML file per insurer product, each with source links
  data/            scripts only: generate, collect, clean, split (no raw scraped text)
  evals/           eval harness, judge prompts, reports
  train/           sft/, dpo/, grpo/: configs + scripts, seeds fixed
  results/         results.md + run logs
  docs/            data card, design notes, decision log (including what failed)
```

**Hugging Face:** the dataset (`claimcheck-bench`), a model for each pipeline stage
(`claimcheck-1.5b-sft`, `-distill`, `-grpo`, and the final `claimcheck-0.5b`) with model cards
that state their limits, and a Gradio demo on Spaces.

**Portfolio site:** a project page plus four posts, one per milestone:
1. *Building a benchmark for Indian health-insurance claims*
2. *Post-training a base model end to end: what each stage added*
3. *Teaching a small model to adjudicate claims with RL*
4. *Distillation vs RL on a rules-heavy task*

**What makes it read as technical:** you wrote the code, so the commit history is real and
built from small commits; the rules engine has tests and CI; every number in
`results.md` can be reproduced with one command; the decision log records what didn't work.

## 7. Execution plan (~19 weeks at 8–10 hours a week)

Each phase pairs learning-plan modules (in the lab repo) with project work (in `claimcheck`).

| Weeks | Learn (lab) | Build (claimcheck) | Public output |
|---|---|---|---|
| **1** | Finish M1 | Create both repos; write the task spec and JSON schemas | Lab repo public |
| **2–4** | M2, M3, M4 | **Rules engine v1** + tests (waiting periods, moratorium, room-rent proportionate deduction, co-pay, sub-limits, deductible); encode 5–10 products; scenario generator; collect and label ~150 real judgments; eval harness; frontier and small-model baselines | `claimcheck` v0, benchmark v0, **post 1** |
| **5–6** | M5, M6, M7 | Keep labelling the real-world test set; fix engine edge cases the baselines expose; compare base vs instruct models on your tasks | Lab: a GPT from scratch |
| **7–10** | M8–M13 | T1 classifier; **post-training stages 0–1 from the base model** (optional continued pretraining, SFT with an instruction mix + T1–T3); LoRA → QLoRA at 7B on GCP; the instruct-model baseline; **D1** traces filtered by the engine; serve locally | SFT + D1 models on Hugging Face, **post 2** |
| **11–13** | M14–M20 | T4 preference pairs + DPO (cut this first if you're behind) | Lab: PPO solving LunarLander |
| **14–16** | M21–M23 | **GRPO on T3** with the engine as the reward, from both SFT and D1; reward-hacking study; held-out product eval | GRPO model, **post 3** |
| **17** | M12 revisited | **D2** logit distillation (Qwen 7B → 1.5B), **D3** on-policy distillation vs GRPO; stage 5: distil the final model to 0.5B | 0.5B model, **post 4** |
| **18–19** | M24, M25 | T5 agentic stretch; DSPy prompt-optimisation comparison; Gradio demo; final results table | Demo Space, project page |

### Compute: train on GCP, run on the laptop

| What | Where | Why |
|---|---|---|
| Lessons, data generation, evals, local inference | M3 MacBook | Free; and "runs on a laptop" is the claim being tested |
| Experiments and the first fine-tunes | **Colab Enterprise** in Vertex AI with a GPU runtime | Notebook-speed iteration, billed to your GCP credits |
| Real training runs (SFT, QLoRA at 7B, DPO, GRPO) | A **Compute Engine VM with an L4 GPU** (24 GB; `g2-standard-8` on a Deep Learning VM image) running scripts from the repo; an A100 for GRPO and the 7B teacher | Reproducible scripts in git, not notebook state |
| Optional comparison row | **Vertex AI supervised tuning of Gemini Flash** | "Fine-tuned frontier model" as one more baseline against the 0.5B |

The steps every model goes through: train on GCP with TRL + PEFT → merge the LoRA adapter →
push to Hugging Face → convert for the Mac (MLX or GGUF via llama.cpp) → evaluate locally.

**Before week 1:** GPU quota on a new GCP project usually starts at 0, so request it on
day one (the `GPUS_ALL_REGIONS` quota plus the L4 quota in a region that offers L4s), and set
a budget alert on the billing account. Stop VMs when you're not training; an idle GPU VM
burns credits.

**Compute estimate:** sprints at 0.5–1.5B take minutes on an L4. QLoRA at 7B takes a few
hours; the 7B teacher for D2 and GRPO on 1.5B take roughly 10–30 A100-hours. Treat these as
estimates and log the real numbers in `results.md`.

**If you fall behind:** cut T4/DPO, then T5, then D3. **Never cut the eval or the real-world
test set.**

## 8. Risks

| Risk | Mitigation |
|---|---|
| The rules are ambiguous or vary by product | v1 covers a documented subset; each rule cites its clause; products outside scope get "needs info" |
| The model learns the synthetic data's quirks, not the rules | Held-out products, a real-world test set, and noise in the generated documents |
| Reward hacking (e.g. always answering "invalid", or formatting tricks) | Balanced verdict classes, format checks, reading samples every N steps (M23) |
| People read it as legal or claims advice | A clear "research, not advice" notice on the README, model cards and demo |
| Data terms and privacy | Publish scripts and derived labels, not scraped text; strip names; no company data |
| Too much scope | Phases cut from the bottom; each phase ships something on its own |

## 9. Decisions before week 1

1. **Domain:** ClaimCheck (recommended) or a generic task (text-to-SQL, tool calling) that's
   easier to explain to anyone but less differentiated.
2. **Repo names and GitHub account:** confirm both repos go public from day one.
3. **Gaido IP:** confirm nothing you build reuses Gaido code, prompts or data. Domain
   knowledge in your head is fine; company assets aren't.
4. ~~**Whose project is this?**~~ **Decided 2026-10-22:** published openly as founder-led
   work that promotes both you and Policygaido, trained on Policygaido's GCP credits. The one
   fixed limit: real customers' claim documents are their health data and never go into a
   public dataset or model. Public artefacts use synthetic data and public sources only.
