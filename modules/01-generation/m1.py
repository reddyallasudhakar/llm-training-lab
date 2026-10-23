"""Module 1 exercises: how LLMs generate.

Fill in each TODO, then run `uv run python modules/01-generation/check_m1.py`.
Experiment freely in a REPL or notebook: `from m1 import *; tok, model = load()`.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
DEVICE = "mps"


def load():
    """Return (tokenizer, model). float32 so greedy decoding is exactly reproducible."""
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).to(DEVICE).eval()
    return tok, model


def chat_ids(tok, user_message):
    """Wrap a user message in the chat template; return input ids of shape [1, n] on DEVICE."""
    messages = [{"role": "user", "content": user_message}]
    enc = tok.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt", return_dict=True)
    return enc["input_ids"].to(DEVICE)


# --- Exercise 1: tokens -------------------------------------------------------


def show_tokens(tok, text):
    """Return a list of (token_id, token_text) pairs for `text`.

    Hint: tok.encode(text) gives ids; tok.decode([one_id]) gives that token's text.
    """
    raise NotImplementedError  # TODO


# --- Exercise 2: the next-token distribution ----------------------------------


@torch.no_grad()
def next_token_probs(model, input_ids):
    """Return a 1-D tensor of probabilities over the whole vocabulary for the NEXT token.

    model(input_ids).logits has shape [batch, seq_len, vocab]. You want the last position.
    """
    raise NotImplementedError  # TODO


# --- Exercise 3: sampling -----------------------------------------------------


def sample_next(logits, temperature=1.0, top_p=1.0, generator=None):
    """Pick one token id (an int) from a 1-D tensor of logits.

    - temperature == 0: greedy, return the argmax.
    - otherwise: divide logits by temperature, softmax, then apply top-p (nucleus):
      sort probabilities descending, keep the smallest set whose cumulative sum reaches
      top_p (include the token that crosses the threshold), renormalise, and sample.
    Use torch.multinomial(probs, 1, generator=generator) for the draw.
    """
    raise NotImplementedError  # TODO


# --- Exercise 4: your own generate() ------------------------------------------


@torch.no_grad()
def generate(model, tok, input_ids, max_new_tokens=50, temperature=0.0, top_p=1.0, generator=None):
    """Autoregressive loop: return the list of NEW token ids (not the prompt).

    Each step: run the model on everything so far, take the last position's logits,
    sample_next(), append, and stop early if the token is an end-of-sequence id.
    The stop ids are in model.generation_config.eos_token_id (an int or a list).
    No KV cache here: recompute the whole sequence every step. That's slow, on purpose.
    """
    raise NotImplementedError  # TODO


# --- Exercise 5: scoring a completion -----------------------------------------


@torch.no_grad()
def completion_logprob(model, tok, prompt_ids, completion):
    """Return log P(completion | prompt) as a float: the sum of each completion token's logprob.

    Tokenise `completion` on its own (tok.encode, no special tokens) and append it to prompt_ids.
    One forward pass gives logits at every position; the logits at position t predict token t+1.
    Use torch.log_softmax, then pick out the logprob of each actual completion token.
    This exact quantity is the SFT loss (negated) and what RL pushes up or down.
    """
    raise NotImplementedError  # TODO


# --- Exercise 6 (stretch): KV cache -------------------------------------------


@torch.no_grad()
def generate_cached(model, tok, input_ids, max_new_tokens=50, temperature=0.0, top_p=1.0, generator=None):
    """Same as generate(), but feed only the newest token each step and reuse the cache.

    First call: out = model(input_ids=input_ids, use_cache=True).
    Then each step: out = model(input_ids=<just the new token, shape [1, 1]>,
                                past_key_values=out.past_key_values, use_cache=True).
    """
    raise NotImplementedError  # TODO (stretch)
