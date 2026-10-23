"""Checks for Module 1. Run: uv run python modules/01-generation/check_m1.py"""

import time

import torch

import m1

results = []


def check(name, fn):
    try:
        fn()
        results.append(("✓", name, ""))
    except NotImplementedError:
        results.append(("·", name, "not started"))
    except AssertionError as e:
        results.append(("✗", name, str(e)))


tok, model = m1.load()
prompt = m1.chat_ids(tok, "Name three rivers in India.")


def ex1():
    pairs = m1.show_tokens(tok, "Claims rejected under clause 4.2")
    assert all(isinstance(i, int) for i, _ in pairs), "ids should be ints"
    assert "".join(t for _, t in pairs) == "Claims rejected under clause 4.2", "pieces should rebuild the text"


def ex2():
    probs = m1.next_token_probs(model, prompt)
    assert probs.dim() == 1, f"expected a 1-D tensor, got shape {tuple(probs.shape)}"
    assert abs(probs.sum().item() - 1) < 1e-3, "probabilities should sum to 1"
    ref = model(prompt).logits[0, -1].argmax().item()
    assert probs.argmax().item() == ref, "most likely token doesn't match the model's"


def ex3():
    logits = torch.tensor([2.0, 1.0, 0.5, 0.0, -1.0])
    g = torch.Generator().manual_seed(0)
    assert all(m1.sample_next(logits, 0, generator=g) == 0 for _ in range(20)), "temperature 0 must be argmax"

    def freqs(**kw):
        draws = torch.tensor([m1.sample_next(logits, generator=g, **kw) for _ in range(20000)])
        return torch.bincount(draws, minlength=5).float() / len(draws)

    for t in (1.0, 0.5):
        f, want = freqs(temperature=t), torch.softmax(logits / t, -1)
        assert (f - want).abs().max() < 0.02, f"T={t}: got {f.tolist()}, want {want.tolist()}"

    # softmax ≈ [.536, .197, .120, .072, .027]; top_p=0.7 keeps tokens 0 and 1 only
    f = freqs(temperature=1.0, top_p=0.7)
    assert f[2:].sum() == 0, f"top_p=0.7 should never pick tokens 2–4, got {f.tolist()}"
    assert abs(f[0] - 0.731) < 0.02, f"top_p=0.7 should renormalise to ~[.73, .27], got {f.tolist()}"


def ex4():
    for msg in ["Name three rivers in India.", "What is 17 * 23?", "Write a haiku about claims."]:
        ids = m1.chat_ids(tok, msg)
        ref = model.generate(ids, max_new_tokens=30, do_sample=False)[0, ids.shape[1]:].tolist()
        mine = m1.generate(model, tok, ids, max_new_tokens=30)
        assert mine == ref, f"greedy output differs for {msg!r}:\n  yours: {tok.decode(mine)!r}\n  ref:   {tok.decode(ref)!r}"


def ex5():
    completion = "Ganga, Yamuna and Godavari."
    got = m1.completion_logprob(model, tok, prompt, completion)
    comp = torch.tensor([tok.encode(completion, add_special_tokens=False)], device=prompt.device)
    full = torch.cat([prompt, comp], 1)
    labels = full.clone()
    labels[:, : prompt.shape[1]] = -100  # ignore prompt tokens in the loss
    want = -model(full, labels=labels).loss.item() * comp.shape[1]
    assert abs(got - want) < 1e-2, f"got {got:.3f}, want {want:.3f}"


def ex6():
    ids = m1.chat_ids(tok, "Explain what a deductible is.")
    t0 = time.perf_counter()
    slow = m1.generate(model, tok, ids, max_new_tokens=60)
    t1 = time.perf_counter()
    fast = m1.generate_cached(model, tok, ids, max_new_tokens=60)
    t2 = time.perf_counter()
    assert fast == slow, "cached output should match uncached output exactly"
    print(f"   uncached {t1 - t0:.2f}s vs cached {t2 - t1:.2f}s")


for name, fn in [
    ("1 show_tokens", ex1),
    ("2 next_token_probs", ex2),
    ("3 sample_next", ex3),
    ("4 generate (matches model.generate)", ex4),
    ("5 completion_logprob (matches the SFT loss)", ex5),
    ("6 generate_cached (stretch)", ex6),
]:
    check(name, fn)

print()
for mark, name, msg in results:
    print(f"{mark} {name}" + (f" — {msg}" if msg else ""))
